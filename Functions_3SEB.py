import numpy as np
from collections import deque
import time

from pyTSEB import TSEB
from pyTSEB import meteo_utils as met
from pyTSEB import resistances as res
from pyTSEB import MO_similarity as MO
from pyTSEB import net_radiation as rad
from pyTSEB import clumping_index as CI


# ==============================================================================
# List of constants used in TSEB/3SEB model and sub-routines
# ==============================================================================
# Threshold for relative change in Monin-Obukhov lengh to stop the iterations
L_thres = 0.001
# mimimun allowed friction velocity
u_friction_min = 0.01
# Maximum number of interations
ITERATIONS = 15
# kB coefficient
kB = 0.0
# Stephan Boltzmann constant (W m-2 K-4)
sb = 5.670373e-8
# von Karman's constant
k = 0.4

# Resistance formulation constants
KUSTAS_NORMAN_1999 = 0
CHOUDHURY_MONTEITH_1988 = 1
MCNAUGHTON_VANDERHURK = 2
CHOUDHURY_MONTEITH_ALPHA_1988 = 3
HADHIGHI_AND_OR_2015 = 4

# Soil heat flux formulation constants
G_CONSTANT = 0
G_RATIO = 1
G_TIME_DIFF = 2
G_TIME_DIFF_SIGMOID = 3

# Flag constants
F_ALL_FLUXES = 0  # All fluxes produced with no reduction of PT parameter (i.e. positive soil evaporation)
F_ZERO_LE_C = 1  # Negative canopy latent heat flux, forced to zero
F_ZERO_H_C = 2  # Negative canopy sensible heat flux, forced to zero
F_ZERO_LE_S = 3  # Negative soil evaporation, forced to zero (the PT parameter is reduced in TSEB-PT and DTD)
F_ZERO_H_S = 4  # Negative soil sensible heat flux, forced to zero
F_ZERO_LE = 5  # No positive latent fluxes found, G recomputed to close the energy balance (G=Rn-H)
F_ALL_FLUXES_OS = 10  # All positive fluxes for soil only, produced using one-source energy balance (OSEB) model.
F_ZERO_LE_OS = 15  # No positive latent fluxes found using OSEB, G recomputed to close the energy balance (G=Rn-H)
F_INVALID = 255  # Arithmetic error. BAD data, it should be discarded



# ==============================================================================
# Function for 3SEB using patch-layer structure with a two (nested) alpha_PT
# ==============================================================================
def ThreeSEB_PT(Tr_K,
                vza,
                T_A_K,
                u,
                ea,
                p,
                Sn_C,
                Sn_S,
                Sn_C_sub,
                L_dn,
                LAI,
                LAI_sub,
                h_C,
                h_C_sub,
                emis_C,
                emis_sub,
                emis_S,
                z_0M,
                z_0M_sub,
                d_0,
                d_0_sub,
                z_u,
                z_T,
                leaf_width=0.01,
                leaf_width_sub=0.01,
                z0_soil=0.01,
                alpha_PT=1.26,
                x_LAD=1,
                x_LAD_sub=1,
                f_c=1.0,
                f_c_sub=1.0,
                f_g=1.0,
                f_g_sub=1.0,
                w_C=1.0,
                w_C_sub=1.0,
                resistance_form=[0, {}],
                calcG_params=[[1],0.35],
                massman_profile=[0.0, []],
                const_L=None):

    '''Three-source model (3SEB) with two nested Priestley-Taylor assumptions

    Calculates the fluxes using a single observation of
    composite radiometric temperature and with a combined patch-layer model using resistances in parrelel for primary foliage and series for substrate foliage.

    Parameters
    ----------
    Tr_K : float
        Radiometric composite temperature (Kelvin).
    vza : float
        View Zenith Angle (degrees).
    T_A_K : float
        Air temperature (Kelvin).
    u : float
        Wind speed above the canopy (m s-1).
    ea : float
        Water vapour pressure above the canopy (mb).
    p : float
        Atmospheric pressure (mb), use 1013 mb by default.
    Sn_C : float
        Primary Canopy net shortwave radiation (W m-2).
    Sn_S : float
        Soil net shortwave radiation (W m-2).
    Sn_C_sub: float
        Secondary canopy (substrate) net shortwave radiation (W m-2)
    L_dn : float
        Downwelling longwave radiation (W m-2).
    LAI : float
        Effective Leaf Area Index of primary/overstory foliage(m2 m-2).
    LAI_sub: float
        Effective Leaf Area Index of secondary/understory foliage(m2 m-2).
    h_C : float
        Canopy height of primary vegetation(m).
    h_C_sub : float
        Canopy height of secondary vegetation (m).
    emis_C : float
        Leaf emissivity.
    emis_sub: float
        understory emissivity
    emis_S : flaot
        Soil emissivity.
    z_0M : float
        Aerodynamic surface roughness length for momentum transfer of primary canopy(m).
    z_0M_sub: float
        Aerodynamic surface roughness length for momentum transfer of secondary canopy (m).
    d_0 : float
        Zero-plane displacement height of primary canopy(m).
    d_0_sub: float
        Zero-plane displacement height of secondary canopy(m).
    z_u : float
        Height of measurement of windspeed (m).
    z_T : float
        Height of measurement of air temperature (m).
    leaf_width : float, optional
        average/effective leaf width of primary canopy (m).
    leaf_width_sub : float, optional
        average/effective leaf width of seconday canopy (m).
    z0_soil : float, optional
        bare soil aerodynamic roughness length (m).
    alpha_PT : float, optional
        Priestley Taylor coeffient for canopy potential transpiration,
        use 1.26 by default.
    x_LAD : float, optional
        Campbell 1990 leaf inclination distribution function chi parameter.
    f_c : float, optional
        Fractional cover of primary canopy.
    f_c_sub : float, optional
        Fractional cover of secondary canopy.
    f_g : float, optional
        Fraction of primary vegetation that is green.
    f_g_sub : float, optional
        Fraction of secondary vegetation that is green.
    w_C : float, optional
        Primary canopy width to height ratio.
    w_C_sub : float, optional
        Secondary canopy width to height ratio.
    resistance_form : int, optional
        Flag to determine which Resistances R_x, R_S model to use.
            * 0 [Default] Norman et al 1995 and Kustas et al 1999.
            * 1 : Choudhury and Monteith 1988.
            * 2 : McNaughton and Van der Hurk 1995.
    calcG_params : list[list,float or array], optional
        Method to calculate soil heat flux,parameters.
            * [[1],G_ratio]: default, estimate G as a ratio of Rn_S, default Gratio=0.35.
            * [[0],G_constant] : Use a constant G, usually use 0 to ignore the computation of G.
            * [[2,Amplitude,phase_shift,shape],time] : estimate G from Santanello and Friedl with
                                                       G_param list of parameters
                                                       (see :func:`~TSEB.calc_G_time_diff`).
    const_L : float or None, optional
        If included, its value will be used to force the Moning-Obukhov stability length.
    Returns
    -------
    flag : int
        Quality flag, see Appendix for description.
    T_S : float
        Soil temperature  (Kelvin).
    T_C : float
        Canopy temperature  (Kelvin).
    T_C_sub: float
        Secondary canopy temperature  (Kelvin).
    T_AC : float
        Air temperature at the canopy interface (Kelvin).
    L_nS : float
        Soil net longwave radiation (W m-2)
    L_nC : float
        Canopy net longwave radiation (W m-2)
    L_nC_sub : float
        Secondary canopy net longwave radiation (W m-2
    LE_C : float
        Canopy latent heat flux (W m-2).
    H_C : float
        Canopy sensible heat flux (W m-2).
    LE_C_sub: float
        Secondary canopy latent heat flux (W m-2).
    H_C_sub: float
        Secondary Canopy sensible heat flux (W m-2
    LE_S : float
        Soil latent heat flux (W m-2).
    H_S : float
        Soil sensible heat flux (W m-2).
    G : float
        Soil heat flux (W m-2).
    R_S : float
        Soil aerodynamic resistance to heat transport (s m-1).
    R_x : float
        Bulk canopy aerodynamic resistance to heat transport (s m-1).
    R_A : float
        Aerodynamic resistance to heat transport (s m-1).
    u_friction : float
        Friction velocity (m s-1).
    L : float
        Monin-Obuhkov length (m).
    n_iterations : int
        number of iterations until convergence of L.
    alpha_final: float
        Retrieved alpha PT value of primary canopy (-)
    alpha_final_sub: float
        Retrieved alpha PT value of secondary canopy (-)

'''
    # Convert input float scalars to arrays and parameters size
    Tr_K = np.asarray(Tr_K)
    (vza,
     T_A_K,
     u,
     ea,
     p,
     Sn_C,
     Sn_S,
     Sn_C_sub,
     L_dn,
     LAI,
     LAI_sub,
     h_C,
     h_C_sub,
     emis_C,
     emis_sub,
     emis_S,
     z_0M,
     z_0M_sub,
     d_0,
     d_0_sub,
     z_u,
     z_T,
     leaf_width,
     leaf_width_sub,
     z0_soil,
     alpha_PT,
     x_LAD,
     x_LAD_sub,
     f_c,
     f_c_sub,
     f_g,
     f_g_sub,
     w_C,
     w_C_sub,
     calcG_array) = map(_check_default_parameter_size,
                        [vza,
                         T_A_K,
                         u,
                         ea,
                         p,
                         Sn_C,
                         Sn_S,
                         Sn_C_sub,
                         L_dn,
                         LAI,
                         LAI_sub,
                         h_C,
                         h_C_sub,
                         emis_C,
                         emis_sub,
                         emis_S,
                         z_0M,
                         z_0M_sub,
                         d_0,
                         d_0_sub,
                         z_u,
                         z_T,
                         leaf_width,
                         leaf_width_sub,
                         z0_soil,
                         alpha_PT,
                         x_LAD,
                         x_LAD_sub,
                         f_c,
                         f_c_sub,
                         f_g,
                         f_g_sub,
                         w_C,
                         w_C_sub,
                         calcG_params[1]],
                        [Tr_K] * 35)
    res_params = resistance_form[1]
    resistance_form = resistance_form[0]
    # Create the output variables
    [T_AC, L_n_sub, L_nC, H, LE, LE_sub, H_sub, LE_C, H_C, LE_C_sub, H_C_sub, LE_S, H_S, G, R_S,R_sub, R_x, R_A, delta_Rn, Rn_sub, Rn_C_sub,
      Ln_C_sub, Ln_S, Rn_S, alpha_final, alpha_final_sub, iterations] = [np.zeros(Tr_K.shape) + np.NaN for i in range(27)]

    Sn_sub = Sn_S + Sn_C_sub
    # iteration of the Monin-Obukhov length
    if const_L is None:
        # Initially assume stable atmospheric conditions and set variables for
        L = np.asarray(np.zeros(Tr_K.shape) + np.inf)
        max_iterations = ITERATIONS
    else:  # We force Monin-Obukhov lenght to the provided array/value
        L = np.asarray(np.ones(Tr_K.shape) * const_L)
        max_iterations = 1  # No iteration

    # Calculate the general parameters
    rho = met.calc_rho(p, ea, T_A_K)  # Air density
    c_p = met.calc_c_p(p, ea)  # Heat capacity of air
    z_0H = res.calc_z_0H(z_0M, kB=kB)  # Roughness length for heat transport

    # Calculate LAI dependent parameters for primary canopy where LAI > 0
    omega0 = CI.calc_omega0_Kustas(LAI, f_c, x_LAD=x_LAD, isLAIeff=True)
    F = np.asarray(LAI / f_c)  # Real LAI
    # Fraction of vegetation observed by the sensor
    f_theta = TSEB.calc_F_theta_campbell(vza, F, w_C=w_C, Omega0=omega0, x_LAD=x_LAD)

    # Calculate LAI dependent parameters for substrate canopy where LAI > 0
    omega0_sub = CI.calc_omega0_Kustas(LAI_sub, f_c_sub, x_LAD=x_LAD_sub, isLAIeff=True)
    F_sub = np.asarray(LAI_sub / f_c_sub)  # Real LAI
    # Fraction of vegetation observed by the sensor
    f_theta_sub = TSEB.calc_F_theta_campbell(vza, F_sub, w_C=w_C_sub, Omega0=omega0_sub, x_LAD=x_LAD_sub)

    # Initially assume stable atmospheric conditions and set variables for
    # iteration of the Monin-Obukhov length
    # assume  atmospheric conditions the same throughout soil-grass-tree system and
    #  calculating friction velocity and related variables with tree characteristics
    u_friction = MO.calc_u_star(u, z_u, L, d_0, z_0M)
    u_friction = np.asarray(np.maximum(u_friction_min, u_friction))

    L_queue = deque([np.array(L)], 6)
    L_converged = np.asarray(np.zeros(Tr_K.shape)).astype(bool)
    L_diff_max = np.inf

    ### patch-layer model --> initial assumption that T_C, T_C_sub and T_S equals T_A
    # First assume that canopy temperature (T_C) equals the minumum of Air or radiometric T
    T_C = np.asarray(np.minimum(Tr_K, T_A_K))
    #T_sub = Tr_K - (f_theta*T_C)/(1-f_theta)
    flag, T_sub = TSEB.calc_T_S(Tr_K, T_C, f_theta)

    # Assume first guess that T_C_sub is mean between T_A and T_sub
    T_C_sub = (T_sub + T_A_K)/2
    flag, T_S = TSEB.calc_T_S(T_sub, T_C_sub, f_theta_sub)

    # Outer loop for estimating stability.
    # Stops when difference in consecutives L is below a given threshold
    start_time = time.time()
    loop_time = time.time()
    for n_iterations in range(max_iterations):
        i = flag != F_INVALID
        j = flag != F_INVALID
        if np.all(L_converged[i]):
            if L_converged[i].size == 0:
                print("Finished iterations with no valid solution")
            else:
                print(f"Finished interations with a max. L diff: {L_diff_max}")
            break
        current_time = time.time()
        loop_duration = current_time - loop_time
        loop_time = current_time
        total_duration = loop_time - start_time
        print("Iteration: %d, non-converged pixels: %d, max L diff: %f, total time: %f, loop time: %f" %(n_iterations, np.sum(~L_converged[i]), L_diff_max, total_duration, loop_duration))
        iterations[np.logical_and(~L_converged, flag != F_INVALID)] = n_iterations

        # Double inner loop to iterativelly reduce alpha_PT in case latent heat flux
        # from the soil or substrate is negative. The initial assumption is of potential
        # canopy transpiration. Nested loop with different alpha for overstory vegetation and understory vegetation.

        flag[np.logical_and(~L_converged, flag != F_INVALID)] = F_ALL_FLUXES
        LE_S[np.logical_and(~L_converged, flag != F_INVALID)] = -1
        LE_sub[np.logical_and(~L_converged, flag != F_INVALID)] = -1

        alpha_PT_rec = np.asarray(alpha_PT + 0.1)
        while np.any(LE_sub[i] < 0):
            i = np.logical_and.reduce((LE_sub < 0, ~L_converged, flag != F_INVALID))
            alpha_PT_rec[i] -= 0.1

            # There cannot be negative transpiration from the vegetation
            alpha_PT_rec[alpha_PT_rec <= 0.0] = 0.0
            flag[np.logical_and(i, alpha_PT_rec == 0.0)] = F_ZERO_LE

            flag[np.logical_and.reduce((i, alpha_PT_rec < alpha_PT, alpha_PT_rec > 0.0))] = F_ZERO_LE_S
            #store alpha value for primary canopy in output array
            alpha_final[i] = alpha_PT_rec[i]



            #calculate aerodynamic resistances (using parallel kustas99 approach)

            R_A[i],_, R_sub[i] = TSEB.calc_resistances(resistance_form,
                                                      {"R_A": {"z_T": z_T[i], "u_friction": u_friction[i], "L": L[i],
                                                               "d_0": d_0[i], "z_0H": z_0H[i]},

                                                       "R_S": {"u_friction": u_friction[i], "h_C": h_C[i],
                                                               "d_0": d_0[i],
                                                               "z_0M": z_0M[i], "L": L[i], "F": F[i],
                                                               "omega0": omega0[i],
                                                               "LAI": LAI[i], "leaf_width": leaf_width[i],"massman_profile": massman_profile,
                                                               "z0_soil": z0_soil[i], "z_u": z_u[i],
                                                               "deltaT": T_sub[i] - T_C[i], 'u': u[i], 'rho': rho[i],
                                                               "c_p": c_p[i], "f_cover": f_c[i], "w_C": w_C[i],
                                                               "res_params": {k: res_params[k][i] for k in
                                                                             res_params.keys()}}
                                                       }
                                                      )


            # Calculate net longwave radiation with current values of T_C, T_C_sub and T_S
            L_nC[i], L_n_sub[i] = rad.calc_L_n_Campbell(T_C[i], T_sub[i], L_dn[i], LAI[i], emis_C[i], emis_sub[i], x_LAD=x_LAD[i])
            # maybe check for convergance of radiation between layers
            delta_Rn[i] = Sn_C[i] + L_nC[i]
            Rn_sub[i] = Sn_sub[i] + L_n_sub[i]



            # apply Cambpell longwave transmittance on substrate
            # (calculate sub-canopy LW and soil LW radiation)
            Ln_C_sub[i], Ln_S[i] = calc_Ln_substrate_Campbell(L_n_sub[i],
                                                              LAI_sub[i],
                                                              x_LAD=LAI_sub[i],
                                                              emiss_c=emis_sub[i],
                                                              emiss_s=emis_S[i])

            Rn_C_sub[i] = Sn_C_sub[i] + Ln_C_sub[i]
            Rn_S[i] = Sn_S[i] + Ln_S[i]
            # Calculate the canopy and subtrate temperatures using the Priestley Taylor approach
            H_C[i] = calc_H_C_PT(
                delta_Rn[i],
                f_g[i],
                T_A_K[i],
                p[i],
                c_p[i],
                alpha_PT_rec[i])

            #get primary canopy temperature with parallel approach
            T_C[i] = calc_T_C_Parallel(H_C[i], R_A[i], T_A_K[i], rho[i], R_sub[i])

            # Calculate substrate temperature
            flag_t = np.zeros(flag.shape) + F_ALL_FLUXES
            flag_t[i], T_sub[i] = TSEB.calc_T_S(Tr_K[i], T_C[i], f_theta[i])
            flag[flag_t == F_INVALID] = F_INVALID
            LE_sub[flag_t == F_INVALID] = 0

            # Recalculate soil resistance using new substrate and canopy temperature
            _, _, R_sub[i] = TSEB.calc_resistances(resistance_form,
                                            {
                                             "R_S": {"u_friction": u_friction[i], "h_C": h_C[i], "d_0": d_0[i],
                                                     "z_0M": z_0M[i], "L": L[i], "F": F[i], "omega0": omega0[i],
                                                     "LAI": LAI[i], "leaf_width": leaf_width[i],"massman_profile": massman_profile,
                                                     "z0_soil": z0_soil[i], "z_u": z_u[i],
                                                     "deltaT": T_sub[i] - T_C[i], "u": u[i], "rho": rho[i],
                                                     "c_p": c_p[i], "f_cover": f_c[i], "w_C": w_C[i],
                                                     "res_params": {k: res_params[k][i] for k in res_params.keys()}}
                                             }
                                            )

            i = np.logical_and.reduce((LE_sub < 0, ~L_converged, flag != F_INVALID))#not exactly sure what this does....

            #get subtrate H flux
            H_sub[i] = rho[i] * c_p[i] * ((T_sub[i] - T_A_K[i]) / (R_sub[i]+R_A[i]))

            # Compute Soil Heat Flux Ratio
            G[i] = TSEB.calc_G([calcG_params[0], calcG_array], Rn_S, i)

            # Estimate latent heat fluxes as residual of energy balance at the
            # soil and the canopy
            LE_sub[i] = Rn_sub[i] - G[i] - H_sub[i]
            LE_C[i] = delta_Rn[i] - H_C[i]

            # Special case if there is no transpiration from vegetation.
            # In that case, there should also be no evaporation from the subsrate
            # and the energy at the soil should be conserved.
            # See end of appendix A1 in Guzinski et al. (2015).
            noT = np.logical_and(i, LE_C == 0)
            H_sub[noT] = np.minimum(H_sub[noT], Rn_C_sub[noT]+Rn_S[noT] - G[noT])
            G[noT] = np.maximum(G[noT], Rn_C_sub[noT]+Rn_S[noT] - H_S[noT])
            LE_sub[noT] = 0
            #alpha_PT[:] = 1.26
            alpha_PT_rec_sub = np.asarray(alpha_PT + 0.1)
            while np.any(LE_S[j] < 0):
                # now do series approach to get sub canopy vegetation and soil temperatures
                j = np.logical_and(i, LE_S < 0)
                alpha_PT_rec_sub[j] -= 0.1
                # There cannot be negative transpiration from the (sub)-vegetation
                alpha_PT_rec_sub[alpha_PT_rec_sub <= 0.0] = 0.0
                flag[np.logical_and(j, alpha_PT_rec_sub == 0.0)] = F_ZERO_LE
                flag[np.logical_and.reduce((j, alpha_PT_rec_sub < alpha_PT, alpha_PT_rec_sub > 0.0))] = F_ZERO_LE_S

                H_C_sub[j] = calc_H_C_PT(
                    Rn_C_sub[j],
                    f_g_sub[j],
                    T_A_K[j],
                    p[j],
                    c_p[j],
                    alpha_PT_rec_sub[j])

                #re-calculate resistances with secondary canopy parameters
                ## you can adjust b coeficient in KN99 Rs calc for different soil conditions
                ## for now hard-coded, later adjust to make it an input/parameter
                ##i.e.
                ###b_coef = np.ones(Tr_K.shape) * 0.012
                ###res_params = {'KN_b': b_coef}

                _, R_x[j], R_S[j] = TSEB.calc_resistances(resistance_form,
                                                               {
                                                                "R_x": {"u_friction": u_friction[j], "h_C": h_C_sub[j],
                                                                        "d_0": d_0_sub[j],
                                                                        "z_0M": z_0M_sub[j], "L": L[j], "F": F_sub[j],
                                                                        "LAI": LAI_sub[j],
                                                                        "leaf_width": leaf_width_sub[j],
                                                                        "massman_profile": massman_profile,
                                                                        "res_params": {k: res_params[k][j] for k in res_params.keys()}},
                                                                "R_S": {"u_friction": u_friction[j], "h_C": h_C_sub[j],
                                                                        "d_0": d_0_sub[j],
                                                                        "z_0M": z_0M_sub[j], "L": L[j], "F": F_sub[j],
                                                                        "omega0": omega0[j],
                                                                        "LAI": LAI_sub[j], "leaf_width": leaf_width_sub[j],
                                                                        "massman_profile": massman_profile,
                                                                        "z0_soil": z0_soil[j], "z_u": z_u[j],
                                                                        "deltaT": T_S[j] - T_C_sub[j], 'u': u[j],
                                                                        'rho': rho[j],
                                                                        "c_p": c_p[j], "f_cover": f_c_sub[j],
                                                                        "w_C": w_C_sub[j],
                                                                        "res_params": {k: res_params[k][j] for k in res_params.keys()}}
                                                                }
                                                               )

                T_C_sub[j] = TSEB.calc_T_C_series(T_sub[j], T_A_K[j], R_A[j], R_x[j], R_S[j],f_theta_sub[j], H_C_sub[j], rho[j], c_p[j])

                # Calculate soil temperature
                flag_t = np.zeros(flag.shape) + F_ALL_FLUXES
                flag_t[j], T_S[j] = TSEB.calc_T_S(T_sub[j], T_C_sub[j], f_theta[j])
                flag[flag_t == F_INVALID] = F_INVALID
                LE_S[flag_t == F_INVALID] = 0

                # Recalculate soil resistance using new soil temperature
                _, R_x[j], R_S[j] = TSEB.calc_resistances(resistance_form,
                                                {"R_x": {"u_friction": u_friction[j], "h_C": h_C_sub[j],
                                                         "d_0": d_0_sub[j], "massman_profile": massman_profile,
                                                         "z_0M": z_0M_sub[j], "L": L[j], "F": F_sub[j], "LAI": LAI_sub[j],
                                                         "leaf_width": leaf_width_sub[j], "res_params": {k: res_params[k][j] for k in res_params.keys()}},

                                                    "R_S": {"u_friction": u_friction[j], "h_C": h_C_sub[j], "d_0": d_0_sub[j],
                                                         "z_0M": z_0M_sub[j], "L": L[j], "F": F_sub[j], "omega0": omega0_sub[j],
                                                         "LAI": LAI_sub[j], "leaf_width": leaf_width_sub[j],"massman_profile": massman_profile,
                                                         "z0_soil": z0_soil[j], "z_u": z_u[j],
                                                         "deltaT": T_S[j] - T_C_sub[j], "u": u[j], "rho": rho[j],
                                                         "c_p": c_p[j], "f_cover": f_c_sub[j], "w_C": w_C_sub[j],
                                                         "res_params": {k: res_params[k][j] for k in res_params.keys()}}
                                                 }
                                                )
                j = np.logical_and(i, LE_S < 0)
                # Get air temperature at canopy interface
                T_AC[j] = ((T_A_K[j] / R_A[j] + T_S[j] / R_S[j] + T_C_sub[j] / R_x[j])
                           / (1.0 / R_A[j] + 1.0 / R_S[j] + 1.0 / R_x[j]))

                #calcualte substrate canopy H
                H_C_sub[j] = rho[j] * c_p[j] * (T_C_sub[j] - T_AC[j]) / R_x[j]
                # Calculate soil fluxes
                H_S[j] = rho[j] * c_p[j] * (T_S[j] - T_AC[j]) / R_S[j]

                # Compute Soil Heat Flux Ratio
                G[j] = TSEB.calc_G([calcG_params[0], calcG_array], Rn_S, j)

                # Estimate latent heat fluxes as residual of energy balance at the primary, secondary and soil sources
                LE_S[j] = Rn_S[j] - G[j] - H_S[j]
                LE_C_sub[j] = Rn_C_sub[j] - H_C_sub[j]
                # store alpha value for secondary canopy in output array
                alpha_final_sub[j] = alpha_PT_rec_sub[j]

                if np.any(alpha_PT_rec_sub[j] <=0):
                    LE_C_sub[j] = 0
                    H_C_sub[j] =  Rn_C_sub[j]

                # Special case if there is no transpiration from vegetation.
                # In that case, there should also be no evaporation from the soil
                # and the energy at the soil should be conserved.
                # See end of appendix A1 in Guzinski et al. (2015).
                noT = np.logical_and(j, LE_C_sub == 0)
                H_S[noT] = np.minimum(H_S[noT], Rn_S[noT] - G[noT])
                G[noT] = np.maximum(G[noT], Rn_S[noT] - H_S[noT])
                LE_S[noT] = 0

                # Calculate total fluxes
                H[j] = np.asarray(H_C[j] + H_C_sub[j] + H_S[j])
                LE[j] = np.asarray(LE_C[j] +LE_C_sub[j]+ LE_S[j])

            # Now L can be recalculated and the difference between iterations
            # derived
            if const_L is None:
                L[i] = MO.calc_L(
                    u_friction[i],
                    T_A_K[i],
                    rho[i],
                    c_p[i],
                    H[i],
                    LE[i])
                # Calculate again the friction velocity with the new stability
                # corrections
                u_friction[i] = MO.calc_u_star(u[i], z_u[i], L[i], d_0[i], z_0M[i])
                u_friction[i] = np.asarray(np.maximum(u_friction_min, u_friction[i]))

        if const_L is None:
            # We check convergence against the value of L from previous iteration but as well
            # against values from 2 or 3 iterations back. This is to catch situations (not
            # infrequent) where L oscillates between 2 or 3 steady state values.
            L_new = np.array(L)
            L_new[L_new == 0] = 1e-36
            L_queue.appendleft(L_new)
            i = np.logical_and(~L_converged, flag != F_INVALID)
            if not np.any(i):
                continue
            L_converged[i] = TSEB._L_diff(L_queue[0][i], L_queue[1][i]) < L_thres
            L_diff_max = np.max(TSEB._L_diff(L_queue[0][i], L_queue[1][i]))
            if len(L_queue) >= 4:
                i = np.logical_and(~L_converged, flag != F_INVALID)
                if not np.any(i):
                    continue
                L_converged[i] = np.logical_and(TSEB._L_diff(L_queue[0][i], L_queue[2][i]) < L_thres,
                                                TSEB._L_diff(L_queue[1][i], L_queue[3][i]) < L_thres)
            if len(L_queue) == 6:
                i = np.logical_and(~L_converged, flag != F_INVALID)
                if not np.any(i):
                    continue
                L_converged[i] = np.logical_and.reduce((TSEB._L_diff(L_queue[0][i], L_queue[3][i]) < L_thres,
                                                        TSEB._L_diff(L_queue[1][i], L_queue[4][i]) < L_thres,
                                                        TSEB._L_diff(L_queue[2][i], L_queue[5][i]) < L_thres))
    print('\n..Finished..\n')

    (flag,
     T_S,
     T_C_sub,
     T_C,
     T_AC,
     L_n_sub,
     L_nC,
     Rn_C_sub,
     Rn_S,
     LE_C,
     H_C,
     LE_sub,
     H_sub,
     LE_C_sub,
     H_C_sub,
     LE_S,
     H_S,
     G,
     R_S,
     R_sub,
     R_x,
     R_A,
     u_friction,
     L,
     n_iterations, alpha_final, alpha_final_sub) = map(np.asarray,
                         (flag,
                         T_S,
                         T_C_sub,
                         T_C,
                         T_AC,
                         L_n_sub,
                         L_nC,
                         Rn_C_sub,
                         Rn_S,
                         LE_C,
                         H_C,
                         LE_sub,
                         H_sub,
                         LE_C_sub,
                         H_C_sub,
                         LE_S,
                         H_S,
                         G,
                         R_S,
                         R_sub,
                         R_x,
                         R_A,
                         u_friction,
                         L,
                         n_iterations, alpha_final, alpha_final_sub))

    return flag, T_S, T_C, T_C_sub, T_AC, L_n_sub, L_nC, Ln_C_sub, Ln_S,  LE_C, H_C, LE_C_sub, H_C_sub, LE_S, H_S, G, R_S, R_sub, R_x, R_A, u_friction,L, n_iterations


def calc_T_C_Parallel(H_C, R_A, T_A, rho, c_p):
    '''
    estimates canopy temperature from canopy sensible heat flux and resistances in parallel

    inversion of equation 14 in Norman 1995
    '''

    T_C = (H_C*R_A)/(rho*c_p) + T_A

    return np.asarray(T_C)


def calc_Sn_Campbell(lai, lai_sub, sza, S_dn_dir, S_dn_dif, fvis, fnir, rho_leaf_vis, rho_leaf_vis_sub,
                     tau_leaf_vis, tau_leaf_vis_sub, rho_leaf_nir, rho_leaf_nir_sub, tau_leaf_nir, tau_leaf_nir_sub,
                     rsoilv, rsoiln, hc, hb, wc, fc,
                     x_LAD=1, x_LAD_sub=1, LAI_eff=None, LAI_eff_sub=None):

    ''' Net shortwave radiation transfer for three sources (overstory, understory and soil)

    Estimate net shorwave radiation for soil, oversotry canopy and understory canopy using the [Campbell1998]_
    Radiative Transfer Model, and implemented in [Kustas1999] [Burchard-Levine2021]

    Parameters
    ----------
    lai : float
        Effective overstory Leaf (Plant) Area Index.
    lai_sub: float
        Effective understory Leaf (Plant) Area Index.
    sza : float
        Sun Zenith Angle (degrees).
    S_dn_dir : float
        Broadband incoming beam shortwave radiation (W m-2).
    S_dn_dif : float
        Broadband incoming diffuse shortwave radiation (W m-2).
    fvis : float
        fration of total visible radiation.
    fnir : float
        fraction of total NIR radiation.
    rho_leaf_vis : float
        Broadband leaf bihemispherical reflectance in the visible region (400-700nm).
    tau_leaf_vis : float
        Broadband leaf bihemispherical transmittance in the visible region (400-700nm).
    rho_leaf_nir : float
        Broadband leaf bihemispherical reflectance in the NIR region (700-2500nm).
    tau_leaf_nir : float
        Broadband leaf bihemispherical transmittance in the NIR region (700-2500nm).
    rsoilv : float
        Broadband soil bihemispherical reflectance in the visible region (400-700nm).
    rsoiln : float
        Broadband soil bihemispherical reflectance in the NIR region (700-2500nm).
    hc : float
        Overstory canopy height (m).
    hb : float
        height of the base of overstory canopy where foliage begins (m).
    wc: float
            width to height ratio (-)
    fc : float
            Fractional cover of primary canopy (tree) (-).
    x_lad : float, optional
        Overstory x parameter for the ellipsoildal Leaf Angle Distribution function of
        Campbell 1988 [default=1, spherical LIDF].
    x_lad_sub : float, optional
        Understory x parameter for the ellipsoildal Leaf Angle Distribution function of
        Campbell 1988 [default=1, spherical LIDF].
    LAI_eff : float or None, optional
        if set, its value is the directional effective understory LAI
        to be used in the beam radiation, if set to None we assume homogeneous canopies.
    LAI_eff_sub : float or None, optional
        if set, its value is the directional effective overstory LAI
        to be used in the beam radiation, if set to None we assume homogeneous canopies.

    Returns
    -------
    Sn_C : float
        Overstory canopy net shortwave radiation (W m-2).
    Sn_S : float
        Soil net shortwave radiation (W m-2).
    Sn_sub_C: float
        Understory canopy net shortwave radioation (W m-2)

    References
    ----------
    .. [Campbell1998] Campbell, G. S. & Norman, J. M. (1998), An introduction to environmental
        biophysics. Springer, New York
        https://archive.org/details/AnIntroductionToEnvironmentalBiophysics.
    .. [Kustas1999] Kustas and Norman (1999) Evaluation of soil and vegetation heat
        flux predictions using a simple two-source model with radiometric temperatures for
        partial canopy cover, Agricultural and Forest Meteorology, Volume 94, Issue 1,
        Pages 13-29, http://dx.doi.org/10.1016/S0168-1923(99)00005-2.
    '''

    rho_leaf = np.array((rho_leaf_vis, rho_leaf_nir))
    tau_leaf = np.array((tau_leaf_vis, tau_leaf_nir))
    rho_leaf_sub = np.array((rho_leaf_vis_sub, rho_leaf_nir_sub))
    tau_leaf_sub = np.array((tau_leaf_vis_sub, tau_leaf_nir_sub))
    rho_soil = np.array((rsoilv, rsoiln))

    # Understory reflectance and transmittance
    albb_sub, albd_sub, taubt_sub, taudt_sub = rad.calc_spectra_Cambpell(lai_sub,
                                                     sza,
                                                     rho_leaf_sub,
                                                     tau_leaf_sub,
                                                     rho_soil,
                                                     x_lad=x_LAD_sub,
                                                     lai_eff=LAI_eff_sub)



    # Inital estimate with Rho_soil to get Tree transmittance
    _, _, taubt, taudt = rad.calc_spectra_Cambpell(lai,
                                                     sza,
                                                     rho_leaf,
                                                     tau_leaf,
                                                     rho_soil,
                                                     x_lad=x_LAD,
                                                     lai_eff=LAI_eff)
    #get percent of shaded area on substrate
    f_shaded = calc_shadow_fraction(sza, hc, hb, wc, fc, np.mean(taubt))
    #the direct substrate albedo dominates over the overstory gaps and there are no shadows
    rho_sub = f_shaded * albd_sub + (1 - f_shaded) * albb_sub

    albb, albd, taubt, taudt = rad.calc_spectra_Cambpell(lai,
                                                     sza,
                                                     rho_leaf,
                                                     tau_leaf,
                                                     rho_sub,
                                                     x_lad=x_LAD,
                                                     lai_eff=LAI_eff)


    S_sub_dir_vis = taubt[0] * S_dn_dir * fvis
    S_sub_dif_vis = taudt[0] * S_dn_dif * fvis
    S_sub_dir_nir = taubt[1] * S_dn_dir * fnir
    S_sub_dif_nir = taudt[1] * S_dn_dif * fnir

    Sn_C = ((1.0 - taubt[0]) * (1.0 - albb[0]) * S_dn_dir * fvis
            + (1.0 - taubt[1]) * (1.0 - albb[1]) * S_dn_dir * fnir
            + (1.0 - taudt[0]) * (1.0 - albd[0]) * S_dn_dif * fvis
            + (1.0 - taudt[1]) * (1.0 - albd[1]) * S_dn_dif * fnir)

    Sn_sub_C = ((1.0 - taubt_sub[0]) * (1.0 - albb_sub[0]) * S_sub_dir_vis
            + (1.0 - taubt_sub[1]) * (1.0 - albb_sub[1]) * S_sub_dir_nir
            + (1.0 - taudt_sub[0]) * (1.0 - albd_sub[0]) * S_sub_dif_vis
            + (1.0 - taudt_sub[1]) * (1.0 - albd_sub[1]) * S_sub_dif_nir)

    Sn_S = (taubt_sub[0] * (1.0 - rsoilv) * S_sub_dir_vis
            + taubt_sub[1] * (1.0 - rsoiln) * S_sub_dir_nir
            + taudt_sub[0] * (1.0 - rsoilv) * S_sub_dif_vis
            + taudt_sub[1] * (1.0 - rsoiln) * S_sub_dif_nir)

    return np.asarray(Sn_C), np.asarray(Sn_S), np.asarray(Sn_sub_C)

def calc_sub_spectra_Cambpell(lai, sza, rho_leaf, tau_leaf, rho_soil, x_lad=1, lai_eff=None):
    """ Canopy spectra
    Estimate canopy spectral using the [Campbell1998]_
    Radiative Transfer Model
    Parameters
    ----------
    lai : float
        Effective Leaf (Plant) Area Index.
    sza : float
        Sun Zenith Angle (degrees).
    rho_leaf : float, or array_like
        Leaf bihemispherical reflectance
    tau_leaf : float, or array_like
        Leaf bihemispherical transmittance
    rho_soil : float
        Soil bihemispherical reflectance
    x_lad : float,  optional
        x parameter for the ellipsoildal Leaf Angle Distribution function of
        Campbell 1988 [default=1, spherical LIDF].
    lai_eff : float or None, optional
        if set, its value is the directional effective LAI
        to be used in the beam radiation, if set to None we assume homogeneous canopies.
    Returns
    -------
    albb : float or array_like
        Beam (black sky) canopy albedo
    albd : float or array_like
        Diffuse (white sky) canopy albedo
    taubt : float or array_like
        Beam (black sky) canopy transmittance
    taudt : float or array_like
        Beam (white sky) canopy transmittance
    References
    ----------
    .. [Campbell1998] Campbell, G. S. & Norman, J. M. (1998), An introduction to environmental
        biophysics. Springer, New York
        https://archive.org/details/AnIntroductionToEnvironmentalBiophysics.
    """

    # calculate aborprtivity
    amean = 1.0 - rho_leaf - tau_leaf
    amean_sqrt = np.sqrt(amean)
    del rho_leaf, tau_leaf, amean

    # Calculate canopy beam extinction coefficient
    # Modification to include other LADs
    if lai_eff is None:
        lai_eff = np.asarray(lai)
    else:
        lai_eff = np.asarray(lai_eff)

    # D I F F U S E   C O M P O N E N T S
    # Integrate to get the diffuse transmitance
    taud = rad._calc_taud(x_lad, lai)

    # Diffuse light canopy reflection coefficients  for a deep canopy
    akd = -np.log(taud) / lai
    rcpy= (1.0 - amean_sqrt) / (1.0 + amean_sqrt)  # Eq 15.7
    rdcpy = 2.0 * akd * rcpy / (akd + 1.0)  # Eq 15.8

    # Diffuse canopy transmission and albedo coeff for a generic canopy (visible)
    expfac = amean_sqrt * akd * lai
    del akd
    neg_exp, d_neg_exp = np.exp(-expfac), np.exp(-2.0 * expfac)
    xnum = (rdcpy * rdcpy - 1.0) * neg_exp
    xden = (rdcpy * rho_soil - 1.0) + rdcpy * (rdcpy - rho_soil) * d_neg_exp
    taudt = xnum / xden  # Eq 15.11
    del xnum, xden
    fact = ((rdcpy - rho_soil) / (rdcpy * rho_soil - 1.0)) * d_neg_exp
    albd = (rdcpy + fact) / (1.0 + rdcpy * fact)  # Eq 15.9
    del rdcpy, fact

    # B E A M   C O M P O N E N T S
    # Direct beam extinction coeff (spher. LAD)
    akb = rad.calc_K_be_Campbell(sza, x_lad)  # Eq. 15.4

    # Direct beam canopy reflection coefficients for a deep canopy
    rbcpy = 2.0 * akb * rcpy / (akb + 1.0)  # Eq 15.8
    del rcpy, sza, x_lad
    # Beam canopy transmission and albedo coeff for a generic canopy (visible)
    expfac = amean_sqrt * akb * lai_eff
    neg_exp, d_neg_exp = np.exp(-expfac), np.exp(-2.0 * expfac)
    del amean_sqrt, akb, lai_eff
    xnum = (rbcpy * rbcpy - 1.0) * neg_exp
    xden = (rbcpy * rho_soil - 1.0) + rbcpy * (rbcpy - rho_soil) * d_neg_exp
    taubt = xnum / xden  # Eq 15.11
    del xnum, xden
    fact = ((rbcpy - rho_soil) / (rbcpy * rho_soil - 1.0)) * d_neg_exp
    del expfac
    albb = (rbcpy + fact) / (1.0 + rbcpy * fact)  # Eq 15.9
    del rbcpy, fact

    taubt[np.isnan(taubt)] = 1
    taudt[np.isnan(taudt)] = 1
    albb[np.isnan(albb)] = rho_soil[np.isnan(albb)]
    albd[np.isnan(albd)] = rho_soil[np.isnan(albd)]

    return albb, albd, taubt, taudt

def _check_default_parameter_size(parameter, input_array):

    parameter = np.asarray(parameter)
    if parameter.size == 1:
        parameter = np.ones(input_array.shape) * parameter
        return np.asarray(parameter)
    elif parameter.shape != input_array.shape:
        raise ValueError(
            'dimension mismatch between parameter array and input array with shapes %s and %s' %
            (parameter.shape, input_array.shape))
    else:
        return np.asarray(parameter)

def calc_shadow_fraction(sza, hc, hb, wc, f_C, tau):
    ''' Shadow fraction below primary (tree) canopy

        Estimate shadow fraction below primary/overstory (e.g. tree) canopy projected onto secondary/understory (e.g. grass vegetation)
        based on sun position (sun zenith angle) and spherical/elipsoidal crown shape
        Parameters
        ----------.
        sza : float
            Sun Zenith Angle (degrees).
        hc : float
            Canopy height (m).
        hb : float
            height of the base of canopy where foliage begins (m).
        wc: float
            width to height ratio (-)
        f_C : float
            Fractional cover of primary canopy (tree) (-).
        tau: float
            proportion of beam/direct radiation that is transmistted through the canopy

        '''

    pi = 3.1416

    # calculate width if not spherical (using width-to-canopy ratio)
    w = hc * wc
    # Shadown length
    shadow_length = (hc-hb)*np.tan(sza)
    # estimate shadow area projected below the canopy, correcting for portion of beam radiation that is transmitted (tau) through the canopy
    A_shade = (pi*(shadow_length/2)*(w/2))*(1-tau)

    # area of tree as seen from above
    A_tree = pi*((w/2)**2)
    ratio = A_shade/A_tree
    # f_shadow as compared to fractional cover of tree area
    f_shade = f_C*ratio
    f_shade[sza >= 90.0] = 1.0
    f_shade[f_shade > 1.0] = 1.0

    return f_shade

def calc_Rn_substrate_BeerLambert(Sn, Ln, LAI, theta, x_LAD=1.0, alpha_s=0.5, kappa_l=0.95 ):
    '''

    calculate substrate net radiation based on beer-lambert
    '''
    #estimate k beam extinction coeffecient
    k = rad.calc_K_be_Campbell(theta,x_LAD)
    rad_extinc = np.exp((-k*np.sqrt(alpha_s)*LAI))#/(np.sqrt(2*np.cos(theta))))
    Sn_S = Sn*rad_extinc
    Sn_C = Sn - Sn_S
    rad_extinc = np.exp((-kappa_l*LAI))#/(np.sqrt(2*np.cos(theta))))
    Ln_S = Ln*rad_extinc
    Ln_C = Ln - Ln_S
    Rn_C = Sn_C + Ln_C
    Rn_S = Sn_S + Ln_S
    return Sn_C, Ln_C, Sn_S, Ln_S


def calc_Ln_substrate_Campbell(Ln, LAI, x_LAD=1.0, emiss_c=0.98, emiss_s=0.95):
    '''

    calculate substrate net radiation based on beer-lambert
    '''
    # estimate Cambpbell diffuse transmittance
    _, _, _, taudt = rad.calc_spectra_Cambpell(LAI,
                                                0,
                                                1 - emiss_c,
                                                0,
                                                1 - emiss_s,
                                                x_lad=x_LAD)

    Ln_S = Ln * taudt
    Ln_C = Ln - Ln_S
    return Ln_C, Ln_S

  
def raupach_94(pai_eff, c_r=0.3, c_s=0.003, max_u_star_u_h=0.3, c_w=2.0, c_d1=7.5):
    """
    Computes the frontal leaf area index based on _[Raupach1992
    :param pai_eff:
        Landscape Canopy Area Inde, i.e. the total (single-sided) area of all canopy
        elements over unit ground area
    :param c_r: float
        Roughness-element drag coefficient.
        A change of + 10% in CR changes zo/h by about + 10% (at pai_eff ~ 0.2,
        where maximum sensitivity occurs) and does not affect d/h.
    :param c_s: float
        Substrate-surface drag coefficient drag coefficient.
        This parameter does not affect d/h and only affects zo/h at low pai_eff (where
        a change in Cs of +10% changes zo/h by about +3%)
    :param max_u_star_u_h: float
        (u_*/U_h)max From drag coefficient data on canopies with
        pai_eff > pai_eff_max (Jarvis et al., 1976; Raupach et al., 1991)
        This parameter does not affect d/h, and affects zo/h only when
        pai_eff > pai_eff_max in this region, a +10% change in (u_*/U_h)max
        increases zo/h by 4%
    :param c_w: float
        characterizes the roughness sublayer depth. The prediction for zo/h is not very
        sensitive to Cw, changing by +2.5% when cw changes by +10%. The prediction
        for d/h is independent of c,.
    :param c_d1: float
        The value cd~ = 7.5 is obtained by requiring Equation (8) to match
        the data in Figure lb. A change of +10% in c_d1 changes d/h by +1.5%
        and zo/h by - 1 % (at pai = 2).
    :return:
    """
    # Convert input scalar to numpy array
    pai_eff = np.asarray(pai_eff)

    # Roughnes element drag coefficient
    u_star_u_h = np.minimum(np.sqrt(c_s + c_r * pai_eff / 2), max_u_star_u_h)
    psi_h = np.log(c_w) - 1.0 + c_w**-1

    # Eq. 8 of ..[Raupach1994]
    d_factor = 1. - (1. - np.exp(-np.sqrt(c_d1 * pai_eff))) / np.sqrt(c_d1 * pai_eff)

    # Eq. 4 of ..[Raupach1994]
    z0M_factor = (1 - d_factor) * np.exp(-k / u_star_u_h - psi_h)
    return np.asarray(z0M_factor), np.asarray(d_factor)
  
def calc_H_C_PT(delta_R_ni, f_g, T_A_K, P, c_p, alpha):
    '''Calculates canopy sensible heat flux based on the Priestley and Taylor formula.

    Parameters
    ----------
    delta_R_ni : float
        net radiation divergence of the vegetative canopy (W m-2).
    f_g : float
        fraction of vegetative canopy that is green.
    T_A_K : float
        air temperature (Kelvin).
    P : float
        air pressure (mb).
    c_p : float
        heat capacity of moist air (J kg-1 K-1).
    alpha : float
        the Priestley Taylor parameter.

    Returns
    -------
    H_C : float
        Canopy sensible heat flux (W m-2).

    References
    ----------
    Equation 14 in [Norman1995]_
    '''

    # slope of the saturation pressure curve (kPa./deg C)
    s = met.calc_delta_vapor_pressure(T_A_K)
    s = s * 10  # to mb
    # latent heat of vaporisation (J./kg)
    Lambda = met.calc_lambda(T_A_K)
    # psychrometric constant (mb C-1)
    gama = met.calc_psicr(c_p, P, Lambda)
    s_gama = s / (s + gama)
    H_C = delta_R_ni * (1.0 - alpha * f_g * s_gama)
    return np.asarray(H_C)

