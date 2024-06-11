import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyTSEB import TSEB
from os.path import  exists, join
from os import mkdir, getcwd
import csv
import scipy.stats as st
from py3seb.py3seb import ThreeSEB_PT, raupach_94, calc_Sn_Campbell



#==============================================================================
# Workspace and output directory
#==============================================================================
workdir = getcwd()
outdir = join(workdir,'Output')
if not exists(outdir):
    mkdir(outdir)

#==============================================================================
# Setting up model configurarions
#==============================================================================

# Set formulae for Resistances computation (0= Norman TSEB; 1= Choudhury and Monteith, 2= McNaughton and Van der Hurk, 3= Nieto, 4=Haghighi)
Resistance_flag=[0,{}]
res_string = 'Norman95'

# Set whether to use measured or estimated soil heat flux[0: measured, 1: constant ratio, 2: diurnal shape]
G_flag = 2
if G_flag == 1:
    G_string = 'G_Ratio'
    G_constant = 0.15
elif G_flag == 2:
    G_string = 'G_TimeDiff'
else:
    G_string = 'G_Obs'

# parameters for estimatig G with 2: diurnal shape]
# Santanello & Friedl (2003) see https://github.com/hectornieto/pyTSEB/blob/600664efd3e5ac4edab84e84fa5cb9d55c58c46f/pyTSEB/TSEB.py#L1652
G_santanello = [0.25, 3.0, 24.0]

#==============================================================================
# Input parameters
#==============================================================================

#constant/static parameters of different sites
site_params = {1: {'site_name':'AU-Dry', 'lat': -15.2588, 'lon':132.3706, 'stdlon':135.0, 'z_u': 15, 'z_t': 15,
                   'f_c': 0.25 , 'f_c_sub': 1.0 ,'h_c': 12.3, 'h_c_sub':1., 'leaf_width': 0.05, 'leaf_width_sub': 0.01},
               2: {'site_name':'ES-LM1', 'lat': 39.945592, 'lon':-5.782977, 'stdlon':-15.0, 'z_u': 15, 'z_t': 15,
                   'f_c': 0.20 , 'f_c_sub': 1.0 ,'h_c': 8.7, 'h_c_sub':0.5, 'leaf_width': 0.05, 'leaf_width_sub': 0.01},
               3: {'site_name':'US-Ton', 'lat': 38.4311 , 'lon':-120.966 , 'stdlon':-120.0, 'z_u': 23.5, 'z_t': 23.5,
                   'f_c': 0.48 , 'f_c_sub': 1.0 ,'h_c': 9.4, 'h_c_sub':0.5, 'leaf_width': 0.05, 'leaf_width_sub': 0.01},
               4: {'site_name':'ES-Abr', 'lat': 38.701839, 'lon':-6.785881, 'stdlon':-15.0, 'z_u': 12, 'z_t': 12,
                   'f_c': 0.24 , 'f_c_sub': 1.0 ,'h_c': 6.6, 'h_c_sub':0.5, 'leaf_width': 0.05, 'leaf_width_sub': 0.01},
              }

#==============================================================================
# Canopy and Soil spectra
#==============================================================================
spectraGrd = {'rsoilv': 0.07, 'rsoiln': 0.28}
spectraVeg_tree = {'rho_leaf_vis': 0.096, 'tau_leaf_vis': 0.014, 'rho_leaf_nir': 0.55, 'tau_leaf_nir': 0.28}
spectraVeg_grass = {'rho_leaf_vis': 0.05, 'tau_leaf_vis': 0.09, 'rho_leaf_nir': 0.24, 'tau_leaf_nir': 0.43}

#Thermal spectra
e_v=0.99        #Leaf emissivity
e_s=0.95        #Soil emissivity
e_v_dry = 0.96  #dry grass emissivity

#sensor viewing angle
vza=0

# bare soil aerodynamic roughness length (m)
z0_soil=0.01


#set header names for output file
outputTxtFieldNames = ['site_ID', 'site_name', 'year','hydro_year', 'DOY', 'time', 'SW_in_obs', 'LW_in_obs', 'SW_out_obs',
                       'LW_out_obs','Rn_obs', 'G_obs', 'H_obs', 'LE_obs', 'T_A1', 'u', 'T_R1', 'ea', 'LAI_C', 'LAI_sub', 'h_C',
                       'h_C_sub','f_c', 'f_c_sub', 'f_g', 'f_g_sub', 'w_C', 'w_C_sub', 'VZA', 'SZA', 'Rn_model', 'SW_model',
                       'LW_model','Rn_C', 'Rn_C_sub', 'Rn_S', 'Rn_lw_C', 'Rn_lw_C_sub', 'Rn_lw_S', 'Tc_model', 'Tc_sub_model',
                       'Ts_model','Tac_model', 'LE_model', 'H_model', 'LE_C', 'H_C', 'LE_C_sub', 'H_C_sub', 'LE_S', 'H_S', 'flag',
                       'zo','d', 'zo_sub', 'd_sub', 'G_model', 'R_s','R_sub', 'R_x', 'R_a', 'u_friction', 'L']


#==============================================================================
# Run 3SEB
#==============================================================================

#set which sites to process

###### 1: AU-Dry, 2: ES-LM1, 3: US-Ton, 4: ES-Abr ######
#default: ES-LM1
sites=[2]

#set years to process

#simulation periods
### AU-Dry: 2012-2015
### ES-LM1: 2015-2018
### US-Ton: 2015-2019
### ES-Abr: 2016-2019
#default: ES-LM1 (all years)
years =  [2015, 2016, 2017, 2018]

#run 3SEB for specified years and sites
for year in years:
    plt.close('all')
    for site in sites:
        site_name = site_params[site]['site_name']

        #create output directory for each site
        outdir_site = join(outdir,site_name)
        if not exists(outdir_site):
            mkdir(outdir_site)

        #create output text file
        output_file = join(outdir_site, '3SEB_output_%s_%s_%s_%s.txt'%(res_string, G_string, site_name, year))
        fp = open(output_file, 'w', newline='')
        writer = csv.writer(fp, delimiter='\t')
        writer.writerow(outputTxtFieldNames)

        print('================= Running Year: '+ str(year)+' for '+ site_name+'=========================')

        # site description
        lat = site_params[site]['lat'] # latitude (degree)
        lon = site_params[site]['lon'] # longitude (degree)
        stdlon = site_params[site]['stdlon'] # Standard longitude (degree)
        z_u = site_params[site]['z_u']      # height of measuremnt of wind spped (m)
        z_t = site_params[site]['z_t']      # height of measuremnt of temperature (m)

        # ==============================================================================
        # Vegetation characteristics
        # ==============================================================================
        alpha_PT = 1.26
        leaf_width = site_params[site]['leaf_width']
        leaf_width_sub = site_params[site]['leaf_width_sub']

        ######### upper layer - Tree canopy ########
        fC_value = site_params[site]['f_c']  # fractional cover
        wC_value = 1                         # width to height ratio
        hC_value = site_params[site]['h_c']  # canopy height
        x_LAD_C = 1  # LAD coefficient for tree

        ######### lower layer - Grass canopy ########
        landcover_low = TSEB.res.GRASS
        fC_sub_value = site_params[site]['f_c_sub']  # fractional cover
        wC_sub_value = 1  # width to height ratio
        hC_sub_value = site_params[site]['h_c_sub']  # canopy height
        x_LAD_sub = 1  # LAD coefficient for grass

        #open vegetation data
        MODIS_file = join(workdir, 'Input', 'Veg_data',site_name, 'VegIndices_MODIS_Filtered_%s.txt' % (site_name))
        MODIS_ds = np.genfromtxt(MODIS_file, delimiter='\t', names=True, dtype=None)
        # filter for year in question
        MODIS_LAI = MODIS_ds[MODIS_ds['hydro_year']==year]

        #LAI
        LAI_grass_daily = MODIS_LAI['LAI_grass']
        LAI_tree_daily = MODIS_LAI['LAI_tree']

        #Fg
        fg_grass_daily = MODIS_LAI['fg_grass']
        fg_tree_daily = MODIS_LAI['fg_tree']

        #create output directory for site
        outdir_site = join(outdir,site_name)
        if not exists(outdir_site):
            mkdir(outdir_site)

        #open meteo data
        meteoFile = join(workdir, 'Input', 'Meteo_data', site_name, '%s_meteo.txt'%(site_name))
        metdata = pd.read_csv(meteoFile, sep='\t')
        # filter for year in question
        metdata = metdata[metdata['year_hydro'] == year]

        #get dates at half hourly time step
        dates_str = pd.to_datetime(metdata['Date'].values).strftime('%Y-%m-%d')
        # get dates at daily time step
        dates_daily = np.unique(dates_str)

        #initialize LAI and fg arrays at half hourly time step
        LAI_tree = np.zeros(dates_str.shape)
        LAI_grass = np.zeros(dates_str.shape)
        fg_C_array = np.zeros(dates_str.shape)
        fg_sub_array = np.zeros(dates_str.shape)

        #mask daily veg data before meteo start date (in case LAI time series has a different initial date)
        start_date = dates_str[0]
        start_mask = pd.to_datetime(MODIS_LAI['date'].astype('U12')) >= pd.to_datetime(start_date)
        #LAI
        LAI_grass_daily = LAI_grass_daily[start_mask]
        LAI_tree_daily = LAI_tree_daily[start_mask]
        #fg
        fg_grass_daily = fg_grass_daily[start_mask]
        fg_tree_daily = fg_tree_daily[start_mask]

        #resample veg data to half-hourly time step
        for day in dates_daily:
            validData_met = dates_str == day
            validData_veg = dates_daily == day
            #lai
            LAI_grass[validData_met] = LAI_grass_daily[validData_veg]
            LAI_tree[validData_met] = LAI_tree_daily[validData_veg]
            #fg
            fg_sub_array[validData_met] = fg_grass_daily[validData_veg]
            fg_C_array[validData_met] = fg_tree_daily[validData_veg]

        #set minimum LAI for understory/grass LAI
        min_lai = 0.5
        LAI_grass[LAI_grass < min_lai] = min_lai

        ############# initialize vegetation structural parameters ################

        # hc, tree canopy height
        hC = np.ones(LAI_tree.shape) * hC_value # tree height

        #hC_sub, grass canopy height
        #grass height depends on LAI (scaled using a power function --> like Sen-ET, see Guzinski et al. (2020): https://doi.org/10.3390/rs12091433)
        LAI_grass_max = np.nanmax(LAI_grass)
        hc_factor = np.minimum((LAI_grass/LAI_grass_max)**2, 1)
        hC_sub = 0.3*hC_sub_value + 0.7*(hC_sub_value*hc_factor)

        #wC, tree canopy height to width ratio
        wC = np.ones(LAI_tree.shape) * wC_value
        wC_ratio = hC / wC

        #wC_sub, grass canopy height to width ratio
        wC_sub = np.ones(LAI_tree.shape) * wC_sub_value
        wC_sub_ratio = hC_sub/wC_sub

        #fC, tree canopy cover
        fC = np.ones(LAI_tree.shape) * fC_value

        #fC_sub, grass canopy cover
        fC_sub = np.ones(LAI_tree.shape) * fC_sub_value

        # dynamic grass emissivity due senescence of vegetation during summer. Dry vegetation has emissivity of 0.96
        e_v_grass = e_v * fg_sub_array + e_v_dry * (1 - fg_sub_array)
        e_surf = fC * e_v + ((1 - fC) * (fC_sub * e_v_grass + (1 - fC_sub) * e_s))

        #LST from LW radiometer
        LST = ((metdata['LW_out'].values - (1. - e_surf) * metdata['LW_in'].values) / (TSEB.rad.SB * e_surf)) ** 0.25

        #sun zenith and azimuth angle
        sza, saa = TSEB.met.calc_sun_angles(np.ones(metdata['DOY'].shape) * lat,
                                            np.ones(metdata['DOY'].shape) * lon,
                                            np.ones(metdata['DOY'].shape) * stdlon, metdata['DOY'], metdata['hour'].values)
        #Local (i.e. individual) tree LAI
        F_tree = LAI_tree / fC
        # calculate clumping index
        Omega0 = TSEB.CI.calc_omega0_Kustas(F_tree, fC, x_LAD=x_LAD_C, isLAIeff=True)
        Omega = TSEB.CI.calc_omega_Kustas(Omega0, sza, w_C=wC_ratio)
        Omega0_sub = TSEB.CI.calc_omega0_Kustas(LAI_grass, fC_sub, x_LAD=x_LAD_sub, isLAIeff=True)
        Omega_sub = TSEB.CI.calc_omega_Kustas(Omega0_sub, sza, w_C=wC_ratio)

        #effective LAI
        LAI_tree_eff = F_tree * Omega
        LAI_grass_eff = LAI_grass*Omega_sub

        #observed G
        G_obs = metdata['G'].values
        # Get flux computation method (0: measured G, 1:ratio, 2:time ratio)
        if G_flag == 0:
            calcG = [[0], G_obs]
        elif G_flag == 1:
            calcG = [[1], G_constant]
        elif G_flag == 2:
            calcG = [[2, G_santanello[0], G_santanello[1], G_santanello[2]], metdata['hour'].values]

        #meteo data
        Ta_K = metdata['T_A'].values
        ea = metdata['ea_mb'].values
        p = metdata['P_mb'].values
        u = metdata['u'].values
        u[u < 0] = 0

        # incoming shortwave radiation
        SW_in = metdata['SW_in'].values
        SW_in[sza > 90] = 0
        sza[sza > 90] = 90
        # incoming longwave radiation
        LW_in = metdata['LW_in'].values

        # Estimates the direct and diffuse solar radiation
        difvis, difnir, fvis, fnir = TSEB.rad.calc_difuse_ratio(SW_in, sza, press=p)
        skyl = fvis * difvis + fnir * difnir
        Sdn_dir = (1. - skyl) * SW_in
        Sdn_dif = skyl * SW_in

        #Calculate roughness parameters
        #using raupach94 method for tree
        z_0M_factor, d_0_factor = raupach_94(LAI_tree)
        d_0 = hC*d_0_factor
        z_0M = hC*z_0M_factor

        d_0[d_0 < 0] = 0
        z_0M[z_0M < z0_soil] = z0_soil

        #use ratio method for grass
        [z_0M_sub, d_0_sub] = TSEB.res.calc_roughness(LAI_grass, hC_sub, wC_sub_ratio, np.ones(LAI_tree.shape) * landcover_low)
        d_0_sub[d_0_sub < 0] = 0
        z_0M_sub[z_0M_sub < z0_soil] = z0_soil

        # height of the base of overstory canopy where foliage begins (m).
        hb = 2.
        #estimate shortwave radiation through 3 sources (overstory+understory+soil)
        Rn_sw_C, Sn_S, Sn_C_sub = calc_Sn_Campbell(LAI_tree, LAI_grass, sza, Sdn_dir, Sdn_dif, fvis, fnir,
                                                   np.full_like(LAI_tree, spectraVeg_tree['rho_leaf_vis']),
                                                   np.full_like(LAI_tree, spectraVeg_grass['rho_leaf_vis']),
                                                   np.full_like(LAI_tree, spectraVeg_tree['tau_leaf_vis']),
                                                   np.full_like(LAI_tree, spectraVeg_grass['tau_leaf_vis']),
                                                   np.full_like(LAI_tree, spectraVeg_tree['rho_leaf_nir']),
                                                   np.full_like(LAI_tree, spectraVeg_grass['rho_leaf_nir']),
                                                   np.full_like(LAI_tree, spectraVeg_tree['tau_leaf_nir']),
                                                   np.full_like(LAI_tree, spectraVeg_grass['tau_leaf_nir']),
                                                   np.full_like(LAI_tree, spectraGrd['rsoilv']),
                                                   np.full_like(LAI_tree, spectraGrd['rsoiln']), hC, hb, wC,fC,
                                                   x_LAD=x_LAD_C, x_LAD_sub=x_LAD_sub, LAI_eff=LAI_tree_eff, LAI_eff_sub=LAI_grass_eff)

        Rn_sw_C[~np.isfinite(Rn_sw_C)] = 0
        Sn_C_sub[~np.isfinite(Sn_C_sub)] = 0
        Sn_S[~np.isfinite(Sn_S)] = 0

        #Run 3SEB
        [flag_PT_all, T_S, T_C, T_C_sub, T_AC, L_n_sub, L_nC, Ln_C_sub, Ln_S,
         LE_C, H_C, LE_C_sub, H_C_sub, LE_S, H_S, G_mod, R_S, R_sub, R_X, R_A, u_friction, L, n_iterations] = ThreeSEB_PT(LST,
                                                                                                                          vza,
                                                                                                                          Ta_K,
                                                                                                                          u,
                                                                                                                          ea,
                                                                                                                          p,
                                                                                                                          Rn_sw_C,
                                                                                                                          Sn_S,
                                                                                                                          Sn_C_sub,
                                                                                                                          LW_in,
                                                                                                                          LAI_tree,
                                                                                                                          LAI_grass,
                                                                                                                          hC,
                                                                                                                          hC_sub,
                                                                                                                          e_v,
                                                                                                                          e_v_grass,
                                                                                                                          e_s,
                                                                                                                          z_0M,
                                                                                                                          z_0M_sub,
                                                                                                                          d_0,
                                                                                                                          d_0_sub,
                                                                                                                          z_u,
                                                                                                                          z_t,
                                                                                                                          leaf_width=leaf_width,
                                                                                                                          leaf_width_sub=leaf_width_sub,
                                                                                                                          f_c = fC,
                                                                                                                          f_c_sub = fC_sub,
                                                                                                                          f_g = fg_C_array,
                                                                                                                          f_g_sub = fg_sub_array,
                                                                                                                          calcG_params = calcG,
                                                                                                                          resistance_form=Resistance_flag)


        validData = np.logical_or.reduce((SW_in < 0, metdata['H'].values == -9999, metdata['SW_in'].values==-9999,
                                          metdata['SW_out'].values==-9999, metdata['LW_out'].values==-9999, metdata['LW_out'].values==-9999,
                                          np.isnan(metdata['H'].values),np.isnan(metdata['Rn'].values), metdata['G'].values == -9999,
                                          np.isnan(metdata['G'].values),
                                          np.isnan(H_C + H_C_sub+H_S)))

                                          
        #=====================================================
        #               Get outputs and plot basic results
        #====================================================

        flag_PT_all[validData] = 255

        #==========================
        # ---- modelled fluxes ----
        #==========================

        #overstory net radiation
        Rn_tree = Rn_sw_C + L_nC
        #Understory net radiation
        Rn_C_sub = Sn_C_sub + Ln_C_sub
        #Soil net radiation
        Rn_S = Sn_S + Ln_S
        # total net radiation
        Rn_PT = Rn_tree + Rn_C_sub + Rn_S

        #total latent heat
        LE_PT = LE_C + LE_C_sub + LE_S
        #total sensible heat
        H_PT = H_C + H_C_sub + H_S

        #total shortwave radation
        Sn_PT = Rn_sw_C + Sn_C_sub + Sn_S
        #total logwave radation
        Ln_PT = L_nC + Ln_C_sub + Ln_S

        #total available energy
        AE_PT = Rn_PT - G_mod

        #==========================
        # ---- observed fluxes ----
        #==========================
        H_obs = metdata['H'].values
        G_obs = metdata['G'].values
        Sn_obs = metdata["SW_in"].values - metdata["SW_out"].values
        Ln_obs = metdata["LW_in"].values - metdata["LW_out"].values
        Rn_obs = Sn_obs + Ln_obs
        AE_obs = Rn_obs - G_obs
        LE_obs = Rn_obs - G_obs - H_obs
        LE_obs[np.logical_or.reduce((Rn_obs==-9999, G_obs==-9999,H_obs==-9999 ))] = -9999
        flag_PT_all[LE_obs==-9999] = 255

        years_actual = metdata['Year'].values

        # Open output file and write the data by rows
        for row in range(LE_PT.size):
            outData = [site, site_name, int(years_actual[row]), year, int(metdata['DOY'].values[row]), metdata['hour'].values[row], SW_in[row],
                       metdata['LW_in'].values[row], metdata['SW_out'].values[row], metdata['LW_out'].values[row], Rn_obs[row],
                       G_obs[row], H_obs[row], LE_obs[row],Ta_K[row], u[row], LST[row], ea[row], LAI_tree[row],
                       LAI_grass[row], hC[row], hC_sub[row], fC[row], fC_sub[row], fg_C_array[row], fg_sub_array[row],
                       wC[row], wC_sub[row], vza, sza[row], Rn_PT[row], Sn_PT[row], Ln_PT[row], Rn_tree[row], Rn_C_sub[row], Rn_S[row],
                       L_nC[row], Ln_C_sub[row], Ln_S[row], T_C[row],T_C_sub[row], T_S[row], T_AC[row], LE_PT[row],
                       H_PT[row], LE_C[row], H_C[row],LE_C_sub[row], H_C_sub[row], LE_S[row], H_S[row], flag_PT_all[row],
                       z_0M[row], d_0[row],z_0M_sub[row], d_0_sub[row],G_mod[row], R_S[row], R_sub[row], R_X[row], R_A[row],
                       u_friction[row], L[row]]

            writer.writerow(outData)

        def model_metrics(X, Y, mask):
            rmse = np.sqrt(np.nanmean((X[mask] - Y[mask]) ** 2))
            cor = st.pearsonr(X[np.logical_and.reduce((mask, ~np.isnan(X), ~np.isnan(Y)))],
                              Y[np.logical_and.reduce((mask, ~np.isnan(Y), ~np.isnan(X)))])[0]
            bias = np.nanmean(X[mask] - Y[mask])
            return rmse, cor, bias


        #plot half-hourly modelled vs observed flux scatter
        noneg = np.logical_and.reduce((H_PT >-50,H_obs>-50, LE_PT>-50,LE_obs>-50,  LE_PT<1500, LE_obs<1500, H_PT<1500, H_obs<1500 ))
        QC = np.logical_and.reduce((flag_PT_all < 5, metdata['SW_in'].values > 25, Sn_PT > 25))
        mask = np.logical_and(noneg, QC )
        plt.figure()
        plt.scatter(H_PT[mask], H_obs[mask], c='r', marker='.', alpha=0.2, label='H', s=3)
        plt.scatter(LE_PT[mask], LE_obs[mask], color='b', marker='.', alpha=0.2, label='LE', s=3)
        plt.xlim(-100, 700)
        plt.ylim(-100, 700)
        plt.xlabel(r'Modeled $\left(W/m^2\right)$', fontsize=14)
        plt.ylabel(r'Observed $\left(W/m^2\right)$', fontsize=14)
        plt.title(str(site_name) + ' ' + str(year))
        plt.plot((-100, 700), (-100, 700), 'k-')
        plt.legend()
        rmse_LE, cor_LE, bias_LE = model_metrics(LE_PT,LE_obs, mask)
        rmse_H, cor_H, bias_H = model_metrics(H_PT, H_obs, mask)

        plt.figtext(0.15, 0.7, 'RMSD: LE = %s    H = %s\nbias: LE = %s    H = %s\nr:    LE = %s  H = %s' % (
            int(rmse_LE), int(rmse_H), int(bias_LE), int(bias_H), np.round(cor_LE, 2), np.round(cor_H, 2)),
                    backgroundcolor='white', linespacing=1.15, family='monospace')

        leg = plt.legend(loc=9, ncol=2)
        for lh in leg.legendHandles:
            lh.set_alpha(1)

        plt.savefig(join(outdir_site, 'FluxPartitioning_%s_%s_%s_%s.png' % (res_string, G_string, year, site_name )))
        plt.close()


        #plot AE  modeled vs observed scatter plot
        plt.figure()
        plt.scatter(AE_PT[mask], AE_obs[mask], c='orange', marker='.', alpha=0.2, s=3)
        plt.xlim(-100, 700)
        plt.ylim(-100, 700)
        plt.xlabel(r'Modeled $\left(W/m^2\right)$', fontsize=14)
        plt.ylabel(r'Observed $\left(W/m^2\right)$', fontsize=14)
        plt.title('AE '+str(site_name) + ' ' + str(year))
        plt.plot((-100, 700), (-100, 700), 'k-')
        rmse_LE = np.sqrt(np.mean((AE_PT[mask] - AE_obs[mask]) ** 2))
        cor_LE = st.pearsonr(AE_PT[mask], AE_obs[mask])[0]
        bias_LE = np.mean(AE_PT[mask] - AE_obs[mask])

        plt.figtext(0.15, 0.7, 'RMSD: AE = %s\nbias: AE = %s\nr:    AE = %s ' % (
            int(rmse_LE), int(bias_LE), np.round(cor_LE, 2)),
                    backgroundcolor='white', linespacing=1.15, family='monospace')

        plt.savefig(join(outdir_site, 'AvailableEnergy_vs_obs_%s_%s_%s_%s.png' % (res_string, G_string, year, site_name)))
        plt.close()
