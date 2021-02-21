import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyTSEB import TSEB
from pyTSEB import meteo_utils as met
from  os.path import  exists, join
from os import mkdir, getcwd
import csv
import scipy.stats as st
from glob import glob
from TSEB_3S_PT import ThreeSEB_PL_singlePT, ThreeSEB_PL_doublePT, raupach_94, calc_Sn_Campbell
from pandas import Timestamp

#==============================================================================
# Setting up model configurarions
#==============================================================================

constantFg=False
# Set formulae for Resistances computation (0= Norman TSEB; 1= Choudhury and Monteith, 2= McNaughton and Van der Hurk, 3= Nieto, 4=Haghighi)
Resistance_flag=[0,{}]
if Resistance_flag[0] == 0:
    res_string = 'Norman95'
elif Resistance_flag[0] == 1:
    res_string = 'ChoudMont88'
elif Resistance_flag[0] == 4:
    res_string = 'Li2019'

# Set whether to use measured or estimated soil heat flux[0: measured, 1: constant ratio, 2: diurnal shape]
G_santanello = [0.3, 3.0, 24.0]
#  Set whether to use measured or estimated soil heat flux[0: measured, 1: constant ratio, 2: diurnal shape]
G_flag = 1
if G_flag == 1:
    G_string = 'G_Ratio'
elif G_flag == 2:
    G_string = 'G_TimeDiff'
else:
    G_string = 'G_Obs'
#set which LAI to use
LAI_flag = 1
if LAI_flag == 0:
    LAI_str = 'LAImodis'
    min_lai = 0.35
else:
    LAI_str = 'LAItotal'
    min_lai = 0.6

LAI_tree_observation = False
if LAI_tree_observation:
    LAI_tree_str ='_LAItree_obs'
else:
    LAI_tree_str = ''

#set temporal period and sites (in hydro years)
#years = [2013, 2014, 2015, 2016]
years = [2015, 2016, 2017, 2018]
#years = [2012, 2013, 2014, 2015]
years = [2016, 2017, 2018, 2019]

years = [2015]
###### 1: CT, 2: NT, 3: NPT ######
sites=[4]

#==============================================================================
# Input parameters
#==============================================================================

#constant/static parameters of different sites
site_params = {1: {'site_name':'AU-Dry', 'lat': -15.2588, 'lon':132.3706, 'stdlon':130.0, 'z_u': 15, 'z_t': 15, 'LAI_tree':0.58,
                   'f_c': 0.25 , 'f_c_sub': 1.0 ,'h_c': 12.3, 'h_c_sub':1.0, 'leaf_width': 0.05, 'leaf_width_sub': 0.005, 'LAI_max':1.5},
               2: {'site_name':'ES-Lm1', 'lat': 39.945592, 'lon':-5.782977, 'stdlon':-15.0, 'z_u': 15, 'z_t': 15, 'LAI_tree':0.4,
                   'f_c': 0.20 , 'f_c_sub': 1.0 ,'h_c': 8.7, 'h_c_sub':0.7, 'leaf_width': 0.05, 'leaf_width_sub': 0.01, 'LAI_max':1.5},
               3: {'site_name':'ZA-Kru', 'lat': -25.0197, 'lon':31.4969, 'stdlon':30.0, 'z_u': 17, 'z_t': 17,
                   'f_c': 0.28 , 'f_c_sub': 1.0 ,'h_c': 9.1, 'h_c_sub':0.7, 'leaf_width': 0.05, 'leaf_width_sub': 0.01},
               4: {'site_name':'US-Ton', 'lat': 38.4311 , 'lon':-120.966 , 'stdlon':-120.0, 'z_u': 23.5, 'z_t': 23.5,'LAI_tree':0.71,
                   'f_c': 0.48 , 'f_c_sub': 1.0 ,'h_c': 9.4, 'h_c_sub':0.7, 'leaf_width': 0.05, 'leaf_width_sub': 0.01, 'LAI_max':2.},
               5: {'site_name':'ES-Abr', 'lat': 38.701839, 'lon':-6.785881, 'stdlon':-15.0, 'z_u': 12, 'z_t': 12, 'LAI_tree':0.4,
                   'f_c': 0.24 , 'f_c_sub': 1.0 ,'h_c': 6.6, 'h_c_sub':0.7, 'leaf_width': 0.05, 'leaf_width_sub': 0.01, 'LAI_max':1.5},
              }

#==============================================================================
# Canopy and Soil spectra
#==============================================================================
#spectraVeg = {'rho_leaf_vis': 0.07, 'tau_leaf_vis': 0.08, 'rho_leaf_nir': 0.32, 'tau_leaf_nir': 0.33}  # from pyTSEB
spectraVeg = {'rho_leaf_vis': 0.05, 'tau_leaf_vis': 0.08, 'rho_leaf_nir': 0.26, 'tau_leaf_nir': 0.33}  # from pyTSEB

#spectraGrd = {'rsoilv': 0.07, 'rsoiln': 0.32}
spectraGrd = {'rsoilv': 0.07, 'rsoiln': 0.28}
#from majadas in situ data
spectraVeg_oak = {'rho_leaf_vis': 0.095759504, 'tau_leaf_vis': 0.013973319, 'rho_leaf_nir': 0.55105422, 'tau_leaf_nir': 0.27949141}
spectraVeg_grass = {'rho_leaf_vis': 0.049839839, 'tau_leaf_vis': 0.089606665, 'rho_leaf_nir': 0.241123781, 'tau_leaf_nir': 0.434988456}
#spectraVeg_S4 = {'rho_leaf_vis': 0.044162915, 'tau_leaf_vis': 0.060733103, 'rho_leaf_nir': 0.303289028,'tau_leaf_nir': 0.514388004}

#Thermal spectra
e_v=0.99        #Leaf emissivity
e_s=0.95        #Soil emissivity
e_v_dry = 0.96  #dry grass emissivity


vza=0

#=========================== resistance related parameters =============================
# for now keep constant for both layers)
z0_soil=0.01

# workspace and output directory
workdir = getcwd()
outdir = join(workdir,'Output', 'DoublePT')

if not exists(outdir):
    mkdir(outdir)

#set header names for output file
outputTxtFieldNames = ['site_ID', 'site_name', 'year','hydro_year', 'DOY', 'time', 'SW_in_obs', 'LW_in_obs', 'SW_out_obs',
                       'LW_out_obs','Rn_obs', 'G_obs', 'H_obs', 'LE_obs', 'T_A1', 'u', 'T_R1', 'ea', 'LAI_C', 'LAI_sub', 'h_C',
                       'h_C_sub','f_c', 'f_c_sub', 'f_g', 'f_g_sub', 'w_C', 'w_C_sub', 'VZA', 'SZA', 'Rn_model', 'SW_model',
                       'LW_model','Rn_C', 'Rn_C_sub', 'Rn_S', 'Rn_lw_C', 'Rn_lw_C_sub', 'Rn_lw_S', 'Tc_model', 'Tc_sub_model',
                       'Ts_model','Tac_model', 'LE_model', 'H_model', 'LE_C', 'H_C', 'LE_C_sub', 'H_C_sub', 'LE_S', 'H_S', 'flag',
                       'zo','d', 'zo_sub', 'd_sub', 'G_model', 'R_s','R_sub', 'R_x', 'R_a', 'u_friction', 'L', 'alphaPT_C',
                       'alphaPT_C_sub']

for year in years:
    plt.close('all')
    for site in sites:
        site_name = site_params[site]['site_name']
        #create output directory for site
        outdir_site = join(outdir,site_name)
        if not exists(outdir_site):
            mkdir(outdir_site)
        #output file
        output_file = join(outdir_site, '3SEB_output_%s_%s_%s_%s_%s%s.txt'%(res_string, G_string, site_name, year, LAI_str, LAI_tree_str))
        fp = open(output_file, 'w', newline='')
        writer = csv.writer(fp, delimiter='\t')
        writer.writerow(outputTxtFieldNames)

        print('================= Running Year: '+ str(year)+' for '+ site_name+'=========================')


        # site description
        ####Majadas####
        lat = site_params[site]['lat']
        lon = site_params[site]['lon']
        stdlon = site_params[site]['stdlon']
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
        wC_value = 1                        # width to height ratio
        hC_value = site_params[site]['h_c'] # canopy height
        x_LAD_C = 1  # LAD coefficient for tree
        ######### lower layer - Grass canopy ########
        landcover_low = TSEB.res.GRASS
        fC_sub_value = site_params[site]['f_c_sub']  # fractional cover
        wC_sub_value = 1  # width to height ratio
        hC_sub_value = site_params[site]['h_c_sub']  # canopy height
        x_LAD_sub = 1  # LAD coefficient for grass


        MODIS_file = join(workdir, 'Veg_data',site_name, 'VegIndices_MODIS_Filtered_%s.txt' % (site_name))
        MODIS_ds = np.genfromtxt(MODIS_file, delimiter='\t', names=True, dtype=None)
        MODIS_LAI = MODIS_ds[MODIS_ds['hydro_year']==year]

        LAI_modis_daily = MODIS_LAI['LAI_MODIS']
        if LAI_tree_observation:
            LAI_tree_contstant = site_params[site]['LAI_tree']
            LAI_tree_daily = np.ones(LAI_modis_daily.shape)*LAI_tree_contstant
            LAI_grass_daily = LAI_modis_daily - LAI_tree_daily

        else:
            LAI_grass_daily = MODIS_LAI['LAI_grass']
            LAI_tree_daily = MODIS_LAI['LAI_tree']

        fg_grass_daily = MODIS_LAI['fg_grass']
        fg_tree_daily = MODIS_LAI['fg_tree']
        #if site == 4 or site == 2:
        #correcting for tree influence on grass fg
        fg_grass_daily[fg_tree_daily==0.9] = (fg_grass_daily[fg_tree_daily==0.9] - (fC_value*fg_tree_daily[fg_tree_daily==0.9]))/(1-fC_value)
        fg_grass_daily[fg_grass_daily<0] = 0
        fg_grass_daily[fg_grass_daily > 1] = 1

        NDVI_daily = MODIS_LAI['NDVI']

        outdir_site = join(outdir,site_name)
        if not exists(outdir_site):
            mkdir(outdir_site)

        #meteoFile = join(workdir, 'Meteo_data', '%s_Above_2015_2018.txt')%(site_name)
        meteoFile = join(workdir, 'Meteo_data', site_name, '%s_meteo_hydroyear.txt'%(site_name))
        metdata = pd.read_csv(meteoFile, sep='\t')
        metdata = metdata[metdata['year_hydro'] == year]
        #delete this
        metdata_eco = metdata

        flag_PT_all = None

        dates_str = pd.to_datetime(metdata['Date'].values).strftime('%Y-%m-%d')
        dates_daily = np.unique(dates_str)
        LAI_oak = np.zeros(dates_str.shape)
        LAI_grass = np.zeros(dates_str.shape)
        LAI_modis = np.zeros(dates_str.shape)
        fg_C_array = np.zeros(dates_str.shape)
        fg_sub_array = np.zeros(dates_str.shape)

        #mask data before start date (for ES-Abr)
        start_date = dates_str[0]
        start_mask = pd.to_datetime(MODIS_LAI['date'].astype('U12')) >= pd.to_datetime(start_date)
        #lai
        LAI_grass_daily = LAI_grass_daily[start_mask]
        LAI_tree_daily = LAI_tree_daily[start_mask]
        LAI_modis_daily = LAI_modis_daily[start_mask]
        #fg
        fg_grass_daily = fg_grass_daily[start_mask]
        fg_tree_daily = fg_tree_daily[start_mask]
        NDVI_daily = NDVI_daily[start_mask]

        NDVI_grass = np.zeros(dates_str.shape)
        fg_sub_array = np.ones(dates_str.shape)
        for day in dates_daily:
            validData_met = dates_str == day
            validData_veg = dates_daily == day
            #lai
            LAI_grass[validData_met] = LAI_grass_daily[validData_veg]
            LAI_oak[validData_met] = LAI_tree_daily[validData_veg]
            LAI_modis[validData_met] = LAI_modis_daily[validData_veg]
            #fg
            fg_sub_array[validData_met] = fg_grass_daily[validData_veg]
            fg_C_array[validData_met] = fg_tree_daily[validData_veg]
            #ndvi
            NDVI_grass[validData_met] = NDVI_daily[validData_veg]

        LAI_grass[LAI_grass < min_lai] = min_lai

        #initialize other parameters
        hC = np.ones(LAI_oak.shape) * hC_value # tree height
        #grass vegetation depends on LAI like Sen-ET

        if site == 1:
            hC_sub = np.ones(LAI_oak.shape) * hC_sub_value
        else:
            LAI_max = site_params[site]['LAI_max']
            hc_factor = np.minimum((LAI_grass/LAI_max)**3, 1)
            hC_sub = 0.1*hC_sub_value + 0.9*(hC_sub_value*hc_factor)

        wC = np.ones(LAI_oak.shape) * wC_value
        wC_sub = np.ones(LAI_oak.shape) * wC_sub_value
        wC_ratio = hC/wC
        wC_sub_ratio = hC_sub/wC_sub

        fC = np.ones(LAI_oak.shape) * fC_value
        fC_sub = np.ones(LAI_oak.shape) * fC_sub_value

        # dynamic grass emissivity due senescence of vegetation during summer. Dry vegetation has emissivity of 0.96
        e_v_grass = e_v * fg_sub_array + e_v_dry * (1 - fg_sub_array)
        e_surf = fC * e_v + ((1 - fC) * (fC_sub * e_v_grass + (1 - fC_sub) * e_s))


        #radiometric and meteo data
        LST = ((metdata['LW_out'].values - (1. - e_surf) * metdata_eco['LW_in'].values) / (TSEB.rad.SB * e_surf)) ** 0.25

        sza, saa = TSEB.met.calc_sun_angles(np.ones(metdata['DOY'].shape) * lat,
                                            np.ones(metdata['DOY'].shape) * lon,
                                            np.ones(metdata['DOY'].shape) * stdlon, metdata['DOY'], metdata['hour'].values)

        F_oak = LAI_oak / fC
        # calculate clumping index
        Omega0 = TSEB.CI.calc_omega0_Kustas(F_oak, fC, x_LAD=x_LAD_C, isLAIeff=True)
        Omega = TSEB.CI.calc_omega_Kustas(Omega0, sza, w_C=wC_ratio)
        Omega0_sub = TSEB.CI.calc_omega0_Kustas(LAI_grass, fC_sub, x_LAD=x_LAD_sub, isLAIeff=True)
        Omega_sub = TSEB.CI.calc_omega_Kustas(Omega0_sub, sza, w_C=wC_ratio)

        LAI_oak_eff = F_oak * Omega
        LAI_grass_eff = LAI_grass*Omega_sub

        '''
        plt.figure()
        plt.plot(pd.to_datetime(dates_str)[metdata['hour']==12], LAI_oak[metdata['hour']==12], c='green', label='LAIoak' )
        plt.plot(pd.to_datetime(dates_str)[metdata['hour']==12], LAI_oak_eff[metdata['hour']==12], c='darkgreen', linestyle='--', label='LAIoak_eff' )
        plt.plot(pd.to_datetime(dates_str)[metdata['hour']==12], Omega[metdata['hour']==12], c='red', label='omega' )
        plt.plot(pd.to_datetime(dates_str)[metdata['hour']==12], Omega0[metdata['hour']==12], c='indianred',linestyle = '--', label='Omega0')
        plt.plot(pd.to_datetime(dates_str)[metdata['hour']==12], F_oak[metdata['hour']==12], c='k', label='F_oak')
        plt.legend()
        plt.show()
        plt.close()
        '''




        G_obs = metdata['G'].values
        # Get flux computation method (0: measured G, 1:ratio, 3:time ratio)
        if G_flag == 0:
            calcG = [[0], G_obs]
        elif G_flag == 1:
            calcG = [[1], 0.15]
        elif G_flag == 2:
            calcG = [[2, G_santanello[0], G_santanello[1], G_santanello[2]], metdata['hour'].values]


        Ta_K = metdata_eco['T_A'].values
        ea = metdata_eco['ea_mb'].values
        p = metdata_eco['P_mb'].values
        #p = 1013.1 * np.ones(metdata['DOY'].shape)
        u = metdata_eco['u'].values
        u[u < 0] = 0

        # retrieve incoming shortwave radiation
        SW_in = metdata_eco['SW_in'].values
        SW_in[sza > 90] = 0
        sza[sza > 90] = 90
        # Retrieve Incoming longwave radiation
        LW_in = metdata_eco['LW_in'].values

        # Estimates the direct and diffuse solar radiation
        difvis, difnir, fvis, fnir = TSEB.rad.calc_difuse_ratio(SW_in, sza, press=p)
        skyl = fvis * difvis + fnir * difnir
        Sdn_dir = (1. - skyl) * SW_in
        Sdn_dif = skyl * SW_in

        #Calculate roughness parameters
        #oak --> using raupach94 method for tree
        z_0M_factor, d_0_factor = raupach_94(LAI_oak)
        d_0 = hC*d_0_factor
        z_0M = hC*z_0M_factor

        d_0[d_0 < 0] = 0
        z_0M[z_0M < z0_soil] = z0_soil

        #grass
        z_0M_factor_sub, d_0_factor_sub = raupach_94(LAI_grass)
        d_0_sub = hC_sub*d_0_factor
        z_0M_sub = hC_sub*z_0M_factor
        #[z_0M_sub, d_0_sub] = TSEB.res.calc_roughness(LAI_grass, hC_sub, wC_sub_ratio, np.ones(LAI_oak.shape) * landcover_low)
        d_0_sub[d_0_sub < 0] = 0
        z_0M_sub[z_0M_sub < z0_soil] = z0_soil


        Rn_sw_C, Sn_S, Sn_C_sub = calc_Sn_Campbell(LAI_oak, LAI_grass, sza, Sdn_dir, Sdn_dif, fvis, fnir,
                                                   np.full_like(LAI_oak, spectraVeg_oak['rho_leaf_vis']),
                                                   np.full_like(LAI_oak, spectraVeg_grass['rho_leaf_vis']),
                                                   np.full_like(LAI_oak, spectraVeg_oak['tau_leaf_vis']),
                                                   np.full_like(LAI_oak, spectraVeg_grass['tau_leaf_vis']),
                                                   np.full_like(LAI_oak, spectraVeg_oak['rho_leaf_nir']),
                                                   np.full_like(LAI_oak, spectraVeg_grass['rho_leaf_nir']),
                                                   np.full_like(LAI_oak, spectraVeg_oak['tau_leaf_nir']),
                                                   np.full_like(LAI_oak, spectraVeg_grass['tau_leaf_nir']),
                                                   np.full_like(LAI_oak, spectraGrd['rsoilv']),
                                                   np.full_like(LAI_oak, spectraGrd['rsoiln']), hC, 2.0, wC,fC,
                                                   x_LAD=x_LAD_C, x_LAD_sub=x_LAD_sub, LAI_eff=LAI_oak_eff, LAI_eff_sub=LAI_grass_eff)



        Rn_sw_C[~np.isfinite(Rn_sw_C)] = 0
        Sn_C_sub[~np.isfinite(Sn_C_sub)] = 0
        Sn_S[~np.isfinite(Sn_S)] = 0

        i = metdata['DOY']>0


        [flag_PT_all, T_S, T_C, T_C_sub, T_AC, L_n_sub, L_nC, Ln_C_sub, Ln_S, LE_C, H_C, LE_C_sub, H_C_sub, LE_S, H_S, G_mod, R_S, R_sub, R_X,R_A, u_friction, L, n_iterations, alpha_tree, alpha_grass] = ThreeSEB_PL_doublePT(LST[i],
                                                                                                                                                                              vza,
                                                                                                                                                                              Ta_K[i],
                                                                                                                                                                              u[i],
                                                                                                                                                                              ea[i],
                                                                                                                                                                              p[i],
                                                                                                                                                                              Rn_sw_C[i],
                                                                                                                                                                              Sn_S[i],
                                                                                                                                                                              Sn_C_sub[i],
                                                                                                                                                                              LW_in[i],
                                                                                                                                                                              LAI_oak[i],
                                                                                                                                                                              LAI_grass[i], #LAI_grass[i]
                                                                                                                                                                              hC[i],
                                                                                                                                                                              hC_sub[i],
                                                                                                                                                                              e_v,
                                                                                                                                                                              e_v_grass[i],
                                                                                                                                                                              e_s,
                                                                                                                                                                              z_0M[i],
                                                                                                                                                                              z_0M_sub[i],
                                                                                                                                                                              d_0[i],
                                                                                                                                                                              d_0_sub[i],
                                                                                                                                                                              z_u,
                                                                                                                                                                              z_t,
                                                                                                                                                                              leaf_width=leaf_width,
                                                                                                                                                                              leaf_width_sub=leaf_width_sub,
                                                                                                                                                                              f_c = fC[i],
                                                                                                                                                                              f_c_sub = fC_sub[i],
                                                                                                                                                                              f_g = fg_C_array[i],
                                                                                                                                                                              f_g_sub = fg_sub_array[i],
                                                                                                                                                                              calcG_params = calcG,
                                                                                                                                                                              resistance_form=Resistance_flag)



        validData = np.logical_or.reduce((SW_in < 0, metdata['H'].values == -9999, metdata['SW_in'].values==-9999,
                                          metdata['SW_out'].values==-9999, metdata['LW_out'].values==-9999, metdata['LW_out'].values==-9999,
                                          np.isnan(metdata['H'].values),np.isnan(metdata['Rn'].values), metdata['G'].values == -9999,
                                          np.isnan(metdata['G'].values),
                                          np.isnan(H_C + H_C_sub+H_S)))

                                          
        #=====================================================
        #               Get outputs and plot resutls
        #====================================================

        flag_PT_all[validData] = 255
        #modelled fluxes
        Rn_C_sub = Sn_C_sub + Ln_C_sub
        Rn_S = Sn_S + Ln_S
        Rn_sw_sub = Rn_C_sub + Rn_S
        LE_PT = LE_C + LE_C_sub + LE_S 
        H_PT = H_C + H_C_sub + H_S
        Rn_oak = Rn_sw_C+L_nC
        Rn_PT = Rn_oak + Rn_C_sub + Rn_S
        Sn_PT = Rn_sw_C + Sn_C_sub + Sn_S
        Ln_PT = L_nC + Ln_C_sub + Ln_S
        AE_PT = Rn_PT - G_mod

        #Observed fluxes
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
        # Open output file and write the data
        for row in range(LE_PT.size):
            outData = [site, site_name, int(years_actual[row]), year, int(metdata['DOY'].values[row]), metdata['hour'].values[row], SW_in[row],
                       metdata['LW_in'].values[row], metdata['SW_out'].values[row], metdata['LW_out'].values[row], Rn_obs[row],
                       G_obs[row], H_obs[row], LE_obs[row],Ta_K[row], u[row], LST[row], ea[row], LAI_oak[row],
                       LAI_grass[row], hC[row], hC_sub[row], fC[row], fC_sub[row], fg_C_array[row], fg_sub_array[row],
                       wC[row], wC_sub[row], vza, sza[row], Rn_PT[row], Sn_PT[row], Ln_PT[row], Rn_oak[row], Rn_C_sub[row], Rn_S[row],
                       L_nC[row], Ln_C_sub[row], Ln_S[row], T_C[row],T_C_sub[row], T_S[row], T_AC[row], LE_PT[row],
                       H_PT[row], LE_C[row], H_C[row],LE_C_sub[row], H_C_sub[row], LE_S[row], H_S[row], flag_PT_all[row],
                       z_0M[row], d_0[row],z_0M_sub[row], d_0_sub[row],G_mod[row], R_S[row], R_sub[row], R_X[row], R_A[row],
                       u_friction[row], L[row], alpha_tree[row], alpha_grass[row]]

            writer.writerow(outData)






        def convert_to_daily_average(subhourly_values, subhourly_DOY, daily_DOY, flags, SW_in, percent_condition):
            Daily_array = np.zeros(daily_DOY.shape)
            i = 0
            for day in daily_DOY:
                Daily_values = subhourly_values[np.logical_and(subhourly_DOY == day, SW_in > 25)]
                # do quality check of number of valid values in day [only interested in daytime hours]
                flag_day = flags[np.logical_and(subhourly_DOY == day, SW_in > 25)]
                noNan = np.logical_and.reduce((~np.isnan(Daily_values), Daily_values != -9999, flag_day))
                # valid = (float(np.sum(noNan)) / float(np.sum(time)))
                if float(Daily_values.size) == 0:
                    Daily_array[i] = None
                else:
                    valid = float(np.sum(noNan)) / float(Daily_values.size)
                    # print('%s/%s'%(np.sum(noNan),Daily_values.size ))
                    if valid > percent_condition:
                        Daily_avg_model = np.nanmean(Daily_values[noNan])
                        Daily_array[i] = Daily_avg_model
                        # print(Daily_values[noNan])
                        # print(Daily_avg_model)
                    else:
                        Daily_array[i] = None
                i = i + 1

            return Daily_array


        def model_metrics(X, Y, mask):
            rmse = np.sqrt(np.nanmean((X[mask] - Y[mask]) ** 2))
            cor = st.pearsonr(X[np.logical_and.reduce((mask, ~np.isnan(X), ~np.isnan(Y)))],
                              Y[np.logical_and.reduce((mask, ~np.isnan(Y), ~np.isnan(X)))])[0]
            bias = np.nanmean(X[mask] - Y[mask])
            return rmse, cor, bias

        #function for daily average for all three sources
        def Threelayers_daily(var_T, var_C, var_sC, var_S, DOY_30mins, DOY, flags, SW_in, percent_condition):
            Var_T_daily = convert_to_daily_average(var_T, DOY_30mins, DOY, flags, SW_in, percent_condition)  #total flux
            Var_C_daily = convert_to_daily_average(var_C, DOY_30mins, DOY, flags,SW_in, percent_condition)   #main canopy
            Var_sC_daily = convert_to_daily_average(var_sC, DOY_30mins, DOY, flags,SW_in, percent_condition) #sub-canopy
            Var_S_daily = convert_to_daily_average(var_S, DOY_30mins, DOY, flags, SW_in, percent_condition)  #soil

            return Var_T_daily, Var_C_daily, Var_sC_daily, Var_S_daily

        #function to plot daily time series of the modelled fluxes from 3 sources
        def plot_FluxPartition_3S_dailyTS(DOY, flux_T, flux_C, flux_sC, flux_S, y_range=None, ax=None, label=None):
            if ax is None:
                fig, ax = plt.subplots(1)
            if y_range is None:
                y_range = [np.nanmin(flux_T), np.nanmax(flux_T)]
            if label is None:
                label = 'flux'

            ax.plot(DOY, flux_T, c='k', label='%s (total)'%(label), alpha=0.8)
            ax.plot(DOY, flux_C, c='forestgreen', label='%s (oak)'%(label), alpha=0.8)
            ax.plot(DOY, flux_sC, c='steelblue', label='%s (grass)'%(label), alpha=0.8)
            ax.plot(DOY, flux_S, c='indianred', label='%s (soil)'%(label), alpha=0.8)
            ax.set_ylim(y_range[0], y_range[1])
            ax.set_ylabel(r'%s $\left(W/m^2\right)$'%(label))
            plt.legend()
            return ax

        #function to plot the daily timeseries of observed and modelled fluxes
        def plot_flux_dailyTS_obsVSmodel(DOY, flux_obs, flux_model,y_range=None, ax=None, label=None, c_mod='red'):
            if ax is None:
                fig, ax = plt.subplots(1)
            if y_range is None:
                y_range = [np.nanmin(flux_obs), np.nanmax(flux_obs)]
            if label is None:
                label = 'flux'

            ax.plot(DOY, flux_obs, c='k', label='%s (observed)'%(label), alpha=0.8)
            ax.plot(DOY, flux_model, c=c_mod, label='%s (modeled)'%(label), alpha=0.8, marker='x')
            ax.set_ylim(y_range[0], y_range[1])
            ax.set_ylabel(r'%s $\left(W/m^2\right)$'%(label))

            mask = np.logical_and(~np.isnan(flux_obs), ~np.isnan(flux_model))
            rmse, cor, bias = model_metrics(flux_model, flux_obs, mask)

            ax.text(Timestamp(DOY[15]), 0.7*y_range[1], 'RMSD:%s\nbias:%s\nr:   %s' % (int(np.round(rmse, 0)), int(np.round(bias, 0)),np.round(cor, 2)),
                                                              backgroundcolor='white', linespacing=1.15, family='monospace')

            leg = plt.legend(loc=9, ncol=2)
            for lh in leg.legendHandles:
                lh.set_alpha(1)
            return ax

        flags = np.logical_and.reduce((flag_PT_all < 5, metdata['SW_in'].values > 25, Sn_PT > 25))
        #get daily observed
        H_obs_daily, LE_obs_daily, Rn_obs_daily, G_obs_daily = Threelayers_daily(H_obs, LE_obs, Rn_obs, G_obs, dates_str, dates_daily,flags, SW_in, 0.5)


        #fet daily average resistances
        R_A_daily, R_sub_daily, R_S_daily, R_X_daily = Threelayers_daily(R_A, R_sub, R_S, R_X, dates_str, dates_daily, flags, SW_in, 0.5)

        dates_daily_plot = pd.to_datetime(dates_daily)
        #plot resistances
        fig, ax = plt.subplots()
        ax.plot(dates_daily_plot, R_A_daily, c='black', label='R_A' )
        ax.plot(dates_daily_plot, R_X_daily, c='grey', label='R_X')
        ax.set_ylim(0,150)
        plt.legend()
        ax2 = ax.twinx()
        ax2.plot(dates_daily_plot,  LAI_grass_daily, c='forestgreen', linestyle='--', label='LAI_grass')
        ax.set_ylabel('Resistances (s/m)')
        ax2.set_ylabel('LAI (m2/m2')
        ax2.set_ylim(0, 3)
        ax.set_xlabel('DOY')
        plt.savefig(join(outdir_site, 'Resistances_RaRx_%s_%s_%s_Site%s_%s%s.png' % (res_string, G_string, year, site, LAI_str, LAI_tree_str)))
        plt.close()

        #plot soil/substrate resistances
        fig, ax = plt.subplots()
        ax.plot(dates_daily_plot, R_sub_daily, c='grey', label='R_sub' )
        ax.plot(dates_daily_plot, R_S_daily, c='black', label='R_S')
        ax.set_ylim(0, 250)
        plt.legend()
        ax2 = ax.twinx()
        ax2.plot(dates_daily_plot,  LAI_grass_daily, c='forestgreen', linestyle='--',label='LAI_grass')
        ax.set_ylabel('Resistances (s/m)')
        ax2.set_ylabel('LAI (m2/m2)')
        ax2.set_ylim(0, 3)
        ax.set_xlabel('DOY')
        plt.savefig(join(outdir_site, 'Resistances_RsRsub_%s_%s_%s_Site%s_%s%s.png' % (res_string, G_string, year, site, LAI_str, LAI_tree_str)))
        plt.close()


        #plot modelled daily Rn for three layers
        Rn_daily, Rn_oak, Rn_grass, Rn_soil = Threelayers_daily(Rn_PT,Rn_oak, Rn_C_sub, Rn_S, dates_str, dates_daily,flags, SW_in, 0.5)
        ax = plot_FluxPartition_3S_dailyTS(dates_daily_plot, Rn_daily, Rn_oak, Rn_grass,Rn_soil, y_range=[0, 500], label='Rn')
        plt.savefig(join(outdir_site, 'RN_partition_3S_%s_%s_%s_Site%s_%s%s.png'%(res_string, G_string, year, site, LAI_str, LAI_tree_str)))
        plt.close()

        #plot modelled daily LE for three layers
        LE_daily, LE_oak, LE_grass, LE_soil = Threelayers_daily(LE_PT, LE_C, LE_C_sub, LE_S, dates_str, dates_daily, flags,SW_in, 0.5)
        ax = plot_FluxPartition_3S_dailyTS(dates_daily_plot, LE_daily, LE_oak, LE_grass,LE_soil, y_range=[-50, 350], label='LE')
        plt.savefig(join(outdir_site, 'LE_partition_3S_%s_%s_%s_Site%s_%s%s.png'%(res_string, G_string, year, site, LAI_str, LAI_tree_str)))
        plt.close()

        #plot modelled daily H for three layers
        H_daily, H_oak, H_grass, H_soil = Threelayers_daily(H_PT, H_C, H_C_sub, H_S, dates_str, dates_daily, flags,SW_in, 0.5)
        ax = plot_FluxPartition_3S_dailyTS(dates_daily_plot, H_daily, H_oak, H_grass,H_soil, y_range=[-50, 350], label='H')
        plt.savefig(join(outdir_site, 'H_partition_3S_%s_%s_%s_Site%s_%s%s.png'%(res_string, G_string, year, site, LAI_str, LAI_tree_str)))
        plt.close()

        #plot daily LE (observed and modelled) time series
        ax = plot_flux_dailyTS_obsVSmodel(dates_daily_plot,LE_obs_daily,LE_daily, y_range=[0, 400], label='LE', c_mod='dodgerblue')
        plt.savefig(join(outdir_site, 'LE_Daily_Time_Series_%s_%s_%s_Site%s_%s%s.png' % (res_string, G_string, year, site, LAI_str, LAI_tree_str)))
        plt.close()

        # plot daily H (observed and modelled) time series
        ax = plot_flux_dailyTS_obsVSmodel(dates_daily_plot, H_obs_daily, H_daily, y_range=[0, 400], label='H',c_mod='indianred')
        plt.savefig(join(outdir_site, 'H_Daily_Time_Series_%s_%s_%s_Site%s_%s%s.png' % (res_string, G_string, year, site, LAI_str, LAI_tree_str)))
        plt.close()

        # plot daily Rn (observed and modelled) time series
        ax = plot_flux_dailyTS_obsVSmodel(dates_daily_plot, Rn_obs_daily, Rn_daily, y_range=[0, 600], label='Rn',c_mod='orange')
        plt.savefig(join(outdir_site, 'Rn_Daily_Time_Series_%s_%s_%s_Site%s_%s%s.png' % (res_string, G_string, year, site, LAI_str, LAI_tree_str)))
        plt.close()

        #plot half-hourly modelled vs observed scatter plot
        mask = np.logical_and.reduce((flag_PT_all < 5, metdata['SW_in'].values > 25, Sn_PT > 25))
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

        plt.savefig(join(outdir_site, 'FluxPartitioning_%s_%s_%s_Site%s_%s%s.png' % (res_string, G_string, year, site, LAI_str,LAI_tree_str )))
        plt.close()

        #plot net radiation modeled vs observedscatter plot
        plt.figure()
        plt.scatter(Rn_PT[mask], Rn_obs[mask], c='r', marker='.', alpha=0.2, label='H', s=3)
        plt.xlim(-100, 700)
        plt.ylim(-100, 700)
        plt.xlabel(r'Modeled $\left(W/m^2\right)$', fontsize=14)
        plt.ylabel(r'Observed $\left(W/m^2\right)$', fontsize=14)
        plt.title(str(site_name) + ' ' + str(year))
        plt.plot((-100, 700), (-100, 700), 'k-')
        plt.legend()
        rmse_LE = np.sqrt(np.mean((Rn_PT[mask] - Rn_obs[mask]) ** 2))
        cor_LE = st.pearsonr(Rn_PT[mask], Rn_obs[mask])[0]
        bias_LE = np.mean(Rn_PT[mask] - Rn_obs[mask])

        plt.figtext(0.15, 0.7, 'RMSD: Rn = %s\nbias: Rn = %s\nr:    Rn = %s ' % (
            int(rmse_LE), int(bias_LE), np.round(cor_LE, 2)), backgroundcolor='white', linespacing=1.15, family='monospace')

        leg = plt.legend(loc=9, ncol=2)
        for lh in leg.legendHandles:
            lh.set_alpha(1)

        plt.savefig(join(outdir_site, 'Rn_Tot_vs_obs_%s_%s_%s_Site%s_%s%s.png' % (res_string, G_string, year, site, LAI_str, LAI_tree_str)))
        plt.close()

        #plot SW radiation modeled vs observedscatter plot

        plt.figure()
        plt.scatter(Sn_PT[mask], Sn_obs[mask], c='r', marker='.', alpha=0.2, label='H', s=3)
        plt.xlim(-100, 700)
        plt.ylim(-100, 700)
        plt.xlabel(r'Modeled $\left(W/m^2\right)$', fontsize=14)
        plt.ylabel(r'Observed $\left(W/m^2\right)$', fontsize=14)
        plt.title(str(site_name) + ' ' + str(year))
        plt.plot((-100, 700), (-100, 700), 'k-')
        plt.legend()
        rmse_LE = np.sqrt(np.mean((Sn_PT[mask] - Sn_obs[mask]) ** 2))
        cor_LE = st.pearsonr(Sn_PT[mask], Sn_obs[mask])[0]
        bias_LE = np.mean(Sn_PT[mask] - Sn_obs[mask])


        plt.figtext(0.15, 0.7, 'RMSD: Rn = %s\nbias: Rn = %s\nr:    Rn = %s ' % (
            int(rmse_LE), int(bias_LE), np.round(cor_LE, 2)),
                    backgroundcolor='white', linespacing=1.15, family='monospace')

        leg = plt.legend(loc=9, ncol=2)
        for lh in leg.legendHandles:
            lh.set_alpha(1)

        plt.savefig(join(outdir_site, 'RN_SW_vs_obs_%s_%s_%s_Site%s_%s%s.png' % (res_string, G_string, year, site, LAI_str, LAI_tree_str)))
        plt.close()

        #plot LW radiation modeled vs observedscatter plot
        plt.figure()
        plt.scatter(Ln_PT[mask], Ln_obs[mask], c='r', marker='.', alpha=0.2, label='H', s=3)
        plt.xlim(-100, 700)
        plt.ylim(-100, 700)
        plt.xlabel(r'Modeled $\left(W/m^2\right)$', fontsize=14)
        plt.ylabel(r'Observed $\left(W/m^2\right)$', fontsize=14)
        plt.title(str(site_name) + ' ' + str(year))
        plt.plot((-100, 700), (-100, 700), 'k-')
        plt.legend()
        rmse_LE = np.sqrt(np.mean((Ln_PT[mask] - Ln_obs[mask]) ** 2))
        cor_LE = st.pearsonr(Ln_PT[mask], Ln_obs[mask])[0]
        bias_LE = np.mean(Ln_PT[mask] - Ln_obs[mask])

        plt.figtext(0.15, 0.7, 'RMSD: Rn = %s\nbias: Rn = %s\nr:    Rn = %s ' % (
            int(rmse_LE), int(bias_LE), np.round(cor_LE, 2)),
                    backgroundcolor='white', linespacing=1.15, family='monospace')

        leg = plt.legend(loc=9, ncol=2)
        for lh in leg.legendHandles:
            lh.set_alpha(1)

        plt.savefig(join(outdir_site, 'RN_LW_vs_obs_%s_%s_%s_Site%s_%s%s.png' % (res_string, G_string, year, site, LAI_str, LAI_tree_str)))
        plt.close()

        # plot G modeled vs observed scatter plot
        plt.figure()
        plt.scatter(G_mod[mask], G_obs[mask], c='black', marker='.', alpha=0.2, s=3)
        plt.xlim(-100, 300)
        plt.ylim(-100, 300)
        plt.xlabel(r'Modeled $\left(W/m^2\right)$', fontsize=14)
        plt.ylabel(r'Observed $\left(W/m^2\right)$', fontsize=14)
        plt.title('G ' + str(site_name) + ' ' + str(year))
        plt.plot((-100, 300), (-100, 300), 'k-')
        rmse_LE = np.sqrt(np.mean((G_mod[mask] - G_obs[mask]) ** 2))
        cor_LE = st.pearsonr(G_mod[mask], G_obs[mask])[0]
        bias_LE = np.mean(G_mod[mask] - G_obs[mask])

        plt.figtext(0.15, 0.7, 'RMSD: Rn = %s\nbias: Rn = %s\nr:    Rn = %s ' % (
            int(rmse_LE), int(bias_LE), np.round(cor_LE, 2)),
                    backgroundcolor='white', linespacing=1.15, family='monospace')

        plt.savefig(join(outdir_site, 'G_vs_obs_%s_%s_%s_Site%s_%s%s.png' % (res_string, G_string, year, site, LAI_str, LAI_tree_str)))
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

        plt.figtext(0.15, 0.7, 'RMSD: Rn = %s\nbias: Rn = %s\nr:    Rn = %s ' % (
            int(rmse_LE), int(bias_LE), np.round(cor_LE, 2)),
                    backgroundcolor='white', linespacing=1.15, family='monospace')

        plt.savefig(join(outdir_site, 'AE_vs_obs_%s_%s_%s_Site%s_%s%s.png' % (res_string, G_string, year, site, LAI_str, LAI_tree_str)))
        plt.close()


        #get daily averge of alpha values for both trees and grasses
        alpha_tree_daily = convert_to_daily_average(alpha_tree, dates_str, dates_daily, flags, SW_in, 0.5)
        alpha_grass_daily = convert_to_daily_average(alpha_grass, dates_str, dates_daily, flags, SW_in, 0.5)
        plt.figure()
        plt.plot(dates_daily_plot, alpha_tree_daily, c='green', label='Alpha_PT (tree)')
        plt.plot(dates_daily_plot, alpha_grass_daily, c='indianred', label='Alpha_PT (grass)')
        plt.ylim(0, 1.3)
        plt.legend()
        plt.ylabel('Alpha_PT (-)')
        plt.savefig(join(outdir_site, 'Alpha_PT_grass_tree_%s_%s_%s_Site%s_%s%s.png'%(res_string, G_string, year, site,LAI_str, LAI_tree_str)))
        plt.close()