import numpy as np
from collections import deque
import time

from pyTSEB import TSEB
from pyTSEB import meteo_utils as met
from pyTSEB import resistances as res
from pyTSEB import MO_similarity as MO
from pyTSEB import net_radiation as rad
from pyTSEB import clumping_index as CI

def calc_effective_resistances_SW_3source(R_A, R_X_ov, R_X_un, R_S,
                                          R_c_ov, R_c_un, R_ss, delta, psicr):
    '''
    calculate effetice resistances to water vapour transport fpr 3-source version of SW model
    as defined in Lhomme et al. 2012

    Parameters
    ----------
    R_A: float
        Aerodynamic resistance to heat transport (s m-1)
    R_X_ov: float
        Bulk overstory canopy aerodynamic resistance to heat transport (s m-1)
    R_X_un: float
        Bulk understory canopy aerodynamic resistance to heat transport (s m-1)
    R_S: float
        Soil aerodynamic resistance to heat transport (s m-1)
    R_c_ov: float
        Overstory Canopy bulk stomatal resistance (s m-1)
    R_c_un: float
        Understory Canopy bulk stomatal resistance (s m-1)
    R_ss: float
        Resistance to water vapour transport in the soil surface (s m-1)
    delta: float
        Slope of the saturation water vapour pressure (kPa K-1)
    psicr:
        Psicrometric constant (mb K-1)

    Returns
    -------

    '''

    delta_psicr = delta + psicr

    R_a_SW = delta_psicr * R_A  # Eq. 16 [Shuttleworth1988]_ # eq.60  Lhomme et al. 2012
    R_s_SW = delta_psicr * R_S + psicr * R_ss  # Eq. 17 [Shuttleworth1988]_
    R_un_SW = delta_psicr * R_X_un + psicr * R_c_un   # Eq.58 Lhomme et al. 2012
    R_ov_SW = delta_psicr * R_X_ov + psicr * R_c_ov  # # Eq.57 Lhomme et al. 2012

    DE = ((R_un_SW * R_ov_SW * R_s_SW) + (R_ov_SW * R_s_SW * R_a_SW) +
          (R_ov_SW * R_un_SW * R_a_SW) + (R_un_SW * R_s_SW * R_a_SW)) # Eq. 64 Lhomme et al. 2012

    C_ov = (R_un_SW * R_s_SW * (R_ov_SW+R_a_SW)) / DE # Eq. 61 Lhomme et al. 2012
    C_un = (R_ov_SW * R_s_SW * (R_un_SW+R_a_SW)) / DE # Eq. 62 Lhomme et al. 2012
    C_s  =  (R_un_SW * R_ov_SW * (R_s_SW+R_a_SW)) / DE # Eq. 63 Lhomme et al. 2012

    '''
    C_c = 1. / (1. + R_ov_SW * R_a_SW / (
            R_s_SW * (R_ov_SW + R_a_SW)))  # Eq. 14 [Shuttleworth1988]_
    C_s = 1. / (1. + R_s_SW * R_a_SW / (
            R_c_SW * (R_s_SW + R_a_SW)))  # Eq. 15 [Shuttleworth1988]_
    '''
    C_ov[np.isnan(C_ov)] = 0
    C_un[np.isnan(C_un)] = 0
    C_s[np.isnan(C_s)] = 0

    return R_a_SW, R_s_SW, R_un_SW, R_ov_SW, C_s, C_un, C_ov

def calc_component_temperature_montes(A, Ra, Rs, T_AC, vpd_0, delta, rho_cp, psicr):
    '''

    Calculate component temperature of i (=overstory, understory or soil) based on Appendix B of Montes et al. 2014

    Ref: Montes, C., Lhomme, J.-P., Demarty, J., Prévot, L., & Jacob, F. (2014).
    A three-source SVAT modeling of evaporation: Application to the seasonal dynamics
    of a grassed vineyard. Agricultural and Forest Meteorology, 191, 64–80. https://doi.org/10.1016/j.agrformet.2014.02.004

    Parameters
    ----------
    A: float
        Available energy of layer i (=overstory, understory or soil)
    Ra: float
        Aerodyamic resistance between evaporative source to mean source height (e.g., R_X_un, R_X_ov or R_S)
    Rs: float
        surface resistance (stomatal or soil surface) of source i (e.g. R_c_ov, R_c_un or Rss)
    T_AC: float
        aerodynamic temperature at mean canopy source height
    vpd_0: float
        vapour pressure deficit at mean canopy source height
    delta: float
        slope of saturated vapor pressure curve at air temperature
    psicr
        Psicrometric constant (mb K-1)
    n: int
        2 if foliage and 1 if subsrate (un,soil)

    Returns
    -------
    T_i float
        component temperature of source i (=overstory, understory or soil)
    '''
    # Eq. B3 in Montes et al. 2014
    T_i = ((Ra + Rs) * (A / rho_cp) - vpd_0 / psicr) / (1 + delta / psicr + Rs / Ra) + T_AC

    return T_i