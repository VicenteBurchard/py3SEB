# 3SEB

## Synopsis

This project contains experimental *Python* code for the *Three Source Energy Balance*  (3SEB) model using a nested double Priestley-Taylor intialization (**ThreeSEB_PT**) for estimating sensible and latent heat flux (evapotranspiration) based on measurements of radiometric surface temperature over tree-grass ecosystems. 

3SEB is a modified version of Two-Source Energy Balance (TSEB) model (Norman et al. 1995). It incorporates an additional vegetation source within its model structure (i.e., overstory vegetation, understory vegetation and soil). It largely uses and is based on functions developed from the python implementation of TSEB (pyTSEB, which can be found here: https://github.com/hectornieto/pyTSEB/).

This repository contains a high-level script (**Main_run3SEB_ground.py**) to run 3SEB with the example datasets provided in **Inputs** using data from:

AU-Dry (Obtained from Ozflux portal: https://www.ozflux.org.au/monitoringsites/dryriver/)

ES-LM1 (Obtained from Zenodo public repository: https://zenodo.org/record/4453567#.YWMEDNpByyw)

ES-Abr (Obtained from Zenodo public repository: https://zenodo.org/record/3707842#.YWMEAtpByyz)

US-Ton (Obtained from Ameriflux portal: https://ameriflux.lbl.gov/sites/siteinfo/US-Ton)

The script **Functions_3SEB.py** provide the core functions necesarry to run 3SEB.


## Installation
The following Python libraries will be required:
- pyTSEB: https://github.com/hectornieto/pyTSEB/
- Numpy
- pandas
- matplotlib

## Basic Contents
### High-level scripts
- *Main_run3SEB_ground.py*  high level scripts for running **ThreeSEB_PT** with the example inputs

### Low-level module
The low-level module in this project is aimed at providing customisation and more flexibility in running 3SEB. 
The following module is included:

- *Functions_3SEB*
> core functions for running different 3SEB models (`ThreeSEB_PT(*args,**kwargs)`). 

## Tests
The folders *.Inputs/Meteo_data* and *.Inputs/Veg_data* contains the example for running 3SEB. Just run the high-level scripts *Main_run3SEB_ground.py* and see the resulting outputs stored in *./Output/*

## Main Scientific References
- Burchard-Levine V, Nieto H, Riaño D, Migliavacca M, El-Madany TS, Perez-Priego O, Carrara A, Martín MP. Seasonal Adaptation of the Thermal-Based Two-Source Energy Balance Model for Estimating Evapotranspiration in a Semiarid Tree-Grass Ecosystem. Remote Sensing. 2020; 12(6):904. doi: 10.3390/rs12060904
- Norman,  J.  M.,  Kustas,  W.  P.,  Prueger,  J.  H.,  and  Diak,  G.  R.: Surface  flux  estimation  using  radiometric  temperature:  a  dual-temperature-difference method to minimize measurement errors, Water  Resour.  Res.,  36,  2263,  doi: 10.1029/2000WR900033, 2000
- Norman,  J.,  Kustas,  W.,  and  Humes,  K.:  A  two-source  approach for estimating soil and vegetation fluxes from observations of directional radiometric surface temperature, Agr. Forest Meteorol., 77, 263–293, doi: 10.1016/0168-1923(95)02265-Y, 1995
- Kustas, W. P. and Norman, J. M.: A two-source approach for estimating turbulent fluxes using multiple angle thermal infrared observations, Water Resour. Res., 33, 1495–1508, 199
- Kustas,  W.  P.  and  Norman,  J.  M.:  Evaluation  of  soil  and  vegetation heat flux prediction using a simple two-source model with radiometric  temperatures  for  partial  canopy  cover,  Agr.  Forest Meteorol., 94, 13–29, 199
- Guzinski, R., Nieto, H., Stisen, S., and Fensholt, R.: Inter-comparison of energy balance and hydrological models for land surface energy flux estimation over a whole river catchment, Hydrol. Earth Syst. Sci., 19, 2017-2036, doi:10.5194/hess-19-2017-2015, 2015.
- William P. Kustas, Hector Nieto, Laura Morillas, Martha C. Anderson, Joseph G. Alfieri, Lawrence E. Hipps, Luis Villagarcía, Francisco Domingo, Monica Garcia: Revisiting the paper “Using radiometric surface temperature for surface energy flux estimation in Mediterranean drylands from a two-source perspective”, Remote Sensing of Environment, In Press. doi:10.1016/j.rse.2016.07.024.

## Contributors
- **Vicente Burchard-Levine** <vicentefelipe.burchard@cchs.csic.es> <vburchardlevine@gmail.com> main developer
- **Hector Nieto** <hector.nieto@complutig.com> <hector.nieto.solana@gmail.com> TSEB modeling, tester 
- **William P. Kustas** TSEB modeling, tester 

## License
3SEB: a Python Three Source Energy Balance Model

Copyright 2020 Vicente Burchard-Levine and contributors.
    
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
