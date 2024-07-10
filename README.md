# Code for Photolysis Parameterisation

Code used in the GEOS-Chem parameterisation of photolysis rate calculations.

## Description

Fast-JX is the photolysis scheme used in GEOS-Chem version 14.2.2 ([10.5281/zenodo10034814](https://zenodo.org/records/10034814)) and in this project we parameterise these photolysis rate calculations through an ensemble of XGBoost models. Using [XGBoost version 1.6.0](https://xgboost.readthedocs.io/en/stable/) we train the models in python and then implement this models in Fortran in GEOS-Chem.

## Usage
### Files in Repo.
* **Dataset_Maker_year.py** - Script used to prepare and engineer the data from the nc4 files from GEOS-Chem to parquet files.
* **Dataset_Maker_year_XGB.py** - Script used to train the collection of xgboost photolysis models.
* **xgb_pred_J.F90** - Fortran file containing the subroutine used in GEOS-Chem to predict during routine with XGBoost models.

### Requirements
In GEOS-Chem version 14.2.2, the [fortran2xgb](https://github.com/christophkeller/fortran2xgb) API is used to enable XGBoost inside Fortran. 
XGBoost is installed in a conda environment, this is pointed to during compilation. 

## License
This project is licensed under the MIT License - see the LICENSE.md file for details.



