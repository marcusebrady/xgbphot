import xarray as xr
import gc
import dask
import pandas as pd
import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import cartopy.feature as cfeature
import math
import xgboost as xgb
from datetime import datetime
import time
from typing import Tuple
import seaborn as sns
import shap
from dateutil.relativedelta import relativedelta

print('Imported Modules')


#USE BELOW FOR JVAL LIST 1-3
'''
def generate_file_paths(date):
    base_path = '/users/vxv505/scratch/GCruns/gc_4x5_merra2_fullchem/OutputDir/'
    date_str = date.strftime('%Y%m%d')
    return {
        'jvals': f'{base_path}GEOSChemDEFAULTRUNYEARSWORTH.JValues.{date_str}_0000z.nc4',
        'state_met': f'{base_path}GEOSChemDEFAULTRUNYEARSWORTH.StateMet.{date_str}_0000z.nc4',
        'aerosols': f'{base_path}GEOSChemDEFAULTRUNYEARSWORTH.Aerosols.{date_str}_0000z.nc4'
    }
'''
#USE BELOW FOR JVAL LIST 4
def generate_file_paths(date):
    base_path = '/users/vxv505/scratch/GCruns/gc_4x5_merra2_fullchem/OutputDir/'
    date_str = date.strftime('%Y%m%d')
    return {
        'jvals': f'{base_path}GEOSChemMODIFIED.JValues.{date_str}_0000z.nc4',
        'state_met': f'{base_path}GEOSChemDEFAULTRUNYEARSWORTH.StateMet.{date_str}_0000z.nc4',
        'aerosols': f'{base_path}GEOSChemDEFAULTRUNYEARSWORTH.Aerosols.{date_str}_0000z.nc4'
    }


#DEFAULTRUNYEARSWORTH


#JVAL LIST 1
#jval_list = ['JvalO3O1D', 'JvalO3O3P', 'Jval_ACET', 'Jval_ALD2', 'Jval_BALD',  'Jval_Br2', 'Jval_BrCl', 'Jval_BrNO2', 'Jval_BrNO3', 'Jval_BrO', 'Jval_CCl4', 'Jval_CFC11', 'Jval_CFC113', 'Jval_CFC114', 'Jval_CFC115', 'Jval_CFC12', 'Jval_CH2Br2', 'Jval_CH2Cl2', 'Jval_CH2I2', 'Jval_CH2IBr', 'Jval_CH2ICl', 'Jval_CH2O', 'Jval_CH3Br', 'Jval_CH3CCl3', 'Jval_CH3Cl', 'Jval_CH3I', 'Jval_CHBr3', 'Jval_CHCl3', 'Jval_Cl2', 'Jval_Cl2O2', 'Jval_ClNO2', 'Jval_ClNO3', 'Jval_ClO', 'Jval_ClOO', 'Jval_ETHLN', 'Jval_ETNO3', 'Jval_ETP', 'Jval_GLYC', 'Jval_GLYX', 'Jval_H1211', 'Jval_H1301', 'Jval_H2402']

#JVAL LIST 2
#jval_list = ['Jval_H2O2', 'Jval_HAC', 'Jval_HCFC123', 'Jval_HCFC141b', 'Jval_HCFC142b', 'Jval_HCFC22', 'Jval_HMHP', 'Jval_HNO2', 'Jval_HNO3', 'Jval_HNO4', 'Jval_HOBr', 'Jval_HOCl', 'Jval_HOI', 'Jval_HONIT', 'Jval_HPALD1', 'Jval_HPALD2', 'Jval_HPALD3', 'Jval_HPALD4', 'Jval_HPETHNL', 'Jval_I2', 'Jval_I2O2', 'Jval_I2O3', 'Jval_I2O4', 'Jval_IBr', 'Jval_ICN', 'Jval_ICl', 'Jval_IDCHP', 'Jval_IDHDP', 'Jval_IDN', 'Jval_IHN1', 'Jval_IHN2', 'Jval_IHN3']

#JVAL LIST 3
#jval_list =['Jval_ICPDH','Jval_IHN4', 'Jval_INO', 'Jval_INPB', 'Jval_IO', 'Jval_IONO', 'Jval_IONO2', 'Jval_IPRNO3', 'Jval_ITCN', 'Jval_ITHN', 'Jval_MACR', 'Jval_MACR1OOH', 'Jval_MCRENOL', 'Jval_MCRHN', 'Jval_MCRHNB', 'Jval_MCRHP', 'Jval_MEK', 'Jval_MENO3', 'Jval_MGLY', 'Jval_MONITS', 'Jval_MONITU', 'Jval_MP', 'Jval_MPN', 'Jval_MVK', 'Jval_MVKHC', 'Jval_MVKHCB', 'Jval_MVKN', 'Jval_MVKPC', 'Jval_N2O', 'Jval_N2O5', 'Jval_NIT', 'Jval_NITs', 'Jval_NO', 'Jval_NO2', 'Jval_NO3', 'Jval_NPHEN', 'Jval_NPRNO3', 'Jval_O2', 'Jval_O3', 'Jval_OCS', 'Jval_OClO', 'Jval_OIO', 'Jval_PAN', 'Jval_PIP',  'Jval_PROPNN', 'Jval_PYAC', 'Jval_R4N2', 'Jval_RCHO',   'Jval_SO4']

#JVAL LIST 4 REQUIRES THE MODIFIED JVAL CODE
jval_list = ['Jval_CF3I', 'Jval_SO2', 'Jval_H2COa', 'Jval_H2COb', 'Jval_ClNO3a', 'Jval_ClNO3b', 'Jval_ActAld', 'Jval_ActAlx', 'Jval_Glyxla', 'Jval_Glyxlb', 'Jval_Glyxlc', 'Jval_Acet-a', 'Jval_Acet-b']

#================================================
#       FUNCTIONS FOR MAIN
#
#================================================


def calculate_cumulative_sums(group):
    # Ensure the group is sorted by 'lev'
    group = group.sort_values(by='lev')

    # Calculate cumulative sums for Met_OPTD
    #group['cumsum_asc'] = group['Met_OPTD'].cumsum()
    #group['cumsum_desc'] = group['Met_OPTD'][::-1].cumsum()
    #group['OPTD_above'] = group['cumsum_asc'] - group['Met_OPTD']
    #group['OPTD_below'] = group['cumsum_desc'].shift(-1).fillna(0)

    # Calculate cumulative sums for Met_TAUCLW
    group['cumsum_asc_tauclw'] = group['Met_TAUCLW'].cumsum()
    group['cumsum_desc_tauclw'] = group['Met_TAUCLW'][::-1].cumsum()
    group['TAUCLW_above'] = group['cumsum_asc_tauclw'] - group['Met_TAUCLW']
    group['TAUCLW_below'] = group['cumsum_desc_tauclw'].shift(-1).fillna(0)

    # Calculate cumulative sums for Met_TAUCLI
    group['cumsum_asc_taucli'] = group['Met_TAUCLI'].cumsum()
    group['cumsum_desc_taucli'] = group['Met_TAUCLI'][::-1].cumsum()
    group['TAUCLI_above'] = group['cumsum_asc_taucli'] - group['Met_TAUCLI']
    group['TAUCLI_below'] = group['cumsum_desc_taucli'].shift(-1).fillna(0)

    # Calculate cumulative sums for Met_CLDF
    group['cumsum_asc_cldf'] = group['Met_CLDF'].cumsum()
    group['cumsum_desc_cldf'] = group['Met_CLDF'][::-1].cumsum()
    group['CLDF_above'] = group['cumsum_asc_cldf'] - group['Met_CLDF']
    group['CLDF_below'] = group['cumsum_desc_cldf'].shift(-1).fillna(0)

    return group

def preprocess3(ds):
    all_vars = set(jval_list + ['Met_SUNCOS', 'Met_UVALBEDO', 'Met_TO3', 'Met_PMID', 'Met_T', 'Met_OPTD', 'Met_AIRDEN', 'Met_CLDF', 'Met_TAUCLI', 'Met_TAUCLW'])

    all_vars.update(['AODDust1000nm_bin1', 'AODDust1000nm_bin7'])

    available_vars = set(ds.variables.keys())
    intersect_vars = list(all_vars.intersection(available_vars))
    return ds[intersect_vars]
print('Functions Defined')

#================================================
#       PREPARING DATA FOR PROCESSING
#
#================================================
#------------------------------------------------
#                 Opening Data
#------------------------------------------------
all_datasets = []

start_date = datetime(2019, 7, 1)
end_date = datetime(2020, 7, 1)


current_date = start_date
while current_date <= end_date:
    paths = generate_file_paths(current_date)
        
    print(f'The current path: {paths}')
    # Open each dataset separately and print dimensions
    #ds_jvals = xr.open_dataset(paths['jvals'])
    #ds_state_met = xr.open_dataset(paths['state_met'])
    #ds_aerosols = xr.open_dataset(paths['aerosols'])

    #print(f"JVALS {current_date}: {ds_jvals.dims}")
    #print(f"State Met {current_date}: {ds_state_met.dims}")
    #print(f"Aerosols {current_date}: {ds_aerosols.dims}") 
    combined_ds = xr.open_mfdataset(
        [paths['jvals'], paths['state_met'], paths['aerosols']],
        preprocess=preprocess3,
        combine='by_coords',
        parallel=True
    )
    all_datasets.append(combined_ds)


    current_date += relativedelta(months=1)





combined_ds = xr.concat(all_datasets, dim='time')
for ds in all_datasets:
    ds.close()

DEBUG = True


if DEBUG:
    print('Selecting the first 90 percent of times')
    length_of_time_dim = combined_ds.dims['time']
    index_times = int(length_of_time_dim * 0.90)
    combined_ds = combined_ds.isel(time=slice(0,index_times))

#------------------------------------------------
#      Turning to DataFrame & Features 
#------------------------------------------------

print('Converting to dataframe')

master_df = combined_ds.to_dataframe()

combined_ds.close()

gc.collect()
if DEBUG:

    #print(master_df.shape)
    #print(master_df.head())
    for col in master_df.columns:
        print(f'Stats for {col}')
        print(master_df[col].describe(), '\n')
master_df = master_df.reset_index()

master_df['SZA'] = np.arccos(master_df['Met_SUNCOS']) * 180 / np.pi
master_df['time'] = master_df['time'].astype('int64') // 10**9
master_df= master_df[master_df['SZA'] <= 98]
print(master_df.shape)


print('Converted time type and calculated SZA')
master_df = master_df.groupby(['lat','lon','time'],group_keys=False).apply(calculate_cumulative_sums).reset_index(drop=True)

cols_to_drop = [ 'cumsum_asc_tauclw', 'cumsum_asc_taucli', 'cumsum_asc_cldf', 'cumsum_desc_tauclw', 'cumsum_desc_taucli', 'cumsum_desc_cldf']

master_df.drop(columns=cols_to_drop, inplace=True)
#feature_order = ['lev','Met_SUNCOS', 'Met_UVALBEDO', 'Met_TO3', 'SZA', 'Met_PMID', 'Met_T', 'Met_OPTD','Met_AIRDEN', 'Met_CLDF','Met_TAUCLI', 'Met_TAUCLW','AODDust1000nm_bin1','AODDust1000nm_bin7']
#feature_order=['Met_SUNCOS', 'Met_UVALBEDO', 'Met_TO3', 'SZA', 'Met_PMID', 'Met_T', 'Met_OPTD']

constants = {}
for jval in jval_list:
    constant = master_df[jval][master_df[jval] > 0].min()
    master_df[jval] = np.log(master_df[jval] + constant)
    constants[jval] = constant
print('==================================================''\n\n\n\n')
print(constants)
print('==================================================')

for col in master_df.columns:
    print(f'Stats for {col}')
    print(master_df[col].describe(), '\n')
#master_df = combined_ds.to_dataframe()
parqeut_file_path = '/users/vxv505/scratch/conda_environments/year4_train.parquet'
master_df.to_parquet(parqeut_file_path, engine='pyarrow')
print('Saved as Parquet')

