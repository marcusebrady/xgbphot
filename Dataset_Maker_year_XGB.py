#print('Imported xr')
import gc
#print('Imported GC')
import dask
#print('Imported dask')
import pandas as pd
#print('Imported Pandas')
import numpy as np
#print('Imported no')
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import cartopy.feature as cfeature #for borders, land etc
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import math
import xgboost as xgb
#print('Imported xgb')
from datetime import datetime
from typing import Tuple
import time
import seaborn as sns
import shap

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def model_maker(species_id, dtrain, dval):
    random_state = 2002
    objective = 'reg:squarederror'
    param = {
        'random_state':random_state,
        'eta': 0.005,
        'max_depth': 12,
        'colsample_bytree':0.8,
        'alpha':1,
        'gamma':1,
        'lambda':1,
        'objective': objective,
        'tree_method': 'gpu_hist',
        'verbosity':0
    }
    num_round = 2000
    evals_result = {}

    model = xgb.train(
        params=param,
        dtrain=dtrain,
        num_boost_round=num_round,
        evals=[(dtrain, 'train'), (dval, 'val')],
        evals_result=evals_result,
        verbose_eval=False
        )

    model.save_model(f'/users/vxv505/scratch/conda_environments/final_model/xgb_model_{species_id}.model')
    return model, evals_result['train']['rmse'], evals_result['val']['rmse']


def dmatrix_maker(X_train, y_train, X_eval, y_eval, X_test, y_test):
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_eval, label=y_eval)
    dtest = xgb.DMatrix(X_test, label=y_test)

    return dtrain, dval, dtest


def plot_eval(train_error, val_error, species_id):
    plt.figure(figsize=(10, 5))
    plt.plot(train_error, label='Training RMSE')
    plt.plot(val_error, label='Validation RMSE')
    plt.xlabel('Boosting Rounds')
    plt.ylabel('RMSE')
    plt.title('XGBoost Learning Curve')
    plt.legend()
    plt.savefig(f'/users/vxv505/scratch/conda_environments/final_model_plots/xgb_model_eval_parquet_{species_id}.png', bbox_inches='tight', dpi=600)
    plt.close()


year_no = 'year4'
master_df = pd.read_parquet(f'/users/vxv505/scratch/conda_environments/{year_no}_train.parquet', engine='pyarrow')
print('Read Parquet')
pd.set_option('display.max_columns', None)
print(master_df.head())
if year_no == 'year1':
    #jval_list = ['JvalO3O1D','JvalO3O3P']
    jval_list = ['JvalO3O1D', 'JvalO3O3P', 'Jval_ACET', 'Jval_ALD2', 'Jval_BALD',  'Jval_Br2', 'Jval_BrCl', 'Jval_BrNO2', 'Jval_BrNO3', 'Jval_BrO', 'Jval_CCl4', 'Jval_CFC11', 'Jval_CFC113', 'Jval_CFC114', 'Jval_CFC115', 'Jval_CFC12', 'Jval_CH2Br2', 'Jval_CH2Cl2', 'Jval_CH2I2', 'Jval_CH2IBr', 'Jval_CH2ICl', 'Jval_CH2O', 'Jval_CH3Br', 'Jval_CH3CCl3', 'Jval_CH3Cl', 'Jval_CH3I', 'Jval_CHBr3', 'Jval_Cl2', 'Jval_Cl2O2', 'Jval_ClNO2', 'Jval_ClNO3', 'Jval_ClO', 'Jval_ClOO', 'Jval_ETHLN', 'Jval_ETNO3', 'Jval_ETP', 'Jval_GLYC', 'Jval_GLYX', 'Jval_H1211', 'Jval_H1301', 'Jval_H2402']
elif year_no == 'year2':
    jval_list = ['Jval_H2O2', 'Jval_HAC', 'Jval_HCFC123', 'Jval_HCFC141b', 'Jval_HCFC142b', 'Jval_HCFC22', 'Jval_HMHP', 'Jval_HNO2', 'Jval_HNO3', 'Jval_HNO4', 'Jval_HOBr', 'Jval_HOCl', 'Jval_HOI', 'Jval_HONIT', 'Jval_HPALD1', 'Jval_HPALD2', 'Jval_HPALD3', 'Jval_HPALD4', 'Jval_HPETHNL', 'Jval_I2', 'Jval_I2O2', 'Jval_I2O3', 'Jval_I2O4', 'Jval_IBr', 'Jval_ICN', 'Jval_ICl', 'Jval_IDCHP', 'Jval_IDHDP', 'Jval_IDN', 'Jval_IHN1', 'Jval_IHN2', 'Jval_IHN3']
elif year_no == 'year3':
    #jval_list = ['Jval_NO3','Jval_NO','Jval_NO2','Jval_O3']
    jval_list = ['Jval_ICPDH','Jval_IHN4', 'Jval_INO', 'Jval_INPB', 'Jval_IO', 'Jval_IONO', 'Jval_IONO2', 'Jval_IPRNO3', 'Jval_ITCN', 'Jval_ITHN', 'Jval_MACR', 'Jval_MACR1OOH', 'Jval_MCRENOL', 'Jval_MCRHN', 'Jval_MCRHNB', 'Jval_MCRHP', 'Jval_MEK', 'Jval_MENO3', 'Jval_MGLY', 'Jval_MONITS', 'Jval_MONITU', 'Jval_MP', 'Jval_MPN', 'Jval_MVK', 'Jval_MVKHC', 'Jval_MVKHCB', 'Jval_MVKN', 'Jval_MVKPC', 'Jval_N2O', 'Jval_N2O5', 'Jval_NIT', 'Jval_NITs', 'Jval_NO', 'Jval_NO2', 'Jval_NO3', 'Jval_NPHEN', 'Jval_NPRNO3', 'Jval_O2', 'Jval_O3', 'Jval_OCS', 'Jval_OClO', 'Jval_OIO', 'Jval_PAN', 'Jval_PIP',  'Jval_PROPNN', 'Jval_PYAC', 'Jval_R4N2', 'Jval_RCHO',   'Jval_SO4']
elif year_no =='year4':
    jval_list = ['Jval_CF3I', 'Jval_H2COa', 'Jval_H2COb', 'Jval_ClNO3a', 'Jval_ClNO3b', 'Jval_ActAld',  'Jval_Glyxla', 'Jval_Glyxlb', 'Jval_Glyxlc', 'Jval_Acet-a', 'Jval_Acet-b']

else:
    print('Test Parquet not defined')


constants = {}
#Below List for Consants from the july month for jval list 1, 2, 3
#constants = {'JvalO3O1D': 1.2237375e-11, 'JvalO3O3P': 1.1831617e-09, 'Jval_ACET': 1.6966958e-13, 'Jval_ALD2': 2.8489476e-12, 'Jval_BALD': 7.5236505e-11, 'Jval_Br2': 1.1601841e-07, 'Jval_BrCl': 5.346989e-08, 'Jval_BrNO2': 3.0139745e-08, 'Jval_BrNO3': 6.898928e-09, 'Jval_BrO': 1.7312884e-07, 'Jval_CCl4': 6.072607e-36, 'Jval_CFC11': 2.2066945e-36, 'Jval_CFC113': 2.9244467e-37, 'Jval_CFC114': 1.7984768e-38, 'Jval_CFC115': 1.21646e-39, 'Jval_CFC12': 1.4267726e-37, 'Jval_CH2Br2': 6.613508e-18, 'Jval_CH2Cl2': 2.341617e-37, 'Jval_CH2I2': 2.7680464e-08, 'Jval_CH2IBr': 1.395895e-09, 'Jval_CH2ICl': 3.770074e-10, 'Jval_CH2O': 2.610891e-10, 'Jval_CH3Br': 1.155432e-25, 'Jval_CH3CCl3': 3.7910016e-36, 'Jval_CH3Cl': 3.497752e-38, 'Jval_CH3I': 1.178823e-11, 'Jval_CHBr3': 1.3854249e-12,  'Jval_Cl2': 1.209261e-08, 'Jval_Cl2O2': 8.263315e-09, 'Jval_ClNO2': 1.481781e-09, 'Jval_ClNO3': 2.1935263e-10, 'Jval_ClO': 6.222837e-11, 'Jval_ClOO': 1.5329666e-06, 'Jval_ETHLN': 3.4546888e-10, 'Jval_ETNO3': 2.057413e-12, 'Jval_ETP': 9.696014e-12, 'Jval_GLYC': 1.31978395e-11, 'Jval_GLYX': 5.702726e-10, 'Jval_H1211': 9.661064e-16, 'Jval_H1301': 1.1605463e-20, 'Jval_H2402': 3.0399666e-16,'Jval_H2O2': 1.9968258e-11, 'Jval_HAC': 4.1262545e-12, 'Jval_HCFC123': 2.9430889e-37, 'Jval_HCFC141b': 3.6372545e-37, 'Jval_HCFC142b': 2.848345e-39, 'Jval_HCFC22': 7.03871e-40, 'Jval_HMHP': 1.3574333e-11, 'Jval_HNO2': 9.547056e-09, 'Jval_HNO3': 8.272876e-13, 'Jval_HNO4': 4.6425617e-11, 'Jval_HOBr': 1.128905e-08, 'Jval_HOCl': 1.2624998e-09, 'Jval_HOI': 4.6213064e-08, 'Jval_HONIT': 5.0670835e-12, 'Jval_HPALD1': 1.4764497e-09, 'Jval_HPALD2': 1.4001184e-09, 'Jval_HPALD3': 1.14643794e-10, 'Jval_HPALD4': 1.14643794e-10, 'Jval_HPETHNL': 1.14643794e-10, 'Jval_I2': 5.229235e-07, 'Jval_I2O2': 1.9553251e-07, 'Jval_I2O3': 1.937503e-07, 'Jval_I2O4': 1.9553251e-07, 'Jval_IBr': 2.2424717e-07, 'Jval_ICN': 1.4051836e-09, 'Jval_ICl': 7.884039e-08, 'Jval_IDCHP': 1.14643794e-10, 'Jval_IDHDP': 3.878883e-11, 'Jval_IDN': 1.0134167e-11, 'Jval_IHN1': 5.0670835e-12, 'Jval_IHN2': 5.0670835e-12, 'Jval_IHN3': 5.0670835e-12,'Jval_ICPDH':1.1453314e-10 ,'Jval_IHN4': 5.0670835e-12, 'Jval_INO': 1.4956237e-07, 'Jval_INPB': 2.445911e-11, 'Jval_IO': 9.4386473e-07, 'Jval_IONO': 1.5672333e-08, 'Jval_IONO2': 5.517457e-08, 'Jval_IPRNO3': 3.787802e-12, 'Jval_ITCN': 3.412832e-10, 'Jval_ITHN': 2.445911e-11, 'Jval_MACR': 7.636825e-12, 'Jval_MACR1OOH': 1.14643794e-10, 'Jval_MCRENOL': 1.4701507e-09, 'Jval_MCRHN': 9.64955e-10, 'Jval_MCRHNB': 2.4614208e-10, 'Jval_MCRHP': 1.14643794e-10, 'Jval_MEK': 4.9985376e-12, 'Jval_MENO3': 1.1317352e-12, 'Jval_MGLY': 8.9409086e-10, 'Jval_MONITS': 5.0670835e-12, 'Jval_MONITU': 5.0670835e-12, 'Jval_MP': 1.9392028e-11, 'Jval_MPN': 4.3990675e-12, 'Jval_MVK': 1.41704435e-11, 'Jval_MVKHC': 8.9409086e-10, 'Jval_MVKHCB': 9.514112e-11, 'Jval_MVKN': 8.019085e-11, 'Jval_MVKPC': 1.14643794e-10, 'Jval_N2O': 1.1980623e-37, 'Jval_N2O5': 1.5224057e-10, 'Jval_NIT': 2.75988e-13, 'Jval_NITs': 2.75988e-12, 'Jval_NO': 1.021373e-39, 'Jval_NO2': 6.090331e-08, 'Jval_NO3': 5.673017e-07, 'Jval_NPHEN': 6.7912974e-11, 'Jval_NPRNO3': 3.4222401e-12, 'Jval_O2': 8.346e-41, 'Jval_O3': 1.2237375e-11, 'Jval_OCS': 3.118778e-21, 'Jval_OClO': 4.936651e-07, 'Jval_OIO': 6.0653554e-07, 'Jval_PAN': 1.5455034e-12, 'Jval_PIP': 1.9968258e-11, 'Jval_PROPNN': 6.7912974e-11, 'Jval_PYAC': 8.9409086e-10, 'Jval_R4N2': 2.2864553e-12, 'Jval_RCHO': 9.514112e-11, 'Jval_SO4': 3.0301065e-18}
#Below list for Constants from the july month for jval list 4
constants = {'Jval_CF3I': 1.1302509e-14, 'Jval_H2COa': 4.2749283e-15, 'Jval_H2COb': 1.845236e-14, 'Jval_ClNO3a': 1.3835245e-13, 'Jval_ClNO3b': 4.2134735e-15, 'Jval_ActAld': 8.7724555e-16, 'Jval_Glyxla': 1.687191e-13, 'Jval_Glyxlb': 9.007525e-15, 'Jval_Glyxlc': 3.837299e-14, 'Jval_Acet-a': 7.532465e-17, 'Jval_Acet-b': 4.073666e-18}



DEBUG = True
#================================================
#       PREPARING DATA FOR TESTING & TRAINING
#              & EVALUATION
#================================================
train_eval_data = master_df.copy()
test_data = master_df.copy()

print('Level Data Split')
#------------------------------------------------
#               TIME Data split
#------------------------------------------------

unique_times= (master_df['time'].unique())
if DEBUG:
    print(unique_times[12])

specific_time_slice_unix_range = unique_times[:100] #the first day excluded
testing_range = unique_times[100:]

specific_time_slice_unix = unique_times[time_index]


#test_data = test_data[test_data['time'] == specific_time_slice_unix] #test data is selected for one specific time slice
test_data = test_data[test_data['time'].isin(specific_time_slice_unix_range)]
train_eval_data = train_eval_data[~train_eval_data['time'].isin(specific_time_slice_unix_range)]
print('Time Data Split')
#------------------------------------------------
#               Train_Eval Data split
#------------------------------------------------

splitting_percent = 0.75

train_eval_data=train_eval_data.sort_values('time')
total_points = len(train_eval_data)
split_index = int(total_points * splitting_percent) #denotes the point at which the data is split

train_data = train_eval_data.iloc[:split_index] #first 75%
eval_data = train_eval_data.iloc[split_index:] #last 25%
print('Train & Eval Data Split')



#feature_order = ['Day_of_year', 'lev','Met_SUNCOS', 'Met_UVALBEDO', 'Met_TO3', 'SZA', 'Met_PMID', 'Met_T', 'Met_OPTD','Met_AIRDEN', 'Met_CLDF']
#feature_order = ['lev','pos_enc','Met_SUNCOS', 'Met_UVALBEDO','SZA', 'Met_PMID', 'Met_T','Met_AIRDEN','AODDust1000nm_bin1' ,'AODDust1000nm_bin6', 'AODDust1000nm_bin7']

feature_order = ['lev','Met_SUNCOS', 'Met_UVALBEDO', 'Met_TO3', 'SZA', 'Met_PMID', 'Met_T','TAUCLW_above','TAUCLW_below','TAUCLI_above','TAUCLI_below','CLDF_above','CLDF_below','Met_AIRDEN', 'Met_CLDF','Met_TAUCLI', 'Met_TAUCLW','AODDust1000nm_bin1','AODDust1000nm_bin7']

print('Features Used:')
for feature in feature_order:
    print(feature_order)

#================================================
#------------------------------------------------
#               Main Loop
#------------------------------------------------

for jval in jval_list:
    y_train = train_data[jval]
    X_train = train_data[feature_order]

    y_eval = eval_data[jval]
    X_eval = eval_data[feature_order]

    y_test = test_data[jval]
    X_test = test_data[feature_order]

    #y_test_log = np.log1p(y_test)
    #y_train_log = np.log1p(y_train)
    #y_eval_log = np.log1p(y_eval)

    #y_test_log = (y_test+0.00000001)
    #y_train_log = (y_train+0.00000001)
    #y_eval_log = (y_eval+0.00000001)
    y_test_log =((y_test))#- constants[jval])
    y_train_log = ((y_train))#- constants[jval])
    y_eval_log = ((y_eval))#- constants[jval])

    dtrain, dval, dtest = dmatrix_maker(X_train, y_train_log, X_eval, y_eval_log, X_test, y_test_log)

    model, train_error, val_error = model_maker(jval, dtrain, dval)
    print('================================================ ')
    print(f'           Loop for {jval}                      ')

    print('================================================ ')
    print(f'The constant used: {constants[jval]}')

    y_pred = model.predict(dtest)
    y_pred_retransformed =np.exp((y_pred) - constants[jval])

    print(f'Predictions head: {y_pred_retransformed[:5]}')
    print(f'Mean Predictions: {np.mean(y_pred_retransformed)}')
    print(f'Median of Predictions: {np.median(y_pred_retransformed)}')
    print(f'Standard Devation of Predictions: {np.std(y_pred_retransformed)}')
    print(f'Minimum of Predictions: {np.min(y_pred_retransformed)}')
    print(f'Maximum of Predictions: {np.max(y_pred_retransformed)}')
    print('')
    print(f'Summary Statistics for Log Test Y:')
    print(f"Mean: {np.mean(np.exp(y_test)- constants[jval])}")
    print(f"Median: {np.median(np.exp(y_test)- constants[jval])}")
    print(f"Standard Deviation: {np.std(np.exp(y_test)- constants[jval])}")
    print(f"Minimum: {np.min(np.exp(y_test)- constants[jval])}")
    print(f"Maximum: {np.max(np.exp(y_test)- constants[jval])}")

    mse = mean_squared_error((np.exp(y_test)-constants[jval]), y_pred_retransformed)

    mae = mean_absolute_error((np.exp(y_test)-constants[jval]), y_pred_retransformed)

    r2 = r2_score((np.exp(y_test)-constants[jval]), y_pred_retransformed)
    print('')
    print('')
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"R-squared (R2): {r2}")

    plot_eval(train_error, val_error, jval)
    
    print(f'Species done {jval}')
    print('')
    print('')
    print('')
    print('')


    del y_train, X_train, y_eval, X_eval, y_test, X_test, dtrain,dval, dtest, model, train_error, val_error, y_pred
                                                                                                                           
