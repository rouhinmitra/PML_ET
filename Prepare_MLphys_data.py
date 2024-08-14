#%% 
import pandas as pd 
import numpy as np 
import os 
import matplotlib.pyplot as plt
#%% Prepare ML physics data
def read_flux(dir_flux):
    """
    Read data from folder: 
    """
    temp=[]
    file_name=[]
    os.chdir(dir_flux)
    flux_list=os.listdir()
    print(flux_list)
    for index in range(len(flux_list)):
        temp.append(pd.read_csv(flux_list[index]))
        file_name.append(flux_list[index].split(".")[0])

    return temp,file_name

def airtemp_calc(df):
    """Calculate Air temp from SEBAL using dT (C)"""
    df["Ta_SEBAL"]=df["T_LST_DEM"]-273.13-df["dT"]
    return df
def calculate_physicsloss_variables(df,Ta,suffix):
    """ Calculate variables reqd as parameters for physics loss
    es: Saturated Vapor pressure (kPa)
    Calculate: del,del2, H taylor where Hpred-Htaylor=0 
    Coefficients of quadratic to solve H: a,b,c 
    Suffix: we can use Ta from ERA or Landsat dT 
    """
    df["rho_"+suffix]=(-0.0046 *(df[Ta]+273.15) ) + 2.5538
    df["es_"+suffix]=0.6108 * np.exp((17.27 * df[Ta]) / (df[Ta] + 237.3))
    df["es_Ts"]=0.6108 * np.exp((17.27 * (df["T_LST_DEM"]-273.13)) / ((df["T_LST_DEM"]-273.13) + 237.3))

    df["del_st_"+suffix] = (4098 * df["es_"+suffix]) / ((df[Ta] + 237.3) ** 2)
    first_term = df["del_st_"+suffix] * ((17.27 * 237.3) / ((df[Ta] + 237.3))**2)
    second_term = df["es_"+suffix] * (-2 * 17.27 * 237.3) / ((df[Ta] + 237.3)**3)
    df["del2_st_"+suffix]=first_term+second_term
    df["b_"+suffix]=df["del_st_"+suffix]*df["rah"]/(df["rho_"+suffix]*1004)
    df["a_"+suffix]=df["del2_st_"+suffix]*(df["rah"]**2)*0.5/((df["rho_"+suffix]*1004)**2)
    df["c_"+suffix]=df["es_"+suffix]-df["es_Ts"]
    df["H_taylor_"+suffix]=(-df["b_"+suffix]+np.sqrt((df["b_"+suffix]**2)-4*df["a_"+suffix]*df["c_"+suffix]))/(2*df["a_"+suffix])
    
    return df 
#%%
data,filename=read_flux("D:\\Backup\\Rouhin_Lenovo\\US_project\\GEE_SEBAL_Project\\Csv_Files\\Validation_data1\\")
data=[airtemp_calc(index) for index in data]
data=[calculate_physicsloss_variables(index,"Ta_SEBAL","SEBAL_Ta") for index in data]
data=[calculate_physicsloss_variables(index,"AirT_G","ERA_Ta") for index in data]

# %%
data[0].columns.tolist()
# %%
data[0][["es_Ts","H_taylor_SEBAL_TA","H_inst_af","Hinst","dT","rah","del_st_SEBAL_TA","rho_SEBAL_TA","es_SEBAL_TA"]]
data[0][["es_Ts","H_taylor_ERA_Ta","H_inst_af","Hinst","dT","rah","del_st_ERA_Ta","rho_ERA_Ta","es_ERA_Ta"]]

# %%
