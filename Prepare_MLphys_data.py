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

def preprocess_data(df):
    """
    Clean the data from outliers of observations 
    LE_closed, LE_inst_af and  H_inst_af and also LEinst 
    """
    print(df[["Lat","Long"]])
    # print(df[["Veg","Name"]].dropna())
    df["doy"]=pd.to_datetime(df['Datetime']).dt.dayofyear
    df=df[df["LE_closed"]>=0]
    df=df[df["LE_inst_af"]>=0]
    df=df[df["H_inst_af"]>=0]
    if df.shape[0]!=0:
        df["Veg"]=df["Veg"].dropna().iloc[0]
        if "latitude" in df.columns:
            print(df.shape)
            df["Lat"]=df["latitude"].dropna().iloc[0]
            df["Long"]=df["longitude"].dropna().iloc[0]
        else:
            df["Lat"]=df["latitude_y"].dropna().iloc[0]
            df["Long"]=df["longitude_y"].dropna().iloc[0]
        print(df[["LEinst","Hinst","ET_24h","Rn","Ginst"]].describe())
    return df

# def airtemp_calc(df):
#     """Calculate Air temp from SEBAL using dT (C)"""
#     df["Ta_SEBAL"]=df["T_LST_DEM"]-273.13-df["dT"]
#     return df

def calculate_mixed_features(df):
    """ Calculate features combining remote sensing inputs from optical and thermal
    """
    print(df[["ST_B10","NDVI_model"]])
    df["R_LST"]=df["R"]*df["ST_B10"]
    df["GR_LST"]=df["GR"]*df["ST_B10"]
    df["B_LST"]=df["B"]*df["ST_B10"]
    df["NIR_LST"]=df["NIR"]*df["ST_B10"]
    df["NDVI_LST"]=df["NDVI_model"]*df["ST_B10"]
    df["NDWI_LST"]=df["NDWI"]*df["ST_B10"]
    df["SWIR1_LST"]=df["SWIR_1"]*df["ST_B10"]
    df["SWIR2_LST"]=df["SWIR_2"]*df["ST_B10"]
    return df 

def add_landcover(df,landcover_data):
    """ Add landcover
    """
    if df.shape[0]!=0:
        for i in range(landcover_data.shape[0]):
            if df.Name.iloc[0]==landcover_data["Name"].iloc[i]:
                df["Landcover_wc"]=landcover_data["Landcover"].iloc[i]

    return df

def add_eto():
    return

def calc_eto_hourly():
    return

def add_terrain(df,terrain_data):
    """ Add landcover
    """
    if df.shape[0]!=0:
        for i in range(terrain_data.shape[0]):
            if df.Name.iloc[0]==terrain_data["Name"].iloc[i]:
                df["Elev_SRTM"]=terrain_data["Elev_SRTM"].iloc[i]
                df["Slope_SRTM"]=terrain_data["Slope_SRTM"].iloc[i]

    return df

def add_canopy_height(df):
    """Simard et al Has missing shrublands  """

    return 
    
def add_soil_data():
    """
    Polaris 
    """
    return 

def summary_df(listofdf,outdir):
    summary_list = []
    
    for df in listofdf:
        # Assuming each dataframe has 'name', 'latitude', and 'longitude' columns
        summary = df[['Name', 'Lat', 'Long']].drop_duplicates()
        summary_list.append(summary)
    
    # Concatenate all dataframes in the summary list
    summary_dataframe = pd.concat(summary_list, ignore_index=True)
    summary_dataframe.to_csv(outdir)
    return summary_dataframe

def remove_na_from_main_features(df):
    """
    Drop nans to make it ML ready 
    """

    main_features=['B',
    'GR',
    'R',
    'NIR',
    'SWIR_1',
    'SWIR_2','ST_B10','NDVI_model','NDWI',
    'R_LST',
    'GR_LST',
    'B_LST',
    'NIR_LST',
    'NDVI_LST',
    'NDWI_LST',
    'SWIR1_LST',
    'SWIR2_LST',
    "Elev_SRTM",
    "Slope_SRTM",
    "Landcover_wc",
    "doy","Rn","Ginst","H_inst_af"]
    if df.shape[0]!=0:
        df=df.dropna(subset=main_features)

    return df

# def calculate_physicsloss_variables(df,Ta,suffix):
#     """ Calculate variables reqd as parameters for physics loss
#     es: Saturated Vapor pressure (kPa)
#     Calculate: del,del2, H taylor where Hpred-Htaylor=0 
#     Coefficients of quadratic to solve H: a,b,c 
#     Suffix: we can use Ta from ERA or Landsat dT 
#     """
#     df["rho_"+suffix]=(-0.0046 *(df[Ta]+273.15) ) + 2.5538
#     df["es_"+suffix]=0.6108 * np.exp((17.27 * df[Ta]) / (df[Ta] + 237.3))
#     df["es_Ts"]=0.6108 * np.exp((17.27 * (df["T_LST_DEM"]-273.13)) / ((df["T_LST_DEM"]-273.13) + 237.3))

#     df["del_st_"+suffix] = (4098 * df["es_"+suffix]) / ((df[Ta] + 237.3) ** 2)
#     first_term = df["del_st_"+suffix] * ((17.27 * 237.3) / ((df[Ta] + 237.3))**2)
#     second_term = df["es_"+suffix] * (-2 * 17.27 * 237.3) / ((df[Ta] + 237.3)**3)
#     df["del2_st_"+suffix]=first_term+second_term
#     df["b_"+suffix]=df["del_st_"+suffix]*df["rah"]/(df["rho_"+suffix]*1004)
#     df["a_"+suffix]=df["del2_st_"+suffix]*(df["rah"]**2)*0.5/((df["rho_"+suffix]*1004)**2)
#     df["c_"+suffix]=df["es_"+suffix]-df["es_Ts"]
#     df["H_taylor_"+suffix]=(-df["b_"+suffix]+np.sqrt((df["b_"+suffix]**2)-4*df["a_"+suffix]*df["c_"+suffix]))/(2*df["a_"+suffix])
    
    # return df 
#%%
# data,filename=read_flux("D:\\Backup\\Rouhin_Lenovo\\US_project\\GEE_SEBAL_Project\\Csv_Files\\Validation_data1\\")
data,filename=read_flux("D:\\Backup\\Rouhin_Lenovo\\US_project\\GEE_SEBAL_Project\\ML\\ML_dT_orig_data\\")
len(data)
data[0]
preprocess_data(data[0])
data=[preprocess_data(index) for index in data]
# data=[calculate_physicsloss_variables(index,"Ta_SEBAL","SEBAL_Ta") for index in data]
# data=[calculate_physicsloss_variables(index,"AirT_G","ERA_Ta") for index in data]
data=[calculate_mixed_features(data[index]) for index in range(len(data)) if data[index].shape[0]!=0]
landcover=pd.read_csv("D:\\Backup\\Rouhin_Lenovo\\US_project\\GEE_SEBAL_Project\\ML\\Landcover\\Worldcover_GEE.csv")
terrain=pd.read_csv("D:\\Backup\\Rouhin_Lenovo\\US_project\\GEE_SEBAL_Project\\ML\\Terrain\\Terrain.csv")

data=[add_landcover(index,landcover) for index in data]
data=[add_terrain(index,terrain) for index in data]
data=[remove_na_from_main_features(index) for index in data]

data_dir="D:\\Backup\\Rouhin_Lenovo\\US_project\\GEE_SEBAL_Project\\ML\\ML_processed_data_all\\"
for df in data:
    if df.shape[0]!=0:
        df.to_csv(data_dir+df.Name.iloc[0]+".csv")
# summary_df(data,"D:\\Backup\\Rouhin_Lenovo\\US_project\\GEE_SEBAL_Project\\Ameriflux_summary_ML.csv")
#%%
for df in data:
    if df.shape[0]!=0:
        print((df["Rn_inst_af"]-df["G_inst_af"]-df["H_inst_af"]-df["LE_inst_af"]).describe(),df["Name"].iloc[0])
        # print(df[["LEinst","Hinst","Ginst","Rn"]].describe())
        # print((df["Rn"]-df["Ginst"]).min())
pd.concat(data).shape
# %%
