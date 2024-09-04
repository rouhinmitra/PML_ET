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
    df["Date"]=pd.to_datetime(df["Datetime"]).dt.date
    df=df[df["LE_closed"]>=0]
    # df=df[df["Rn"]>=0]
    # df=df[df["H_inst_af"]>=0]
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

# def add_openET(list_ameriflux,list_openet):
#     """Merge OpenET daily results with the ameriflux data available"""
#     st=[]
#     for i in range(len(list_ameriflux)):
#         if "Name_x" in list_ameriflux[i].columns:
#             list_ameriflux[i]=list_ameriflux[i].drop(columns={"Name_x","Name_y",'Unnamed: 0_x'})
#         if list_ameriflux[i].shape[0]!=0:
#             for j in range(len(list_openet)):
#                 if list_ameriflux[i]["Name"].iloc[0]==list_openet[j]["Name"].iloc[0]:
#                     print(list_ameriflux[i]["Name"].iloc[0])
#                     st.append(pd.merge(list_ameriflux[i],list_openet[j],on="Date",how="left"))
#     return st

def add_openET(list_ameriflux, list_openet):
    """Merge OpenET daily results with the AmeriFlux data available."""
    st = []
    
    # Convert list_openet to a dictionary for faster lookup by Name
    openet_dict = {df["Name"].iloc[0]: df for df in list_openet}
    
    for ameriflux_df in list_ameriflux:
        if ameriflux_df.shape[0]!=0:
            ameriflux_name = ameriflux_df["Name"].iloc[0]
            
            if ameriflux_name in openet_dict:
                openet_df = openet_dict[ameriflux_name]
                
                # Drop the 'Name' column from openet_df to avoid duplication
                openet_df_clean = openet_df.drop(columns=["Name", "Unnamed: 0"], errors='ignore')
                
                # Merge the dataframes on the "Date" column
                merged_df = pd.merge(ameriflux_df, openet_df_clean, on="Date", how="left", suffixes=('_ameriflux', '_openet'))
                
                # Keep the 'Name' column from ameriflux_df
                merged_df["Name"] = ameriflux_name
                print(merged_df)
                st.append(merged_df)
            else:
                print(f"No matching OpenET data for {ameriflux_name}")
    
    return st

def calc_eto_hourly():
    return

def add_eto():

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

def add_canopy_height(df,canopy_data):
    """WRI and meta data """
    if df.shape[0]!=0:
        for i in range(canopy_data.shape[0]):
            if df.Name.iloc[0]==canopy_data["Name"].iloc[i]:
                df["Canopy_height"]=canopy_data["Canopy_height"].iloc[i]

    return df
    
def add_soil_data(df,soil_data):
    """
    Polaris data: soil, clay and silt percent and Ks 
    """
    if df.shape[0]!=0:
        for i in range(soil_data.shape[0]):
            if df.Name.iloc[0]==soil_data["Name"].iloc[i]:
                df["clay_percent"]=soil_data["Clay_percent"].iloc[i]
                df["silt_percent"]=soil_data["Silt_percent"].iloc[i]
                df["sand_percent"]=soil_data["Sand_percent"].iloc[i]
                df["Ksat"]=soil_data["Ksat"].iloc[i]

    return df

def summary_df(listofdf,outdir):
    summary_list = []
    
    for df in listofdf:
        print(df["Name"])
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
    "doy","Rn","Ginst","LE_closed"]
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
canopy_height=pd.read_csv("D:\\Backup\\Rouhin_Lenovo\\US_project\\GEE_SEBAL_Project\\ML\\Canopy_height\\Global_canopy_height.csv")
soil=pd.read_csv("D:\\Backup\\Rouhin_Lenovo\\US_project\\GEE_SEBAL_Project\\ML\\Soil_data\\Soil_polaris.csv")
os.chdir("D:\\Backup\\Rouhin_Lenovo\\US_project\\GEE_SEBAL_Project\\ML\\OpenET_New\\")
## Read open ET
file_list_openet=os.listdir()
openet=[]
for i in range(len(file_list_openet)):
    openet.append(pd.read_csv(file_list_openet[i]))
    openet[i]["Date"]=pd.to_datetime(openet[i]["Date"]).dt.date
    openet[i]["Name"]=file_list_openet[i].split(".")[0]
# print(soil,canopy_height)
# data=[add_landcover(index,landcover) for index in data]
# data=[add_canopy_height(index,canopy_height) for index in data]
# data=[add_terrain(index,terrain) for index in data]
# data=[add_soil_data(index,soil) for index in data]
# data=[remove_na_from_main_features(index) for index in data]
data_dir="D:\\Backup\\Rouhin_Lenovo\\US_project\\GEE_SEBAL_Project\\ML\\ML_processed_data_all\\"
for df in data:
    if df.shape[0]!=0:
        print(df.shape[0])
        df.to_csv(data_dir+df.Name.iloc[0]+".csv")
data=add_openET(data,openet)
for df in data:
    if df.shape[0]!=0:
        print(df.shape[0])
        df.to_csv(data_dir+df.Name.iloc[0]+".csv")
summary_df(data,"D:\\Backup\\Rouhin_Lenovo\\US_project\\GEE_SEBAL_Project\\Ameriflux_summary_ML.csv")
#%%
for df in data:
    if df.shape[0]!=0:
        print(df.columns)
        # print((df["Rn_inst_af"]-df["G_inst_af"]-df["H_inst_af"]-df["LE_inst_af"]).describe(),df["Name"].iloc[0])
        # print(df[["LEinst","Hinst","Ginst","Rn"]].describe())
        # print((df["Rn"]-df["Ginst"]).min())
# pd.concat(data).shape
# file_list[0].split(".")[0]
# %%
data[1][["ST_B10","ETa"]]