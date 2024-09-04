#%%
import ee
import pandas as pd 
import numpy as np
import os
#%%
"""Import pixel data for ameriflux stations for ML using earth engine"""
ee.Initialize()
aflux=pd.read_csv("D:\\Backup\\Rouhin_Lenovo\\US_project\\GEE_SEBAL_Project\\Ameriflux_summary_ML.csv")
print(aflux)
def landcover(df,outdir):
    """Get pixel values for WorldCover from earth engine for each ameriflxu station"""
    lc=[]
    for i in range(df.shape[0]):
        geometry=ee.Geometry.Point([df.Long.iloc[i],df.Lat.iloc[i]]) # Neutral to accomodate Bi1 and Vaira
        col = ee.ImageCollection("ESA/WorldCover/v200")
        
        # .filterDate("2022-01-01", "2022-01-02")
        # print(col.first().bandNames())
        means=col.toBands().reduceRegion(reducer=ee.Reducer.mean(),
                              geometry=geometry,
                              scale= 30,
                              crs="EPSG:4326",
                              maxPixels=10e9).getInfo()
        # print(means)
        df_mean = pd.DataFrame.from_dict(means,orient='index')
        df_mean=df_mean.reset_index() 
        df_mean=df_mean.rename(columns={"index":"ID",0:"Landcover"})
        print(df_mean.shape) 
        lc.append(df_mean.reset_index())
    lc_data=pd.concat(lc)
    lc_data=lc_data.reset_index()
    lc_data["Name"]=df["Name"]
    print(lc_data)
    lc_data.to_csv(outdir)
    # return lc_data
landcover(aflux,"D:\\Backup\\Rouhin_Lenovo\\US_project\\GEE_SEBAL_Project\\ML\\Landcover\\Worldcover_GEE.csv")

#%%
def elev_slope(df,outdir):
    """Get pixel values for WorldCover from earth engine for each ameriflxu station"""
    terrain=[]
    for i in range(df.shape[0]):
        geometry=ee.Geometry.Point([df.Long.iloc[i],df.Lat.iloc[i]]) # Neutral to accomodate Bi1 and Vaira
        elev = ee.Image("USGS/SRTMGL1_003")
        slope=ee.Terrain.slope(elev)
        # elev=elev.addBands(slope)
        # .filterDate("2022-01-01", "2022-01-02")
        # print(col.first().bandNames())
        srtm_elev=elev.reduceRegion(reducer=ee.Reducer.mean(),
                              geometry=geometry,
                              scale= 30,
                              crs="EPSG:4326",
                              maxPixels=10e9).getInfo()
        srtm_slope=slope.reduceRegion(reducer=ee.Reducer.mean(),
                        geometry=geometry,
                        scale= 30,
                        crs="EPSG:4326",
                        maxPixels=10e9).getInfo()
        # print(means)
        df_elev = pd.DataFrame.from_dict(srtm_elev,orient='index')
        df_elev=df_elev.reset_index()
        df_elev=df_elev.rename(columns={0:"Elev_SRTM"}).drop(columns={"index"})
    ##-- Slope 
        df_slope = pd.DataFrame.from_dict(srtm_slope,orient='index')
        df_slope=df_slope.reset_index()
        df_slope=df_slope.rename(columns={0:"Slope_SRTM"}).drop(columns={"index"})
        df_mean=pd.concat([df_elev,df_slope],axis=1)
        # print(df_elev)
        # print(df_slope)
        print(df_mean) 
        terrain.append(df_mean.reset_index())
    terrain_data=pd.concat(terrain)
    terrain_data=terrain_data.reset_index()
    terrain_data["Name"]=df["Name"]
    print(terrain_data)
    terrain_data.to_csv(outdir)
    return terrain_data
elev_slope(aflux,"D:\\Backup\\Rouhin_Lenovo\\US_project\\GEE_SEBAL_Project\\ML\\Terrain\\Terrain.csv")

# %%
def soil_data(df,outdir):
    """Get pixel values for WorldCover from earth engine for each ameriflxu station"""
    soil=[]
    for i in range(df.shape[0]):
        geometry=ee.Geometry.Point([df.Long.iloc[i],df.Lat.iloc[i]]) # Neutral to accomodate Bi1 and Vaira
        clay_mean = ee.ImageCollection('projects/sat-io/open-datasets/polaris/clay_mean').first();
        ksat_mean = ee.ImageCollection('projects/sat-io/open-datasets/polaris/ksat_mean').first();
        sand_mean = ee.ImageCollection('projects/sat-io/open-datasets/polaris/sand_mean').first();
        silt_mean = ee.ImageCollection('projects/sat-io/open-datasets/polaris/silt_mean').first();

        clay_polaris=clay_mean.reduceRegion(reducer=ee.Reducer.mean(),
                              geometry=geometry,
                              scale= 30,
                              crs="EPSG:4326",
                              maxPixels=10e9).getInfo()
        ksat_polaris=ksat_mean.reduceRegion(reducer=ee.Reducer.mean(),
                        geometry=geometry,
                        scale= 30,
                        crs="EPSG:4326",
                        maxPixels=10e9).getInfo()
        sand_polaris=sand_mean.reduceRegion(reducer=ee.Reducer.mean(),
                    geometry=geometry,
                    scale= 30,
                    crs="EPSG:4326",
                    maxPixels=10e9).getInfo()
        silt_polaris=silt_mean.reduceRegion(reducer=ee.Reducer.mean(),
                    geometry=geometry,
                    scale= 30,
                    crs="EPSG:4326",
                    maxPixels=10e9).getInfo()
        # clay(means)
        df_clay = pd.DataFrame.from_dict(clay_polaris,orient='index')
        df_clay=df_clay.reset_index()
        df_clay=df_clay.rename(columns={0:"Clay_percent"}).drop(columns={"index"})
    ##-- Ksat 
        df_ksat = pd.DataFrame.from_dict(ksat_polaris,orient='index')
        df_ksat=df_ksat.reset_index()
        df_ksat=df_ksat.rename(columns={0:"Ksat"}).drop(columns={"index"})
        ##-- sand 
        df_sand = pd.DataFrame.from_dict(sand_polaris,orient='index')
        df_sand=df_sand.reset_index()
        df_sand=df_sand.rename(columns={0:"Sand_percent"}).drop(columns={"index"})
        ##-- Silt 
        df_silt = pd.DataFrame.from_dict(silt_polaris,orient='index')
        df_silt=df_silt.reset_index()
        df_silt=df_silt.rename(columns={0:"Silt_percent"}).drop(columns={"index"})
        df_mean=pd.concat([df_clay,df_ksat,df_sand,df_silt],axis=1)
        # print(df_elev)
        # print(df_slope)
        print(df_mean) 
        soil.append(df_mean.reset_index())
    soil_data=pd.concat(soil)
    soil_data=soil_data.reset_index()
    soil_data["Name"]=df["Name"]
    print(soil_data)
    soil_data.to_csv(outdir)
    return soil_data
soil_data(aflux,"D:\\Backup\\Rouhin_Lenovo\\US_project\\GEE_SEBAL_Project\\ML\\Soil_data\\Soil_polaris.csv")

#%% Canopy height 
def canopy_height(df,outdir):
    """Get pixel values for WorldCover from earth engine for each ameriflxu station"""
    lc=[]
    for i in range(df.shape[0]):
        geometry=ee.Geometry.Point([df.Long.iloc[i],df.Lat.iloc[i]]) # Neutral to accomodate Bi1 and Vaira
        col = ee.ImageCollection("projects/meta-forest-monitoring-okw37/assets/CanopyHeight").mosaic()
        
        # .filterDate("2022-01-01", "2022-01-02")
        # print(col.first().bandNames())
        means=col.reduceRegion(reducer=ee.Reducer.mean(),
                              geometry=geometry,
                              scale= 30,
                              crs="EPSG:4326",
                              maxPixels=10e9).getInfo()
        # print(means)
        df_mean = pd.DataFrame.from_dict(means,orient='index')
        df_mean=df_mean.reset_index() 
        df_mean=df_mean.rename(columns={"index":"ID",0:"Canopy_height"})
        print(df_mean.shape) 
        lc.append(df_mean.reset_index())
    lc_data=pd.concat(lc)
    lc_data=lc_data.reset_index()
    lc_data["Name"]=df["Name"]
    print(lc_data)
    lc_data.to_csv(outdir)
    return lc_data
canopy_height(aflux,"D:\\Backup\\Rouhin_Lenovo\\US_project\\GEE_SEBAL_Project\\ML\\Canopy_height\\Global_canopy_height.csv")

# %%
