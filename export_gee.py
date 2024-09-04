#%%
import ee
import pandas as pd 
import numpy as np
import folium
from folium import plugins
import geopandas as gpd
import geemap
import time 
#%%
ee.Initialize()
import sys
sys.path.append('D:\\Backup\\Rouhin_Lenovo\\US_project\\GEE_SEBAL_Project\\geeSEBAL_copy_edits\\etbrasil\\')
import geesebal
from geesebal import (tools,landsatcollection,masks,meteorology,endmembers, 
evapotranspiration,collection,timeseries,image,ET_Collection_mod)
# %% Set an image and get details
# Get a <10% cloudy landsat image of the central valley 
# LANDSAT/LC08/C02/T1_L2/LC08_043034_20220402
geometry=ee.Geometry.Point([ -121.550491,38.382768]) # Neutral to accomodate Bi1 and Vaira
# Get the image collection 
ls=ee.ImageCollection("LANDSAT/LC08/C02/T1_L2").filterDate("2022-04-01","2022-09-30").filterMetadata('CLOUD_COVER', 'less_than', 5).filterBounds(geometry);
print("The image collection has", ls.size().getInfo(), "images")
## Scale the images 
## Why do we need to scale? (Refer the slides)
def applyScaleFactors(image):
        opticalBands = image.select('SR_B.').multiply(0.0000275).add(-0.2);
        thermalBands = image.select('ST_B.*').multiply(0.00341802).add(149.0);
        return image.addBands(opticalBands, None, True).addBands(thermalBands, None, True)
ls=ls.map(applyScaleFactors)
#Get the first image
ls_first=ls.first()
### Get Image details 
landsat_version=ls_first.get('L1_LANDSAT_PRODUCT_ID').getInfo()
print(landsat_version)
sun_elevation=ls_first.get("SUN_ELEVATION")
print("Sun Elevation ", sun_elevation.getInfo())
time_start=ls_first.get('system:time_start')
date=ee.Date(time_start)
year=ee.Number(date.get('year'))
month=ee.Number(date.get('month'))
day=ee.Number(date.get('day'))
hour=ee.Number(date.get('hour'))
minuts = ee.Number(date.get('minutes'))
print("Time of Image ", day.getInfo(),"/",month.getInfo(),"/",year.getInfo(), "@" , hour.getInfo(), ":",minuts.getInfo())
#%% Model
# image= ls.filterMetadata('system:index','equals',ls_list[n]).first()
def runsebal(image):
        image.getInfo()
        image=ee.Image(image)
                # et=image.Image(image)
        NDVI_cold=5
        Ts_cold=20
        NDVI_hot=10
        Ts_hot=20
        index=image.get('system:index')
        cloud_cover=image.get('CLOUD_COVER')
        LANDSAT_ID=image.get('L1_LANDSAT_PRODUCT_ID').getInfo()
        print(LANDSAT_ID)
        landsat_version=image.get('SATELLITE').getInfo()
        sun_elevation=image.get("SUN_ELEVATION")
        print(sun_elevation.getInfo())
        time_start=image.get('system:time_start')
        date=ee.Date(time_start)
        year=ee.Number(date.get('year'))
        month=ee.Number(date.get('month'))
        day=ee.Number(date.get('day'))
        hour=ee.Number(date.get('hour'))
        minuts = ee.Number(date.get('minutes'))
        print(str(hour.getInfo())+str(minuts.getInfo()))
        crs = image.projection().crs()
        transform=ee.List(ee.Dictionary(ee.Algorithms.Describe(image.projection())).get('transform'))
        date_string=date.format('YYYY-MM-dd').getInfo()
        #ENDMEMBERS
        p_top_NDVI=ee.Number(NDVI_cold)
        p_coldest_Ts=ee.Number(Ts_cold)
        p_lowest_NDVI=ee.Number(NDVI_hot)
        p_hottest_Ts=ee.Number(Ts_hot)
        ls_trial=image.select([0,1,2,3,4,5,6,8,17], ["UB","B","GR","R","NIR","SWIR_1","SWIR_2","ST_B10","pixel_qa"])
        #       ls.first_toa=ee.Image('LANDSAT/LC08/C01/T1/'+index.getInfo())
        print(ls_trial.bandNames().getInfo())

        #col_rad = ee.Algorithms.Landsat.calibratedRadiance(ls.first_toa)
        #col_rad = ls_trial.addBands(col_rad.select([9],["T_RAD"]))
        #CLOUD REMOVAL
        #ls_trial=ee.ImageCollection(col_rad).map(masks.f_cloudMaskL8_SR)
        ls_trial=masks.f_cloudMaskL8_SR(ls_trial)
        #         print("Cloud masking Complete")
        print(ls_trial.bandNames().getInfo())

        #ALBEDO TASUMI ET AL. (2008) METHOD WITH KE ET AL. (2016) COEFFICIENTS
        # ls_trial=ls_trial.map(masks.f_albedoL8)
        ls_trial=masks.f_albedoL8(ls_trial)
        print(ls_trial.bandNames().getInfo())
        #         print("Albedo calc done")

        #------ Meteorology
                #GEOMETRY
        geometryReducer=ls_trial.geometry().bounds().getInfo()
        #         print("sun elevation check")

        geometry_download=geometryReducer['coordinates']

        # camada_clip=ls_trial.select('BRT').first()
        #         camada_clip=ls_trial.select('BRT')
        #         sun_elevation=ee.Number(90).subtract(ee.Number(azimuth_angle))
        print(sun_elevation.getInfo())
        #METEOROLOGY PARAMETERS
        col_meteorology= meteorology.get_meteorology(ls_trial,time_start);
        #AIR TEMPERATURE [C]
        T_air = col_meteorology.select('AirT_G');
        print(T_air.bandNames().getInfo())
        #WIND SPEED [M S-1]
        ux= col_meteorology.select('ux_G');
        #RELATIVE HUMIDITY [%]
        UR = col_meteorology.select('RH_G');

        #NET RADIATION 24H [W M-2]
        Rn24hobs = col_meteorology.select('Rn24h_G');

        ## print("Metorology ready")

        #------
        #------ Elevation
        #SRTM DATA ELEVATION
        SRTM_ELEVATION ='USGS/SRTMGL1_003'
        srtm = ee.Image(SRTM_ELEVATION).clip(geometryReducer);
        z_alt = srtm.select('elevation')
        ## print(z_alt) 
        ls_trial=tools.fexp_spec_ind(ls_trial)
        ls_trial=tools.LST_DEM_correction(ls_trial, z_alt, T_air, UR,sun_elevation,hour,minuts)
        print("It's a miracle")
        ## GET IMAGE
        ## COLD PIXEL
        d_cold_pixel=endmembers.fexp_cold_pixel(ls_trial, geometryReducer, p_top_NDVI, p_coldest_Ts)
        print(d_cold_pixel.getInfo())
        ## COLD PIXEL NUMBER
        n_Ts_cold = ee.Number(d_cold_pixel.get('temp').getInfo())
        ##INSTANTANEOUS OUTGOING LONG-WAVE RADIATION [WM-2]
        ls_trial=tools.fexp_radlong_up(ls_trial)
        ##INSTANTANEOUS INCOMING SHORT-WAVE RADIATION [WM-2]
        ls_trial=tools.fexp_radshort_down(ls_trial,z_alt,T_air,UR, sun_elevation)

        ## INSTANTANEOUS INCOMING LONGWAVE RADIATION [W M-2]
        ls_trial=tools.fexp_radlong_down(ls_trial, n_Ts_cold)
        ##INSTANTANEOUS NET RADIATON BALANCE [W M-2]

        ls_trial=tools.fexp_radbalance(ls_trial)

        ##SOIL HEAT FLUX (G) [W M-2]
        ls_trial=tools.fexp_soil_heat(ls_trial)
        ##HOT PIXEL
        d_hot_pixel=endmembers.fexp_hot_pixel(ls_trial, geometryReducer,p_lowest_NDVI, p_hottest_Ts)
        ##SENSIBLE HEAT FLUX (H) [W M-2]
        ls_trial=tools.fexp_sensible_heat_flux(ls_trial, ux, UR,Rn24hobs,n_Ts_cold,
                                       d_hot_pixel, date_string,geometryReducer)
#         cold_pixel_lat.append(d_cold_pixel.get("y").getInfo())
#         cold_pixel_lon.append(d_cold_pixel.get("x").getInfo())
#         cold_pixel_temp.append(d_cold_pixel.get("temp").getInfo())
#         cold_pixel_ndvi.append(d_cold_pixel.get("ndvi").getInfo())
#         cold_pixel_sum.append(d_cold_pixel.get("sum").getInfo())
# ## Get info about hot pixl
#         hot_pixel_lat.append(d_hot_pixel.get("y").getInfo())
#         hot_pixel_lon.append(d_hot_pixel.get("x").getInfo())
#         hot_pixel_temp.append(d_hot_pixel.get("temp").getInfo())
#         hot_pixel_ndvi.append(d_hot_pixel.get("ndvi").getInfo())
#         hot_pixel_sum.append(d_hot_pixel.get("sum").getInfo())
#         hot_pixel_Rn.append(d_hot_pixel.get("Rn").getInfo())
#         hot_pixel_G.append(d_hot_pixel.get("G").getInfo())
#         zenith_angle.append(90-sun_elevation.getInfo())

        ##DAILY EVAPOTRANSPIRATION (ET_24H) [MM DAY-1]
        ls_trial=evapotranspiration.fexp_et(ls_trial,Rn24hobs)
        return ls_trial,col_meteorology,d_cold_pixel.get("temp").getInfo(),d_hot_pixel.get("temp").getInfo(),d_hot_pixel.get("Rn").getInfo(),d_hot_pixel.get("G").getInfo()
#%% Run model
b,met,ts_cold,ts_hot,rn_hot,g_hot=runsebal(ls_first)
## bstores the image
# b_export=b.select(["B","R","GR","NIR","SWIR_1","SWIR_2",'ST_B10',"NDVI","NDWI","ALFA",'Tao_sw_1',"Rs_down","Rl_down","Rl_up","Rn","G","H","LE","ET_24h"])
# b_export.geometry().getInfo()["coordinates"]
# b_export.projection().getInfo()
#%% Exporting 
bands=["B","R","GR","NIR","SWIR_1","SWIR_2",'ST_B10',"NDVI","NDWI","ALFA",'Tao_sw_1',"Rs_down","Rl_down","Rl_up","Rn","G","H","LE","ET_24h"]

def make_features(img_collection,band):
    tmp = img_collection.expression(
      str(band)+'*LST', {
              band : img_collection.select(band),
              'LST': img_collection.select("ST_B10") }).rename(band+'_LST')
    return img_collection.addBands([tmp])  
b=make_features(b,"B")
b=make_features(b,"GR")
b=make_features(b,"R")
b=make_features(b,"NIR")
b=make_features(b,"NDVI")
b=make_features(b,"NDWI")
b=make_features(b,"SWIR_1")
b=make_features(b,"SWIR_2")

#%%
b.select("R")
#%%       

# bands=["B","R","GR","NIR","SWIR_1","SWIR_2",'ST_B10',"NDVI","NDWI","ALFA",'Tao_sw_1',"Rs_down","Rl_down","Rl_up","Rn","G","H","LE","ET_24h"]
# bands=["B","R","GR","NIR","SWIR_1","SWIR_2",'ST_B10',"NDVI","NDWI",\
#     "B_LST","GR_LST","R_LST","NIR_LST","NDVI_LST","NDWI_LST",\
#         "SWIR_1_LST","SWIR_2_LST","Rn","G","H","LE","ET_24h"]
bands=["B_LST","GR_LST","R_LST","NIR_LST","NDVI_LST","NDWI_LST",\
        "SWIR_1_LST","SWIR_2_LST"]
for band in bands:
    # Select the band
    band_image = b.select(band)
    
    # Define export parameters
    export_params = {
        'image': band_image.toFloat(),
        'description': f'Landsat_{band}',
        'folder': 'EarthEngineImages_crs',
        'scale': 30,  # Resolution in meters
        'fileFormat': 'GeoTIFF',
        'region': b.geometry().bounds().getInfo()['coordinates'],
        'crs': "EPSG:4326"
    }
    
    # Export the band to Google Drive
    task = ee.batch.Export.image.toDrive(**export_params)
    task.start()

    print(f"Exporting {band} band to Google Drive...")
#%% Check if task is running 
import time
while task.active():
  print('Polling for task (id: {}).'.format(task.id))
  time.sleep(5)
#%% See if it is completed
def print_task_status():
    task_list = ee.batch.Task.list()
    for task in task_list:
        print(f"Task {task.id} ({task.config['description']}): {task.state}")
print_task_status()
#%% Export landcover
landcover = ee.ImageCollection("ESA/WorldCover/v200")
landcover.select("Map")

# Select the band
band_image = landcover.select("Map").first()

# Define export parameters
export_params = {
    'image': band_image.toFloat(),
    'description': "Landcover",
    'folder': 'EarthEngineImages_crs',
    'scale': 30,  # Resolution in meters
    'fileFormat': 'GeoTIFF',
    'region': b.geometry().bounds().getInfo()['coordinates'],
    'crs': "EPSG:4326"
}

# Export the band to Google Drive
task = ee.batch.Export.image.toDrive(**export_params)
task.start()

print(f"Exporting Landcover band to Google Drive...")
# %%
elev=ee.Image("USGS/SRTMGL1_003")
band_image = elev.select("elevation")

# Define export parameters
export_params = {
    'image': band_image.toFloat(),
    'description': "Elevation",
    'folder': 'EarthEngineImages_crs',
    'scale': 30,  # Resolution in meters
    'fileFormat': 'GeoTIFF',
    'region': b.geometry().bounds().getInfo()['coordinates'],
    'crs': "EPSG:4326"
}

# Export the band to Google Drive
task = ee.batch.Export.image.toDrive(**export_params)
task.start()

print(f"Exporting elevation band to Google Drive...")
slope=ee.Terrain.slope(elev)

# Define export parameters
export_params = {
    'image': slope.toFloat(),
    'description': "Slope",
    'folder': 'EarthEngineImages_crs',
    'scale': 30,  # Resolution in meters
    'fileFormat': 'GeoTIFF',
    'region': b.geometry().bounds().getInfo()['coordinates'],
    'crs': "EPSG:4326"
}

# Export the band to Google Drive
task = ee.batch.Export.image.toDrive(**export_params)
task.start()
print(f"Exporting Slope band to Google Drive...")
# %% EXport Canopy Height resampled to 30m
canopy_height = ee.ImageCollection("projects/meta-forest-monitoring-okw37/assets/CanopyHeight").mosaic()

# Select the band
# band_image = landcover.select("Map").first()

# Define export parameters
export_params = {
    'image': canopy_height.toFloat(),
    'description': "canopy_height",
    'folder': 'EarthEngineImages_crs',
    'scale': 30,  # Resolution in meters
    'fileFormat': 'GeoTIFF',
    'region': b.geometry().bounds().getInfo()['coordinates'],
    'crs': "EPSG:4326"
}
# Export the band to Google Drive
task = ee.batch.Export.image.toDrive(**export_params)
task.start()
print(f"Exporting canopy height band to Google Drive...")
#%%  Soil data
clay_mean = ee.ImageCollection('projects/sat-io/open-datasets/polaris/clay_mean').first();
ksat_mean = ee.ImageCollection('projects/sat-io/open-datasets/polaris/ksat_mean').first();
sand_mean = ee.ImageCollection('projects/sat-io/open-datasets/polaris/sand_mean').first();
silt_mean = ee.ImageCollection('projects/sat-io/open-datasets/polaris/silt_mean').first();

# Define export parameters
def export_fn(image,description):
        export_params = {
        'image': image.toFloat(),
        'description': description,
        'folder': 'EarthEngineImages_crs',
        'scale': 30,  # Resolution in meters
        'fileFormat': 'GeoTIFF',
        'region': b.geometry().bounds().getInfo()['coordinates'],
        'crs': "EPSG:4326"
        }
        # Export the band to Google Drive
        task = ee.batch.Export.image.toDrive(**export_params)
        task.start()
        print(f"Exporting band to Google Drive...")
export_fn(clay_mean,"clay")
export_fn(sand_mean,"sand")
export_fn(silt_mean,"silt")
export_fn(ksat_mean,"ksat")
# %%
