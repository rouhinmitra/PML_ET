#%%
import numpy as np
import xarray
import pandas as pd 
import rioxarray as rio
import os 
import matplotlib.pyplot as plt
## Let's get some torchgeo 
import os
import tempfile
from urllib.parse import urlparse
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchgeo.datasets import RasterDataset, stack_samples, unbind_samples
from torchgeo.datasets.utils import download_url
from torchgeo.samplers import RandomGeoSampler
import joblib
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_percentage_error as MAPE
from torchgeo.samplers import GridGeoSampler
from tensorflow.keras.models import load_model

#Let's use this cell to read the landsat data 
dir="D:\\Backup\\Rouhin_Lenovo\\US_project\\GEE_SEBAL_Project\\ML\\Image_features\\EE_images_ET_daily"
os.chdir(dir)
file_list=os.listdir()
print(file_list[21])
xds = xarray.open_dataset(file_list[21], engine="rasterio")
# xds.band_data.plot(vmin=0,vmax=1)
# print(xds.band_data.to_numpy())
print(xds.band_data.to_numpy()[~np.isnan(xds.band_data.to_numpy())])
## 
da_reprojected = xds.rio.reproject("EPSG:4326")
# da_reprojected.band_data.plot()
# da_reprojected.band_data.to_numpy()
ds_masked = da_reprojected.where((da_reprojected['band_data']<10000) ) 
print(da_reprojected)
#%% 
# ds_masked
# ds_masked.band_data.plot(vmin=0,vmax=8,cmap="RdYlBu")
## Let's do some torch geo 
class landsat_gee(RasterDataset):
    filename_glob = '*.tif'
    filename_regex = r'^Landsat_(?P<band>[A-Z_0-9]+)\.tif$'  # Regex to capture band names
    # date_format = '%Y%m%dT%H%M%S'
    is_image = True
    separate_files = True
    all_bands=["Landcover","Elevation","Slope","B","R","GR","NIR","SWIR_1","SWIR_2",'ST_B10',"NDVI","NDWI",\
    "B_LST","GR_LST","R_LST","NIR_LST","NDVI_LST","NDWI_LST",\
        "SWIR_1_LST","SWIR_2_LST","Rn","G","H","LE","ET_24h"]
    # features = ["B","R","GR","NIR","SWIR_1","SWIR_2",'ST_B10',"NDVI","NDWI","Rn","G"]
    rgb_bands = ['R', 'GR', 'B']
    def plot(self, sample):
        # Find the correct band index order
        rgb_indices = []
        for band in self.rgb_bands:
            rgb_indices.append(self.all_bands.index(band))
        # Reorder and rescale the image
        image = sample['image'][rgb_indices].permute(1, 2, 0)
        # Inspect the raw range of the image
        print(f"Raw image range: min={image.min()}, max={image.max()}")
        # print(image[:,:,0]==image[:,:,1])
        image = torch.clamp(image, 0, 1).numpy()
        fig, ax = plt.subplots()
        ax.imshow(image*4)
        return fig
    def plot_other(self,sample,vmin,vmax,cmap="RdYlBu_r"):
        # print(sample)
        fig, ax = plt.subplots()
        map=ax.imshow(sample,vmin=vmin,vmax=vmax,cmap=cmap)
        plt.colorbar(map)
        plt.show()
torch.manual_seed(6)
dataset = landsat_gee(dir)
sampler = RandomGeoSampler(dataset, size=2098, length=1)
# sampler=GridGeoSampler(dataset,size=512,stride=512)
def custom_collate_fn(batch):
    # This assumes batch is a list of samples
    images = [item['image'] for item in batch]
    samples = {
        'image': torch.stack(images)
    }
    return samples

dataloader = DataLoader(dataset,sampler=sampler,collate_fn=custom_collate_fn)
for batch in dataloader:
    sample = unbind_samples(batch)[0]
    # sample2=unbind_samples(sample)[0]
    # print(sample["image"][6].shape)
    dataset.plot(sample)
    # print(sample["image"][6])
    # a=sample["image"][18].flatten().transpose(0,1)
    # print(a.shape)
    # tmp=a.reshape(256,25624).detach().numpy()
    dataset.plot_other(sample["image"][-1],0,7)
    print(sample["image"][2])
    # print(sample["image"][5].shape)
    # print(batch["image"])
    plt.axis("off")
    plt.show()
print(dataset)

# %%
## Claude inference 
import tensorflow as tf
from tensorflow import keras
import joblib
import numpy as np
from datetime import datetime
import keras.backend as K
# Load the pre-trained Keras model
# model = keras.models.load_model('D:\\Backup\\Rouhin_Lenovo\\US_project\\GEE_SEBAL_Project\\ML\\ML_model\\model_optimizer_str_dropout_0.2_lambda_0.5.h5')
# Load the saved StandardScaler
scaler = joblib.load('D:\\Backup\\Rouhin_Lenovo\\US_project\\GEE_SEBAL_Project\\ML\\ML_model\\scaler_0.save')
# Preprocess the image data
features = ['B',
 'GR',
 'R',
 'NIR',
 'SWIR_1',
 'SWIR_2','ST_B10','NDVI','NDWI',
 'R_LST',
 'GR_LST',
 'B_LST',
 'NIR_LST',
 'NDVI_LST',
 'NDWI_LST',
 'SWIR_1_LST',
 'SWIR_2_LST',
 "Elevation",
 "Slope",
 "Landcover",
 "Rn",
 "G"]
print("No of features",len(features))
feature_indices = [dataset.all_bands.index(band) for band in features if band in dataset.all_bands]
print("No of images",len(feature_indices))

# # Extract the relevant features
image_data = sample['image'][feature_indices]
print(len(feature_indices),len(dataset.all_bands))
# # Get the image shape
height, width = image_data.shape[1], image_data.shape[2]
print(height,width)
# # # Create doy feature
# # # Replace this with the actual date of your image
image_date = datetime(2022, 4, 9)  # Example date
doy = image_date.timetuple().tm_yday
doy_array = np.full((1,height, width), doy)
print("Feature array shape",image_data.shape)
print("DOY array shape",doy_array.shape)
# Split the image_data at the insert position
insert_position=-2
image_data_before = image_data[:insert_position,:,:]
image_data_after = image_data[insert_position:,:,:]
# Concatenate the parts with doy in between
image_data_with_doy = np.concatenate([image_data_before, doy_array, image_data_after], axis=0)
# # Concatenate doy to image_data (DOY has to go after landcover for the standard sclaer to apply)
# image_data_with_doy = np.concatenate([image_data, doy_array], axis=0)
print(image_data_with_doy.shape,image_data.shape)
##---Standardize 
# Step 1: Transpose the image to get shape (X, Y, bands)
transposed_image = image_data_with_doy.transpose(1, 2, 0)
# Step 2: Reshape to flatten the spatial dimensions, resulting in shape (X * Y, bands)
flattened_image = transposed_image.reshape(height*width, 23)
scaled_data = scaler.transform(flattened_image)
scaled_image=scaled_data.reshape(height,width,23)
input_scaled=scaled_image
input_shape=input_scaled.shape
X_input = input_scaled.reshape(height*width, input_shape[2]) 
print(X_input.shape)
model = load_model('D:\\Backup\\Rouhin_Lenovo\\US_project\\GEE_SEBAL_Project\\ML\\ML_model\\LE_closed_model_optimizer_str_dropout_0.2_fold_0.h5')
predictions = model.predict(X_input)
# ## Reshape predictions back to the original image shape
predictions_reshaped = predictions.reshape(height, width)
#%%
# print(predictions)
# # Visualize the predictions
dataset.plot_other(predictions_reshaped/28.36, 0,7)
# plt.title("Model Predictions (H_inst)")
# plt.show()
#%%

# %%
