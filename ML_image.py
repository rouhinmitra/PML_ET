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
#Let's use this cell to read the landsat data 
dir="D:\\Backup\\Rouhin_Lenovo\\US_project\\Untitled_Folder\\Data\\Image_test\\Landsat_test\\EarthEngineImages"
os.chdir(dir)
file_list=os.listdir()
# print(file_list)
xds = xarray.open_dataset(file_list[1], engine="rasterio")
# xds.band_data.plot(vmin=0,vmax=1)
# print(xds.band_data.to_numpy())
print(xds.band_data.to_numpy()[~np.isnan(xds.band_data.to_numpy())])
## 
da_reprojected = xds.rio.reproject("EPSG:4326")
# da_reprojected.band_data.plot()

# da_reprojected.band_data.to_numpy()
ds_masked = da_reprojected.where((da_reprojected['band_data']<10000) )  
# ds_masked
# ds_masked.band_data.plot(vmin=0,vmax=8,cmap="RdYlBu")
## Let's do some torch geo 
class landsat_gee(RasterDataset):
    filename_glob = '*.tif'
    filename_regex = r'^Landsat_(?P<band>[A-Z_0-9]+)\.tif$'  # Regex to capture band names
    # date_format = '%Y%m%dT%H%M%S'
    is_image = True
    separate_files = True
    all_bands = ["B","R","GR","NIR","SWIR_1","SWIR_2",'ST_B10',"NDVI","NDWI","ALFA",'Tao_sw_1',"Rs_down","Rl_down","Rl_up","Rn","G","H","LE","ET_24h"]
    features = ["B","R","GR","NIR","SWIR_1","SWIR_2",'ST_B10',"NDVI","NDWI","ALFA",'Tao_sw_1',"Rs_down","Rl_down","Rl_up","Rn","G","H","LE"]
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
    def plot_other(self,sample,vmin,vmax):
        # print(sample)
        fig, ax = plt.subplots()
        map=ax.imshow(sample,vmin=vmin,vmax=vmax)
        plt.colorbar(map)
        plt.show()
    def transform(self,sample,scaler):
        '''
        scaling the images using standard scaler imported from sckit learn
        '''
        feature_indices=[]
        for band in self.features:
            feature_indices.append(self.all_bands.index(band))
        image = sample['image'][feature_indices]
        print(image.shape,"input image")
        print(image)
        print("flattened image",image.flatten(start_dim=1))
        resized_feature=image.flatten(start_dim=1).transpose(0,1)
        print("trasnposed image",resized_feature)

        print(resized_feature.shape,"resized shape")
        # print(image.flatten(start_dim=1).transpose(0,1).shape)
        # print(scaler.transform(resized_feature))
        return scaler.transform(resized_feature)
        # return image.flatten(start_dim=1).transpose(0,1).numpy()
    def inference(self,model,image):
        '''
        Model prediction and reconstruction of the 
        '''
        pred=model(image)
        pred=torch.div(pred,28.36)
        print("op predicted image shape",pred.shape)
        print("pred",pred)
        # print(pred.reshape(1024,1024))
        # print(pred.transpose(0,1).reshape(1024,1024).detach().numpy())
        print("pred shape",np.argwhere(np.isnan(pred.reshape(256,256).detach())))
        print("Number of nans",np.count_nonzero(np.isnan(pred.reshape(256,256).detach())))
        print("reshaped pred",pred.reshape(256,256).detach())
        self.plot_other(pred.reshape(256,256).detach(),0,3)
        return np.argwhere(np.isnan(pred.reshape(256,256).detach()))

torch.manual_seed(6)
dataset = landsat_gee(dir)
sampler = RandomGeoSampler(dataset, size=256, length=1)
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
    dataset.plot_other(sample["image"][17],0,400)
    plt.axis("off")
    plt.show()

## Load the model and scalar transforms 
# Load

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class NeuralNetwork(nn.Module):
    def __init__(self, input_shape, hidden_units=[128, 128, 128, 128, 128, 64], output_units=1, dropout_rate=0.5):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        # Define the layers dynamically based on the hidden_units list
        layers = []
        in_features = input_shape
        for hidden in hidden_units:
            layers.append(nn.Linear(in_features, hidden))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            in_features = hidden
        # Add the final output layer
        layers.append(nn.Linear(in_features, output_units))
        # Use nn.Sequential to combine the layers
        self.network = nn.Sequential(*layers)
    def forward(self, x):
        x=torch.from_numpy(x).to(torch.float32)
        # print(x.dtype)
        x = self.network(x)
        return x
model = NeuralNetwork(18)
model.load_state_dict(torch.load("D:\\Backup\\Rouhin_Lenovo\\US_project\\Untitled_Folder\\Data\\ML_models\\Phys5000.pt"))
model = model.to(device) # Set model to gpu
model.eval()
# model = torch.load("D:\\Backup\\Rouhin_Lenovo\\US_project\\Untitled_Folder\\Data\\ML_models\\Phys5000.pt")
# And now to load...
scaler = joblib.load("D:\\Backup\\Rouhin_Lenovo\\US_project\\Untitled_Folder\\Data\\ML_models\\standard_scaler_phys.bin") 
# print(model.eval())
for batch in dataloader:
    sample = unbind_samples(batch)[0]
    print(sample["image"][18].numpy())

    # print(scaler.transform(sample["image"][6]))
    # print(sample)
    scaled_image=dataset.transform(sample,scaler)
    print(scaled_image)
    a=dataset.inference(model,scaled_image)
    print("Number of nans",np.count_nonzero(np.isnan(sample["image"][18])))
    print("array",np.argwhere(np.isnan(sample["image"][18])))

    # print(a==np.argwhere(np.isnan(sample["image"][18])))
#%%
 