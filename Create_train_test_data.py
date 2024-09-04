#%%
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import os
from collections import defaultdict 
import random
#%%
"""
This py file is used to create a 3 separate sets of data with  equal distribtuion of landcover
for cross validation and generating out of sample estimates to test the model 
"""
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
def remove_na_for_Hinstpred(df):
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
        "Landcover_wc",
        "doy","Hinst","LE_closed","ET_24h","Rn24h_G",
        "Rn","Ginst","H_inst_af"]
    if df.shape[0]!=0:
        df=df.dropna(subset=main_features)

    return df

def veg_mapping(df):
    """
    As some vegetation/landcover types are sparse we combine some of them for better stratified sampling
    """
    print(df)
    map = {
        'CRO': 'CRO',
        'ENF': 'ENF',
        'WET': 'WET',
        'CSH': 'shrublands',
        'OSH': 'shrublands',
        'CVM': 'shrublands',
        'BSV': 'shrublands',
        'GRA': 'GRA',
        'WSA': 'GRA',
        'SAV': 'GRA',
        'DBF': 'DBF',
        'MF': 'DBF'
    }
    df["Veg_ML"]=df['Veg'].map(map)
    return df

# def save_train_test(df,base_dir):
#     # Define the base directory where the train and test folders will be created
#     train_dir = os.path.join(base_dir, 'train')
#     test_dir = os.path.join(base_dir, 'test')

#     # Create directories if they do not exist
#     os.makedirs(train_dir, exist_ok=True)
#     os.makedirs(test_dir, exist_ok=True)

#     # Iterate over the folds
#     for fold in range(3):
#         # Get train and test site IDs for the current fold
#         test_sites = test_sites_folds[fold]
#         train_sites = train_sites_folds[fold]
        
#         # Filter the DataFrame for test and train sites
#         test_df = geo_df[geo_df['Site_ID'].isin(test_sites)]
#         train_df = geo_df[geo_df['Site_ID'].isin(train_sites)]
        
#         # Define file paths
#         test_file_path = os.path.join(test_dir, f'test_fold_{fold + 1}.csv')
#         train_file_path = os.path.join(train_dir, f'train_fold_{fold + 1}.csv')
        
#         # Save DataFrames to CSV
#         test_df.to_csv(test_file_path, index=False)
#         train_df.to_csv(train_file_path, index=False)
        
#         print(f'Saved test sites for fold {fold + 1} to {test_file_path}')
#         print(f'Saved train sites for fold {fold + 1} to {train_file_path}')
#%%
data,file_name=read_flux("D:\\Backup\\Rouhin_Lenovo\\US_project\\GEE_SEBAL_Project\\ML\\ML_processed_data_all\\")
data=[remove_na_for_Hinstpred(df) for df in data]

data=[veg_mapping(df) for df in data]
#%%
## Instead on concatenating the data we will split the data clustered on landcover (This has an issue that we might be using spatial autocorrelated datasets for predction)
stations_lc=[]
count=[]
for i in pd.concat(data)["Veg_ML"].unique():
    counter=0
    print(i)
    for st in range(len(data)):
        if i==data[st]["Veg_ML"].iloc[0]:
            counter=counter+1
    count.append(counter)
print(count)
# %%
# %%
for i in data:
    if i.Veg.iloc[0]=="CRO":
        print(i.Name.iloc[0],i.Lat.iloc[0],i.Long.iloc[0])
## Convert the landcovers to 6 main classes: 
# CRO, ENF, WET , Shrubland (CSH, OSH and CVM,BSV), GRA(WSA and SAV to GRA), DBF (MF to DBF),
#%%
import random
from collections import defaultdict

# Step 1: Create dictionary of Site_ID and Land_cover
site_id = []
landcover = []

for i in range(len(data)):
    site_id.append(data[i]["Name"].iloc[0])
    landcover.append(data[i]["Veg_ML"].iloc[0])

station_landcover_dict = {k: v for k, v in zip(site_id, landcover)}

# Step 2: Group stations by landcover
landcover_groups = defaultdict(list)
for station, landcover in station_landcover_dict.items():
    landcover_groups[landcover].append(station)

print("Before shuffling:")
print(dict(landcover_groups))  # Converting defaultdict to dict for a clean print

# Step 3: Set random seed and shuffle
random.seed(6)
for landcover, stations in landcover_groups.items():
    random.shuffle(stations)

# Step 4: Split stations into 3 unique test sets
test_sites_folds = defaultdict(list)
train_sites_folds = defaultdict(list)

for landcover, stations in landcover_groups.items():
    # Determine number of stations per fold
    fold_size = len(stations) // 3
    
    for fold in range(3):
        # Assign test sites for the current fold
        start_idx = fold * fold_size
        if fold == 2:  # The last fold takes the remainder of the list
            test_sites_folds[fold].extend(stations[start_idx:])
        else:
            test_sites_folds[fold].extend(stations[start_idx:start_idx + fold_size])
    
    # Assign train sites for each fold
    for fold in range(3):
        train_sites_folds[fold].extend([s for s in stations if s not in test_sites_folds[fold]])


# Step 5: Calculate and print land cover distribution for each fold
for fold in range(3):
    # Count land cover distribution in test set
    test_landcover_distribution = pd.Series([station_landcover_dict[site] for site in test_sites_folds[fold]]).value_counts(normalize=True)
    
    # Count land cover distribution in train set
    train_landcover_distribution = pd.Series([station_landcover_dict[site] for site in train_sites_folds[fold]]).value_counts(normalize=True)
    print(f"Fold {fold + 1} - Test sites:")
    print(test_sites_folds[fold])
    print(f"Fold {fold + 1} - Train sites:")
    print(train_sites_folds[fold])
    # Print the distributions
    print(f"Fold {fold + 1} - Land cover distribution in Test set:")
    print(test_landcover_distribution)
    print(f"Fold {fold + 1} - Land cover distribution in Train set:")
    print(train_landcover_distribution)
    print("\n")
# You can now use `test_sites_folds` and `train_sites_folds` to extract the corresponding data
 
# %% Plotting on a map 
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point

# Assuming data is a list of DataFrames where each DataFrame contains 'Name', 'Veg_ML', 'Latitude', and 'Longitude'
site_id = []
landcover = []
latitudes = []
longitudes = []

for i in range(len(data)):
    site_id.extend(data[i]["Name"].tolist())
    landcover.extend(data[i]["Veg_ML"].tolist())
    latitudes.extend(data[i]["Lat"].tolist())  # Assuming 'Latitude' column exists
    longitudes.extend(data[i]["Long"].tolist())  # Assuming 'Longitude' column exists

# Create a DataFrame
df = pd.DataFrame({
    'Site_ID': site_id,
    'Land_cover': landcover,
    'Latitude': latitudes,
    'Longitude': longitudes
})

# Create a GeoDataFrame
geometry = [Point(xy) for xy in zip(df['Longitude'], df['Latitude'])]
geo_df = gpd.GeoDataFrame(df, geometry=geometry)

# Set up the plot
fig, axs = plt.subplots(3, 2, figsize=(15, 15), sharex=True, sharey=True)
axs = axs.flatten()

for fold in range(3):
    # Test and train sites
    test_sites = test_sites_folds[fold]
    train_sites = train_sites_folds[fold]
    
    # Filter the GeoDataFrame
    test_geo_df = geo_df[geo_df['Site_ID'].isin(test_sites)]
    train_geo_df = geo_df[geo_df['Site_ID'].isin(train_sites)]
    
    # Plot train and test sites
    ax = axs[fold]
    base = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))  # Use a background map
    base.plot(ax=ax, color='lightgrey', edgecolor='black')
    
    train_geo_df.plot(ax=ax, color='blue', markersize=15, label='Train Sites', alpha=0.7)
    test_geo_df.plot(ax=ax, color='red', markersize=15, label='Test Sites', alpha=0.7)
    # Set plot limits to focus on the US
    plt.xlim(-130, -65)  # Longitude limits for CONUS
    plt.ylim(24, 50)  
    ax.set_title(f'Fold {fold + 1}')
    ax.legend()
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

plt.tight_layout()
plt.show()


# %%Save test train 
# Define the base directory where the train and test folders will be created
base_dir = 'D:\\Backup\\Rouhin_Lenovo\\US_project\\GEE_SEBAL_Project\\ML\\Test_train\\'
# Create directories if they do not exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Extract site data from the list of dataframes
site_data = {}
for df in data:
    site_id = df["Name"].iloc[0]  # Assuming Name is a unique identifier
    site_data[site_id] = df

# Iterate over the folds
for fold in range(3):
    # Get train and test site IDs for the current fold
    test_sites = test_sites_folds[fold]
    train_sites = train_sites_folds[fold]
    
    # Prepare lists to hold the filtered dataframes
    test_data_list = []
    train_data_list = []
    
    # Filter site data based on site IDs
    for site_id in test_sites:
        if site_id in site_data:
            test_data_list.append(site_data[site_id])
    
    for site_id in train_sites:
        if site_id in site_data:
            train_data_list.append(site_data[site_id])
    
    # Concatenate the filtered dataframes
    test_df = pd.concat(test_data_list, ignore_index=True)
    train_df = pd.concat(train_data_list, ignore_index=True)
    
    # Define file paths
    test_file_path = os.path.join(test_dir, f'test_fold_{fold + 1}.csv')
    train_file_path = os.path.join(train_dir, f'train_fold_{fold + 1}.csv')
    
    # Save DataFrames to CSV
    test_df.to_csv(test_file_path, index=False)
    train_df.to_csv(train_file_path, index=False)
    
    print(f'Saved test sites for fold {fold + 1} to {test_file_path}')
    print(f'Saved train sites for fold {fold + 1} to {train_file_path}')
#%% 
