#%% 
from __future__ import print_function
import scipy.io as spio
import numpy as np
from tensorflow.keras.models import Model
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Input
from keras.optimizers import RMSprop, Adadelta, Adagrad, Adam, Nadam, SGD
from keras.callbacks import EarlyStopping, TerminateOnNaN
from keras import backend as K
from keras.losses import mean_squared_error
import os
import pandas as pd
import tensorflow as tf
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from keras.utils.layer_utils import count_params
from keras import backend as K
from sklearn.metrics import r2_score
# Define root mean squared error
def root_mean_squared_error(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true)))
#%%
## Read the data
os.chdir("D:\\Backup\\Rouhin_Lenovo\\US_project\\GEE_SEBAL_Project\\ML\\All_Data\\")
file_list=os.listdir()
list_files=[]
for i in range(len(file_list)):
    list_files.append(pd.read_csv(file_list[i]))
    list_files[i]=list_files[i][list_files[i]["H_inst_af"]>=0]
    list_files[i]=list_files[i][list_files[i]["Hinst"]>=0]
list_files[0].columns.tolist()
df=pd.concat(list_files)
print(df.shape[0])
# df["Hinst"].hist()
# df.columns.tolist()
# # df["R"]*df["ST_B10"]
# df["H_inst_af"]
# df["Landcover"]
df.columns.tolist()
#%% Just split to test and train in random fashion: no stratified sampling 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

##
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
 "doy","Rn","Ginst"]
features=['B',
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
 "doy","Hinst","LE_closed","ET_24h","Rn24h_G","eto","ETo_model_hourly",\
    "Rn","Ginst"]
labels=["H_inst_af"]
df[features].notna()
df = df.dropna(subset=main_features + labels)
print(df[main_features].describe())
X=df[features]
y=df[labels]
# ## Split data into features and labels

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Initialize the StandardScaler
scaler = StandardScaler()
# Fit the scaler on the training data and transform it
X_train_scaled_all = scaler.fit_transform(X_train.drop(columns={"Hinst","LE_closed","ET_24h","Rn24h_G","eto","ETo_model_hourly"}))
# Remove the last 3 columns to create X_train_scaled
X_train_scaled = X_train_scaled_all[:, :-2]
# Select the last column to create rn_train
rn_train = X_train_scaled_all[:, -2]
# Optionally, if you also want to select the second-to-last column as g_train
g_train = X_train_scaled_all[:, -1]
X_test_scaled_all =scaler.transform(X_test.drop(columns={"Hinst","LE_closed","ET_24h","Rn24h_G","eto","ETo_model_hourly"}))
X_test_scaled = X_test_scaled_all[:, :-2]
# Select the last column to create rn_test
rn_test = X_test_scaled_all[:, -2]
# Optionally, if you also want to select the second-to-last column as g_test
g_test = X_test_scaled_all[:, -1]
H_train_sebal=np.array(X_train)[:,-3]
H_test_sebal=np.array(X_test)[:,-3]
# print(y_test.describe())
# X_train_scaled = scaler.fit_transform(X_train))
# rn_train=tf.convert_to_tensor(scaler.fit_transform(X_train_scaled_all[:,-2])))
# rn_test=tf.convert_to_tensor(scaler.fit_transform(X_test["Rn"])))
# g_train=tf.convert_to_tensor(scaler.fit_transform(X_train_scaled_all[:,:-1])))
# g_test=tf.convert_to_tensor(scaler.fit_transform(X_test["Ginst"])))
# # Use the same scaler to transform the test data
# X_test_scaled = tf.convert_to_tensor(scaler.transform(X_test).drop(columns={"Rn","Ginst"})))
y_train=tf.convert_to_tensor(y_train)
y_test=tf.convert_to_tensor(y_test)
# G, tf.float32)
print(X_train_scaled_all.shape,X_test_scaled_all.shape)
print(X_train_scaled.shape,rn_train.shape,g_train.shape)
print(X_train[["SWIR1_LST","SWIR2_LST","NDWI_LST","NDVI_LST"]])
print(X_test_scaled_all[-2,:])
import joblib
scaler_filename = "D:\\Backup\\Rouhin_Lenovo\\US_project\\GEE_SEBAL_Project\\ML\\ML_model\\scaler.save"
joblib.dump(scaler, scaler_filename) 
#%% Using Keras for H prediction
from sklearn.metrics import root_mean_squared_error 
def rmse(Y_actual,Y_Predicted):
    return root_mean_squared_error(Y_actual,Y_Predicted)
def MAPE(Y_actual,Y_Predicted):
    mape = np.mean(( Y_Predicted-Y_actual))*100/np.mean(Y_actual)
    return mape

class NN:
    """
    Best modek 10,256 , do=0.2, Nadam, lamda=0.5 R2=0.7
    """
    def __init__(self, input_shape, rn_input_shape=(1,), g_input_shape=(1,)):
        self.rn_input_shape = rn_input_shape
        self.g_input_shape = g_input_shape
        self.model = self.create_model(input_shape, 10, 256)

    def create_model(self, input_shape,n_layers,n_nodes,drop_frac=0.2):
        main_input = Input(shape=input_shape)  
        rn_input = Input(shape=self.rn_input_shape)
        g_input = Input(shape=self.g_input_shape)

        x = main_input
        inputs=x
        # inputs=[main_input,rn_input,g_input]
        for _ in range(n_layers):
            x = Dense(n_nodes, activation='relu')(x)
            x = Dropout(drop_frac)(x)
        output = Dense(1, activation='linear')(x)
        model = Model(inputs=[main_input, rn_input, g_input], outputs=output)
        return model

    def phy_loss_mean(self,Rn, G):
        # print("shape",len(inputs))
        def loss(y_true, y_pred):
            # Rn, G = inputs[0], inputs[1]
            phy_loss = K.relu(y_pred - Rn + G)
            return K.mean(phy_loss)
        return loss

    def combined_loss(self, Rn, G, lamda=1):
        lam = K.constant(value=lamda)  # Regularization hyper-parameter
        def loss(y_true, y_pred):
            # Cast Rn and G to the same type as y_pred
            Rn_cast = K.cast(Rn, dtype=y_pred.dtype)
            G_cast = K.cast(G, dtype=y_pred.dtype)

            mse_loss = tf.keras.losses.mean_squared_error(y_true, y_pred)
            phy_loss = K.relu(y_pred - Rn_cast + G_cast)
            return mse_loss + lam * K.mean(phy_loss)
        return loss

    def compile_model(self, Rn, G, lamda,optimizer):
        loss_fn = self.combined_loss(Rn, G, lamda)
        phys_loss=self.phy_loss_mean(Rn,G)
        self.model.compile(optimizer=optimizer,
                            # loss='mean_squared_error',
                            loss=loss_fn,
                            metrics=[keras.metrics.RootMeanSquaredError(),loss_fn])

    def train_model(self, x_train, y_train, epochs=1000):
        self.model.fit(x_train, y_train, epochs=epochs,batch_size=500)

    def evaluate_model(self, x_test, y_test):
        return self.model.evaluate(x_test, y_test)
    def r2_calc(self,x_test,y_test):
        print(self.model.predict(x_test).shape,y_test.shape)
        return r2_score(y_test, self.model.predict(x_test))
    def bias_calc(self,x_test,y_test):
        return MAPE(y_test, self.model.predict(x_test).squeeze())
    def save_model(self, optimizer, drop_frac, lamda,outdir):
        # Create a filename based on the model configuration
        model_filename = f"model_optimizer_{optimizer.__class__.__name__}_dropout_{drop_frac}_lambda_{lamda}.h5"
        # Save the model
        self.model.save(outdir+model_filename)
        print(f"Model saved as {model_filename}")
# Example usage
# Assuming x_train, y_train, x_test, y_test are your training and testing data
input_shape = (None,)  # Input shape is flexible
model = NN(X_train_scaled.shape[1])
model.compile_model(Rn=rn_train, G=g_train,lamda=0.5,optimizer="Nadam")
# Prepare your training and testing data
x_train = [X_train_scaled, rn_train, g_train]
x_test = [X_test_scaled, rn_test, g_test]
# Train the model
model.train_model(x_train, y_train)
accuracy = model.evaluate_model(x_test, y_test)
print(f"Test accuracy: {accuracy}")
print(f"Test R2: {model.r2_calc(x_test, y_test)}")
print(f"Test Bias (%): {model.bias_calc(x_test, y_test)}")
model.save_model(optimizer='Nadam', drop_frac=0.2, lamda=0.5,\
    outdir='D:\\Backup\\Rouhin_Lenovo\\US_project\\GEE_SEBAL_Project\\ML\\ML_model\\')

# model.save('D:\\Backup\\Rouhin_Lenovo\\US_project\\GEE_SEBAL_Project\\ML\\ML_model\\model_do0.2_lamda0.5.keras')

# model.compile_model()
# model.train_model(X_train_scaled_all, y_train)
# accuracy = model.evaluate_model(X_test_scaled_all, y_test)
# print(f"Test accuracy: {accuracy}")
# print(f"Test R2: {model.r2_calc(X_test_scaled_all, y_test)}")
# print(f"Test Bias (%): {model.bias_calc(X_test_scaled_all, y_test)}")
print("GEESEBAL accuracy (RMSE)=",rmse(y_test,H_test_sebal) )
print("GEESEBAL R2 =",r2_score(y_test,H_test_sebal) )
print("GEESEBAL MAPE =",MAPE(y_test,H_test_sebal) )
#%%
## How does model affect the results of Daily ET 
## Energy balance: 
def calc_etdaily(Hpred,df,suffix):
    """
    Calculate ET daily using H pred from ML 
    """
    df_copy=df.copy().reset_index()
    # print(df_copy)
    df_copy["Hpred_"+suffix]=Hpred
    df_copy["LEinst_ml_"+suffix]=df_copy["Rn"]-df_copy["Ginst"]-df_copy["Hpred_"+suffix]
    df_copy["LE_daily_EF_"+suffix]=df_copy["LEinst_ml_"+suffix]*df_copy["Rn24h_G"]/(df_copy["Rn"]-df_copy["Ginst"])
    df_copy["LE_daily_ETo_"+suffix]=df_copy["LEinst_ml_"+suffix]*df_copy["eto"]/(df_copy["ETo_model_hourly"])
    return df_copy 

def call_model_MLET(ml_model,ml_features,external_data,suffix):
    """
    Call model prediction for ML H for given dataframes with features
    mll_features is for preidcting the model 
    external data contains other varibales from the geesebal model reqd for ET calc
    """
    Hpred=ml_model.model.predict(ml_features)
    my_list = map(lambda x: x[0], Hpred)
    series = pd.Series(my_list)
    # print(series)
    print(series.shape,external_data.shape[0])
    final_df=calc_etdaily(series,external_data,suffix)
    return final_df 

test_val=call_model_MLET(model,x_test,X_test,"test")
train_val=call_model_MLET(model,x_train,X_train,"train")

import matplotlib.pyplot as plt
def plot_scatter_LE(df,suffix):
    plt.plot(df["LE_closed"],1.2*df["LE_daily_EF_"+suffix],"o",c="r",label="ML ET EF")
    plt.plot(df["LE_closed"],df["LE_daily_ETo_"+suffix],"o",c="b",label="ML ET ETo")
    plt.plot(df["LE_closed"],df["ET_24h"]*28.36,"o",c="y",label="GEESEBAL",alpha=0.5)
    plt.plot(np.arange(0,250,2),np.arange(0,250,2))
    print("RMSE of LEd SEBAL=",rmse(df[df["LE_closed"]<15]["LE_closed"],df[df["LE_closed"]<15]["ET_24h"]*28.36) )
    print("RMSE of LEd ML ETo=",rmse(df[df["LE_closed"]<15]["LE_closed"],df[df["LE_closed"]<15]["LE_daily_ETo_"+suffix]) )
    print("RMSE of LEd ML EF=",rmse(df[df["LE_closed"]<15]["LE_closed"],df[df["LE_closed"]<15]["LE_daily_EF_"+suffix]) )


    plt.legend()
    plt.show()
    return
# plot_scatter_LE(test_val,"test")
plot_scatter_LE(test_val,"test")

def plot_scatter_H(df,suffix):
    plt.plot(y_test,df["Hpred_"+suffix],"o",c="r",label="ML ET EF")
    plt.plot(y_test,df["Hinst"],"o",c="y",label="GEESEBAL")
    plt.plot(np.arange(0,800,2),np.arange(0,800,2))
    print(r2_score(y_test,df["Hpred_"+suffix]))
    plt.legend()
    plt.show()
    return
plot_scatter_H(test_val,"test")
# plot_scatter_H(train_val,"train")

test_val["Hpred_test"]
X_test["Landcover_wc"].hist()
#%%
# Example usage with looping over hyperparameters and saving results
input_shape = (X_train_scaled.shape[1],)

# Define the ranges for the hyperparameters
optimizers = ['adam', 'nadam', 'sgd']
dropout_rates = [0.1,0.15,0.2, 0.25]
lambda_values = [0.1,0.5,1,10]

# Create a directory to save the results
results_dir = "D:\\Backup\\Rouhin_Lenovo\\model_results_H_non_stratified"
os.makedirs(results_dir, exist_ok=True)

# Loop over the different hyperparameter combinations
for optimizer in optimizers:
    for dropout_rate in dropout_rates:
        for lamda in lambda_values:
            print(f"Training with optimizer={optimizer}, dropout_rate={dropout_rate}, lambda={lamda}")
            
            # Create the model
            model = NN(input_shape)
            
            # Compile the model with the current set of hyperparameters
            model.compile_model(Rn=rn_train, G=g_train, lamda=lamda, optimizer=optimizer)
            
            # Prepare your training and testing data
            x_train = [X_train_scaled, rn_train, g_train]
            x_test = [X_test_scaled, rn_test, g_test]
            
            # Train the model
            model.train_model(x_train, y_train)
            
            # Evaluate the model
            accuracy = model.evaluate_model(x_test, y_test)
            r2 = model.r2_calc(x_test, y_test)
            bias = model.bias_calc(x_test, y_test)
            
            # Save the model
            model_name = f"model_{optimizer}_dropout{dropout_rate}_lambda{lamda}.h5"
            model_path = os.path.join(results_dir, model_name)
            model.model.save(model_path)
            
            # Save the evaluation results
            results = {
                "optimizer": optimizer,
                "dropout_rate": dropout_rate,
                "lambda": lamda,
                "accuracy": accuracy,
                "r2": r2,
                "bias": bias
            }
            results_filename = f"results_{optimizer}_dropout{dropout_rate}_lambda{lamda}.npz"
            np.savez(os.path.join(results_dir, results_filename), **results)
            
            # Print the results for this combination
            print(f"Test accuracy: {accuracy}")
            print(f"Test R2: {r2}")
            print(f"Test Bias (%): {bias}\n")
#%%
y_test
#%%
def phy_loss_mean(inputs):
    print("shape",len(inputs))
    def loss(y_true, y_pred):
        Rn, G = inputs[0], inputs[1]
        phy_loss = K.relu(y_pred - Rn - G)
        return K.mean(phy_loss)
    return loss

def combined_loss(inputs):
    lam = K.constant(value=lamda)  # Regularization hyper-parameter
    def loss(y_true, y_pred):
        Rn, G = inputs[0], inputs[1]
        mse_loss = tf.keras.losses.mean_squared_error(y_true, y_pred)
        phy_loss = K.relu(y_pred - Rn - G)
        return mse_loss + lam * K.mean(phy_loss)
    return loss
#%%
def PGNN_train_test(trainX, trainY, testX, testY, optimizer_name, optimizer_val, drop_frac,
                    use_YPhy, iteration, n_layers, n_nodes, lamda, physics_loss_on,
                    train_Rn=None, train_G=None, test_Rn=None, test_G=None):
    # Hyper-parameters of the training process
    batch_size = 1000
    num_epochs = 1000
    val_frac = 0.1
    patience_val = 500

    # Initializing results filename
    exp_name = optimizer_name + '_drop' + str(drop_frac) + '_usePhy' + str(use_YPhy) +  '_nL' + str(n_layers) + '_nN' + str(n_nodes) +  '_lamda' + str(lamda) + '_iter' + str(iteration)
    exp_name = exp_name.replace('.','pt')
    results_dir = '../results/'
    model_name = results_dir + exp_name + '_model.h5' # storing the trained model
    results_name = results_dir + exp_name + '_results.mat' # storing the results of the model

    # Placeholder for inputs and model
    input_shape = (trainX.shape[1],)  # Main features input shape

    # Define the main input
    main_input = Input(shape=input_shape, dtype=tf.float32)

    if physics_loss_on:
        if train_Rn is None or train_G is None or test_Rn is None or test_G is None:
            raise ValueError("Rn and G must be provided for both train and test sets when physics_loss_on is True")
        
        # Define Rn and G as additional inputs
        Rn_input = Input(shape=(1,), dtype=tf.float32)
        G_input = Input(shape=(1,), dtype=tf.float32)
        inputs = [main_input, Rn_input, G_input]  # Main features and physics inputs
    else:
        # Use Rn and G as part of the features if physics loss is off
        Rn = trainX[:, -2].reshape(-1, 1)
        G = trainX[:, -1].reshape(-1, 1)
        trainX = trainX[:, :-2]
        testX = testX[:, :-2]
        
        inputs = [main_input, Input(shape=(1,), dtype=tf.float32), Input(shape=(1,), dtype=tf.float32)]

    # Building the model
    x = main_input
    for _ in range(n_layers):
        x = Dense(n_nodes, activation='relu')(x)
        x = Dropout(drop_frac)(x)
    output = Dense(1, activation='linear')(x)
    
    model = Model(inputs=inputs, outputs=output)

    if physics_loss_on:
        totloss = combined_loss([train_Rn, train_G, lamda])
        phyloss = phy_loss_mean([train_Rn, train_G])
    else:
        totloss = mean_squared_error
        phyloss = root_mean_squared_error

    model.compile(loss=totloss,
                  optimizer=optimizer_val,
                  metrics=[phyloss, root_mean_squared_error])

    early_stopping = EarlyStopping(monitor='val_loss', patience=patience_val, verbose=1)

    print('Running...' + optimizer_name)
    
    if physics_loss_on:
        print(f"train_Rn shape: {train_Rn.shape}")
        print(f"train_G shape: {train_G.shape}")
        history = model.fit([trainX, train_Rn, train_G], trainY,
                            batch_size=batch_size,
                            epochs=num_epochs,
                            verbose=1,
                            validation_split=val_frac, callbacks=[early_stopping, TerminateOnNaN()])
    else:
        history = model.fit([trainX, Rn, G], trainY,
                            batch_size=batch_size,
                            epochs=num_epochs,
                            verbose=1,
                            validation_split=val_frac, callbacks=[early_stopping, TerminateOnNaN()])

    test_score = model.evaluate([testX, test_Rn, test_G] if physics_loss_on else [testX, test_Rn, test_G], testY, verbose=0)
    print('iter: ' + str(iteration) + ' useYPhy: ' + str(use_YPhy) + ' nL: ' + str(n_layers) + ' nN: ' + str(n_nodes) + ' lamda: ' + str(lamda) + ' trsize: ' + str(tr_size) + ' TestRMSE: ' + str(test_score[2]) + ' PhyLoss: ' + str(test_score[1]))
    model.save(model_name)
    spio.savemat(results_name, {'train_loss': history.history['loss'], 'val_loss': history.history['val_loss'], 'train_rmse': history.history['root_mean_squared_error'], 'val_rmse': history.history['val_root_mean_squared_error'], 'test_rmse': test_score[2]})
if __name__ == '__main__':
    # Main Function

    # List of optimizers to choose from    
    optimizer_names = ['Adagrad', 'Adadelta', 'Adam', 'Nadam', 'RMSprop', 'SGD', 'NSGD']
    optimizer_vals = [Adagrad(clipnorm=1), Adadelta(clipnorm=1), Adam(clipnorm=1), Nadam(clipnorm=1), RMSprop(clipnorm=1), SGD(clipnorm=1.), SGD(clipnorm=1, nesterov=True)]

    # Selecting the optimizer
    optimizer_num = 2  # Adam
    optimizer_name = optimizer_names[optimizer_num]
    optimizer_val = optimizer_vals[optimizer_num]

    # Selecting Other Hyper-parameters
    drop_frac = 0  # Fraction of nodes to be dropped out
    use_YPhy = 1  # Whether YPhy is used as another feature in the NN model or not
    n_layers = 2  # Number of hidden layers
    n_nodes = 12  # Number of nodes per hidden layer

    # Set lamda=0 for pgnn0
    lamda = 1000 * 0.5  # Physics-based regularization constant

    # Iterating over different training fractions and splitting indices for train-test splits
    # trsize_range = [5000, 2500, 1000, 500, 100]
    iter_range = np.arange(1)  # range of iteration numbers for random initialization of NN parameters

    # Default training size = 5000
    # tr_size = trsize_range[0]

    # List of lakes to choose from
    # lake = ['mendota', 'mille_lacs']
    # lake_num = 0  # 0 : mendota , 1 : mille_lacs
    # lake_name = lake[lake_num]

    # Iterating through all possible params
    for iteration in iter_range:
        PGNN_train_test(X_train_scaled,y_train,X_test_scaled,y_test,optimizer_name, 
        optimizer_val, drop_frac, use_YPhy, iteration, n_layers, n_nodes, lamda, 
        physics_loss_on=True,train_Rn=tf.convert_to_tensor(X_train_scaled[:-2]),
         train_G=tf.convert_to_tensor(X_train_scaled[:-2]),
         test_Rn=tf.convert_to_tensor(X_test_scaled[:-2]),
          test_G=tf.convert_to_tensor(X_test_scaled[:-1]))
# %%


# X_train_scaled.shape
y_train.shape