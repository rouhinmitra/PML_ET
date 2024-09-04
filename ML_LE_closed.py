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
import sklearn
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from keras.utils.layer_utils import count_params
from keras import backend as K
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import joblib
from sklearn.metrics import root_mean_squared_error 

#%% Using Keras for H prediction
def rmse(Y_actual,Y_Predicted):
    return root_mean_squared_error(Y_actual,Y_Predicted)
def rmse_per(Y_actual,Y_Predicted):
    return root_mean_squared_error(Y_actual,Y_Predicted)*100/Y_actual.mean()
def MAPE(Y_actual,Y_Predicted):
    mape = np.mean(( Y_Predicted-Y_actual))*100/np.mean(Y_actual)
    return mape
#%%
# Directory where the folds are saved
train_dir = 'D:\\Backup\\Rouhin_Lenovo\\US_project\\GEE_SEBAL_Project\\ML\\Test_train\\train\\'
test_dir = 'D:\\Backup\\Rouhin_Lenovo\\US_project\\GEE_SEBAL_Project\\ML\\Test_train\\test\\'
# Main features and labels
main_features = ['B', 'GR', 'R', 'NIR', 'SWIR_1', 'SWIR_2', 'ST_B10', 'NDVI_model', 'NDWI',
                 'R_LST', 'GR_LST', 'B_LST', 'NIR_LST', 'NDVI_LST', 'NDWI_LST', 'SWIR1_LST',
                 'SWIR2_LST', "Elev_SRTM", "Landcover_wc","Canopy_height","doy", "Rn", "Ginst"]
labels = ["LE_closed"]
#%%
# Custom Neural Network class (same as in your script)
class NN:
    """
    Best modek 10,256 , do=0., batch size=32, Nadam, R2=0.7 w physics based inputs 
    Also model 10,256, do=0.2, batch size=512, 
    """
    def __init__(self, input_shape, rn_input_shape=(1,), g_input_shape=(1,)):
        self.rn_input_shape = rn_input_shape
        self.g_input_shape = g_input_shape
        self.model = self.create_model(input_shape, 10, 256)

    def create_model(self, input_shape,n_layers,n_nodes,drop_frac=0.0):
        main_input = Input(shape=input_shape)  
        # rn_input = Input(shape=self.rn_input_shape)
        # g_input = Input(shape=self.g_input_shape)
        x = main_input
        inputs=x
        # inputs=[main_input,rn_input,g_input]
        for _ in range(n_layers):
            x = Dense(n_nodes, activation='relu')(x)
            x = Dropout(drop_frac)(x)
        output = Dense(1, activation='linear')(x)
        model = Model(inputs=[main_input], outputs=output)
        return model

    def compile_model(self,optimizer):
        # loss_fn = self.combined_loss(Rn, G, lamda)
        # phys_loss=self.phy_loss_mean(Rn,G)
        self.model.compile(optimizer=optimizer,
                            loss='mean_squared_error',
                            # loss=loss_fn,
                            metrics=[keras.metrics.RootMeanSquaredError()])

    def train_model(self, x_train, y_train, epochs=500):
        self.model.fit(x_train, y_train, epochs=epochs,batch_size=32)

    def evaluate_model(self, x_test, y_test):
        return self.model.evaluate(x_test, y_test)
    def r2_calc(self,x_test,y_test):
        print(self.model.predict(x_test).shape,y_test.shape)
        return r2_score(y_test, self.model.predict(x_test))
    def bias_calc(self,x_test,y_test):
        return MAPE(y_test, self.model.predict(x_test).squeeze())
    def save_model(self,fold, optimizer, drop_frac, outdir):
        # Create a filename based on the model configuration
        model_filename = f"LE_closed_model_optimizer_{optimizer.__class__.__name__}_dropout_{drop_frac}_fold_{fold}.h5"
        # Save the model
        self.model.save(outdir+model_filename)
        print(f"Model saved as {model_filename}")

#%%Model call 
fold_results = []
for fold in range(3):
    print(fold)
    # Load train and test data
    train_df = pd.read_csv(os.path.join(train_dir, f'train_fold_{fold + 1}.csv'))
    # print(train_df.columns.tolist())
    test_df = pd.read_csv(os.path.join(test_dir, f'test_fold_{fold + 1}.csv'))
    ## Scaling the data 
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
            "Slope_SRTM",
            "Landcover_wc",
            "Canopy_height",
            "sand_percent",
            "silt_percent",
            "clay_percent",
            "Ksat",
            "doy",
            "ALFA","Tao_sw","Rs_down","Rl_down","Rl_up",
            "LEinst","ET_24h","Rn24h_G",\
            "Rn","Ginst"]
    X_train=train_df[features]
    # print(X_train.isna().any())
    X_test=test_df[features]
    y_train=train_df[labels]
    y_test=test_df[labels]
    # Initialize the StandardScaler
    scaler = StandardScaler()
    # Fit the scaler on the training data and transform it
    X_train_scaled_all = scaler.fit_transform(X_train.drop(columns={"LEinst","ET_24h","Rn24h_G"}))
    X_test_scaled_all =scaler.transform(X_test.drop(columns={"LEinst","ET_24h","Rn24h_G"}))
    # scaler_filename = "D:\\Backup\\Rouhin_Lenovo\\US_project\\GEE_SEBAL_Project\\ML\\ML_model\\scaler_"+str(fold)+".save"
    # joblib.dump(scaler, scaler_filename) 
    LE_train_sebal=np.array(X_train)[:,-4]
    LE_test_sebal=np.array(X_test)[:,-4]                
    # print(H_test_sebal)
    input_shape = (None,)  # Input shape is flexible
    model = NN(X_train_scaled_all.shape[1])
    model.compile_model(optimizer="Nadam")
    # Prepare your training and testing data
    x_train = X_train_scaled_all
    x_test = X_test_scaled_all
    # Train the model
    model.train_model(x_train, y_train)
    accuracy = model.evaluate_model(x_test, y_test)
    r2=model.r2_calc(x_test, y_test)
    print(f"Test accuracy: {accuracy}")
    print(f"Test R2: {model.r2_calc(x_test, y_test)}")
    print("RMSE Percent ML", rmse_per(y_test,model.model.predict(x_test)))
    # print(f"Test Bias (%): {model.bias_calc(x_test, y_test)}")
    # model.save_model(fold,optimizer='Nadam', drop_frac=0.1, \
    # outdir='D:\\Backup\\Rouhin_Lenovo\\US_project\\GEE_SEBAL_Project\\ML\\ML_model\\')
    # model.save('D:\\Backup\\Rouhin_Lenovo\\US_project\\GEE_SEBAL_Project\\ML\\ML_model\\model_do0.2_lamda0.5.keras')
    print("GEESEBAL accuracy (RMSE)=",rmse(y_test,LE_test_sebal*28.36) )
    print("GEESEBAL R2 =",r2_score(y_test,LE_test_sebal*28.36) )
    print("RMSE Percent SEBAL", rmse_per(y_test,LE_test_sebal*28.36))
    # print("GEESEBAL MAPE =",MAPE(y_test,H_test_sebal) )
    ##  Save the predictions 
    # Step 1: Make Predictions
    et_predictions_all = model.model.predict(x_test)

    # Step 2: Add Predictions to X_test DataFrame
    test_df['ET_predictions_all'] = et_predictions_all

    # Optional: Save the DataFrame with predictions to a CSV file if needed
    # test_df.to_csv('D:\\Backup\\Rouhin_Lenovo\\US_project\\GEE_SEBAL_Project\\ML\\Test_train\\Test_w_predictions\\X_test_with_predictions_fold_'+str(fold+1)+'.csv', index=False)
    # Store the results
    fold_results.append({
        'fold': fold + 1,
        'accuracy': accuracy,
        'r2': r2,
        # 'bias': bias
    })

# %%
X_train[features].describe()
y_test[labels].describe()
fold_results
# %%
print(model.model.predict(x_test))
