# (c) 2020 The Regents of the University of Michigan, 
# Michigan Center for Integrative Research in Critical Care
# https://mcircc.umich.edu/ 

import keras
import keras.backend as K
import tensorflow as tf
from keras.layers import Lambda
from keras.backend import slice
import numpy as np
import pandas as pd
from keras.models import load_model
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer, MissingIndicator
import pickle

tf.random.set_seed(0)
np.random.seed(0)

mask_value = 0.0
def masked_loss_function_weighted(y_true, y_pred):
    #n_sample = K.int_shape(y_pred)[0]
    mask = K.cast(K.not_equal(y_true, mask_value), K.floatx())
    sum_not_imputed = K.sum(mask,axis=0)
    col_sum = K.sum(K.square( mask * (y_true - y_pred) ),axis=0)
    col_mean = col_sum / sum_not_imputed
    return K.mean( col_mean )

def masked_loss_function(y_true, y_pred):
    #n_sample = K.int_shape(y_pred)[0]
    mask = K.cast(K.not_equal(y_true, mask_value), K.floatx())
    return K.mean(K.square(y_true * mask - y_pred * mask))



class Sampling(keras.layers.Layer):
    def call(self, inputs):
        mean, log_var = inputs
        return K.random_normal(tf.shape(log_var)) * K.exp(log_var / 2) + mean

    
class DenseTranspose(keras.layers.Layer):
    def __init__(self, dense, activation=None, **kwargs):
        self.dense = dense
        self.activation = keras.activations.get(activation)
        super().__init__(**kwargs)
    def build(self, batch_input_shape):
        self.biases = self.add_weight(name="bias", initializer="zeros",
                                      shape=[self.dense.input_shape[-1]])
        super().build(batch_input_shape)
    def call(self, inputs):
        z = tf.matmul(inputs, self.dense.weights[0], transpose_b=True)
        return self.activation(z + self.biases)

    
def merge_X_y(X,y):
    X_train_with_y = X
    X_train_with_y['Target'] = y.values
    return X_train_with_y
       
def sigmoid(x):
    return 1.0/(1 + np.exp(-x))

class VariationalAutoencoder(object):
    # input_features includes the target
    def __init__(self,codings_size=16,input_features=46,patience=7,optimizer="adam",dropout_p=0.15,
                     X_colnames=None,Y_colname=None,
                 d1_size = 256,
                 d2_size = 128,
                 d3_size = 64,
                 d4_size = 32,
                 use_mse_loss=False):
        
        self.d1_size = d1_size
        self.d2_size = d2_size
        self.d3_size = d3_size
        self.d4_size = d4_size
        self.optimizer = optimizer
        self.dropout_p = dropout_p
        self.codings_size=codings_size
        self.input_features=input_features
        self.patience=patience
        self.variational_encoder = None
        self.variational_decoder = None
        self.variational_ae = None
        self.codings_mean = None
        self.codings_log_var = None
        self.latent_loss = None
        self.history = None
        self.X_colnames = X_colnames
        self.Y_colname = Y_colname
        self.use_mse_loss = use_mse_loss
        self.create_model()
    
    def fit(self,X_train,Y_train,X_val,Y_val,epochs=100,batch_size=128):
        X_train_with_y = X_train.copy()
        X_train_with_y['Target'] = Y_train.values
        X_val_with_y = X_val.copy()
        X_val_with_y['Target'] = Y_val.values
        callbacks = [keras.callbacks.callbacks.EarlyStopping(patience=self.patience)]
        history = self.variational_ae.fit(X_train_with_y.values, [X_train.values, Y_train.values], epochs=epochs, batch_size=batch_size, validation_data=(X_val_with_y.values, [X_val.values, Y_val.values]), callbacks=callbacks)
        self.history = history
        return history
    
    def predict_encoder(self,X,y):
        X_with_y = merge_X_y(X.copy(),y.copy())
        means, log_vars, codings = self.variational_encoder.predict(X_with_y.values)
        return means, log_vars, codings
    
    def predict_decoder(self,codings,split_y=False):
        X_with_y_pred = self.variational_decoder.predict(codings)
        X_with_y_pred[:,self.input_features - 1] = sigmoid(X_with_y_pred[:,self.input_features - 1])
        if not split_y:
            return X_with_y_pred
        else:
            X = X_with_y_pred[:,0:(self.input_features - 1)]
            y = X_with_y_pred[:,self.input_features - 1]
            return X, y
    
    def predict(self,X,y, split_y=False):
        means, log_vars, codings = self.predict_encoder(X,y)
        return self.predict_decoder(codings, split_y)
        
    def simulate(self,size=1, as_df=False):
        codings = np.random.normal(size=[size, self.codings_size])
        X, y = self.predict_decoder(codings,split_y=True)
        y = np.random.binomial(n=1,p=y,size=y.shape)
        
        if not as_df:
            return X, y
        else:
            return pd.DataFrame(X,columns=self.X_colnames),pd.Series(y,name=self.Y_colname)
    
    def save(self,path):
        self.variational_ae.save_weights(path)
        #Save class info
        save_dict_pickle(path + '.info.pickle',
        d1_size=self.d1_size,
        d2_size=self.d2_size,
        d3_size=self.d3_size,
        d4_size=self.d4_size,
        dropout_p=self.dropout_p,
        codings_size=self.codings_size,
        input_features=self.input_features,
        patience=self.patience)
    
    def load(self,path):
        #load would not work so swtiched to load_weights
        #load_model(path)#(path,custom_objects={'Sampling': Sampling})
        #self.variational_decoder = self.variational_ae.get_layer('decoder')
        #self.variational_encoder = self.variational_ae.get_layer('encoder')
        #self.input_features = self.variational_encoder.get_layer('encoder_input').output.shape[1]
        #self.codings_size = self.variational_encoder.get_layer('sampling_layer').output.shape[1]
        loaded_dict = load_dict_pickle(path + '.info.pickle')
        self.d1_size = loaded_dict['d1_size']
        self.d2_size = loaded_dict['d2_size']
        self.d3_size = loaded_dict['d3_size']
        self.d4_size = loaded_dict['d4_size']
        self.dropout_p = loaded_dict['dropout_p']
        self.codings_size = loaded_dict['codings_size']
        self.input_features = loaded_dict['input_features']
        self.patience = loaded_dict['patience']
        
        self.create_model()
        self.variational_ae.load_weights(path) 
        
        return self
    
    def create_model(self):
        codings_size = self.codings_size
        input_features = self.input_features
        patience = self.patience
        dropout_p = self.dropout_p
        optimizer = self.optimizer
        input_features_float = np.float32(input_features)
        d1_size = self.d1_size
        d2_size = self.d2_size
        d3_size = self.d3_size
        d4_size = self.d4_size
        inputs = keras.layers.Input(shape=(input_features,),name="encoder_input")
        dropout = keras.layers.Dropout(dropout_p)
        d1 = keras.layers.Dense(d1_size, activation="selu")
        d2 = keras.layers.Dense(d2_size, activation="selu")
        d3 = keras.layers.Dense(d3_size, activation="selu")
        d4 = keras.layers.Dense(d4_size, activation="selu")
        z = dropout(inputs)
        z = d1(z)
        z = d2(z)
        z = d3(z)
        z = d4(z)
        codings_mean = keras.layers.Dense(codings_size,name="codings_mean")(z)
        codings_log_var = keras.layers.Dense(codings_size,name="codings_log_var")(z)
        sampling_layer = Sampling(name="sampling_layer")
        codings = sampling_layer([codings_mean, codings_log_var])
        variational_encoder = keras.models.Model(
            inputs=[inputs], outputs=[codings_mean, codings_log_var, codings], name="encoder")
        
        #variational_encoder._name = "encoder"
        
        decoder_inputs = keras.layers.Input(shape=[codings_size],name="decoder_input")
        dc = keras.layers.Dense(d4_size, activation="selu")
        d4_t = DenseTranspose(d4, activation="selu")
        d3_t = DenseTranspose(d3, activation="selu")
        d2_t = DenseTranspose(d2, activation="selu")
        d1_t = DenseTranspose(d1, activation="linear", name="decoder_outputs")
        x = dc(decoder_inputs)
        x = d4_t(x)
        x = d3_t(x)
        x = d2_t(x)
        outputs = d1_t(x)
        
        #outputs = keras.layers.Dense(input_features, activation="linear", name="decoder_outputs")(x)
        variational_decoder = keras.models.Model(inputs=[decoder_inputs], outputs=[outputs], name="decoder")
        #variational_decoder._name = "decoder"
        
        _, _, codings = variational_encoder(inputs)
        reconstructions = variational_decoder(codings)
        
        #LAST COLUMN IS THE TARGET
        input_features_m_1 = self.input_features - 1
        def apply_slice_numeric(x):
            return slice(x, (0,0), (-1,input_features_m_1) )
        
        def apply_slice_target(x):
            return slice(x, (0,input_features_m_1), (-1,1))
        
        
        l1 = Lambda( apply_slice_numeric , name='numeric')
        l2 = Lambda( apply_slice_target, name='lambda_categorical')
        d2a_layer = keras.layers.Activation(keras.activations.sigmoid, name='categorical')
        
        self.l1 = l1
        self.l2 = l2
        self.d2a_layer = d2a_layer
        
        
        d1 = l1(reconstructions)
        d2 = l2(reconstructions)
        d2a = d2a_layer(d2)
    
        variational_ae = keras.models.Model(inputs=[inputs], outputs=[d1,d2a])

        latent_loss = -0.5 * K.sum(
            1 + codings_log_var - K.exp(codings_log_var) - K.square(codings_mean),
            axis=-1)
        variational_ae.add_loss(K.mean(latent_loss) / np.float32(input_features))
        if self.use_mse_loss:
            variational_ae.compile(loss=[keras.losses.mse,keras.losses.binary_crossentropy], optimizer=optimizer, loss_weights=[(input_features_float-1.0)/input_features_float, 10/input_features_float])
        else:
            variational_ae.compile(loss=[masked_loss_function_weighted,keras.losses.binary_crossentropy], optimizer=optimizer, loss_weights=[(input_features_float-1.0)/input_features_float, 10/input_features_float])
        #variational_ae.compile(loss=[masked_loss_function,keras.losses.binary_crossentropy], optimizer=optimizer, loss_weights=[(input_features_float-1.0)/input_features_float, 10/input_features_float])
        #variational_ae.compile(loss=[keras.losses.mse,keras.losses.binary_crossentropy], optimizer=optimizer, loss_weights=[(input_features_float-1.0)/input_features_float, 10/input_features_float])
        
        self.variational_ae = variational_ae
        self.variational_encoder = variational_encoder
        self.variational_decoder = variational_decoder
        self.latent_loss = latent_loss
        self.codings_mean = codings_mean
        self.codings_log_var = codings_log_var


class Winsorizer(object):
    def __init__(self,l=0.005,u=0.995):
        self.col_info = {}
        self.l = l
        self.u = u
    
    def fit(self,X):
        col_info = {}
        for col in list(X.columns):
            x = X[col]
            q_l, q_u = np.nanquantile(x,(self.l,self.u))
            col_info[col] = (q_l, q_u)
        
        self.col_info = col_info
        return self
    
    def transform(self,X):
        X = X.copy()
        for col in list(X.columns):
            q_l, q_u = self.col_info[col]
            x = X[col]
            X.loc[x < q_l,col] = q_l
            X.loc[x > q_u,col] = q_u
        
        return X
        
    
    def fit_transform(self,X):
        self.fit(X)
        return self.transform(X)

    


    
"""
An object of this class will do the following
1 apply a winsorizor (if params are set )
2 return indicators for imputation
3 applies a scaler
4 does a simple imputation
""" 
class SimulationDataScaler(object):
    def __init__(self,l=0.0,u=1.0,fill_value=0.0,winsorize=False, return_df = True):
        self.l = l
        self.u = u
        self.columns = []
        self.is_fit = False
        self.return_df = return_df
        
        if not winsorize:
            self.l = 0.0
            self.u = 1.0
        
        self.winsorize = winsorize
        self.winsorizor = Winsorizer(l=l,u=u)
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='constant',fill_value=fill_value,add_indicator=False)
        self.indicator_imputer = MissingIndicator(features="all")
    
    def fit(self,X):
        #X = X.copy()
        X[~np.isfinite(X)] = np.nan
        self.is_fit = True
        if type(X) is pd.DataFrame:
            self.columns = list(X.columns)
        
        X_w = self.winsorizor.fit_transform(X)
        #print( (X[~np.isfinite(X)]).sum())
        self.indicator_imputer.fit(X)
        self.scaler.fit(X_w)
        self.imputer.fit(X_w)
        
        return self
    
    def transform(self,X):
        if not self.is_fit:
            raise "Please fit the before running"
        X[~np.isfinite(X)] = np.nan
        X_w = self.winsorizor.transform(X)
        X_imp_ind = self.indicator_imputer.transform(X)
        X_w_s = self.scaler.transform(X_w)
        X_w_s_i = self.imputer.transform(X_w_s)
        
        if self.return_df:
            return pd.DataFrame(X_w_s_i,columns=self.columns), pd.DataFrame(X_imp_ind,columns=self.columns)
        else:
            return X_w_s_i, X_imp_ind
    
    def fit_transform(self,X):            
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self,X_w_s_i):
        if not self.is_fit:
            raise "Please fit the before running"
        X_w_i = self.scaler.inverse_transform(X_w_s_i)
        if self.return_df:
            return pd.DataFrame(X_w_i,columns=self.columns)
        else:
            return X_w_i

def mark_data_missing(X,p):
    """
    X is a dataframe
    p is a vector of probabilities
    """
    assert type(X) is pd.DataFrame
    assert p.shape[0] == X.shape[1]
    mask = np.bool8(np.random.binomial(n=1,p=p,size=X.shape))
    return X.mask(mask)

def mark_data_missing_by_target(X,y,p0,p1):
    """
    X is a dataframe
    y a vector of 0s and 1s
    p0 is a vector of probabilities when y = 0
    p1 is a vector of probabilities when y = 1
    """
    assert type(X) is pd.DataFrame
    assert type(y) is pd.Series
    assert X.shape[0] == y.shape[0]
    
    X0 = mark_data_missing( X[y.values == 0], p0 )
    X1 = mark_data_missing( X[y.values == 1], p1 )
    y0 = y[y.values == 0]
    y1 = y[y.values == 1]
    return pd.concat( [X0, X1], axis=0), pd.concat([y0, y1 ])

        
def save_dict_pickle(path,**kwargs):
    data = kwargs
    assert type(data) is dict
    pickle.dump( data, open( path, "wb" ) )

def load_dict_pickle(path):
    data = pickle.load( open( path, "rb" ) )
    return data


