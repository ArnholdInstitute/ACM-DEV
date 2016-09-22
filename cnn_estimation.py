"""

Using the model weights to estimate population in Kenya

"""
import os
import numpy as np
import pickle
from scipy import stats
import h5py
import parmap
from keras.models import Sequential, model_from_json
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.utils.io_utils import HDF5Matrix
from keras import callbacks
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam, SGD
from keras.preprocessing.image import ImageDataGenerator
import theano
print ""
print "Estimating Kenya's population using the mini Convnet"
print ""
processes = 2

print "Loading things ..."

model_json = '../data/model/med_architecture.json'
model_weights = '../data/model/med_weights.h5'


params = pickle.load(open('../data/model/preprocessing.p', 'r'))
X_mean = params['x_mean']
X_std = params['x_std']
y_mean = params['y_mean']
y_std = params['y_std']

def unnorm_y(y, y_std, y_mean): 
    "Unnormalises y"
    tmp = np.exp(y) - 1e-6
    tmp *= y_std
    tmp += y_mean
    return tmp

#loading data and doing some final cleaning
f = h5py.File('../data/estimates/db_post_KE_250.hdf5', 'r')
X_test = np.array(f['features'])
f.close()
X_test -= X_mean
X_test /= X_std
X_test = X_test.astype('float32')
print "... things loaded (#^.^#)"
print ""

# the model in short
print 'Constructing the model...'
model = Sequential()
model = model_from_json(open(model_json).read())
model.load_weights(model_weights)
model.compile(optimizer='adagrad', loss='mse')
print '!Model constructed!'
print ''
print 'Predicting population for Kenya'
preds = model.predict(X_test)
print 'Kenyan population predicted \o/'
print ''
print 'Unnormalising estimates ...'
y_est_norm = np.array(preds).flatten()
y_est = parmap.map(unnorm_y, y_est_norm, y_std, y_mean, processes=processes)
print 'Saving data'
pickle.dump(y_est, open('kenyan_estimates.p', 'r'))
print 'Niiiiiceeee'




