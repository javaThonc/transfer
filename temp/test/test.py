import build
import time
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import argparse
import distutils.util
import os
os.environ['KERAS_BACKEND'] = 'theano'
import sys
np.set_printoptions(threshold=sys.maxsize)
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.layers import Dense, TimeDistributed
def plot_results(predicted_data, true_data, i):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.show()
    plt.savefig('figure' + str(i) + '.png')

#Main Run Thread
if __name__=='__main__':
    parser = argparse.ArgumentParser()
	# n-step prediction
    parser.add_argument('-s','--step', type=int, default=1)
	# data path
    parser.add_argument('-d','--data_file', type=str, default='../dataset/crsp.npy')
	# visualization
    parser.add_argument('-v','--visualization', type=distutils.util.strtobool, default='false')
    args = parser.parse_args()
    step = args.step
	
    global_start_time = time.time()

    print '> Loading data... '
    
    data_file = args.data_file
    X_train, y_train, X_val, y_val, X_test, y_test, gt_test, max_data, min_data = build.load_data(data_file, step)
    test_len = X_test.shape[1]

    print '> Data Loaded. Compiling...'
    #dimension of hidden states
    if step == 1:
	    hidden_dim = 10
    elif step == 3:
	    hidden_dim = 50
    elif step == 5:
        hidden_dim = 50
    else:
        raise Exception("Don't have the model pretrained with the n-step prediction.")
    #number of frequencies
    freq = 10
    model = Sequential()
    model.add(LSTM(output_dim = 10, input_shape = (None,1) , return_sequences=True))
    model.add(TimeDistributed(Dense(1)))
    rms = keras.optimizers.RMSprop(lr=0.01)
    model.compile(loss="mse", optimizer=rms)
#loading model
    if step == 1:
	    model_path = './snapshot/weights400.hdf5'
    elif step == 3:
	    model_path = './snapshot/3d_50_10_17.00_0.00233.hdf5'
    elif step == 5:
        model_path = './snapshot/5d_50_10_28.90_0.00384.hdf5'
    else:
        raise Exception("Don't have the model pretrained with the n-step prediction.")
	
    model.load_weights(model_path)
    #predition
    print '> Predicting... '
    prediction = model.predict(X_test)
    #denormalization   
    # max_data = np.max(0_test, axis=1)
    # min_data = np.min(X_test, axis=1)
    prediction = (np.squeeze(prediction) * (max_data - min_data) + (max_data + min_data))/2
    print prediction.shape
    error = np.sum((np.squeeze(prediction) - gt_test)**2) / (len(prediction)* prediction.shape[0])
    print 'The mean square error is: %f' % error
    
    if args.visualization:
        for ii in range(0,len(prediction)):
            plot_results(prediction[ii,:], gt_test[ii,:], ii)
