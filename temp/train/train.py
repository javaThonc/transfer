import build
import time
import matplotlib.pyplot as plt
import numpy as np
import argparse
import distutils.util
import sys
from keras.layers.core import Dense, Activation, Dropout 
from keras.layers.recurrent import LSTM
from keras.layers import Dense
from keras.models import Sequential
from keras.layers import Input
import keras
np.set_printoptions(threshold=sys.maxsize)

#Main Run Thread
if __name__=='__main__':
    parser = argparse.ArgumentParser()
	# n-step prediction
    parser.add_argument('-s','--step', type=int, default=3)
	# data path
    parser.add_argument('-d','--data_file', type=str, default='../dataset/crsp.npy')
	# dimension
    parser.add_argument('-hd','--hidden_dim', type=int, default=50)
    parser.add_argument('-f','--freq_dim', type=int, default=10)
	# training parameter
    parser.add_argument('-n','--niter', type=int, default=4000)
    parser.add_argument('-ns', '--nsnapshot', type=int, default=20)
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.01)
    
    args = parser.parse_args()
    step = args.step
	
    global_start_time = time.time()

    print '> Loading data... '
    
    data_file = args.data_file
    X_train, y_train, X_val, y_val, X_test, y_test, gt_test, max_data, min_data = build.load_data(data_file, step)
    train_len = X_train.shape[1]
    val_len = X_val.shape[1]
    test_len = X_test.shape[1]

    print '> Data Loaded. Compiling...'
    #model = build.build_model([1, args.hidden_dim, 1], args.freq_dim, args.learning_rate)
    model = Sequential()
    model.add(LSTM(output_dim = args.hidden_dim, input_shape = (2012, 1)))
    model.add(Dense(output_dim = 1))
    rms = keras.optimizers.RMSprop(lr=args.learning_rate) 
    model.compile(loss="mse", optimizer=rms)
    best_error = np.inf
    best_epoch = 0
    print "hello"
    for ii in range(int(args.niter/args.nsnapshot)):
        model.fit(X_train, y_train, batch_size = 45, nb_epoch=args.nsnapshot, validation_split=0)
        
            
        
        num_iter = str(args.nsnapshot * (ii+1))
        model.save_weights('./snapshot/weights{}.hdf5'.format(num_iter), overwrite = True)
        
        predicted = model.predict(X_train)
        train_error = np.sum((predicted[:,:,0] - y_train[:,:,0])**2) / (predicted.shape[0] * predicted.shape[1])
        
        print num_iter, ' training error ', train_error

        predicted = model.predict(X_val)
        val_error = np.sum((predicted[:,:,0] - y_val[:,:,0])**2) / (val_len * predicted.shape[0])
        
        print ' val error ', val_error
        
        if(val_error < best_error):
            best_error = val_error
            best_iter = args.nsnapshot * (ii+1)
    
    print 'Training duration (s) : ', time.time() - global_start_time
    print 'best iteration ', best_iter
    print 'smallest error ', best_error
