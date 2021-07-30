import numpy as np
# Modeling & Preprocessing
from keras.layers import Conv2D, BatchNormalization, Activation, Flatten, Dense, Dropout, LSTM, Input, TimeDistributed, Bidirectional
from keras import initializers, Model, optimizers, callbacks
from keras.models import load_model
#from keras.utils.training_utils import multi_gpu_model
from glob import glob
from keras import optimizers
# Get Paths
from glob import glob
# EEG package
from mne import  pick_types
from mne.io import read_raw_edf
import mne
import os
import numpy as np
# Save the model
import pickle
import tensorflow as tf
from tensorflow import keras
import tensorboard
import h5py

for i in range(1,16):
    FNAMES = os.listdir("Session3")
    for file in FNAMES:
        if ".edf" not in file:
            FNAMES.remove(file)
    new_FNAMES = []
    for file in FNAMES:
        if ".edf" in file and "P{:02d}".format(i) in file:
            new_FNAMES.append(file)

    FNAMES = new_FNAMES
    del new_FNAMES
    import random
    print(FNAMES)


    def get_data(subj_num=FNAMES, epoch_sec=0.0625):
        """ Import from edf files data and targets in the shape of 3D tensor

            Output shape: (Trial*Channel*TimeFrames)

            Some edf+ files recorded at low sampling rate, 128Hz, are excluded. 
            Majority was sampled at 160Hz.

            epoch_sec: time interval for one segment of mashes
            """

        # To calculated completion rate
        count = 0

        # Initiate X, y
        X = []
        #y = []

        # fixed numbers
        nChan = 61
        sfreq = 250
        sliding = epoch_sec/2 

        # Sub-function to assign X and X, y
        def append_X(n_segments, old_x):
            '''This function generate a tensor for X and append it to the existing X'''
            new_x = old_x + [data[:, int(sfreq*sliding*n):int(sfreq*sliding*(n+2))] for n in range(n_segments)\
                         if data[:, int(sfreq*sliding*n):int(sfreq*sliding*(n+2))].shape==(nChan, int(sfreq*epoch_sec))]
            return new_x
        print(subj_num)
        for i, subj in enumerate(subj_num):
            # Return completion rate
            count+=1
            displayStep = max(len(subj_num)//10, 1)

            if i%displayStep == 0:
                print('working on {}, {:.1%} completed'.format(subj, count/len(subj_num)))

            # Get file names
            #fnames = os.listdir()
            #for file in fnames:
            #    if ".edf" not in file:
            #        fnames.remove(file)
            #if ".ipynb_checkpoints" in fnames:
            #    fnames.remove(".ipynb_checkpoints")
            #for i, fname in enumerate(fnames):

                # Import data into MNE raw object
            fname = subj
            print(fname)
            raw = read_raw_edf("Session3/" + subj, preload=True, verbose=False)
            try:
                ch_names = raw.ch_names
            except:
                continue
                #channels = raw.ch_names
                #for i in channels:
                #    if i != "Fc3." and i != "Fcz." and i != "Fc4." and i != "C3.." and i != "Cz.." and i != "C4.." and i != "Cp3." and i != "Cpz." and i != "Cp4.":
                #        raw.drop_channels(i)
                #raw.filter(7.5,12)

            picks = pick_types(raw.info, eeg=True)

            if raw.info['sfreq'] != 250:
                print('{} is sampled at 128Hz so will be excluded.'.format(subj))
                break

                # High-pass filtering
            raw.filter(l_freq=1, h_freq=None, picks=picks)

                # Get annotation
            #try:
                    #### for  mne .18
                    # events = raw.find_edf_events()
                    #### for  mne .19
            #    events = mne.find_events(raw, verbose = False)
            #except:
            #    continue
                # Get data
            data = raw.get_data(picks=picks)

            """ Assignment Starts """ 
            #if "Difficult" in fname:

                    # Number of sliding windows
            #    n_segments = int((raw.n_times/(epoch_sec*sfreq))*2-1)

                #y.extend([2]*n_segments)
            #    X = append_X(n_segments, X)

            #elif "Med" in fname:

                    # Number of sliding windows
             #   n_segments = int((raw.n_times/(epoch_sec*sfreq))*2-1)

                #y.extend([1]*n_segments)
              #  X = append_X(n_segments, X)

            #elif "Easy" in fname:

                    # Number of sliding windows
             #   n_segments = int((raw.n_times/(epoch_sec*sfreq))*2-1)

                #y.extend([0]*n_segments)
              #  X = append_X(n_segments, X)
        n_segments = int((raw.n_times/(epoch_sec*sfreq))*2-1)
        print("Raw.n_times is: ", str(raw.n_times))
        print("epoch_sec is: ", str(epoch_sec))
        print("sfreq is: ", str(sfreq))
        print(n_segments)
        X = append_X(n_segments, X)
        X = np.stack(X)
        #y = np.array(y).reshape((-1,1))
        return X, ch_names, raw.n_times

    import numpy as np
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    #%%
    def convert_mesh(X):

        mesh = np.zeros((X.shape[0], X.shape[2], 9, 11))
        X = np.swapaxes(X, 1, 2)

        # 1st line
        mesh[:, :, 0, 4:7] = X[:,:,21:24]; print('1st finished')

        # 2nd line
        mesh[:, :, 1, 3:8] = X[:,:,24:29]; print('2nd finished')

        # 3rd line
        mesh[:, :, 2, 1:10] = X[:,:,29:38]; print('3rd finished')

        # 4th line
        mesh[:, :, 3, 1:10] = np.concatenate((X[:,:,38].reshape(-1, X.shape[1], 1),\
                                              X[:,:,0:7], X[:,:,39].reshape(-1, X.shape[1], 1)), axis=2)
        print('4th finished')

        # 5th line
        mesh[:, :, 4, 0:11] = np.concatenate((X[:,:,(42, 40)],\
                                            X[:,:,7:14], X[:,:,(41, 43)]), axis=2)
        print('5th finished')

        # 6th line
        mesh[:, :, 5, 1:10] = np.concatenate((X[:,:,44].reshape(-1, X.shape[1], 1),\
                                            X[:,:,14:21], X[:,:,45].reshape(-1, X.shape[1], 1)), axis=2)
        print('6th finished')

        # 7th line
        mesh[:, :, 6, 1:10] = X[:,:,46:55]; print('7th finished')

        # 8th line
        mesh[:, :, 7, 3:8] = X[:,:,55:60]; print('8th finished')

        # 9th line
        #mesh[:, :, 8, 4:7] = X[:,:,60]; print('9th finished')

        # 10th line
        #mesh[:, :, 9, 5] = X[:,:,63]; print('10th finished')

        return mesh

    #%%
    def prepare_data(X, test_ratio=0.2, return_mesh=True, set_seed=42):

        # y encoding
        #oh = OneHotEncoder()
        #y = oh.fit_transform(y).toarray()

        # Shuffle trials
        #np.random.seed(set_seed)
        #trials = X.shape[0]
        #shuffle_indices = np.random.permutation(trials)
        #X = X[shuffle_indices]
        #y = y[shuffle_indices]

        # Test set seperation
        #train_size = int(trials*(1-test_ratio)) 
        #X_train, X_test = X[:train_size,:,:], X[train_size:,:,:]
        
        # Z-score Normalization
        def scale_data(X):
            shape = X.shape
            scaler = StandardScaler()
            scaled_X = np.zeros((shape[0], shape[1], shape[2]))
            displayStep = max(int(shape[0]/10), 1)
            for i in range(shape[0]):
                for z in range(shape[2]):
                    scaled_X[i, :, z] = np.squeeze(scaler.fit_transform(X[i, :, z].reshape(-1, 1)))
                if i%displayStep == 0:
                    print('{:.1%} done'.format((i+1)/shape[0]))   
            return scaled_X

        X_train = scale_data(X)
        if return_mesh:
            X_train = convert_mesh(X_train)
        return X_train

    epoch_sec = 0.25
    nomeschema = "full"

    ##   GET PREPROCESSED DATA FROM FILE  (OR PREPROCES IT ON THE SPOT)
    preprocessed_data_file = './preprocessed_dataset/dataIMM_epochsec-'+str(epoch_sec)+'_schema_'+nomeschema+'_P{:02d}Session3'.format(i)+'.h5'
    if os.path.exists(preprocessed_data_file):
        print('true')
        hf = h5py.File(preprocessed_data_file, 'r')
        hf.keys()
        X_train =  np.array(hf.get('X_train'))
        hf.close()

    else:
        print('false')
        X,ch_names, n = get_data(FNAMES, epoch_sec=epoch_sec)
        X_train = prepare_data(X)
        del X
        #del y
        hf = h5py.File(preprocessed_data_file, 'w')
        hf.create_dataset('X_train', data=X_train)
        hf.keys()
        hf.close()

    # Make another dimension, 1, to apply CNN for each time frame.
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], X_train.shape[3], 1)

    #modelname = "model_all_electrodes_double_fully_connected_bidirectional_P{:02d}".format(i)
    modelname = "model_all_electrodes_double_fully_connected_bidirectional_P{:02d}".format(i)
    # Get past models
    MODEL_LIST = glob('model/'+modelname+'*')
    print(MODEL_LIST)
    if MODEL_LIST:
        print('A model that already exists detected and loaded.')
        model = load_model(MODEL_LIST[-1])

    y_pred = model.predict(X_train)
    sec = 0.25
    varlist = []
    for m in y_pred:
        varlist.append([m, sec])
        sec += 0.25
    header = ["Class", "Time"]
    import csv
    with open("P{:02d}S3Classes.csv".format(i), 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)

        # write the header
        writer.writerow(header)

        # write multiple rows
        writer.writerows(varlist)
    