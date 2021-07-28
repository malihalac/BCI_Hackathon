import numpy as np
# Modeling & Preprocessing
from keras.layers import Conv2D, BatchNormalization, Activation, Flatten, Dense, Dropout, LSTM, Input, TimeDistributed, Bidirectional
from keras import initializers, Model, optimizers, callbacks
from keras.models import load_model
#from keras.utils.training_utils import multi_gpu_model
from glob import glob
from keras import optimizers
import h5py

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

FNAMES = os.listdir()

for file in FNAMES:
    if "edf" not in file:
        FNAMES.remove(file)

if "model" in FNAMES:
    FNAMES.remove("model")
new_FNAMES = []
for file in FNAMES:
    if ".edf" in file:
        new_FNAMES.append(file)

FNAMES = new_FNAMES
        
nomeschema = 'single'

#Choose the electrode to analyze between 0-60
#NumeroElettrodoDaAnalizzare = 6
val_list = []

#Just some variables for the model
nnpool = [ 8 ,8 ]
llpool = [ 16 , 16]
ccpool = [ 32 ]
ffpool = [ 32 ]

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
    y = []
    
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
            raw = read_raw_edf(subj, preload=True, verbose=False)
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
            try:
                #### for  mne .18
                # events = raw.find_edf_events()
                #### for  mne .19
                events = mne.find_events(raw, verbose = False)
            except:
                continue
            # Get data
            data = raw.get_data(picks=picks)
            
            """ Assignment Starts """ 
            if "Difficult" in fname:

                # Number of sliding windows
                n_segments = int((raw.n_times/(epoch_sec*sfreq))*2-1)
                
                y.extend([2]*n_segments)
                X = append_X(n_segments, X)
                      
            elif "Med" in fname:
                
                # Number of sliding windows
                n_segments = int((raw.n_times/(epoch_sec*sfreq))*2-1)
                
                y.extend([1]*n_segments)
                X = append_X(n_segments, X)
                        
            elif "Easy" in fname:
                   
                # Number of sliding windows
                n_segments = int((raw.n_times/(epoch_sec*sfreq))*2-1)
                
                y.extend([0]*n_segments)
                X = append_X(n_segments, X)
    
    X = np.stack(X)
    y = np.array(y).reshape((-1,1))
    return X, y, ch_names

import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def prepare_data_single(X, y, test_ratio=0.2, set_seed=42):
    # y encoding
    oh = OneHotEncoder()
    y = oh.fit_transform(y).toarray()
    
    # Shuffle trials
    np.random.seed(set_seed)
    trials = X.shape[0]
    shuffle_indices = np.random.permutation(trials)
    X = X[shuffle_indices]
    y = y[shuffle_indices]
    
    # Test set seperation
    train_size = int(trials*(1-test_ratio)) 
    X_train, X_test, y_train, y_test = X[:train_size,:,:], X[train_size:,:,:],\
                                    y[:train_size,:], y[train_size:,:]
                                    
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
            
    X_train, X_test  = scale_data(X_train), scale_data(X_test)
    X_train, X_test = np.transpose(X_train,(1,2,0)) , np.transpose(X_test,(1,2,0)) 
    
    return X_train, y_train, X_test, y_test

##   GET PREPROCESSED DATA FROM FILE  (OR PREPROCES IT ON THE SPOT)
for NumeroElettrodoDaAnalizzare in range(61):
    for e_s in [0.0625]: ## ADD HERE DIFFERENT TIME WINDOWS (0.125, 0.25 etc.). INSERT MANUALLY DIFFERENT VALUES EACH RUN I WAS HAVING PROBLEMS ADDING VALUES ALL TOGETHER IN THE LOOP
        preprocessed_data_file = '/Users/malihalac/desktop/BCI_Hackathon/github/BCI_Hackathon/edf_data/' + 'preprocessed_dataset/dataIMM_epochsec-'+str(e_s)+'_schema_'+nomeschema+'_nsub_'+str(len(FNAMES))+'.h5'
        print(preprocessed_data_file)
        if os.path.exists(preprocessed_data_file):
            try:
                NumeroElettrodoDaAnalizzare
            except NameError:
                print("NumeroElettrodoDaAnalizzare not defined")
            else:
                print('true')
                hf = h5py.File(preprocessed_data_file, 'r')
                hf.keys()
                asciiList = hf.get('ch_names')
                ch_names = [n.decode('utf-8') for n in asciiList]
                print(ch_names)
                y_train =  np.array(hf.get('y_train'))
                y_test =  np.array(hf.get('y_test'))
                name = ch_names[NumeroElettrodoDaAnalizzare]
                print(name)
                X_train =  np.array(hf.get(name+'X_train'))
                X_test =  np.array(hf.get(name+'X_test'))
                hf.close()
        else:
            print('preprocessing...')
            print(FNAMES)
            X,y,ch_names = get_data(FNAMES, epoch_sec=e_s)
            X_train, y_train, X_test, y_test = prepare_data_single(X, y)
            del X
            del y
            hf = h5py.File(preprocessed_data_file, 'w')
            asciiList = [n.encode("ascii", "ignore") for n in ch_names]
            hf.create_dataset('ch_names', data=asciiList)
            for iii, name in enumerate(ch_names):
                try:
                    print(f' {iii:02d}',name)
                    hf.create_dataset(name+'X_train', data=X_train[iii])
                    hf.create_dataset(name+'X_test' , data=X_test[iii] )
                except:
                    continue
            hf.create_dataset('y_train', data=y_train)
            hf.create_dataset('y_test' , data=y_test)
            hf.keys()
            hf.close()

    X_train = np.transpose(X_train, [1,0])
    X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    X_test = np.transpose(X_test, (1,0))
    X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

    ## Complicated Model - the same as Zhang`s
    input_shape = X_train.shape[1:5]
    lecun = initializers.lecun_normal(seed=42)

    # Input layer
    inputs = Input(shape=input_shape)
    x = inputs
    # TimeDistributed Wrapper
    def timeDist(layer, prev_layer, name):
        return TimeDistributed(layer, name=name)(prev_layer)  

    y = x


    # Convolutional layers block
    if ffpool[0]>0:
        for idx, val in enumerate(ffpool):
            y = Dense(val, kernel_initializer=lecun, name='full'+str(idx+1))(y)
            y = Dropout(0.5, name='dropout0_'+str(idx+1))(y)
            y = BatchNormalization(name='batch0_'+str(idx+1))(y)
            y = Activation(activation='elu')(y)

    z = y

    # Recurrent layers block
    for idx, val in enumerate(llpool[0:-1]):
        z = LSTM(val, kernel_initializer=lecun, return_sequences=True, name='LSTM'+str(idx+1))(z)
    z = LSTM(llpool[-1], kernel_initializer=lecun, name='LSTM'+str(len(llpool)))(z)
    h = z

    # Fully connected layer block
    for idx, val in enumerate(nnpool):
        h = Dense(val, kernel_initializer=lecun, activation='elu', name='FC'+str(idx+1))(h)
        h = Dropout(0.2, name='dropout'+str(idx+1))(h)

    # Output layer
    outputs = Dense(3, activation='softmax')(h)

    # Model compile
    model = Model(inputs=inputs, outputs=outputs)
    model.summary()

    modelname = "model_"+ ch_names[NumeroElettrodoDaAnalizzare] + "_fully_connected"
    # Get past models
    MODEL_LIST = glob('model/'+modelname+'*')
    print(MODEL_LIST)
    if MODEL_LIST:
        print('A model that already exists detected and loaded.')
        model = load_model(MODEL_LIST[-1])

    callbacks_list = [callbacks.ModelCheckpoint('./model/'+modelname+'.h5', 
                                                save_best_only=True, 
                                                monitor='val_loss'),
                     callbacks.EarlyStopping(monitor='acc', patience=10),
                     callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=50),
                     callbacks.TensorBoard(log_dir='./tensorboard_dir/'+modelname, 
                                           histogram_freq=0, 
                                           write_graph=True,
                                           write_images=True)]

    model.compile(loss='categorical_crossentropy', optimizer=optimizers.SGD(lr=0.001), metrics=['acc'])
    hist = model.fit(X_train, y_train, batch_size=64, epochs=1000, 
                    callbacks=callbacks_list, validation_data=(X_test, y_test))
                    
    val_list.append([ch_names[NumeroElettrodoDaAnalizzare], max(hist.history["acc"]), max(hist.history["val_acc"])])

header = ["electrode", "Train Accuracy", "Validation Accuracy"]
import csv
with open('accuracies_electrodes.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)

    # write the header
    writer.writerow(header)

    # write multiple rows
    writer.writerows(val_list)
