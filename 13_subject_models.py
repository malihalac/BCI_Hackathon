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

for epoch_sec in [0.25, 0.0625]:
  for i in range(1,16):
      FNAMES = os.listdir()
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
      random.shuffle(FNAMES)


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

      #%%
      def convert_mesh_44(X,ch_names,mioschema):

        index = []
        for elettrodo in mioschema:
            index = index + [ ch_names.index(elettrodo) ]

        print(X.shape[0], X.shape[2])
        mesh = np.zeros((X.shape[0], X.shape[2], 4, 13))
        X = np.swapaxes(X, 1, 2)

        # 1st line

        mesh[:, :, 0, 0:4] = X[:,:,index[0:4]]; print('1st finished')

        # 2nd line
        mesh[:, :, 1, 4:8] = X[:,:,index[4:8]]; print('2nd finished')

        # 3rd line
        mesh[:, :, 2, 8:12] = X[:,:,index[8:12]]; print('3rd finished')

        # 4th line
        mesh[:, :, 2, 12] = X[:,:,index[12]]; print('4th finished')

        return mesh

      #%%
      def prepare_data(X, y, ch_names = [], mioschema = [], test_ratio=0.2, return_mesh=True, set_seed=42):
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

          if return_mesh:
              X_train, X_test = convert_mesh_44(X_train, ch_names, mioschema), convert_mesh_44(X_test, ch_names, mioschema)

          return X_train, y_train, X_test, y_test

          nomeschema = "13_ele"

      ##   GET PREPROCESSED DATA FROM FILE  (OR PREPROCES IT ON THE SPOT)
      preprocessed_data_file = './preprocessed_dataset/dataIMM_epochsec-'+str(epoch_sec)+'_schema_'+nomeschema+'_P{:02d}'.format(i)+'.h5'
      if os.path.exists(preprocessed_data_file):
          print('true')
          hf = h5py.File(preprocessed_data_file, 'r')
          hf.keys()
          X_train =  np.array(hf.get('X_train'))
          y_train =  np.array(hf.get('y_train'))
          X_test =  np.array(hf.get('X_test'))
          y_test =  np.array(hf.get('y_test'))
          hf.close()

      else:
          print('false')
          X,y,ch_names = get_data(FNAMES, epoch_sec=epoch_sec)
          X_train, y_train, X_test, y_test = prepare_data(X, y)
          del X
          del y
          hf = h5py.File(preprocessed_data_file, 'w')
          hf.create_dataset('X_train', data=X_train)
          hf.create_dataset('y_train', data=y_train)
          hf.create_dataset('X_test', data=X_test)
          hf.create_dataset('y_test', data=y_test)
          hf.keys()
          hf.close()

      # Make another dimension, 1, to apply CNN for each time frame.
      X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], X_train.shape[3], 1)
      X_test = X_test.reshape(X_test.shape[0], X_train.shape[1], X_train.shape[2], X_train.shape[3], 1)

      ## Complicated Model - the same as Zhang`s
      input_shape = X_train.shape[1:5]
      lecun = initializers.lecun_normal(seed=42)

      # Input layer
      inputs = Input(shape=input_shape)
      x = inputs
      # TimeDistributed Wrapper
      def timeDist(layer, prev_layer, name):
          return TimeDistributed(layer, name=name)(prev_layer)  

      x = timeDist(Conv2D(32, (3,3), padding='same', 
                          data_format='channels_last', kernel_initializer=lecun), inputs, name='CNN1')
      x = BatchNormalization(name='batch1')(x)
      x = Activation('elu', name='act1')(x)
      x = timeDist(Conv2D(64, (3,3), padding='same', data_format='channels_last', kernel_initializer=lecun), x, name='CNN2')
      x = BatchNormalization(name='batch2')(x)
      x = Activation('elu', name='act2')(x)
      x = timeDist(Conv2D(128, (3,3), padding='same', data_format='channels_last', kernel_initializer=lecun), x, name='CNN3')
      x = BatchNormalization(name='batch3')(x)
      x = Activation('elu', name='act3')(x)
      x = timeDist(Flatten(), x, name='flatten')

      # Fully connected layer block
      y = Dense(512, kernel_initializer=lecun, name='FC')(x)
      y = Dropout(0.5, name='dropout1')(y)
      y = BatchNormalization(name='batch4')(y)
      y = Activation(activation='elu')(y)

      # Recurrent layers block
      z = Bidirectional(LSTM(64, kernel_initializer=lecun, return_sequences=True, name='LSTM1'))(y)
      z = Bidirectional(LSTM(64, kernel_initializer=lecun, name='LSTM2'))(z)

      # Fully connected layer block
      h = Dense(512, kernel_initializer=lecun, activation='elu', name='FC2')(z)
      h = Dropout(0.5, name='dropout2')(h)

      # Output layer
      outputs = Dense(3, activation='softmax')(h)

      # Model compile
      model = Model(inputs=inputs, outputs=outputs)
      model.summary()

      modelname = "model_13_elec_"+str(epoch_sec)+"_fully_connected_bidirectional_P{:02d}".format(i)
      # Get past models
      MODEL_LIST = glob('model/'+modelname+'*')
      print(MODEL_LIST)
      if MODEL_LIST:
          print('A model that already exists detected and loaded.')
          model = load_model(MODEL_LIST[-1])

      callbacks_list = [callbacks.ModelCheckpoint('./model/'+modelname+'.h5', 
                                                  save_best_only=True, 
                                                  monitor='val_loss'),
                       callbacks.EarlyStopping(monitor='acc', patience=50),
                       callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=100),
                       callbacks.TensorBoard(log_dir='./tensorboard_dir/'+modelname, 
                                             histogram_freq=0, 
                                             write_graph=True,
                                             write_images=True)]

      model.compile(loss='categorical_crossentropy', optimizer=optimizers.SGD(lr=0.001), metrics=['acc'])
      hist = model.fit(X_train, y_train, batch_size=64, epochs=1000, 
                      callbacks=callbacks_list, validation_data=(X_test, y_test))
