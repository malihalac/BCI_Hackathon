import os
import sys
import csv
import pywt
import pyedflib
import numpy as np
from spectrum import *
from os import listdir
from nitime import utils
import scipy.stats as sp
from os.path import isfile, join
from nitime.viz import plot_tseries
from matplotlib import pyplot as plt
from nitime import algorithms as alg
from nitime.timeseries import TimeSeries
from mne.io import read_raw_edf
from mne import  pick_types
import mne
import pandas as pd
import scipy
import warnings
import collections

# ignore annoying deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 


################################   Note!   ##########################
"""
Extract the features for each electrode separately.

Author: Mali Halac
        BS Electrical Engineering | Computer Science
        Department of Electrical and Computer Engineering
        Drexel University
"""
####################################################################


# New function to extract the band powers
def bandpower(x, fs, fmin, fmax):
    my_awesome_list = []
    for i in x:
        f, Pxx = scipy.signal.periodogram(i, fs=fs)
        ind_min = np.argmax(f > fmin) - 1
        ind_max = np.argmax(f > fmax) - 1
        return_val = np.trapz(Pxx[ind_min: ind_max], f[ind_min: ind_max])
        my_awesome_list.append(return_val)
    return my_awesome_list



# Extract the features for each electrode 
# Then save them to a csv file
def run_all(data_file, save_path, data_name):

    features = collections.defaultdict(dict)
    
    raw = read_raw_edf(data_file, preload=True, verbose=False)
    
    i=0
    for channel in raw.ch_names:
        print("Channel number:", i)
        print("Channel:", channel)

        # pick EEG channels
        # dont forget to remove the stimulation channel from the csv file!
        picks = pick_types(raw.info, eeg=True, stim=True)
        #print(raw.ch_names)
    
        events = mne.find_events(raw, verbose=False)
        epochs = mne.Epochs(raw,events,event_id=None,preload=True,picks=picks)

        #print("\n\n\n\n\n")
        #print("Channel:", channel)
        #print()
        # pick channels
        pickydipick = epochs.pick_channels([channel])     
        
        data = pickydipick.get_data()
        data = np.squeeze(data)
        #print(data)
        #print(data.shape)

        bandpower_alpha = bandpower(data, 500, 8, 12)
        bandpower_beta = bandpower(data, 500, 12, 30)
        bandpower_delta = bandpower(data, 500, 1, 4)
        bandpower_gamma = bandpower(data, 500, 30, 64)
        bandpower_theta = bandpower(data, 500, 4, 8)

        features[channel]["alpha"] = bandpower_alpha
        features[channel]["beta"] = bandpower_beta
        features[channel]["delta"] = bandpower_delta
        features[channel]["gamma"] = bandpower_gamma
        features[channel]["theta"] = bandpower_theta
        print()

        i+=1

    df = pd.DataFrame.from_dict(features)
    features_csv_file = save_path + "/" + data_name + "features.csv"
    df.to_csv(features_csv_file, index=True)

    return




if __name__ == '__main__':

    directory = "/Users/malihalac/desktop/BCI_Hackathon/github/BCI_Hackathon/edf_data/"
    save_path = "/Users/malihalac/desktop/BCI_Hackathon/github/BCI_Hackathon/features"

    for file in os.listdir(directory):
        if ".edf" in file:
            file_name = file.strip(".edf")
            file_path = directory + file

            print()
            print("Reading the file:", file_name)
            run_all(file_path,save_path,file_name)
            print("Extracted the futures for", file_name)
            print()
            
    







