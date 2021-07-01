'''
This feature_extraction.py script is based on (modified)
https://github.com/BCI-HCI-IITKGP/Cognitive-Mental-workload-Classification/blob/master/FeatureExtraction.ipynb
'''

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

"""
Extract the features for each electrode separately
"""

### Coefficient Variation
def coeff_var(a):
    b = a #Extracting the data from the 61 channels
    output = np.zeros(len(b)) #Initializing the output array with zeros
    k = 0; #For counting the current row no.
    for i in b:
        mean_i = np.mean(i) #Saving the mean of array i
        std_i = np.std(i) #Saving the standard deviation of array i
        output[k] = std_i/mean_i #computing coefficient of variation
        k=k+1
    return output



### Mean of vertex to slope
import heapq
from scipy.signal import argrelextrema

def first_diff(i):
    b=i    
    
    out = np.zeros(len(b))
    
    for j in range(len(i)):
        out[j] = b[j-1]-b[j]# Obtaining the 1st Diffs
        
        j=j+1
        c=out[1:len(out)]
    return c #returns first diff

def slope_mean(p):
    b = p #Extracting the data from the 61 channels
    output = np.zeros(len(b)) #Initializing the output array with zeros
    res = np.zeros(len(b)-1)
    
    k = 0; #For counting the current row no.
    for i in b:
        x=i
        amp_max = i[argrelextrema(x, np.greater)[0]]
        t_max = argrelextrema(x, np.greater)[0]
        amp_min = i[argrelextrema(x, np.less)[0]]
        t_min = argrelextrema(x, np.less)[0]
        t = np.concatenate((t_max,t_min),axis=0)
        t.sort()#sort on the basis of time

        h=0
        amp = np.zeros(len(t))
        res = np.zeros(len(t)-1)
        for l in range(len(t)):
            amp[l]=i[t[l]]
           
        
        amp_diff = first_diff(amp)
        
        t_diff = first_diff(t)
        
        for q in range(len(amp_diff)):
            res[q] = amp_diff[q]/t_diff[q]         
        output[k] = np.mean(res) 
        k=k+1
    return output



### Variance of vertex to slope
def first_diff(i):
    b=i    
    
    out = np.zeros(len(b))
    
    for j in range(len(i)):
        out[j] = b[j-1]-b[j]# Obtaining the 1st Diffs
        
        j=j+1
        c=out[1:len(out)]
    return c #returns first diff


def slope_var(p):
    b = p #Extracting the data from the 61 channels
    output = np.zeros(len(b)) #Initializing the output array with zeros
    res = np.zeros(len(b)-1)
    
    k = 0; #For counting the current row no.
    for i in b:
        x=i
        amp_max = i[argrelextrema(x, np.greater)[0]]#storing maxima value
        t_max = argrelextrema(x, np.greater)[0]#storing time for maxima
        amp_min = i[argrelextrema(x, np.less)[0]]#storing minima value
        t_min = argrelextrema(x, np.less)[0]#storing time for minima value
        t = np.concatenate((t_max,t_min),axis=0) #making a single matrix of all matrix
        t.sort() #sorting according to time

        h=0
        amp = np.zeros(len(t))
        res = np.zeros(len(t)-1)
        for l in range(len(t)):
            amp[l]=i[t[l]]
           
        
        amp_diff = first_diff(amp)
        
        t_diff = first_diff(t)
        
        for q in range(len(amp_diff)):
            res[q] = amp_diff[q]/t_diff[q] #calculating slope        
    
        output[k] = np.var(res) 
        k=k+1#counting k
    return output



###FFT Max Power - Delta, Theta, Alpha & Beta Band!
from scipy import signal
def maxPwelch(data_win,Fs):
	BandF = [0.5, 4, 8, 12, 35]
	PMax = np.zeros([61,(len(BandF)-1)]);
	for j in range(61):
		f,Psd = signal.welch(data_win[j,:], Fs)
		for i in range(len(BandF)-1):
			fr = np.where((f>BandF[i]) & (f<=BandF[i+1]))
			PMax[j,i] = np.max(Psd[fr])
	return PMax[:,0],PMax[:,1],PMax[:,2],PMax[:,3]



### Hjorth parameters
def hjorth(input):                                             # function for hjorth 
    realinput = input
    hjorth_activity = np.zeros(len(realinput))
    hjorth_mobility = np.zeros(len(realinput))
    hjorth_diffmobility = np.zeros(len(realinput))
    hjorth_complexity = np.zeros(len(realinput))
    diff_input = np.diff(realinput)
    diff_diffinput = np.diff(diff_input)
    k = 0
    for j in realinput:
        hjorth_activity[k] = np.var(j)
        hjorth_mobility[k] = np.sqrt(np.var(diff_input[k])/hjorth_activity[k])
        hjorth_diffmobility[k] = np.sqrt(np.var(diff_diffinput[k])/np.var(diff_input[k]))
        hjorth_complexity[k] = hjorth_diffmobility[k]/hjorth_mobility[k]
        k = k+1
    return hjorth_activity, hjorth_mobility, hjorth_complexity        #returning hjorth activity, hjorth mobility , hjorth complexity



### Kurtosis
def kurtosis(a):
    b = a # Extracting the data from the 61 channels
    output = np.zeros(len(b)) # Initializing the output array with zeros (length = 14)
    k = 0; # For counting the current row no.
    for i in b:
        mean_i = np.mean(i) # Saving the mean of array i
        std_i = np.std(i) # Saving the standard deviation of array i
        t = 0.0
        for j in i:
            t += (pow((j-mean_i)/std_i,4)-3)
        kurtosis_i = t/len(i) # Formula: (1/N)*(summation(x_i-mean)/standard_deviation)^4-3
        output[k] = kurtosis_i # Saving the kurtosis in the array created
        k +=1 # Updating the current row no.
    return output



### Second difference mean
def secDiffMean(a):
    b = a # Extracting the data of the 61 channels
    output = np.zeros(len(b)) # Initializing the output array with zeros (length = 14)
    temp1 = np.zeros(len(b[0])-1) # To store the 1st Diffs
    k = 0; # For counting the current row no.
    for i in b:
        t = 0.0
        for j in range(len(i)-1):
            temp1[j] = abs(i[j+1]-i[j]) # Obtaining the 1st Diffs
        for j in range(len(i)-2):
            t += abs(temp1[j+1]-temp1[j]) # Summing the 2nd Diffs
        output[k] = t/(len(i)-2) # Calculating the mean of the 2nd Diffs
        k +=1 # Updating the current row no.
    return output



### Second difference max
def secDiffMax(a):
    b = a # Extracting the data from the 61 channels
    output = np.zeros(len(b)) # Initializing the output array with zeros (length = 14)
    temp1 = np.zeros(len(b[0])-1) # To store the 1st Diffs
    k = 0; # For counting the current row no.
    t = 0.0
    for i in b:
        for j in range(len(i)-1):
            temp1[j] = abs(i[j+1]-i[j]) # Obtaining the 1st Diffs
        t = temp1[1] - temp1[0]
        for j in range(len(i)-2):
            if abs(temp1[j+1]-temp1[j]) > t :
                t = temp1[j+1]-temp1[j] # Comparing current Diff with the last updated Diff Max

        output[k] = t # Storing the 2nd Diff Max for channel k
        k +=1 # Updating the current row no.
    return output



### Skewness
def skewness(arr):
    data = arr 
    skew_array = np.zeros(len(data)) #Initialinling the array as all 0s
    index = 0; #current cell position in the output array
   
    for i in data:
        skew_array[index]=sp.stats.skew(i,axis=0,bias=True)
        index+=1 #updating the cell position
    return skew_array



### First Difference Mean
def first_diff_mean(arr):
    data = arr 
    diff_mean_array = np.zeros(len(data)) #Initialinling the array as all 0s
    index = 0; #current cell position in the output array
   
    for i in data:
        sum=0.0#initializing the sum at the start of each iteration
        for j in range(len(i)-1):
            sum += abs(i[j+1]-i[j]) # Obtaining the 1st Diffs
           
        diff_mean_array[index]=sum/(len(i)-1)
        index+=1 #updating the cell position
    return diff_mean_array



### First Difference Max
def first_diff_max(arr):
    data = arr 
    diff_max_array = np.zeros(len(data)) #Initialinling the array as all 0s
    first_diff = np.zeros(len(data[0])-1)#Initialinling the array as all 0s 
    index = 0; #current cell position in the output array
   
    for i in data:
        max=0.0#initializing at the start of each iteration
        for j in range(len(i)-1):
            first_diff[j] = abs(i[j+1]-i[j]) # Obtaining the 1st Diffs
            if first_diff[j]>max: 
                max=first_diff[j] # finding the maximum of the first differences
        diff_max_array[index]=max
        index+=1 #updating the cell position
    return diff_max_array




def run_all(data_file, save_path, data_name):
    raw = read_raw_edf(data_file, preload=True, verbose=False)
    picks = pick_types(raw.info, eeg=True)
    data = raw.get_data(picks=picks)
    events = mne.find_events(raw, verbose = False)
    epochs = mne.Epochs(raw,events)
    
    print("\n\n\n\n\n")
    # Band Powers
    delta, theta, alpha, beta = maxPwelch(data,500)
    # Coefficient of variation
    coefficient_variation = coeff_var(data)
    # Mean of vertex to vertex slope
    mean_vertex = slope_mean(data)
    # Variance of vertex to vertex slope
    var_vertex = slope_var(data)
    # Hjorth Parameters
    activity, mobility, complexity = hjorth(data)
    # Kurtosis
    kurtosis_ = kurtosis(data)
    # Second Difference Mean
    sec_dif_mean = secDiffMean(data)
    # Second Difference Max
    sec_dif_max = secDiffMax(data)
    # Skewness
    skewness_ = skewness(data)
    # First Difference Mean
    first_dif_mean = first_diff_mean(data)
    # First Difference Max
    first_dif_max = first_diff_max(data)


    features_dict = {
                    "Delta": delta,
                    "Theta": theta,
                    "Alpha": alpha,
                    "Beta": beta,
                    "Coefficient of variation": coefficient_variation,
                    "Mean of vertex to vertex slope": mean_vertex,
                    "Variance of vertex to vertex slope": var_vertex,
                    "Hjorth activity": activity,
                    "Hjorth mobility": mobility,
                    "Hjorth complexity": complexity,
                    "Kurtosis": kurtosis_,
                    "Second difference mean": sec_dif_mean,
                    "Second difference max": sec_dif_max,
                    "Skewness": skewness_,
                    "First difference mean": first_dif_mean,
                    "First difference max": first_dif_max
                        }

    ##################### IMPORTANT ####################################
    ### Columns in the generated features.csv files are the features
    ### Rows are the individual electrodes
    ####################################################################

    df = pd.DataFrame.from_dict(features_dict)
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







