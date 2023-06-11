#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Excitation/Inhibition Ratio in Python 

The following script is a translation in python of Bruining et al (2020) MATLAB script
to compute excitation/inhibition ratio fEI:
Bruining, H., Hardstone, R., Juarez-Martinez, E.L. et al. 
Measurement of excitation-inhibition ratio in autism spectrum disorder using critical brain dynamics. 
Sci Rep 10, 9195 (2020). https://doi.org/10.1038/s41598-020-65500-4
Original fEI Matlab script : https://github.com/rhardstone/fEI/blob/master/calculateFEI.m

@author: Julien Pichot
 """
import numpy as np
import pandas as pd
import mne
import matplotlib
import pathlib
import matplotlib.pyplot as plt
import os
import os.path as op
import sys
import logging
import scipy
from scipy import signal
from scipy.signal import detrend
from scipy.stats import pearsonr
from scipy.signal import hilbert
from functools import partial
%matplotlib qt
import glob
from collections import OrderedDict
from sklearn.preprocessing import MinMaxScaler
from mne.channels import make_standard_montage
from numpy.matlib import repmat
from PyAstronomy.pyasl import generalizedESD
import multiprocessing
from mne.filter import next_fast_len
from joblib import Parallel, delayed


def calculate_fei(Signal, window_size, window_overlap):
    

    """
    Parameters
    ----------
    signal: array, shape(n_channels,n_times) amplitude envelope for all channels
    sfreq: integer sampling frequency of the signal
    window_size: float window size (i.e 5000)
    window_overlap: float fraction of overlap between windows (0-1)
    DFAExponent: array, shape(n_channels) array of DFA values, with corresponding value for each channel, used for thresholding fEI


    Returns
    -------
    EI : array, shape(n_channels) fEI values, with wAmp and wDNF outliers 
    fEI_outliers_removed: array, shape(n_channels) fEI values, with outliers removed
    wAmp: array, shape(n_channels, num_windows) windowed amplitude, computed across all channels/windows
    wDNF: array, shape(n_channels, num_windows) windowed detrended normalized fluctuation, computed across all channels/windows
    """   

    
    num_channels, length_signal = Signal.shape
    window_offset = int(window_size * (1-window_overlap))
    all_window_index = create_window_indices(length_signal, window_size, window_offset)
    num_windows = all_window_index.shape[0]
    
    EI = np.zeros((num_channels,))
    EI[:] = np.NAN
    
    fEI_outliers_removed = np.zeros((num_channels,))
    fEI_outliers_removed[:] = np.NAN
    
    num_outliers = np.zeros((num_channels,num_windows))
    num_outliers[:] = np.NAN
    
    wAmp = np.zeros((num_channels, num_windows))
    wAmp[:] = np.NAN
    
    wDNF = np.zeros((num_channels, num_windows))
    wDNF[:] = np.NAN
    
  

    for i_channel in range(num_channels):
        original_amplitude = Signal[i_channel,:]  
        
        if np.min(original_amplitude) == np.max(original_amplitude):
            print('Problem computing fEI for i_channel'+str(i_channel))
            continue
        signal_profile = np.cumsum(original_amplitude - np.mean(original_amplitude))  
         
        w_original_amplitude = np.mean(original_amplitude[all_window_index], axis=1)
        xAmp = np.repeat(w_original_amplitude[:, np.newaxis], window_size, axis=1) 
        
        xSignal = signal_profile[all_window_index]                                        
        xSignal = np.divide(xSignal, xAmp).T   
                     
        _, fluc, _, _, _ = np.polyfit(np.arange(window_size), xSignal, deg=1, full=True) # arthur
        # Convert to root-mean squared error, from squared error
        w_detrended_normalized_fluctuations = np.sqrt(fluc / window_size) #arthur 
    
        EI[i_channel] = 1 - pearsonr(w_detrended_normalized_fluctuations, w_original_amplitude)[0]
        #EI[i_channel] = 1 - np.corrcoef(w_original_amplitude, w_detrended_normalized_fluctuations)[0, 1] arthur script 
        # np.corrcoef et pearsonr font exactement la même chose aussi 
        
        
        gesd_alpha = 0.05
        max_outliers_percentage = 0.025  # this is set to 0.025 per dimension (2-dim: wAmp and wDNF), so 0.05 is max
        max_num_outliers = int(np.round(max_outliers_percentage * len(w_original_amplitude)))
        outlier_indexes_wAmp = generalizedESD(w_original_amplitude, max_num_outliers, gesd_alpha)[1] #1 
        outlier_indexes_wDNF = generalizedESD(w_detrended_normalized_fluctuations, max_num_outliers, gesd_alpha)[1] #1
        outlier_union = outlier_indexes_wAmp + outlier_indexes_wDNF
        num_outliers[i_channel, :] = len(outlier_union)
        not_outlier_both = np.setdiff1d(np.arange(len(w_original_amplitude)), np.array(outlier_union))
        fEI_outliers_removed[i_channel] = 1 - np.corrcoef(w_original_amplitude[not_outlier_both], \
                                                       w_detrended_normalized_fluctuations[not_outlier_both])[0, 1]

        wAmp[i_channel,:] = w_original_amplitude
        wDNF[i_channel,:] = w_detrended_normalized_fluctuations

        EI[DFAExponent <= 0.6] = np.nan
        fEI_outliers_removed[DFAExponent <= 0.6] = np.nan
        
    return EI, fEI_outliers_removed, wAmp, wDNF



def calculate_DFA(Signal, windowSizes, windowOverlap):

    
""" 
This function calculates the Detrended Fluctuation Analysis (DFA) exponent in Python based on the provided input signal and parameters. It takes the following parameters:

Parameters: 
    Signal: 
        signal in numpy array with shape (numChannels, lengthSignal) where numChannels is the number of channels and lengthSignal is the length of the signal.
        
    windowSizes: 
        A numpy array of integers representing the window sizes used for analysis. 
    windowOverlap: 
        A float representing the overlap between consecutive windows (between 0-1).

Returns:

    DFAExponent: A numpy array of shape (numChannels,) containing the calculated DFA exponents for each channel.
    meanDF: A numpy array of shape (numChannels, numWindowSizes) containing the mean detrended fluctuations for each channel and window size.
    windowSizes: The same input windowSizes array.

"""

    numChannels,lengthSignal = Signal.shape
    meanDF = np.zeros((numChannels, len(windowSizes)))
    DFAExponent = np.zeros((numChannels,))
    #windowSizes = windowSizes.reshape(-1, 1) 
    
    for i_channel in range(numChannels):
        for i_windowSize in range(len(windowSizes)):
            windowOffset = int(windowSizes[i_windowSize] * (1 - windowOverlap))
            allWindowIndex = create_window_indices(lengthSignal, windowSizes[i_windowSize], windowOffset)
            originalAmplitude = Signal[i_channel,:]
            signalProfile = np.cumsum(originalAmplitude - np.mean(originalAmplitude))
            xSignal = signalProfile[allWindowIndex]
            
            # Calculate local trend, as the line of best fit within the time window -> fluc is the sum of squared residuals
            _, fluc, _, _, _ = np.polyfit(np.arange(windowSizes[i_windowSize]), xSignal.T, deg=1, full=True)
            # Convert to root-mean squared error, from squared error
            det_fluc = np.sqrt(np.mean(fluc / windowSizes[i_windowSize]))
            meanDF[i_channel, i_windowSize] = det_fluc       
        
        # get the positions of the first and last window sizes used for fitting
        fit_interval_first_window = np.argwhere(windowSizes >= fit_interval[0] * sfreq)[0][0]
        fit_interval_last_window = np.argwhere(windowSizes <= fit_interval[1] * sfreq)[-1][0]

        x = np.log10(windowSizes[fit_interval_first_window:fit_interval_last_window]).reshape(-1)
        y = np.log10(meanDF[i_channel, fit_interval_first_window:fit_interval_last_window]).reshape(-1)

        model = np.polyfit(x, y, 1)
        #dfa_intercept[ch_idx] = model[1]
        DFAExponent[i_channel] = model[0]
        

    return (DFAExponent,meanDF, windowSizes)

def create_window_indices(length_signal, length_window, window_offset):
    window_starts = np.arange(0, length_signal-length_window+1, window_offset)
    num_windows = window_starts.shape

    one_window_index = np.arange(length_window)
    all_window_index = np.repeat(one_window_index[np.newaxis, :], num_windows, axis=0)

    all_window_index += np.repeat(window_starts[:, np.newaxis], length_window, axis=1)
    return all_window_index

#%%
""" Example to use functions """
# First, define parameters 

#Specify frequency band
frequency_band = [8,13]

#fEI parameters
window_size_sec = 5
sfreq = 1000
window_size = int(window_size_sec * sfreq)
window_overlap = 0.8 

#DFA parameters
windowOverlap = 0.5
compute_interval = [1,10] #interval over which to compute DFA, 
fit_interval = [1,10]
# compute DFA window sizes: 20 windows sizes per order of magnitude
windowSizes = np.floor(np.logspace(-1, 3, 81) * sfreq).astype(int)  # %logspace from 0.1 seccond (10^-1) to 1000 (10^3) seconds
# make sure there are no duplicates after rounding
windowSizes = np.sort(np.unique(windowSizes))
windowSizes = windowSizes[(windowSizes >= compute_interval[0] * sfreq) & \
                            (windowSizes <= compute_interval[1] * sfreq)]



""" Exemple to use the functions and compute fEI for several EEG """
# Define the folder containing the EEG files
folder = '/path/to/eeg'

## Define channel names
@WARNING 
# You need to load a eeg and extract ch_names as follow : 
#ch_names=raw.info['ch_names'] 
#Or you can load a text file with ch_names 
#np.savetxt('ch_names.txt', ch_names,delimiter=',', fmt='%s') 
ch_names= np.loadtxt('ch_names_txt_file',dtype=str)


# Create df to store results 
df=pd.DataFrame()
# create list for files with bads empty = False and DFA <0.6 
bad_files = []
fEI_bad_files = []


# Loop through all EEG files in a folder
for file in glob.glob(os.path.join(folder, '*.fif')): ## modify according to your eeg format
    
    # Load the EEG file
    raw = mne.io.read_raw_fif(file, preload=True) 
    # Drop channels out of the scalp
    raw = raw.drop_channels(['E14','E17','E21','E48','E119','E126','E127']) # modify to drop electrode out of the scalp for your montage
    
    # Check if there are still bad channels
    if raw.info['bads']:
    # If empty= False, then Append the filename to the list of bad files
        bad_files.append(op.basename(file).split('.')[0])
        continue

    # Pre-process 
    #Filter 
    raw = raw.filter(l_freq=frequency_band[0],h_freq=frequency_band[1],
                                 filter_length='auto', l_trans_bandwidth='auto', h_trans_bandwidth='auto',
                                 fir_window='hamming',phase='zero',fir_design="firwin",
                                 pad='reflect_limited', verbose=0)

    # remove first and last second for edge effects 
    start_time = raw.times[0] + 1.0
    end_time = raw.times[-1] - 1.0
    raw=raw.crop(tmin=start_time, tmax=end_time)
    
    #Compute Hilbert tranform for amplitude envelope of the filtered signal 
    raw =raw.apply_hilbert(picks=['eeg'], envelope = True)
    
    # Select the data to use in the calculation of fEi and DFA_exponent (convert signal in numpy array)
    Signal = raw.get_data(reject_by_annotation='omit',picks='eeg')

    #compute DFA first to removed fEI_outliers with DFAExponent<0.6
    DFAExponent, meanDF, windowSizes = calculate_DFA(Signal, windowSizes, windowOverlap)
    DFA_mean = np.nanmean(DFAExponent)
    
    # Compute EI ratio 
    try:
        EI, fEI_outliers_removed, wAmp, wDNF = calculate_fei(Signal, window_size, window_overlap)
        
    except :
        fEI_bad_files.append(op.basename(file).split('.')[0])
        continue
    
    
    fEI = np.nanmean(fEI_outliers_removed)
    #else:
    #    fEI = np.nan
    
    # Add the subject ID, fEI, DFA to DF 
    subject = op.basename(file).split('.')[0]
    new_row = {
        'subject': subject, 
        'DFA_criterion': DFA_mean,
        'fEI': fEI, 
        **dict(zip(ch_names, fEI_outliers_removed)),
        **{f'EI_{ch}': val for ch, val in zip(ch_names, EI)},
}
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)



""" Exemple to compute fEI for single EEG """
# Load eeg 
filename = '/Path/to/eeg_file'
raw = mne.io.read_raw_fif(filename, preload=(True),verbose=True) 

#Drop bad channels 
raw= raw.drop_channels(['E14','E17','E21','E48','E119','E126','E127'])
raw.info['bads'] # check that there are no more bads

#""" Filtered between 8 and 13 Hz """
raw = raw.filter(l_freq=frequency_band[0],h_freq=frequency_band[1],
                             filter_length='auto', l_trans_bandwidth='auto', h_trans_bandwidth='auto',
                             fir_window='hamming',phase='zero',fir_design="firwin",
                             pad='reflect_limited', verbose=0)



# remove first and last second for edge effects 
start_time = raw.times[0] +1
end_time = raw.times[-1] - 1
raw=raw.crop(tmin=start_time, tmax=end_time)

# Compute Hilbert amplitude envelope of the filtered signal
raw=raw.apply_hilbert(picks=['eeg'], envelope = True,verbose='debug')


## Extract data in numpy array
Signal = raw.get_data(reject_by_annotation='omit', picks = 'eeg')

# Compute DFA and fEI 
DFAExponent, meanDF, windowSizes = calculate_DFA(Signal, windowSizes, windowOverlap)
DFA_criterion = np.mean(DFAExponent)
EI,fEI_outliers_removed, wAmp, wDNF = calculate_fei(Signal, window_size, window_overlap)




""" Example to plot topomap with fEI for each electrode """

# First, extract x, y coordinates 
#EXTRACT x,y coordinates for each electrodes that fit to 2D topomap : Plot sensors then extract coordinates from the figure
# you need to load one EEG
fig = raw.plot_sensors(show_names=True) ## also works with fig=mne.viz.plot_montage(your_montage)
ax = fig.axes[0]
coordinates = []
for point in ax.collections[0].get_offsets():
    x, y = point
    coordinates.append([x, y])
xy_pos= np.array(coordinates)
np.savetxt('electrode_positions.txt', xy_pos) # Save coordinates in txt as references coordinates

xy_pos= np.loadtxt('electrode_positions.txt')

# determine colormap/colorbar intervall as Bruining et al 2020 
def plot_interval(values):
    # Determine the minimum and maximum of the range
    min_range = np.nanpercentile(values, 5)
    max_range = np.nanpercentile(values, 95)

    # Make the interval symmetric around 1
    range_diff = max(1 - min_range, max_range - 1)
    min_range_s = 1 - range_diff
    max_range_s = 1 + range_diff

    # Round the range to one decimal place
    min_range_r = np.round(min_range_s, 1)
    max_range_r = np.round(max_range_s, 1)

    return   min_range_r, max_range_r


ei_all = df.iloc[:, 3:124] ## to obtain the ei ratio for each electrodes for all subject
ei_all= np.nanmean(ei_all,axis=0)
 min_range_r, max_range_r = plot_interval(ei_all)

#topomap 
#WARNING : If the plot not displayed, load one record, try raw.plot() then retry
ig, ax = plt.subplots()
im, _ = mne.viz.plot_topomap(ei_all, pos=xy_pos, vlim=(min_range_r, max_range_r),  cmap='bwr'  ,contours=0,axes=ax)
cbar = plt.colorbar(im, ax=ax)
plt.legend(loc='lower center')
plt.gcf().set_size_inches(7, 6)
plt.subplots_adjust(top=0.94,
bottom=0.048,
left=0.053,
right=0.985,
hspace=0.2,
wspace=0.2)
ax.set_title('Excitation/Inhibition ratio', fontsize= 18, fontweight='bold')
plt.show()





