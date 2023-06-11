#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The following script contains all the code used to run the analyses, as well as the code to reproduce all the figures

Excitation/Inhibition Ratio in Python. 
Matlab to Python translation from Bruining et al (2020) :
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

import seaborn as sns
import glob
from collections import OrderedDict
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from mne.channels import make_standard_montage


# for fEI and DFA 
from numpy.matlib import repmat
from PyAstronomy.pyasl import generalizedESD
import multiprocessing
from mne.filter import next_fast_len
from joblib import Parallel, delayed

#stats
import pingouin as pg
import dabest
import statsmodels.api as sm
from statsmodels.multivariate.manova import MANOVA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda
from PIL import Image

# clustering 
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.mixture import GaussianMixture
from sklearn.cluster import SpectralClustering
from sklearn.cluster import MeanShift
from sklearn.mixture import BayesianGaussianMixture


"""fEI function"""  
"""
Input : 
Signal= amplitude envelope with dimensions (num_samples,num_channels)
window_size= in samples
window_overlap= is fraction of overlap between windows (0-1)

EI, wAmp, wDNF = calculate_fei(Signal, window_size, window_overlap) 
"""
"""fEI  parameters"""
frequency_band = [8,13]
window_size_sec = 5 
sfreq = 1000
window_size = int(window_size_sec * sfreq)
window_size = 5000
sampling_frequency = 1000
window_overlap = 0.8 

""" DFA parameters"""
windowOverlap = 0.5
compute_interval = [1,10] #interval over which to compute DFA, 
fit_interval = [1,10]
# compute DFA window sizes: 20 windows sizes per order of magnitude
windowSizes = np.floor(np.logspace(-1, 3, 81) * sfreq).astype(int)  # %logspace from 0.1 seccond (10^-1) to 1000 (10^3) seconds
# make sure there are no duplicates after rounding
windowSizes = np.sort(np.unique(windowSizes))
windowSizes = windowSizes[(windowSizes >= compute_interval[0] * sfreq) & \
                            (windowSizes <= compute_interval[1] * sfreq)]

def calculate_fei(Signal, window_size, window_overlap):
    
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
                     
        _, fluc, _, _, _ = np.polyfit(np.arange(window_size), xSignal, deg=1, full=True) 
        # Convert to root-mean squared error, from squared error
        w_detrended_normalized_fluctuations = np.sqrt(fluc / window_size)  
    
        EI[i_channel] = 1 - pearsonr(w_detrended_normalized_fluctuations, w_original_amplitude)[0]


        gesd_alpha = 0.05
        max_outliers_percentage = 0.025  # this is set to 0.025 per dimension (2-dim: wAmp and wDNF), so 0.05 is max
        max_num_outliers = int(np.round(max_outliers_percentage * len(w_original_amplitude)))
        outlier_indexes_wAmp = generalizedESD(w_original_amplitude, max_num_outliers, gesd_alpha)[1] #1 
        outlier_indexes_wDNF = generalizedESD(w_detrended_normalized_fluctuations, max_num_outliers, gesd_alpha)[1] #1
        outlier_union = outlier_indexes_wAmp + outlier_indexes_wDNF
        num_outliers[i_channel, :] = len(outlier_union)
        not_outlier_both = np.setdiff1d(np.arange(len(w_original_amplitude)), np.array(outlier_union))
        fEI_outliers_removed[i_channel] =1 - np.corrcoef(w_original_amplitude[not_outlier_both], \
                                                       w_detrended_normalized_fluctuations[not_outlier_both])[0, 1]

        wAmp[i_channel,:] = w_original_amplitude
        wDNF[i_channel,:] = w_detrended_normalized_fluctuations
        EI[DFAExponent <= 0.6] = np.nan
        fEI_outliers_removed[DFAExponent <= 0.6] = np.nan
        
    return EI, fEI_outliers_removed, wAmp, wDNF

def calculate_DFA(Signal, windowSizes, windowOverlap):

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

DFAExponent, meanDF, windowSizes = calculate_DFA(Signal, windowSizes, windowOverlap)
DFA_criterion = np.mean(DFAExponent)

EI,fEI_outliers_removed, wAmp, wDNF = calculate_fei(Signal, window_size, window_overlap)


""" RUN fei ratio script For all subject""" 

# Define the folder containing the EEG files
folder = '/Volumes/G-DRIVE/EEG_rdb/EEG_final_and_f7/'
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

## Define channel names
#ch_names=raw.info['ch_names'] 
@WARNING = ## Make sure to load a record, then drop bads channelsn then create ch_names
#np.savetxt('ch_names.txt', ch_names,delimiter=',', fmt='%s') 
ch_names= np.loadtxt('/Users/julienpichot/Documents/fEI_2023/txt_file/ch_names.txt',dtype=str)

# Create df to store results 
df=pd.DataFrame()
# create list to append bad files 
bad_files = []
fEI_bad_files = []

# Loop through all EEG files in the folder
for file in glob.glob(os.path.join(folder, '*.fif')):
    
    # Load the EEG file
    raw = mne.io.read_raw_fif(file, preload=True)
    sfreq = raw.info['sfreq']
    # Drop channels out of the scalp
    raw = raw.drop_channels(['E14','E17','E21','E48','E119','E126','E127'])
    
    # Check if there are still bad channels
    if raw.info['bads']:
    # If empty= False, then Append the filename to the list of bad files
        bad_files.append(op.basename(file).split('.')[0])
        continue

    # Pre-process 
    raw = raw.filter(l_freq=frequency_band[0],h_freq=frequency_band[1],
                                 filter_length='auto', l_trans_bandwidth='auto', h_trans_bandwidth='auto',
                                 fir_window='hamming',phase='zero',fir_design="firwin",
                                 pad='reflect_limited', verbose=0)

    #remove first and last second for edge effects 
    start_time = raw.times[0] + 1.0
    end_time = raw.times[-1] - 1.0
    raw=raw.crop(tmin=start_time, tmax=end_time)
    
    #Compute Hilbert tranform for amplitude envelope of the filtered signal 
    raw =raw.apply_hilbert(picks=['eeg'], envelope = True)
    
    #Select the data to use in the calculation of fEi and DFA_exponent
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
    
    nan_percent = np.sum(np.isnan(fEI_outliers_removed)) / len(fEI_outliers_removed)
    if nan_percent <= 0.6:
        fEI = np.nanmean(fEI_outliers_removed)
    else:
        fEI = np.nan
    
    # Add the subject ID, fEI, DFA, ROI etc to DF 
    subject = op.basename(file).split('.')[0]
    new_row = {
        'subject': subject, 
        'DFA_criterion': DFA_mean,
        'fEI': fEI, 
        **dict(zip(ch_names, fEI_outliers_removed)),
        'frontal_fEI': np.nanmean(fEI_outliers_removed[[2,   3,   4,   8,   9,  10,  11,  13,  14,  15,  16,  17,  18,
                19,  20, 113]]),
        'parietal_fEI': np.nanmean(fEI_outliers_removed[[ 5,   6,  12,  25,  26,  27,  33,  49,  50,  56,  73,  74,  75,
                92, 100, 101, 106, 107]]),
        'occipital_fEI': np.nanmean(fEI_outliers_removed[[61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 77, 78, 79, 83, 84,
               89]]),
        'r_temporal_fEI': np.nanmean(fEI_outliers_removed[[86,  87,  88,  91,  92,  93,  96,  97,  98,  99, 102, 103, 104,
               105, 108, 109, 110, 111]]),
        'l_temporal_fEI': np.nanmean(fEI_outliers_removed[[30, 31, 32, 35, 36, 37, 38, 40, 41, 42, 43, 44, 45, 46, 47, 51, 53,
               54]]),
        'mean_roi': np.nanmean(fEI_outliers_removed[[2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                                                     17, 18, 19, 20, 25, 26, 27, 30, 31, 32, 33, 35,
                                                     36, 37, 38, 40, 41, 42, 43, 44, 45, 46, 47, 49,
                                                     50, 51, 53, 54, 56, 61, 62, 63, 64, 65, 66, 67,
                                                     68, 69, 70, 71, 72, 73, 74, 75, 77, 78, 79, 83, 
                                                     84, 86, 87, 88, 89, 91, 92, 92, 93, 96, 97, 98,
                                                     99, 100, 101, 102, 103, 104, 105, 106, 107, 108,
                                                     109, 110, 111, 113]]),
        **{f'EI_{ch}': val for ch, val in zip(ch_names, EI)},
        **{f'wamp_{ch}': val for ch, val in zip(ch_names, wAmp)},
        **{f'wdnf_{ch}': val for ch, val in zip(ch_names, wDNF)},

}
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df['subject'] = df['subject'].str.replace('-RS_eeg', '')

#create the full dataset with EI and clinical variables 
pheno = pd.read_excel('/mypath) ##open dataset with clinical variables
df = pd.merge(df, pheno, on='subject') # merged the two dataframe 
df.to_csv('/mypath/file.csv', index=False)

#%% PLOT 
"""1) Define channel locations """
    """ Set parameters """ 
#montage = mne.channels.make_standard_montage('GSN-HydroCel-128')
#ch_names = raw.info['ch_names']
#n_channels = len(EI)
#sfreq = raw.info['sfreq']
#ch_types = ['eeg'] * n_channels


    """ USE THIS METHOD TO EXTRACT x,y coordinates for each electrodes that fit to 2D topomap """
#fig = raw.plot_sensors(show_names=True) ## also works with fig=mne.viz.plot_montage(your_montage)
#ax = fig.axes[0]
                      
#coordinates = []
#for point in ax.collections[0].get_offsets():
#    x, y = point
#    coordinates.append([x, y])
#xy_pos= np.array(coordinates)
#np.savetxt('electrode_positions.txt', xy_pos) # Save coordinates in txt as references coordinates

    """ Next time you just need to load x,y coordinates from .txt file previously save  """
xy_pos= np.loadtxt('/Users/julienpichot/Documents/fEI_2023/txt_file/electrode_positions.txt')


"""2)Determine topomap colorbar interval between 5th, 95th percentiles
Use on all subjects """ 
# Function to determine interval between 5th and 95th percentiles
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

    return    min_range_r, max_range_r


#%% 
"""Plot for all electrodes"""
""" Define group """
    """All subjects  """
    
ei_all = df.iloc[:, 3:124] ## to obtain the ei ratio for each electrodes for all subject
ei_all= np.nanmean(ei_all,axis=0)
# determine interval for plot all electrodes with function plot_interval 
min_range_r, max_range_r = plot_interval(ei_all)


    """ASD """
asd_ei = df.loc[df['group'] == 'ASD', df.columns[3:124]]
asd_ei = asd_ei.mean(skipna=True)

    """Relatives""" 
relatives_ei= df.loc[df['group'] == 'Relatives', df.columns[3:124]]
relatives_ei = relatives_ei.mean(skipna=True)

    """Controls"""
controls_ei=df.loc[df['group'] == 'Controls', df.columns[3:124]]
controls_ei = controls_ei.mean(skipna=True)


""" Plot three groups for all electrodes """
fig, axs = plt.subplots(1, 3, figsize=(12, 6))

im, _ = mne.viz.plot_topomap(asd_ei, pos=xy_pos, vlim=(min_range_r, max_range_r), cmap='bwr', contours=0, axes=axs[0])
axs[0].set_title('ASD\nn=118', fontsize= 14, fontweight='bold')

im, _ = mne.viz.plot_topomap(controls_ei, pos=xy_pos, vlim=(min_range_r, max_range_r), cmap='bwr', contours=0, axes=axs[1])
axs[1].set_title('Controls\nn=21', fontsize= 14, fontweight='bold')

im,_= mne.viz.plot_topomap(relatives_ei, pos=xy_pos, vlim=(min_range_r, max_range_r), cmap='bwr', contours=0, axes=axs[2])
cbar = plt.colorbar(im, ax=axs, location='bottom', pad=0.5, shrink=0.5)
cbar.set_label('Excitation/Inhibition ratio', fontsize=18)
cbar.ax.tick_params(labelsize=14)
axs[2].set_title('Relatives\nn=26', fontsize= 14, fontweight='bold')

plt.subplots_adjust(top=1.0, bottom=0.3, left=0.05, right=0.98, hspace=0.2, wspace=0.2)
plt.show()



#%% Descriptives stats and plot 
#Define Hypo and Hyper variable (based on Lefebvre et al (2022) method)
hypo_columns = ['DUNN' + str(col) for col in [2, 15, 16, 17, 18, 19, 20, 21, 23, 26, 28, 29, 30, 31, 32, 33]]
hyper_columns = ['DUNN' + str(col) for col in [1, 3, 4, 5, 6, 8, 9, 10, 11, 12, 14, 22, 24, 25, 34, 35, 36, 37, 38]]

df['hypo'] = df[hypo_columns].mean(axis=1)
df['hyper'] = df[hyper_columns].mean(axis=1)

df['hyper']=df['hyper'].replace(0,np.nan)
df['hypo']=df['hypo'].replace(0,np.nan)

# Compute dSSP score for each row in df_filtered
df_filtered['dSSP'] = df_filtered['hypo'] / df_filtered['hyper']
# Calculate the mean and standard deviation of the dSSP scores
mean_dSSP = df_filtered['dSSP'].mean()
std_dSSP = df_filtered['dSSP'].std()
# Calculate the centralized and normalized dSSP score
df_filtered['dSSP'] = (df_filtered['dSSP'] - mean_dSSP) / std_dSSP

"""Descriptive stats""""
# To create excel tables 
summary_demo = df.groupby('group')[['age_years', 'sex', 'qi_total', 'qi_icv', 'qi_irf', 'qi_ivs', 'qi_imt', 'qi_ivt', 'ados_css']].describe().round(2)
s= summary_demo.T


def count_sex(x):
    homme_count = (x == 'Homme').sum()
    femme_count = (x == 'Femme').sum()
    return f"{homme_count} Homme, {femme_count} Femme"

summary = df.groupby('group').agg(
    n_subjects=('subject', 'nunique'),
    mean_age=('age_years', 'mean'),
    range_age=('age_years', lambda x: f"{x.min()} - {x.max()}"),
    mean_qi_total=('qi_total', 'mean'),
    range_qi_total=('qi_total', lambda x: f"{x.min()} - {x.max()}"),
    sex=('sex', count_sex),
    ados_css_std=('ados_css', 'std'),
    ados_css_mean=('ados_css', 'mean'),
    ados_css_range=('ados_css', lambda x: f"{x.min()} - {x.max()}"),
    qi_icv_std=('qi_icv', 'std'),
    qi_icv_mean=('qi_icv', 'mean'),
    qi_icv_range=('qi_icv', lambda x: f"{x.min()} - {x.max()}"),
    qi_irf_std=('qi_irf', 'std'),
    qi_irf_mean=('qi_irf', 'mean'),
    qi_irf_range=('qi_irf', lambda x: f"{x.min()} - {x.max()}"),
    qi_ivs_std=('qi_ivs', 'std'),
    qi_ivs_mean=('qi_ivs', 'mean'),
    qi_ivs_range=('qi_ivs', lambda x: f"{x.min()} - {x.max()}"),
    qi_imt_std=('qi_imt', 'std'),
    qi_imt_mean=('qi_imt', 'mean'),
    qi_imt_range=('qi_imt', lambda x: f"{x.min()} - {x.max()}"),
    qi_ivt_std=('qi_ivt', 'std'),
    qi_ivt_mean=('qi_ivt', 'mean'),
    qi_ivt_range=('qi_ivt', lambda x: f"{x.min()} - {x.max()}"),
    adi_com_mean =('adi_communication', 'mean'),
    adi_com_std =('adi_communication', 'std'),
    adi_com_range=('adi_communication', lambda x: f"{x.min()} - {x.max()}")
)

summary= summary.round(2)
summary = summary.reindex(['ASD', 'Controls', 'Relatives'])
summary = summary.T


summary = df.groupby('group').agg({
    'subject': 'nunique',
    'age_years': ['mean', lambda x: f"{x.min()} - {x.max()}"],
    'qi_total': ['mean', lambda x: f"{x.min()} - {x.max()}"],
    'sex': count_sex,
    'ados_css': ['std', 'mean', lambda x: f"{x.min()} - {x.max()}"],
    'qi_icv': ['std', 'mean', lambda x: f"{x.min()} - {x.max()}"],
    'qi_irf': ['std', 'mean', lambda x: f"{x.min()} - {x.max()}"],
    'qi_ivs': ['std', 'mean', lambda x: f"{x.min()} - {x.max()}"],
    'qi_imt': ['std', 'mean', lambda x: f"{x.min()} - {x.max()}"],
    'qi_ivt': ['std', 'mean', lambda x: f"{x.min()} - {x.max()}"],
})

summary.columns = ['n_subjects', 'mean_age', 'range_age', 'mean_qi_total', 'range_qi_total', 'sex',
                   'ados_css_std', 'ados_css_mean', 'ados_css_range', 'qi_icv_std', 'qi_icv_mean', 'qi_icv_range',
                   'qi_irf_std', 'qi_irf_mean', 'qi_irf_range', 'qi_ivs_std', 'qi_ivs_mean', 'qi_ivs_range',
                   'qi_imt_std', 'qi_imt_mean', 'qi_imt_range', 'qi_ivt_std', 'qi_ivt_mean', 'qi_ivt_range',]

summary = summary.round(2)
summary = summary.reindex(['ASD', 'Controls', 'Relatives'])
summary = summary.T


agg_funcs = [
    ('mean', 'mean'),
    ('std', 'std'),
    ('range', lambda x: f"{x.min()} - {x.max()}")
]


cols = df.columns[279:321]
summary_pheno = df.groupby('group')[cols].agg([np.mean, np.std, np.ptp])


new_cols = []
for col in cols:
    new_cols.append(col + '_mean')
    new_cols.append(col + '_std')
    new_cols.append(col + '_range')

summary_pheno.columns = new_cols
summary_pheno= summary_pheno.T

summary_fEI = df.groupby('group')[['fEI', 'frontal_fEI', 'parietal_fEI', 'occipital_fEI', 'l_temporal_fEI', 'r_temporal_fEI']].agg(['mean', 'std']).T.round(2)



"""Plot """
colors = {'ASD': '#2c6fbb', 'Relatives': '#40a368', 'Controls': '#be0119'}
df_pl = df[['subject','group','fEI','age_years', 'sex','frontal_fEI','parietal_fEI','occipital_fEI','l_temporal_fEI','r_temporal_fEI','mean_roi']]
sns.pairplot(data=df_pl, hue='group', dropna=True, palette= 'muted')


df_filtered= df
#fEI average and fEi roi by groups 
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 8))
colors= {'ASD': '#ab1239',  'Controls':'#2c6fbb', 'Relatives': '#2bb179'}
colors3 = {'ASD': '#070d0d', 'Relatives': '#070d0d', 'Controls': '#070d0d'}

sns.boxplot(ax=axes[0, 0], data=df_filtered, x='group', y='fEI', palette=colors, hue='group', dodge=False, width=0.4,order=['ASD','Controls','Relatives'])
sns.swarmplot(ax=axes[0, 0], data=df_filtered, x='group', y='fEI', palette=colors3, hue='group',dodge=False, alpha=0.4,s=4,order=['ASD','Controls','Relatives'])
axes[0, 0].set_title('fEI total',fontweight='bold')

sns.boxplot(ax=axes[0, 1], data=df_filtered, x='group', y='frontal_fEI', palette=colors, hue='group',dodge=False, width=0.4,order=['ASD','Controls','Relatives'])
sns.swarmplot(ax=axes[0, 1], data=df_filtered, x='group', y='frontal_fEI', palette=colors3, hue='group',dodge=False, alpha=0.4,s=4,order=['ASD','Controls','Relatives'])
axes[0, 1].set_title('Frontal',fontweight='bold')

sns.boxplot(ax=axes[0, 2], data=df_filtered, x='group', y='parietal_fEI', palette=colors, hue='group',dodge=False, width=0.4,order=['ASD','Controls','Relatives'])
sns.swarmplot(ax=axes[0, 2], data=df_filtered, x='group', y='parietal_fEI', palette=colors3, hue='group',dodge=False, alpha=0.4,s=4,order=['ASD','Controls','Relatives'])
axes[0, 2].set_title('Parietal',fontweight='bold')

sns.boxplot(ax=axes[1, 0], data=df_filtered, x='group', y='occipital_fEI', palette=colors, hue='group',dodge=False, width=0.4,order=['ASD','Controls','Relatives'])
sns.swarmplot(ax=axes[1, 0], data=df_filtered,x='group', y='occipital_fEI', palette=colors3, hue='group',dodge=False, alpha=0.4,s=4,order=['ASD','Controls','Relatives'])
axes[1, 0].set_title('Occipital',fontweight='bold')

sns.boxplot(ax=axes[1, 1], data=df_filtered, x='group', y='l_temporal_fEI', palette=colors, hue='group',dodge=False, width=0.4,order=['ASD','Controls','Relatives'])
sns.swarmplot(ax=axes[1, 1], data=df_filtered, x='group', y='l_temporal_fEI', palette=colors3, hue='group',dodge=False, alpha=0.4,s=4,order=['ASD','Controls','Relatives'])
axes[1, 1].set_title('Left Temporal',fontweight='bold')

sns.boxplot(ax=axes[1, 2], data=df_filtered, x='group', y='r_temporal_fEI', palette=colors, hue='group',dodge=False, width=0.4,order=['ASD','Controls','Relatives'])
sns.swarmplot(ax=axes[1, 2], data=df_filtered,x='group',y='r_temporal_fEI', palette=colors3, hue='group',dodge=False, alpha=0.4,s=4,order=['ASD','Controls','Relatives'])
axes[1, 2].set_title('Right Temporal',fontweight='bold')

handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=len(labels), bbox_to_anchor=(0.5, 1.1))

for ax in axes.flat:
    ax.legend().remove()

plt.tight_layout()
plt.show()
       

#socio_demo 
df_filtered = df.loc[df['group'] == 'ASD']
df_filtered =df           
df_filtered = df[df['group'] != 'Relatives']

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 8))
colors= {'ASD': '#ab1239',  'Controls':'#2c6fbb', 'Relatives': '#2bb179'}

sns.scatterplot(ax=axes[0, 0], data=df_filtered, x='age_years', y='fEI', palette=colors, hue='group',s=10)
axes[0, 0].set_title('Age',fontweight='bold')

sns.scatterplot(ax=axes[0, 1], data=df_filtered, y='qi_icv', x='fEI', palette=colors, hue='group',s=10)
axes[0, 1].set_title('Verbal Communication Index',fontweight='bold')

sns.scatterplot(ax=axes[0, 2], data=df_filtered, x='fEI', y='qi_irf', palette=colors, hue='group',s=10)
axes[0, 2].set_title('Fluid Reasoning Index',fontweight='bold')

sns.scatterplot(ax=axes[1, 0], data=df_filtered, x='fEI', y='qi_ivs', palette=colors, hue='group',s=10)
axes[1, 0].set_title('Spatial Visual Index',fontweight='bold')

sns.scatterplot(ax=axes[1, 1], data=df_filtered, x='fEI', y='qi_ivt', palette=colors, hue='group',s=10)
axes[1, 1].set_title('Speed Processing Index',fontweight='bold')

sns.scatterplot(ax=axes[1, 2], data=df_filtered, x='fEI', y='qi_total', palette=colors, hue='group',s=10)
axes[1, 2].set_title('QI total',fontweight='bold')

handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=len(labels), bbox_to_anchor=(0.5, 1.1))

for ax in axes.flat:
    ax.legend().remove()

plt.tight_layout()
plt.show()
    

# Sensory 
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(24, 10))
colors = {'ASD': '#2c6fbb', 'Relatives': '#40a368', 'Controls': '#be0119'}

# Plot first boxplot
sns.scatterplot(ax=axes[0, 0], data=df_filtered, x='fEI', y='Tactile Sensitivity', palette=colors, hue='group',s=10)
axes[0, 0].set_title('Tactile Sensitivity',fontweight='bold')

# Plot second boxplot
sns.scatterplot(ax=axes[0, 1], data=df_filtered, x='fEI', y='Taste/Smell Sensitivity', palette=colors, hue='group',s=10)
axes[0, 1].set_title('Taste/Smell Sensitivity',fontweight='bold')

sns.scatterplot(ax=axes[0, 2], data=df_filtered, x='fEI', y='Movement Sensitivity', palette=colors, hue='group',s=10)
axes[0, 2].set_title('Movement Sensitivity',fontweight='bold')

sns.scatterplot(ax=axes[1, 0], data=df_filtered, x='fEI', y='Underresponsive/Seeks Sensation', palette=colors, hue='group',s=10)
axes[1, 0].set_title('Underresponsive/Seeks Sensation',fontweight='bold')

sns.scatterplot(ax=axes[1, 1], data=df_filtered, x='fEI', y='Auditory Filtering', palette=colors, hue='group',s=10)
axes[1, 1].set_title('Auditory Filtering',fontweight='bold')

sns.scatterplot(ax=axes[1, 2], data=df_filtered, x='fEI', y='Low Energy/Weak', palette=colors, hue='group',s=10)
axes[1, 2].set_title('Low Energy/Weak',fontweight='bold')

sns.scatterplot(ax=axes[2, 0], data=df_filtered, x='fEI', y='Visual Auditory Sensitivity', palette=colors, hue='group',s=10)
axes[2, 0].set_title('Visual Auditory Sensitivity',fontweight='bold')

sns.scatterplot(ax=axes[2, 1], data=df_filtered, x='fEI', y='hypo', palette=colors, hue='group',s=10)
axes[2, 1].set_title('Hyposensitivity',fontweight='bold')

sns.scatterplot(ax=axes[2, 2], data=df_filtered, x='fEI', y='hyper', palette=colors, hue='group',s=10)
axes[2, 2].set_title('Hypersensitivity',fontweight='bold')

# Create the legend outside the subplots
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=len(labels), bbox_to_anchor=(0.5, 1.1))

for ax in axes.flat:
    ax.legend().remove()

plt.tight_layout()
plt.show()


#SRS
df_filtered = df.loc[df['group'] == 'ASD']

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(24, 10))
colors = {'ASD': '#2c6fbb', 'Relatives': '#40a368', 'Controls': '#be0119'}
#colors = {'ASD': '#000000'}

sns.scatterplot(ax=axes[0, 0], data=df_filtered, x='fEI', y='srs_social_awareness_t', palette=colors, hue='group',s=8)
axes[0, 0].set_title('Social Awareness',fontweight='bold')

sns.scatterplot(ax=axes[0, 1], data=df_filtered, x='fEI', y='srs_social_cognition_t', palette=colors, hue='group',s=8)
axes[0, 1].set_title('Social Cognition',fontweight='bold')

sns.scatterplot(ax=axes[0, 2], data=df_filtered, x='fEI', y='srs_social_communication_t', palette=colors, hue='group',s=8)
axes[0, 2].set_title('Social Communication',fontweight='bold')

sns.scatterplot(ax=axes[1, 0], data=df_filtered, x='fEI', y='srs_social_motivation_t', palette=colors, hue='group',s=8)
axes[1, 0].set_title('Social Motivation',fontweight='bold')

sns.scatterplot(ax=axes[1, 1], data=df_filtered, x='fEI', y='srs_RRB_t', palette=colors, hue='group',s=8)
axes[1, 1].set_title('Repetitve and Restrained Behavior',fontweight='bold')

sns.scatterplot(ax=axes[1, 2], data=df_filtered, x='fEI', y='srs_total_t', palette=colors, hue='group',s=8)
axes[1, 2].set_title('Total SRS',fontweight='bold')

handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=len(labels), bbox_to_anchor=(0.5, 1.1))

for ax in axes.flat:
    ax.legend().remove()

plt.tight_layout()
plt.show()



#%% statistical analysis 
# Dabest
# create df for dabest
df['fEI_asd'] = df.loc[df['group'] == 'ASD', 'fEI']
df['fEI_frontal_asd'] = df.loc[df['group'] == 'ASD', 'frontal_fEI']
df['fEI_parietal_asd'] = df.loc[df['group'] == 'ASD', 'parietal_fEI']
df['fEI_occipital_asd'] = df.loc[df['group'] == 'ASD', 'occipital_fEI']
df['fEI_l_temporal_asd'] = df.loc[df['group'] == 'ASD', 'l_temporal_fEI']
df['fEI_r_temporal_asd'] = df.loc[df['group'] == 'ASD', 'r_temporal_fEI']
df['fEI_mean_roi_asd'] = df.loc[df['group'] == 'ASD', 'mean_roi']


df['fEI_controls'] = df.loc[df['group'] == 'Controls', 'fEI']
df['fEI_frontal_controls'] = df.loc[df['group'] == 'Controls', 'frontal_fEI']
df['fEI_parietal_controls'] = df.loc[df['group'] == 'Controls', 'parietal_fEI']
df['fEI_occipital_controls'] = df.loc[df['group'] == 'Controls', 'occipital_fEI']
df['fEI_l_temporal_controls'] = df.loc[df['group'] == 'Controls', 'l_temporal_fEI']
df['fEI_r_temporal_controls'] = df.loc[df['group'] == 'Controls', 'r_temporal_fEI']
df['fEI_mean_roi_controls'] = df.loc[df['group'] == 'Controls', 'mean_roi']


df['fEI_relatives'] = df.loc[df['group'] == 'Relatives', 'fEI']
df['fEI_frontal_relatives'] = df.loc[df['group'] == 'Relatives', 'frontal_fEI']
df['fEI_parietal_relatives'] = df.loc[df['group'] == 'Relatives', 'parietal_fEI']
df['fEI_occipital_relatives'] = df.loc[df['group'] == 'Relatives', 'occipital_fEI']
df['fEI_l_temporal_relatives'] = df.loc[df['group'] == 'Relatives', 'l_temporal_fEI']
df['fEI_r_temporal_relatives'] = df.loc[df['group'] == 'Relatives', 'r_temporal_fEI']
df['fEI_mean_roi_relatives'] = df.loc[df['group'] == 'Relatives', 'mean_roi']


df_test=df[['fEI_asd','fEI_frontal_asd','fEI_parietal_asd','fEI_occipital_asd',
          'fEI_l_temporal_asd','fEI_r_temporal_asd','fEI_controls',
          'fEI_frontal_controls','fEI_parietal_controls','fEI_occipital_controls',
          'fEI_l_temporal_controls','fEI_r_temporal_controls',
          'fEI_relatives','fEI_frontal_relatives','fEI_parietal_relatives',
          'fEI_occipital_relatives','fEI_l_temporal_relatives','fEI_r_temporal_relatives'
]]




# fEI average 
group_names = {
    'fEI_asd': 'ASD\n(n=118)',
    'fEI_controls': 'Controls\n(n=21)',
    'fEI_relatives': 'Relatives*\n(n=26)',
}

one_way = dabest.load(df_test, idx=('fEI_asd', 'fEI_controls', 'fEI_relatives'))
fig = one_way.hedges_g.plot(raw_marker_size=5, fig_size=(5, 6), swarm_label='fEI average', custom_palette=['#ab1239', '#2c6fbb','#2bb179'])
fig.suptitle("fEI average", fontsize=14, fontweight='bold')
# Set the x tick label names for the first subplot
ax1 = fig.get_axes()[0]
ax1.set_xticks(range(len(group_names)))
ax1.set_xticklabels(group_names.values())

# Set the x tick label names for the second subplot
ax2 = fig.get_axes()[1]
ax2.set_xticks(range(1,len(group_names)))
#ax2.set_xticklabels(['ASD - ' + g for g in group_names.values() if g != 'ASD'])
ax2.set_xticklabels(['ASD - Controls','ASD - Relatives'])

# Modify the plot dimensions
fig.subplots_adjust(top=0.945,
bottom=0.05,
left=0.155,
right=0.885,
hspace=0.2,
wspace=0.2)
plt.show()


#Frontal
group_names = {
    'fEI_frontal_asd': 'ASD\n(n=118)',
    'fEI_frontal_controls': 'Controls\n(n=21)',
    'fEI_frontal_relatives': 'Relatives*\n(n=26)'
}

frontal_dabest=dabest.load(df_test ,idx=('fEI_frontal_asd','fEI_frontal_controls','fEI_frontal_relatives'))
fig = frontal_dabest.hedges_g.plot(raw_marker_size=5, fig_size=(5, 6),swarm_label='Frontal fEI',custom_palette=['#ab1239', '#2c6fbb','#2bb179'])
fig.suptitle("Frontal fEI", fontsize=14, fontweight='bold')

ax1 = fig.get_axes()[0]
ax1.set_xticks(range(len(group_names)))
ax1.set_xticklabels(group_names.values())


ax2 = fig.get_axes()[1]
ax2.set_xticks(range(1,len(group_names)))
ax2.set_xticklabels(['ASD - Controls', 'ASD - Relatives' ])


fig.subplots_adjust(top=0.945,
bottom=0.05,
left=0.155,
right=0.885,
hspace=0.2,
wspace=0.2)
plt.show()

#Parietal
group_names = {
    'fEI_parietal_asd': 'ASD\n(n=118)',
    'fEI_parietal_controls': 'Controls\n(n=21)',
    'fEI_parietal_relatives': 'Relatives*\n(n=26)'
}

frontal_dabest=dabest.load(df_test ,idx=('fEI_parietal_asd','fEI_parietal_controls','fEI_parietal_relatives'))
fig = frontal_dabest.hedges_g.plot(raw_marker_size=5, fig_size=(5, 6),swarm_label='Parietal fEI',custom_palette=['#ab1239', '#2c6fbb','#2bb179'])
fig.suptitle("Parietal fEI", fontsize=14, fontweight='bold')
# Set the x tick label names for the first subplot
ax1 = fig.get_axes()[0]
ax1.set_xticks(range(len(group_names)))
ax1.set_xticklabels(group_names.values())

# Set the x tick label names for the second subplot
ax2 = fig.get_axes()[1]
ax2.set_xticks(range(1,len(group_names)))
ax2.set_xticklabels(['ASD - Controls', 'ASD - Relatives'])

# Modify the plot dimensions
fig.subplots_adjust(top=0.945,
bottom=0.05,
left=0.155,
right=0.885,
hspace=0.2,
wspace=0.2)
plt.show()


#Occipital 
group_names = {
    'fEI_occipital_asd': 'ASD\n(n=118)',
    'fEI_occipital_controls': 'Controls\n(n=21)',
    'fEI_occipital_relatives': 'Relatives*\n(n=26)'
}

frontal_dabest=dabest.load(df_test ,idx=('fEI_occipital_asd','fEI_occipital_controls','fEI_occipital_relatives'))
fig = frontal_dabest.hedges_g.plot(raw_marker_size=5, fig_size=(5, 6),swarm_label='Occipital fEI',custom_palette=['#ab1239', '#2c6fbb','#2bb179'])
fig.suptitle("Occipital fEI", fontsize=14, fontweight='bold')
# Set the x tick label names for the first subplot
ax1 = fig.get_axes()[0]
ax1.set_xticks(range(len(group_names)))
ax1.set_xticklabels(group_names.values())

# Set the x tick label names for the second subplot
ax2 = fig.get_axes()[1]
ax2.set_xticks(range(1,len(group_names)))
ax2.set_xticklabels(['ASD - Controls','ASD - Relatives'])

# Modify the plot dimensions
fig.subplots_adjust(top=0.945,
bottom=0.05,
left=0.155,
right=0.885,
hspace=0.2,
wspace=0.2)
plt.show()

#Left Temporal
group_names = {
    'fEI_l_temporal_asd': 'ASD\n(n=118)',
    'fEI_l_temporal_controls': 'Controls\n(n=21)',
    'fEI_l_temporal_relatives': 'Relatives*\n(n=26)'
}

frontal_dabest=dabest.load(df_test ,idx=('fEI_l_temporal_asd','fEI_l_temporal_controls','fEI_l_temporal_relatives'))
fig = frontal_dabest.hedges_g.plot(raw_marker_size=5, fig_size=(5, 6),swarm_label='Left Temporal fEI',custom_palette=['#ab1239', '#2c6fbb','#2bb179'])
fig.suptitle("Left Temporal fEI", fontsize=14, fontweight='bold')
# Set the x tick label names for the first subplot
ax1 = fig.get_axes()[0]
ax1.set_xticks(range(len(group_names)))
ax1.set_xticklabels(group_names.values())

# Set the x tick label names for the second subplot
ax2 = fig.get_axes()[1]
ax2.set_xticks(range(1,len(group_names)))
ax2.set_xticklabels(['ASD - Controls', 'ASD - Relatives'])

# Modify the plot dimensions
fig.subplots_adjust(top=0.945,
bottom=0.05,
left=0.155,
right=0.885,
hspace=0.2,
wspace=0.2)
plt.show()


# Right Temporal
group_names = {
    'fEI_r_temporal_asd': 'ASD\n(n=118)',
    'fEI_r_temporal_controls': 'Controls\n(n=21)',
    'fEI_r_temporal_relatives': 'Relatives*\n(n=26)'
}

frontal_dabest=dabest.load(df_test ,idx=('fEI_r_temporal_asd','fEI_r_temporal_controls','fEI_r_temporal_relatives'))
fig = frontal_dabest.hedges_g.plot(raw_marker_size=5, fig_size=(5, 6),swarm_label='Right Temporal fEI',custom_palette=['#ab1239', '#2c6fbb','#2bb179'])
fig.suptitle("Right Temporal fEI", fontsize=14, fontweight='bold')
ax1 = fig.get_axes()[0]
ax1.set_xticks(range(len(group_names)))
ax1.set_xticklabels(group_names.values())

ax2 = fig.get_axes()[1]
ax2.set_xticks(range(1,len(group_names)))
ax2.set_xticklabels(['ASD - Controls', 'ASD - Relatives'])

fig.subplots_adjust(top=0.945,
bottom=0.05,
left=0.155,
right=0.885,
hspace=0.2,
wspace=0.2)
plt.show()







#%% Pingouin stats 
df_h = df[df['group'] != 'Relatives']
# t-test 
df_t = df[df['group'] != 'Relatives']

# fEI
normality = pg.normality(df_t, dv='fEI', group='group').round(3)
print(normality)
ttest =pg.ttest(df_t['fEI_asd'], df_t['fEI_controls']).round(3)
print(ttest)
# if normality = false do mwu :
fEI_mwu= pg.mwu(df_t['fEI_asd'], df_t['fEI_controls'], alternative='two-sided').round(3)
print(fEI_mwu) 


# frontal_fEI
normality = pg.normality(df_t, dv='frontal_fEI', group='group')
print(normality)
frontal_ttest =pg.ttest(df_t['fEI_frontal_asd'], df_t['fEI_frontal_controls'], correction=True).round(2)
print(frontal_ttest)
#


# parietal_fEI  
normality = pg.normality(df_t, dv='parietal_fEI', group='group')
print(normality)
parietal_ttest =pg.ttest(df_t['fEI_parietal_asd'], df_t['fEI_parietal_controls'], correction=True).round(2)
print(parietal_ttest)

# occipital_fEI
normality = pg.normality(df_t, dv='occipital_fEI', group='group')
print(normality)
occipital_ttest =pg.ttest(df_t['fEI_occipital_asd'], df_t['fEI_occipital_controls'], correction=True).round(2)
print(occipital_ttest)

# l_temporal_fEI 
normality = pg.normality(df_t, dv='l_temporal_fEI', group='group')
print(normality)
l_temporal_ttest =pg.ttest(df_t['fEI_l_temporal_asd'], df_t['fEI_l_temporal_controls'], correction=True).round(2)
print(l_temporal_ttest)


# r_temporal_fEI 
normality = pg.normality(df_t, dv='r_temporal_fEI', group='group')
print(normality)
r_temporal_ttest =pg.ttest(df_t['fEI_r_temporal_asd'], df_t['fEI_r_temporal_controls'], correction=True).round(2)
print(r_temporal_ttest)

#FDR 
pvals = [.....]
reject, pvals_corr = pg.multicomp(pvals, method='fdr_bh')
print(reject, pvals_corr)

# create excel table with t-test results 
ttest_results = pd.concat([ttest, frontal_ttest, parietal_ttest, occipital_ttest, l_temporal_ttest, r_temporal_ttest], axis=0)

                      
 
df_t = df[df['group'] != 'Relatives']           
#ANCOVA 
#fEI
fEI_ancova =pg.ancova(df_t, dv='fEI', between='group',covar=['age_years']).round(3)
print(fEI_ancova)
fEI_pairwise = df_t.pairwise_tukey(dv='fEI', between='group').round(3)
print(fEI_pairwise)

#mean roi
roi_ancova =pg.ancova(df_t, dv='mean_roi', between='group',covar=['age_years']).round(3)
print(roi_ancova)
roi_pairwise = df_t.pairwise_tukey(dv='mean_roi', between='group').round(3)
print(roi_pairwise)

#frontal
frontal_ancova =pg.ancova(df_t, dv='frontal_fEI', between='group',covar=['age_years']).round(3)
print(frontal_ancova)
frontal_pairwise = df_t.pairwise_tukey(dv='frontal_fEI', between='group').round(3)
print(frontal_pairwise)

#parietal
parietal_ancova =pg.ancova(df_t, dv='parietal_fEI', between='group',covar=['age_years']).round(3)
print(parietal_ancova)
parietal_pairwise = df_t.pairwise_tukey(dv='parietal_fEI', between='group').round(3)
print(parietal_pairwise)

#occipital
occipital_ancova =pg.ancova(df_t, dv='occipital_fEI', between='group',covar=['age_years']).round(3)
print(occipital_ancova)
occipital_pairwise = df_t.pairwise_tukey(dv='occipital_fEI', between='group').round(3)
print(occipital_pairwise)

#l_temporal 
l_temporal_ancova =pg.ancova(df_t, dv='l_temporal_fEI', between='group',covar=['age_years']).round(3)
print(l_temporal_ancova)
l_temporal_pairwise = df_t.pairwise_tukey(dv='l_temporal_fEI', between='group').round(3)
print(l_temporal_pairwise)

#r_temporal 
r_temporal_ancova =pg.ancova(df_t, dv='r_temporal_fEI', between='group',covar=['age_years']).round(3)
print(r_temporal_ancova)
r_temporal_pairwise = df_t.pairwise_tukey(dv='l_temporal_fEI', between='group').round(3)
print(r_temporal_pairwise)


#FDR 
pvals = [......]
reject, pvals_corr = pg.multicomp(pvals, method='fdr_bh')
print(reject, pvals_corr)



#Linear regression for association with clinical variables 
df_h= df[df['group'] == 'ASD']


#Lm age
# Linear model 
regions = ['fEI', 'frontal_fEI', 'parietal_fEI', 'occipital_fEI', 'l_temporal_fEI', 'r_temporal_fEI']
results = pd.DataFrame(columns=['Region', 'Model', 'Intercept', 'Slope', 'Slope^2', 'p-value', 'R-squared', 'AIC'])

for region in regions:
    x = df_h['age_years']
    y = df_h[region]
    x_linear = sm.add_constant(x)  # Linear model: Add a constant term for the intercept
    
    # Linear regression model
    model_linear = sm.OLS(y, x_linear, missing='drop')
    results_summary_linear = model_linear.fit()
    aic_linear = results_summary_linear.aic
    
    # Polynomial regression model (second order)
    x_poly = np.column_stack((x, x**3))  # Add polynomial terms
    x_poly = sm.add_constant(x_poly)  # Add a constant term for the intercept
    
    model_poly = sm.OLS(y, x_poly, missing='drop')
    results_summary_poly = model_poly.fit()
    aic_poly = results_summary_poly.aic
    
    results = results.append({
        'Region': region,
        'Model': 'Linear',
        'Intercept': results_summary_linear.params[0],
        'Slope': results_summary_linear.params[1],
        'Slope^2': '-',
        'p-value': results_summary_linear.pvalues[1],
        'R-squared': results_summary_linear.rsquared,
        'AIC': aic_linear
    }, ignore_index=True)
    
    results = results.append({
        'Region': region,
        'Model': 'Polynomial',
        'Intercept': results_summary_poly.params[0],
        'Slope': results_summary_poly.params[1],
        'Slope^2': results_summary_poly.params[2],
        'p-value': results_summary_poly.pvalues[2],
        'R-squared': results_summary_poly.rsquared,
        'AIC': results_summary_poly.aic
    }, ignore_index=True)

print(results)

#FDR 
pvals = [....]
reject, pvals_corr = pg.multicomp(pvals, method='fdr_bh')
print(reject, pvals_corr)





#Lm QI total 
regions = ['fEI', 'frontal_fEI', 'parietal_fEI', 'occipital_fEI', 'l_temporal_fEI', 'r_temporal_fEI']
results = pd.DataFrame(columns=['Region', 'Intercept', 'Slope', 'R-squared', 'p-value'])
for region in regions:
    x = df_h['qi_total']
    y = df_h[region]
    x = sm.add_constant(x)  # Add a constant term for the intercept
    
    model = sm.OLS(y, x, missing='drop')
    results_summary = model.fit()
    
    results = results.append({
        'Region': region,
        'Intercept': results_summary.params[0],
        'Slope': results_summary.params[1],
        'R-squared': results_summary.rsquared,
        'p-value': results_summary.pvalues[1]
    }, ignore_index=True).round(2)
print(results)

#FDR 
pvals = [....]
reject, pvals_corr = pg.multicomp(pvals, method='fdr_bh')
print(reject, pvals_corr)


regions = ['fEI', 'frontal_fEI', 'parietal_fEI', 'occipital_fEI', 'l_temporal_fEI', 'r_temporal_fEI']
results = pd.DataFrame(columns=['Region', 'Model', 'Intercept', 'Slope', 'Slope^2', 'p-value', 'R-squared', 'AIC'])

for region in regions:
    x = df_h['qi_total']
    y = df_h[region]
    x_linear = sm.add_constant(x)  # Linear model: Add a constant term for the intercept
    
    # Linear regression model
    model_linear = sm.OLS(y, x_linear, missing='drop')
    results_summary_linear = model_linear.fit()
    aic_linear = results_summary_linear.aic
    
    # Polynomial regression model (second order)
    x_poly = np.column_stack((x, x**3))  # Add polynomial terms
    x_poly = sm.add_constant(x_poly)  # Add a constant term for the intercept
    
    model_poly = sm.OLS(y, x_poly, missing='drop')
    results_summary_poly = model_poly.fit()
    aic_poly = results_summary_poly.aic
    
    results = results.append({
        'Region': region,
        'Model': 'Linear',
        'Intercept': results_summary_linear.params[0],
        'Slope': results_summary_linear.params[1],
        'Slope^2': '-',
        'p-value': results_summary_linear.pvalues[1],
        'R-squared': results_summary_linear.rsquared,
        'AIC': aic_linear
    }, ignore_index=True)
    
    results = results.append({
        'Region': region,
        'Model': 'Polynomial',
        'Intercept': results_summary_poly.params[0],
        'Slope': results_summary_poly.params[1],
        'Slope^2': results_summary_poly.params[2],
        'p-value': results_summary_poly.pvalues[2],
        'R-squared': results_summary_poly.rsquared,
        'AIC': results_summary_poly.aic
    }, ignore_index=True)

print(results)


#Lm QI IRF
regions = ['fEI', 'frontal_fEI', 'parietal_fEI', 'occipital_fEI', 'l_temporal_fEI', 'r_temporal_fEI']
results = pd.DataFrame(columns=['Region', 'Intercept', 'Slope', 'R-squared', 'p-value'])
for region in regions:
    x = df_h['qi_irf']
    y = df_h[region]
    x = sm.add_constant(x)  # Add a constant term for the intercept
    
    model = sm.OLS(y, x, missing='drop')
    results_summary = model.fit()
    #print(results_summary.summary())
    results = results.append({
        'Region': region,
        'Intercept': results_summary.params[0],
        'Slope': results_summary.params[1],
        'R-squared': results_summary.rsquared,
        'p-value': results_summary.pvalues[1]
    }, ignore_index=True).round(2)
print(results)


regions = ['fEI', 'frontal_fEI', 'parietal_fEI', 'occipital_fEI', 'l_temporal_fEI', 'r_temporal_fEI']
results = pd.DataFrame(columns=['Region', 'Model', 'Intercept', 'Slope', 'Slope^2', 'p-value', 'R-squared', 'AIC'])

for region in regions:
    x = df_h['qi_irf']
    y = df_h[region]
    x_linear = sm.add_constant(x)  # Linear model: Add a constant term for the intercept
    
    # Linear regression model
    model_linear = sm.OLS(y, x_linear, missing='drop')
    results_summary_linear = model_linear.fit()
    aic_linear = results_summary_linear.aic
    
    # Polynomial regression model (second order)
    x_poly = np.column_stack((x, x**3))  # Add polynomial terms
    x_poly = sm.add_constant(x_poly)  # Add a constant term for the intercept
    
    model_poly = sm.OLS(y, x_poly, missing='drop')
    results_summary_poly = model_poly.fit()
    aic_poly = results_summary_poly.aic
    
    results = results.append({
        'Region': region,
        'Model': 'Linear',
        'Intercept': results_summary_linear.params[0],
        'Slope': results_summary_linear.params[1],
        'Slope^2': '-',
        'p-value': results_summary_linear.pvalues[1],
        'R-squared': results_summary_linear.rsquared,
        'AIC': aic_linear
    }, ignore_index=True)
    
    results = results.append({
        'Region': region,
        'Model': 'Polynomial',
        'Intercept': results_summary_poly.params[0],
        'Slope': results_summary_poly.params[1],
        'Slope^2': results_summary_poly.params[2],
        'p-value': results_summary_poly.pvalues[2],
        'R-squared': results_summary_poly.rsquared,
        'AIC': results_summary_poly.aic
    }, ignore_index=True)

print(results)

#Lm QI ICV
regions = ['fEI', 'frontal_fEI', 'parietal_fEI', 'occipital_fEI', 'l_temporal_fEI', 'r_temporal_fEI']
results = pd.DataFrame(columns=['Region', 'Intercept', 'Slope', 'R-squared', 'p-value'])
for region in regions:
    x = df_h['qi_icv']
    y = df_h[region]
    x = sm.add_constant(x)  # Add a constant term for the intercept
    
    model = sm.OLS(y, x, missing='drop')
    results_summary = model.fit()
    
    results = results.append({
        'Region': region,
        'Intercept': results_summary.params[0],
        'Slope': results_summary.params[1],
        'R-squared': results_summary.rsquared,
        'p-value': results_summary.pvalues[1]
    }, ignore_index=True).round(2)
print(results)

regions = ['fEI', 'frontal_fEI', 'parietal_fEI', 'occipital_fEI', 'l_temporal_fEI', 'r_temporal_fEI']
results = pd.DataFrame(columns=['Region', 'Model', 'Intercept', 'Slope', 'Slope^2', 'p-value', 'R-squared', 'AIC'])
for region in regions:
    x = df_h['qi_icv']
    y = df_h[region]
    x_linear = sm.add_constant(x)  # Linear model: Add a constant term for the intercept
    
    # Linear regression model
    model_linear = sm.OLS(y, x_linear, missing='drop')
    results_summary_linear = model_linear.fit()
    aic_linear = results_summary_linear.aic
    
    # Polynomial regression model (second order)
    x_poly = np.column_stack((x, x**3))  # Add polynomial terms
    x_poly = sm.add_constant(x_poly)  # Add a constant term for the intercept
    
    model_poly = sm.OLS(y, x_poly, missing='drop')
    results_summary_poly = model_poly.fit()
    aic_poly = results_summary_poly.aic
    
    results = results.append({
        'Region': region,
        'Model': 'Linear',
        'Intercept': results_summary_linear.params[0],
        'Slope': results_summary_linear.params[1],
        'Slope^2': '-',
        'p-value': results_summary_linear.pvalues[1],
        'R-squared': results_summary_linear.rsquared,
        'AIC': aic_linear
    }, ignore_index=True)
    
    results = results.append({
        'Region': region,
        'Model': 'Polynomial',
        'Intercept': results_summary_poly.params[0],
        'Slope': results_summary_poly.params[1],
        'Slope^2': results_summary_poly.params[2],
        'p-value': results_summary_poly.pvalues[2],
        'R-squared': results_summary_poly.rsquared,
        'AIC': results_summary_poly.aic
    }, ignore_index=True)





#Lm QI IVS
regions = ['fEI', 'frontal_fEI', 'parietal_fEI', 'occipital_fEI', 'l_temporal_fEI', 'r_temporal_fEI']
results = pd.DataFrame(columns=['Region', 'Intercept', 'Slope', 'R-squared', 'p-value'])
for region in regions:
    x = df_h['qi_ivs']
    y = df_h[region]
    x = sm.add_constant(x)  # Add a constant term for the intercept
    
    model = sm.OLS(y, x, missing='drop')
    results_summary = model.fit()
    
    results = results.append({
        'Region': region,
        'Intercept': results_summary.params[0],
        'Slope': results_summary.params[1],
        'R-squared': results_summary.rsquared,
        'p-value': results_summary.pvalues[1]
    }, ignore_index=True).round(2)
print(results)


regions = ['fEI', 'frontal_fEI', 'parietal_fEI', 'occipital_fEI', 'l_temporal_fEI', 'r_temporal_fEI']
results = pd.DataFrame(columns=['Region', 'Model', 'Intercept', 'Slope', 'Slope^2', 'p-value', 'R-squared', 'AIC'])

for region in regions:
    x = df_h['qi_ivs']
    y = df_h[region]
    x_linear = sm.add_constant(x)  # Linear model: Add a constant term for the intercept
    
    # Linear regression model
    model_linear = sm.OLS(y, x_linear, missing='drop')
    results_summary_linear = model_linear.fit()
    aic_linear = results_summary_linear.aic
    
    # Polynomial regression model (second order)
    x_poly = np.column_stack((x, x**3))  # Add polynomial terms
    x_poly = sm.add_constant(x_poly)  # Add a constant term for the intercept
    
    model_poly = sm.OLS(y, x_poly, missing='drop')
    results_summary_poly = model_poly.fit()
    aic_poly = results_summary_poly.aic
    
    results = results.append({
        'Region': region,
        'Model': 'Linear',
        'Intercept': results_summary_linear.params[0],
        'Slope': results_summary_linear.params[1],
        'Slope^2': '-',
        'p-value': results_summary_linear.pvalues[1],
        'R-squared': results_summary_linear.rsquared,
        'AIC': aic_linear
    }, ignore_index=True)
    
    results = results.append({
        'Region': region,
        'Model': 'Polynomial',
        'Intercept': results_summary_poly.params[0],
        'Slope': results_summary_poly.params[1],
        'Slope^2': results_summary_poly.params[2],
        'p-value': results_summary_poly.pvalues[2],
        'R-squared': results_summary_poly.rsquared,
        'AIC': results_summary_poly.aic
    }, ignore_index=True)



#Lm QI IMT 
regions = ['fEI', 'frontal_fEI', 'parietal_fEI', 'occipital_fEI', 'l_temporal_fEI', 'r_temporal_fEI']
results = pd.DataFrame(columns=['Region', 'Intercept', 'Slope', 'R-squared', 'p-value'])
for region in regions:
    x = df_h['qi_imt']
    y = df_h[region]
    x = sm.add_constant(x)  # Add a constant term for the intercept
    
    model = sm.OLS(y, x, missing='drop')
    results_summary = model.fit()
    #print(results_summary.summary())    
    results = results.append({
        'Region': region,
        'Intercept': results_summary.params[0],
        'Slope': results_summary.params[1],
        'R-squared': results_summary.rsquared,
        'p-value': results_summary.pvalues[1]
    }, ignore_index=True).round(2)
print(results)


regions = ['fEI', 'frontal_fEI', 'parietal_fEI', 'occipital_fEI', 'l_temporal_fEI', 'r_temporal_fEI']
results = pd.DataFrame(columns=['Region', 'Model', 'Intercept', 'Slope', 'Slope^2', 'p-value', 'R-squared', 'AIC'])

for region in regions:
    x = df_h['qi_imt']
    y = df_h[region]
    x_linear = sm.add_constant(x)  # Linear model: Add a constant term for the intercept
    
    # Linear regression model
    model_linear = sm.OLS(y, x_linear, missing='drop')
    results_summary_linear = model_linear.fit()
    aic_linear = results_summary_linear.aic
    
    # Polynomial regression model (second order)
    x_poly = np.column_stack((x, x**3))  # Add polynomial terms
    x_poly = sm.add_constant(x_poly)  # Add a constant term for the intercept
    
    model_poly = sm.OLS(y, x_poly, missing='drop')
    results_summary_poly = model_poly.fit()
    aic_poly = results_summary_poly.aic
    
    results = results.append({
        'Region': region,
        'Model': 'Linear',
        'Intercept': results_summary_linear.params[0],
        'Slope': results_summary_linear.params[1],
        'Slope^2': '-',
        'p-value': results_summary_linear.pvalues[1],
        'R-squared': results_summary_linear.rsquared,
        'AIC': aic_linear
    }, ignore_index=True)
    
    results = results.append({
        'Region': region,
        'Model': 'Polynomial',
        'Intercept': results_summary_poly.params[0],
        'Slope': results_summary_poly.params[1],
        'Slope^2': results_summary_poly.params[2],
        'p-value': results_summary_poly.pvalues[2],
        'R-squared': results_summary_poly.rsquared,
        'AIC': results_summary_poly.aic
    }, ignore_index=True)


#Lm QI IVT 
regions = ['fEI', 'frontal_fEI', 'parietal_fEI', 'occipital_fEI', 'l_temporal_fEI', 'r_temporal_fEI']
results = pd.DataFrame(columns=['Region', 'Intercept', 'Slope', 'R-squared', 'p-value'])
for region in regions:
    x = df_h['qi_ivt']
    y = df_h[region]
    x = sm.add_constant(x)  # Add a constant term for the intercept
    
    model = sm.OLS(y, x, missing='drop')
    results_summary = model.fit()
    
    results = results.append({
        'Region': region,
        'Intercept': results_summary.params[0],
        'Slope': results_summary.params[1],
        'R-squared': results_summary.rsquared,
        'p-value': results_summary.pvalues[1]
    }, ignore_index=True).round(2)
print(results)

regions = ['fEI', 'frontal_fEI', 'parietal_fEI', 'occipital_fEI', 'l_temporal_fEI', 'r_temporal_fEI']
results = pd.DataFrame(columns=['Region', 'Model', 'Intercept', 'Slope', 'Slope^2', 'p-value', 'R-squared', 'AIC'])

for region in regions:
    x = df_h['qi_ivt']
    y = df_h[region]
    x_linear = sm.add_constant(x)  # Linear model: Add a constant term for the intercept
    
    # Linear regression model
    model_linear = sm.OLS(y, x_linear, missing='drop')
    results_summary_linear = model_linear.fit()
    aic_linear = results_summary_linear.aic
    
    # Polynomial regression model (second order)
    x_poly = np.column_stack((x, x**3))  # Add polynomial terms
    x_poly = sm.add_constant(x_poly)  # Add a constant term for the intercept
    
    model_poly = sm.OLS(y, x_poly, missing='drop')
    results_summary_poly = model_poly.fit()
    aic_poly = results_summary_poly.aic
    
    results = results.append({
        'Region': region,
        'Model': 'Linear',
        'Intercept': results_summary_linear.params[0],
        'Slope': results_summary_linear.params[1],
        'Slope^2': '-',
        'p-value': results_summary_linear.pvalues[1],
        'R-squared': results_summary_linear.rsquared,
        'AIC': aic_linear
    }, ignore_index=True)
    
    results = results.append({
        'Region': region,
        'Model': 'Polynomial',
        'Intercept': results_summary_poly.params[0],
        'Slope': results_summary_poly.params[1],
        'Slope^2': results_summary_poly.params[2],
        'p-value': results_summary_poly.pvalues[2],
        'R-squared': results_summary_poly.rsquared,
        'AIC': results_summary_poly.aic
    }, ignore_index=True)


#Lm Hyper 
regions = ['fEI', 'frontal_fEI', 'parietal_fEI', 'occipital_fEI', 'l_temporal_fEI', 'r_temporal_fEI']
results = pd.DataFrame(columns=['Region', 'Intercept', 'Slope', 'R-squared', 'p-value'])
for region in regions:
    x = df_h['hyper']
    y = df_h[region]
    x = sm.add_constant(x)  # Add a constant term for the intercept
    
    model = sm.OLS(y, x, missing='drop')
    results_summary = model.fit()
    
    results = results.append({
        'Region': region,
        'Intercept': results_summary.params[0],
        'Slope': results_summary.params[1],
        'R-squared': results_summary.rsquared,
        'p-value': results_summary.pvalues[1]
    }, ignore_index=True).round(2)
print(results)

regions = ['fEI', 'frontal_fEI', 'parietal_fEI', 'occipital_fEI', 'l_temporal_fEI', 'r_temporal_fEI']
results = pd.DataFrame(columns=['Region', 'Model', 'Intercept', 'Slope', 'Slope^2', 'p-value', 'R-squared', 'AIC'])

for region in regions:
    x = df_h['hyper']
    y = df_h[region]
    x_linear = sm.add_constant(x)  # Linear model: Add a constant term for the intercept
    
    # Linear regression model
    model_linear = sm.OLS(y, x_linear, missing='drop')
    results_summary_linear = model_linear.fit()
    aic_linear = results_summary_linear.aic
    
    # Polynomial regression model (second order)
    x_poly = np.column_stack((x, x**3))  # Add polynomial terms
    x_poly = sm.add_constant(x_poly)  # Add a constant term for the intercept
    
    model_poly = sm.OLS(y, x_poly, missing='drop')
    results_summary_poly = model_poly.fit()
    aic_poly = results_summary_poly.aic
    
    results = results.append({
        'Region': region,
        'Model': 'Linear',
        'Intercept': results_summary_linear.params[0],
        'Slope': results_summary_linear.params[1],
        'Slope^2': '-',
        'p-value': results_summary_linear.pvalues[1],
        'R-squared': results_summary_linear.rsquared,
        'AIC': aic_linear
    }, ignore_index=True)
    
    results = results.append({
        'Region': region,
        'Model': 'Polynomial',
        'Intercept': results_summary_poly.params[0],
        'Slope': results_summary_poly.params[1],
        'Slope^2': results_summary_poly.params[2],
        'p-value': results_summary_poly.pvalues[2],
        'R-squared': results_summary_poly.rsquared,
        'AIC': results_summary_poly.aic
    }, ignore_index=True)


#Lm Hypo 
regions = ['fEI', 'frontal_fEI', 'parietal_fEI', 'occipital_fEI', 'l_temporal_fEI', 'r_temporal_fEI']
results = pd.DataFrame(columns=['Region', 'Intercept', 'Slope', 'R-squared', 'p-value'])
for region in regions:
    x = df_h['hypo']
    y = df_h[region]
    x = sm.add_constant(x)  # Add a constant term for the intercept
    
    model = sm.OLS(y, x, missing='drop')
    results_summary = model.fit()
    
    results = results.append({
        'Region': region,
        'Intercept': results_summary.params[0],
        'Slope': results_summary.params[1],
        'R-squared': results_summary.rsquared,
        'p-value': results_summary.pvalues[1]
    }, ignore_index=True).round(2)
print(results)
results.to_excel('/Users/julienpichot/Documents/fEI_2023/figure/5s/significatif_lm/lm_hypo.xlsx', index=True )

pvals = [.14, .04, .06, .7, 0.3, 0.1]
reject, pvals_corr = pg.multicomp(pvals, method='fdr_bh')
print(reject, pvals_corr)


regions = ['fEI', 'frontal_fEI', 'parietal_fEI', 'occipital_fEI', 'l_temporal_fEI', 'r_temporal_fEI']
results = pd.DataFrame(columns=['Region', 'Model', 'Intercept', 'Slope', 'Slope^2', 'p-value', 'R-squared', 'AIC'])

for region in regions:
    x = df_h['hypo']
    y = df_h[region]
    x_linear = sm.add_constant(x)  # Linear model: Add a constant term for the intercept
    
    # Linear regression model
    model_linear = sm.OLS(y, x_linear, missing='drop')
    results_summary_linear = model_linear.fit()
    aic_linear = results_summary_linear.aic
    
    # Polynomial regression model (second order)
    x_poly = np.column_stack((x, x**3))  # Add polynomial terms
    x_poly = sm.add_constant(x_poly)  # Add a constant term for the intercept
    
    model_poly = sm.OLS(y, x_poly, missing='drop')
    results_summary_poly = model_poly.fit()
    aic_poly = results_summary_poly.aic
    
    results = results.append({
        'Region': region,
        'Model': 'Linear',
        'Intercept': results_summary_linear.params[0],
        'Slope': results_summary_linear.params[1],
        'Slope^2': '-',
        'p-value': results_summary_linear.pvalues[1],
        'R-squared': results_summary_linear.rsquared,
        'AIC': aic_linear
    }, ignore_index=True)
    
    results = results.append({
        'Region': region,
        'Model': 'Polynomial',
        'Intercept': results_summary_poly.params[0],
        'Slope': results_summary_poly.params[1],
        'Slope^2': results_summary_poly.params[2],
        'p-value': results_summary_poly.pvalues[2],
        'R-squared': results_summary_poly.rsquared,
        'AIC': results_summary_poly.aic
    }, ignore_index=True)


sns.regplot(x=df_h['r_temporal_fEI'],y=df_h['hyper'],scatter_kws={'s': 10},order=2)
sns.regplot(x=df_h['frontal_fEI'],y=df_h['hyper'],scatter_kws={'s': 10},order=2)



#Lm SRS total 
regions = ['fEI', 'frontal_fEI', 'parietal_fEI', 'occipital_fEI', 'l_temporal_fEI', 'r_temporal_fEI']
results = pd.DataFrame(columns=['Region', 'Intercept', 'Slope', 'R-squared', 'p-value'])
for region in regions:
    x = df_h['srs_total_t']
    y = df_h[region]
    x = sm.add_constant(x)  # Add a constant term for the intercept
    
    model = sm.OLS(y, x, missing='drop')
    results_summary = model.fit()
    
    results = results.append({
        'Region': region,
        'Intercept': results_summary.params[0],
        'Slope': results_summary.params[1],
        'R-squared': results_summary.rsquared,
        'p-value': results_summary.pvalues[1]
    }, ignore_index=True).round(2)
print(results)
                      
pvals = [.....]
reject, pvals_corr = pg.multicomp(pvals, method='fdr_bh')
print(reject, pvals_corr)


regions = ['fEI', 'frontal_fEI', 'parietal_fEI', 'occipital_fEI', 'l_temporal_fEI', 'r_temporal_fEI']
results = pd.DataFrame(columns=['Region', 'Model', 'Intercept', 'Slope', 'Slope^2', 'p-value', 'R-squared', 'AIC'])

for region in regions:
    x = df_h['srs_total_t']
    y = df_h[region]
    x_linear = sm.add_constant(x)  # Linear model: Add a constant term for the intercept
    
    # Linear regression model
    model_linear = sm.OLS(y, x_linear, missing='drop')
    results_summary_linear = model_linear.fit()
    aic_linear = results_summary_linear.aic
    
    # Polynomial regression model (second order)
    x_poly = np.column_stack((x, x**3))  # Add polynomial terms
    x_poly = sm.add_constant(x_poly)  # Add a constant term for the intercept
    
    model_poly = sm.OLS(y, x_poly, missing='drop')
    results_summary_poly = model_poly.fit()
    aic_poly = results_summary_poly.aic
    
    results = results.append({
        'Region': region,
        'Model': 'Linear',
        'Intercept': results_summary_linear.params[0],
        'Slope': results_summary_linear.params[1],
        'Slope^2': '-',
        'p-value': results_summary_linear.pvalues[1],
        'R-squared': results_summary_linear.rsquared,
        'AIC': aic_linear
    }, ignore_index=True)
    
    results = results.append({
        'Region': region,
        'Model': 'Polynomial',
        'Intercept': results_summary_poly.params[0],
        'Slope': results_summary_poly.params[1],
        'Slope^2': results_summary_poly.params[2],
        'p-value': results_summary_poly.pvalues[2],
        'R-squared': results_summary_poly.rsquared,
        'AIC': results_summary_poly.aic
    }, ignore_index=True)


#Lm SRS social motivation
regions = ['fEI', 'frontal_fEI', 'parietal_fEI', 'occipital_fEI', 'l_temporal_fEI', 'r_temporal_fEI']
results = pd.DataFrame(columns=['Region', 'Model', 'Intercept', 'Slope', 'Slope^2', 'p-value', 'R-squared', 'AIC'])

for region in regions:
    x = df_h['srs_social_motivation_t']
    y = df_h[region]
    x_linear = sm.add_constant(x)  # Linear model: Add a constant term for the intercept
    
    # Linear regression model
    model_linear = sm.OLS(y, x_linear, missing='drop')
    results_summary_linear = model_linear.fit()
    aic_linear = results_summary_linear.aic
    
    # Polynomial regression model (second order)
    x_poly = np.column_stack((x, x**3))  # Add polynomial terms
    x_poly = sm.add_constant(x_poly)  # Add a constant term for the intercept
    
    model_poly = sm.OLS(y, x_poly, missing='drop')
    results_summary_poly = model_poly.fit()
    aic_poly = results_summary_poly.aic
    
    results = results.append({
        'Region': region,
        'Model': 'Linear',
        'Intercept': results_summary_linear.params[0],
        'Slope': results_summary_linear.params[1],
        'Slope^2': '-',
        'p-value': results_summary_linear.pvalues[1],
        'R-squared': results_summary_linear.rsquared,
        'AIC': aic_linear
    }, ignore_index=True)
    
    results = results.append({
        'Region': region,
        'Model': 'Polynomial',
        'Intercept': results_summary_poly.params[0],
        'Slope': results_summary_poly.params[1],
        'Slope^2': results_summary_poly.params[2],
        'p-value': results_summary_poly.pvalues[2],
        'R-squared': results_summary_poly.rsquared,
        'AIC': results_summary_poly.aic
    }, ignore_index=True)





#Lm SRS social communication
regions = ['fEI', 'frontal_fEI', 'parietal_fEI', 'occipital_fEI', 'l_temporal_fEI', 'r_temporal_fEI']
results = pd.DataFrame(columns=['Region', 'Model', 'Intercept', 'Slope', 'Slope^2', 'p-value', 'R-squared', 'AIC'])

for region in regions:
    x = df_h['srs_social_communication_t']
    y = df_h[region]
    x_linear = sm.add_constant(x)  # Linear model: Add a constant term for the intercept
    
    # Linear regression model
    model_linear = sm.OLS(y, x_linear, missing='drop')
    results_summary_linear = model_linear.fit()
    aic_linear = results_summary_linear.aic
    
    # Polynomial regression model (second order)
    x_poly = np.column_stack((x, x**3))  # Add polynomial terms
    x_poly = sm.add_constant(x_poly)  # Add a constant term for the intercept
    
    model_poly = sm.OLS(y, x_poly, missing='drop')
    results_summary_poly = model_poly.fit()
    aic_poly = results_summary_poly.aic
    
    results = results.append({
        'Region': region,
        'Model': 'Linear',
        'Intercept': results_summary_linear.params[0],
        'Slope': results_summary_linear.params[1],
        'Slope^2': '-',
        'p-value': results_summary_linear.pvalues[1],
        'R-squared': results_summary_linear.rsquared,
        'AIC': aic_linear
    }, ignore_index=True)
    
    results = results.append({
        'Region': region,
        'Model': 'Polynomial',
        'Intercept': results_summary_poly.params[0],
        'Slope': results_summary_poly.params[1],
        'Slope^2': results_summary_poly.params[2],
        'p-value': results_summary_poly.pvalues[2],
        'R-squared': results_summary_poly.rsquared,
        'AIC': results_summary_poly.aic
    }, ignore_index=True)


#Lm SRS social awarenes
regions = ['fEI', 'frontal_fEI', 'parietal_fEI', 'occipital_fEI', 'l_temporal_fEI', 'r_temporal_fEI']
results = pd.DataFrame(columns=['Region', 'Model', 'Intercept', 'Slope', 'Slope^2', 'p-value', 'R-squared', 'AIC'])

for region in regions:
    x = df_h['srs_social_awareness_t']
    y = df_h[region]
    x_linear = sm.add_constant(x)  # Linear model: Add a constant term for the intercept
    
    # Linear regression model
    model_linear = sm.OLS(y, x_linear, missing='drop')
    results_summary_linear = model_linear.fit()
    aic_linear = results_summary_linear.aic
    
    # Polynomial regression model (second order)
    x_poly = np.column_stack((x, x**3))  # Add polynomial terms
    x_poly = sm.add_constant(x_poly)  # Add a constant term for the intercept
    
    model_poly = sm.OLS(y, x_poly, missing='drop')
    results_summary_poly = model_poly.fit()
    aic_poly = results_summary_poly.aic
    
    results = results.append({
        'Region': region,
        'Model': 'Linear',
        'Intercept': results_summary_linear.params[0],
        'Slope': results_summary_linear.params[1],
        'Slope^2': '-',
        'p-value': results_summary_linear.pvalues[1],
        'R-squared': results_summary_linear.rsquared,
        'AIC': aic_linear
    }, ignore_index=True)
    
    results = results.append({
        'Region': region,
        'Model': 'Polynomial',
        'Intercept': results_summary_poly.params[0],
        'Slope': results_summary_poly.params[1],
        'Slope^2': results_summary_poly.params[2],
        'p-value': results_summary_poly.pvalues[2],
        'R-squared': results_summary_poly.rsquared,
        'AIC': results_summary_poly.aic
    }, ignore_index=True)

#Lm SRS social cognition
regions = ['fEI', 'frontal_fEI', 'parietal_fEI', 'occipital_fEI', 'l_temporal_fEI', 'r_temporal_fEI']
results = pd.DataFrame(columns=['Region', 'Model', 'Intercept', 'Slope', 'Slope^2', 'p-value', 'R-squared', 'AIC'])

for region in regions:
    x = df_h['srs_social_cognition_t']
    y = df_h[region]
    x_linear = sm.add_constant(x)  # Linear model: Add a constant term for the intercept
    
    # Linear regression model
    model_linear = sm.OLS(y, x_linear, missing='drop')
    results_summary_linear = model_linear.fit()
    aic_linear = results_summary_linear.aic
    
    # Polynomial regression model (second order)
    x_poly = np.column_stack((x, x**3))  # Add polynomial terms
    x_poly = sm.add_constant(x_poly)  # Add a constant term for the intercept
    
    model_poly = sm.OLS(y, x_poly, missing='drop')
    results_summary_poly = model_poly.fit()
    aic_poly = results_summary_poly.aic
    
    results = results.append({
        'Region': region,
        'Model': 'Linear',
        'Intercept': results_summary_linear.params[0],
        'Slope': results_summary_linear.params[1],
        'Slope^2': '-',
        'p-value': results_summary_linear.pvalues[1],
        'R-squared': results_summary_linear.rsquared,
        'AIC': aic_linear
    }, ignore_index=True)
    
    results = results.append({
        'Region': region,
        'Model': 'Polynomial',
        'Intercept': results_summary_poly.params[0],
        'Slope': results_summary_poly.params[1],
        'Slope^2': results_summary_poly.params[2],
        'p-value': results_summary_poly.pvalues[2],
        'R-squared': results_summary_poly.rsquared,
        'AIC': results_summary_poly.aic
    }, ignore_index=True)

#Lm SRS RRB
regions = ['fEI', 'frontal_fEI', 'parietal_fEI', 'occipital_fEI', 'l_temporal_fEI', 'r_temporal_fEI']
results = pd.DataFrame(columns=['Region', 'Model', 'Intercept', 'Slope', 'Slope^2', 'p-value', 'R-squared', 'AIC'])

for region in regions:
    x = df_h['srs_RRB_t']
    y = df_h[region]
    x_linear = sm.add_constant(x)  # Linear model: Add a constant term for the intercept
    
    # Linear regression model
    model_linear = sm.OLS(y, x_linear, missing='drop')
    results_summary_linear = model_linear.fit()
    aic_linear = results_summary_linear.aic
    
    # Polynomial regression model (second order)
    x_poly = np.column_stack((x, x**3))  # Add polynomial terms
    x_poly = sm.add_constant(x_poly)  # Add a constant term for the intercept
    
    model_poly = sm.OLS(y, x_poly, missing='drop')
    results_summary_poly = model_poly.fit()
    aic_poly = results_summary_poly.aic
    
    results = results.append({
        'Region': region,
        'Model': 'Linear',
        'Intercept': results_summary_linear.params[0],
        'Slope': results_summary_linear.params[1],
        'Slope^2': '-',
        'p-value': results_summary_linear.pvalues[1],
        'R-squared': results_summary_linear.rsquared,
        'AIC': aic_linear
    }, ignore_index=True)
    
    results = results.append({
        'Region': region,
        'Model': 'Polynomial',
        'Intercept': results_summary_poly.params[0],
        'Slope': results_summary_poly.params[1],
        'Slope^2': results_summary_poly.params[2],
        'p-value': results_summary_poly.pvalues[2],
        'R-squared': results_summary_poly.rsquared,
        'AIC': results_summary_poly.aic
    }, ignore_index=True)
    
    
   
#ADHD
regions = ['fEI', 'frontal_fEI', 'parietal_fEI', 'occipital_fEI', 'l_temporal_fEI', 'r_temporal_fEI']
results = pd.DataFrame(columns=['Region', 'Model', 'Intercept', 'Slope', 'Slope^2', 'p-value', 'R-squared', 'AIC'])

for region in regions:
    x = df_h['adhd_rs']
    y = df_h[region]
    x_linear = sm.add_constant(x)  # Linear model: Add a constant term for the intercept
    
    # Linear regression model
    model_linear = sm.OLS(y, x_linear, missing='drop')
    results_summary_linear = model_linear.fit()
    aic_linear = results_summary_linear.aic
    
    # Polynomial regression model (second order)
    x_poly = np.column_stack((x, x**3))  # Add polynomial terms
    x_poly = sm.add_constant(x_poly)  # Add a constant term for the intercept
    
    model_poly = sm.OLS(y, x_poly, missing='drop')
    results_summary_poly = model_poly.fit()
    aic_poly = results_summary_poly.aic
    
    results = results.append({
        'Region': region,
        'Model': 'Linear',
        'Intercept': results_summary_linear.params[0],
        'Slope': results_summary_linear.params[1],
        'Slope^2': '-',
        'p-value': results_summary_linear.pvalues[1],
        'R-squared': results_summary_linear.rsquared,
        'AIC': aic_linear
    }, ignore_index=True)
    
    results = results.append({
        'Region': region,
        'Model': 'Polynomial',
        'Intercept': results_summary_poly.params[0],
        'Slope': results_summary_poly.params[1],
        'Slope^2': results_summary_poly.params[2],
        'p-value': results_summary_poly.pvalues[2],
        'R-squared': results_summary_poly.rsquared,
        'AIC': results_summary_poly.aic
    }, ignore_index=True)



# RBS Total
regions = ['fEI', 'frontal_fEI', 'parietal_fEI', 'occipital_fEI', 'l_temporal_fEI', 'r_temporal_fEI']
results = pd.DataFrame(columns=['Region', 'Intercept', 'Slope', 'R-squared', 'p-value'])
for region in regions:
    x = df_h['RBS_total']
    y = df_h[region]
    x = sm.add_constant(x)  # Add a constant term for the intercept
    
    model = sm.OLS(y, x, missing='drop')
    results_summary = model.fit()
    
    results = results.append({
        'Region': region,
        'Intercept': results_summary.params[0],
        'Slope': results_summary.params[1],
        'R-squared': results_summary.rsquared,
        'p-value': results_summary.pvalues[1]
    }, ignore_index=True).round(2)
print(results)


# RBS stereotypies
regions = ['fEI', 'frontal_fEI', 'parietal_fEI', 'occipital_fEI', 'l_temporal_fEI', 'r_temporal_fEI']
results = pd.DataFrame(columns=['Region', 'Intercept', 'Slope', 'R-squared', 'p-value'])
for region in regions:
    x = df_h['RBS_cpts_stereotypes']
    y = df_h[region]
    x = sm.add_constant(x)  # Add a constant term for the intercept
    
    model = sm.OLS(y, x, missing='drop')
    results_summary = model.fit()
    
    results = results.append({
        'Region': region,
        'Intercept': results_summary.params[0],
        'Slope': results_summary.params[1],
        'R-squared': results_summary.rsquared,
        'p-value': results_summary.pvalues[1]
    }, ignore_index=True).round(2)
    
print(results)



# RBS Self injury 
regions = ['fEI', 'frontal_fEI', 'parietal_fEI', 'occipital_fEI', 'l_temporal_fEI', 'r_temporal_fEI']
results = pd.DataFrame(columns=['Region', 'Intercept', 'Slope', 'R-squared', 'p-value'])
for region in regions:
    x = df_h['RBS_cpts_automutilatoires']
    y = df_h[region]
    x = sm.add_constant(x)  # Add a constant term for the intercept
    
    model = sm.OLS(y, x, missing='drop')
    results_summary = model.fit()
    
    results = results.append({
        'Region': region,
        'Intercept': results_summary.params[0],
        'Slope': results_summary.params[1],
        'R-squared': results_summary.rsquared,
        'p-value': results_summary.pvalues[1]
    }, ignore_index=True).round(2)
    
print(results)


# RBS Compulsive behaviors
regions = ['fEI', 'frontal_fEI', 'parietal_fEI', 'occipital_fEI', 'l_temporal_fEI', 'r_temporal_fEI']
results = pd.DataFrame(columns=['Region', 'Intercept', 'Slope', 'R-squared', 'p-value'])
for region in regions:
    x = df_h['RBS_cpts_compulsifs']
    y = df_h[region]
    x = sm.add_constant(x)  # Add a constant term for the intercept
    
    model = sm.OLS(y, x, missing='drop')
    results_summary = model.fit()
    
    results = results.append({
        'Region': region,
        'Intercept': results_summary.params[0],
        'Slope': results_summary.params[1],
        'R-squared': results_summary.rsquared,
        'p-value': results_summary.pvalues[1]
    }, ignore_index=True).round(2)  
print(results)


# RBS ritualistic 
regions = ['fEI', 'frontal_fEI', 'parietal_fEI', 'occipital_fEI', 'l_temporal_fEI', 'r_temporal_fEI']
results = pd.DataFrame(columns=['Region', 'Intercept', 'Slope', 'R-squared', 'p-value'])
for region in regions:
    x = df_h['RBS_cpts_ritualisés']
    y = df_h[region]
    x = sm.add_constant(x)  # Add a constant term for the intercept
    
    model = sm.OLS(y, x, missing='drop')
    results_summary = model.fit()
    
    results = results.append({
        'Region': region,
        'Intercept': results_summary.params[0],
        'Slope': results_summary.params[1],
        'R-squared': results_summary.rsquared,
        'p-value': results_summary.pvalues[1]
    }, ignore_index=True).round(2)
print(results)


# RBS Sameness
regions = ['fEI', 'frontal_fEI', 'parietal_fEI', 'occipital_fEI', 'l_temporal_fEI', 'r_temporal_fEI']
results = pd.DataFrame(columns=['Region', 'Intercept', 'Slope', 'R-squared', 'p-value'])
for region in regions:
    x = df_h['RBS_cpts_immuables']
    y = df_h[region]
    x = sm.add_constant(x)  # Add a constant term for the intercept
    
    model = sm.OLS(y, x, missing='drop')
    results_summary = model.fit()
    
    results = results.append({
        'Region': region,
        'Intercept': results_summary.params[0],
        'Slope': results_summary.params[1],
        'R-squared': results_summary.rsquared,
        'p-value': results_summary.pvalues[1]
    }, ignore_index=True).round(2)
    
print(results)


# RBS restricted behaviors
regions = ['fEI', 'frontal_fEI', 'parietal_fEI', 'occipital_fEI', 'l_temporal_fEI', 'r_temporal_fEI']
results = pd.DataFrame(columns=['Region', 'Intercept', 'Slope', 'R-squared', 'p-value'])
for region in regions:
    x = df_h['RBS_cpts_restreints']
    y = df_h[region]
    x = sm.add_constant(x)  # Add a constant term for the intercept
    
    model = sm.OLS(y, x, missing='drop')
    results_summary = model.fit()
    
    results = results.append({
        'Region': region,
        'Intercept': results_summary.params[0],
        'Slope': results_summary.params[1],
        'R-squared': results_summary.rsquared,
        'p-value': results_summary.pvalues[1]
    }, ignore_index=True).round(4)
    
print(results)

sns.regplot(x=df_h['r_temporal_fEI'],y=df_h['hyper'],scatter_kws={'s': 10},order=2
pvals = [.2, .2, .02, .8, .5, .4])
reject, pvals_corr = pg.multicomp(pvals, method='fdr_bh')
print(reject, pvals_corr)


#ADOS css
lm_ados_css =pg.linear_regression(df_h[['fEI']], df_h['ados_css'], remove_na=True).round(2)
print(lm_ados_css)

regions = ['fEI', 'frontal_fEI', 'parietal_fEI', 'occipital_fEI', 'l_temporal_fEI', 'r_temporal_fEI']
results = pd.DataFrame(columns=['Region', 'Intercept', 'Slope', 'R-squared', 'p-value'])
for region in regions:
    x = df_h['ados_css']
    y = df_h[region]
    x = sm.add_constant(x)  # Add a constant term for the intercept
    
    model = sm.OLS(y, x, missing='drop')
    results_summary = model.fit()
    
    results = results.append({
        'Region': region,
        'Intercept': results_summary.params[0],
        'Slope': results_summary.params[1],
        'R-squared': results_summary.rsquared,
        'p-value': results_summary.pvalues[1]
    }, ignore_index=True).round(2)
print(results)


#ADI social interaction 
regions = ['fEI', 'frontal_fEI', 'parietal_fEI', 'occipital_fEI', 'l_temporal_fEI', 'r_temporal_fEI']
results = pd.DataFrame(columns=['Region', 'Model', 'Intercept', 'Slope', 'Slope^2', 'p-value', 'R-squared', 'AIC'])

for region in regions:
    x = df_h['adi_social_interaction']
    y = df_h[region]
    x_linear = sm.add_constant(x)  # Linear model: Add a constant term for the intercept
    
    # Linear regression model
    model_linear = sm.OLS(y, x_linear, missing='drop')
    results_summary_linear = model_linear.fit()
    aic_linear = results_summary_linear.aic
    
    # Polynomial regression model (second order)
    x_poly = np.column_stack((x, x**3))  # Add polynomial terms
    x_poly = sm.add_constant(x_poly)  # Add a constant term for the intercept
    
    model_poly = sm.OLS(y, x_poly, missing='drop')
    results_summary_poly = model_poly.fit()
    aic_poly = results_summary_poly.aic
    
    results = results.append({
        'Region': region,
        'Model': 'Linear',
        'Intercept': results_summary_linear.params[0],
        'Slope': results_summary_linear.params[1],
        'Slope^2': '-',
        'p-value': results_summary_linear.pvalues[1],
        'R-squared': results_summary_linear.rsquared,
        'AIC': aic_linear
    }, ignore_index=True)
    
    results = results.append({
        'Region': region,
        'Model': 'Polynomial',
        'Intercept': results_summary_poly.params[0],
        'Slope': results_summary_poly.params[1],
        'Slope^2': results_summary_poly.params[2],
        'p-value': results_summary_poly.pvalues[2],
        'R-squared': results_summary_poly.rsquared,
        'AIC': results_summary_poly.aic
    }, ignore_index=True)


#ADI communication  
regions = ['fEI', 'frontal_fEI', 'parietal_fEI', 'occipital_fEI', 'l_temporal_fEI', 'r_temporal_fEI']
results = pd.DataFrame(columns=['Region', 'Model', 'Intercept', 'Slope', 'Slope^2', 'p-value', 'R-squared', 'AIC'])

for region in regions:
    x = df_h['adi_communication']
    y = df_h[region]
    x_linear = sm.add_constant(x)  # Linear model: Add a constant term for the intercept
    
    # Linear regression model
    model_linear = sm.OLS(y, x_linear, missing='drop')
    results_summary_linear = model_linear.fit()
    aic_linear = results_summary_linear.aic
    
    # Polynomial regression model (second order)
    x_poly = np.column_stack((x, x**3))  # Add polynomial terms
    x_poly = sm.add_constant(x_poly)  # Add a constant term for the intercept
    
    model_poly = sm.OLS(y, x_poly, missing='drop')
    results_summary_poly = model_poly.fit()
    aic_poly = results_summary_poly.aic
    
    results = results.append({
        'Region': region,
        'Model': 'Linear',
        'Intercept': results_summary_linear.params[0],
        'Slope': results_summary_linear.params[1],
        'Slope^2': '-',
        'p-value': results_summary_linear.pvalues[1],
        'R-squared': results_summary_linear.rsquared,
        'AIC': aic_linear
    }, ignore_index=True)
    
    results = results.append({
        'Region': region,
        'Model': 'Polynomial',
        'Intercept': results_summary_poly.params[0],
        'Slope': results_summary_poly.params[1],
        'Slope^2': results_summary_poly.params[2],
        'p-value': results_summary_poly.pvalues[2],
        'R-squared': results_summary_poly.rsquared,
        'AIC': results_summary_poly.aic
    }, ignore_index=True)

#ADI CRR 
regions = ['fEI', 'frontal_fEI', 'parietal_fEI', 'occipital_fEI', 'l_temporal_fEI', 'r_temporal_fEI']
results = pd.DataFrame(columns=['Region', 'Model', 'Intercept', 'Slope', 'Slope^2', 'p-value', 'R-squared', 'AIC'])

for region in regions:
    x = df_h['adi_crr']
    y = df_h[region]
    x_linear = sm.add_constant(x)  # Linear model: Add a constant term for the intercept
    
    # Linear regression model
    model_linear = sm.OLS(y, x_linear, missing='drop')
    results_summary_linear = model_linear.fit()
    aic_linear = results_summary_linear.aic
    
    # Polynomial regression model (second order)
    x_poly = np.column_stack((x, x**3))  # Add polynomial terms
    x_poly = sm.add_constant(x_poly)  # Add a constant term for the intercept
    
    model_poly = sm.OLS(y, x_poly, missing='drop')
    results_summary_poly = model_poly.fit()
    aic_poly = results_summary_poly.aic
    
    results = results.append({
        'Region': region,
        'Model': 'Linear',
        'Intercept': results_summary_linear.params[0],
        'Slope': results_summary_linear.params[1],
        'Slope^2': '-',
        'p-value': results_summary_linear.pvalues[1],
        'R-squared': results_summary_linear.rsquared,
        'AIC': aic_linear
    }, ignore_index=True)
    
    results = results.append({
        'Region': region,
        'Model': 'Polynomial',
        'Intercept': results_summary_poly.params[0],
        'Slope': results_summary_poly.params[1],
        'Slope^2': results_summary_poly.params[2],
        'p-value': results_summary_poly.pvalues[2],
        'R-squared': results_summary_poly.rsquared,
        'AIC': results_summary_poly.aic
    }, ignore_index=True)


#Lm ADHD-RS
lm_adhd =pg.linear_regression(df_h[['fEI']], df_h['adhd_rs'], remove_na=True).round(2)
print(lm_adhd)

regions = ['fEI', 'frontal_fEI', 'parietal_fEI', 'occipital_fEI', 'l_temporal_fEI', 'r_temporal_fEI']
results = pd.DataFrame(columns=['Region', 'Intercept', 'Slope', 'R-squared', 'p-value'])
for region in regions:
    x = df_h['adhd_rs']
    y = df_h[region]
    x = sm.add_constant(x)  # Add a constant term for the intercept
    
    model = sm.OLS(y, x, missing='drop')
    results_summary = model.fit()
    
    results = results.append({
        'Region': region,
        'Intercept': results_summary.params[0],
        'Slope': results_summary.params[1],
        'R-squared': results_summary.rsquared,
        'p-value': results_summary.pvalues[1]
    }, ignore_index=True).round(4)
    
print(results)




# Regression plot 
# Create the figure and axes
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(36, 24))

# Set the background color to light gray for all subplots
#fig.patch.set_facecolor('#F0F0F0')
background_color = 'white'
for ax in axes.flatten():
    ax.set_facecolor(background_color)
# Plot first boxplot
sns.regplot(ax=axes[0, 0], data=df_filtered, x='fEI', y='adi_communication', scatter_kws={'s': 10}, scatter=True, color='#343837')
axes[0, 0].set_title('fEI average\n(n=115)', fontweight='bold')
axes[0, 0].set_xlabel('fEI')
axes[0, 0].set_ylabel('ADI communication')

# Add text box and annotation
axes[0, 0].text(0.71, 0.97, "R2=0.07; p=0.01", transform=axes[0, 0].transAxes, fontsize=11, verticalalignment='top', bbox=dict(facecolor='white', edgecolor='black'))

# Plot second boxplot
sns.regplot(ax=axes[0, 1], data=df_filtered, x='frontal_fEI', y='adi_communication', scatter_kws={'s': 10}, color='#343837')
axes[0, 1].set_title('Frontal fEI\n(n=115)', fontweight='bold')
axes[0, 1].set_xlabel('frontal fEI')
axes[0, 1].set_ylabel('ADI communication')

# Add text box and annotation
axes[0, 1].text(0.71, 0.97, "R2=0.04; p=0.04", transform=axes[0, 1].transAxes,
                fontsize=11, verticalalignment='top', bbox=dict(facecolor='white', edgecolor='black'))

# Plot third boxplot
sns.regplot(ax=axes[0, 2], data=df_filtered, x='parietal_fEI', y='adi_communication', scatter_kws={'s': 10}, color='#343837')
axes[0, 2].set_title('Parietal EI\n(n=115)', fontweight='bold')
axes[0, 2].set_xlabel('parietal fEI')
axes[0, 2].set_ylabel('ADI communication')

# Add text box and annotation
axes[0, 2].text(0.71, 0.97, "R2=0.06; p=0.01", transform=axes[0, 2].transAxes,
                fontsize=11, verticalalignment='top', bbox=dict(facecolor='white', edgecolor='black'))

# Plot fourth boxplot
sns.regplot(ax=axes[1, 0], data=df_filtered, x='occipital_fEI', y='adi_communication', scatter_kws={'s': 10}, color='#343837')
axes[1, 0].set_title('Occipital fEI\n(n=115)', fontweight='bold')
axes[1, 0].set_xlabel('occipital fEI')
axes[1, 0].set_ylabel('ADI communication')

axes[1, 0].text(0.71, 0.97, "R2=0.06; p=0.01", transform=axes[1, 0].transAxes,
                fontsize=11, verticalalignment='top', bbox=dict(facecolor='white', edgecolor='black'))

# Plot sixth boxplot
sns.regplot(ax=axes[1, 1], data=df_filtered, x='r_temporal_fEI', y='adi_communication', scatter_kws={'s': 10}, color='#343837')
axes[1, 1].set_title('Right temporal fEI\n(n=115)', fontweight='bold')
axes[1, 1].set_xlabel('right temporal fEI')
axes[1, 1].set_ylabel('ADI communication')

# Add text box and annotation
axes[1, 1].text(0.71, 0.97, "R2=0.07; p=0.01", transform=axes[1, 1].transAxes,
                fontsize=11, verticalalignment='top', bbox=dict(facecolor='white', edgecolor='black'))

plt.delaxes(axes[1, 2])
plt.tight_layout()
plt.show()

                      
# ADI Social interaction
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 12))
background_color = 'white'
for ax in axes.flatten():
    ax.set_facecolor(background_color)
# Plot first boxplot
sns.regplot(ax=axes[0, 0], data=df_filtered, x='fEI', y='adi_social_interaction', scatter_kws={'s': 10}, scatter=True, color='#343837')
axes[0, 0].set_title('fEI average\n(n=115)', fontweight='bold')
axes[0, 0].set_xlabel('fEI')
axes[0, 0].set_ylabel('ADI social interaction')

# Add text box and annotation
axes[0, 0].text(0.71, 0.97, "R2=0.06; p=0.03", transform=axes[0, 0].transAxes, fontsize=11, verticalalignment='top', bbox=dict(facecolor='white', edgecolor='black'))

# Plot second boxplot
sns.regplot(ax=axes[0, 1], data=df_filtered, x='frontal_fEI', y='adi_social_interaction', scatter_kws={'s': 10}, color='#343837')
axes[0, 1].set_title('Frontal fEI\n(n=115)', fontweight='bold')
axes[0, 1].set_xlabel('frontal fEI')
axes[0, 1].set_ylabel('ADI social interaction')

# Add text box and annotation
axes[0, 1].text(0.71, 0.97, "R2=0.03; p=0.06", transform=axes[0, 1].transAxes,
                fontsize=11, verticalalignment='top', bbox=dict(facecolor='white', edgecolor='black'))

# Plot third boxplot
sns.regplot(ax=axes[0, 2], data=df_filtered, x='parietal_fEI', y='adi_social_interaction', scatter_kws={'s': 10}, color='#343837')
axes[0, 2].set_title('Parietal EI\n(n=115)', fontweight='bold')
axes[0, 2].set_xlabel('parietal fEI')
axes[0, 2].set_ylabel('ADI social interaction')

# Add text box and annotation
axes[0, 2].text(0.71, 0.97, "R2=0.05; p=0.03", transform=axes[0, 2].transAxes,
                fontsize=11, verticalalignment='top', bbox=dict(facecolor='white', edgecolor='black'))

# Plot fourth boxplot
sns.regplot(ax=axes[1, 0], data=df_filtered, x='occipital_fEI', y='adi_social_interaction', scatter_kws={'s': 10}, color='#343837')
axes[1, 0].set_title('Occipital fEI\n(n=115)', fontweight='bold')
axes[1, 0].set_xlabel('occipital fEI')
axes[1, 0].set_ylabel('ADI social interaction')

axes[1, 0].text(0.71, 0.97, "R2=0.03; p=0.03", transform=axes[1, 0].transAxes,
                fontsize=11, verticalalignment='top', bbox=dict(facecolor='white', edgecolor='black'))

# Plot sixth boxplot
sns.regplot(ax=axes[1, 1], data=df_filtered, x='r_temporal_fEI', y='adi_social_interaction', scatter_kws={'s': 10}, color='#343837')
axes[1, 1].set_title('Right temporal fEI\n(n=115)', fontweight='bold')
axes[1, 1].set_xlabel('right temporal fEI')
axes[1, 1].set_ylabel('ADI social interaction')

# Add text box and annotation
axes[1, 1].text(0.71, 0.97, "R2=0.05; p=0.03", transform=axes[1, 1].transAxes,
                fontsize=11, verticalalignment='top', bbox=dict(facecolor='white', edgecolor='black'))

plt.delaxes(axes[1, 2])
plt.tight_layout()
plt.show()


#Age lm regplot 
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(36, 25))
background_color = 'white'
for ax in axes.flatten():
    ax.set_facecolor(background_color)
# Plot first boxplot
sns.regplot(ax=axes[0, 0], data=df_filtered, y='fEI', x='age_years', scatter_kws={'s': 10}, scatter=True, color='#343837')
axes[0, 0].set_title('fEI average\n(n=118)', fontweight='bold')
axes[0, 0].set_ylabel('fEI')
axes[0, 0].set_xlabel('Age in years')

# Add text box and annotation
axes[0, 0].text(0.5, 0.97, "R2=0.03; p=0.05; FDR=0.1", transform=axes[0, 0].transAxes, fontsize=11, verticalalignment='top', bbox=dict(facecolor='white', edgecolor='black'))

# Plot second boxplot
sns.regplot(ax=axes[0, 1], data=df_filtered, y='frontal_fEI', x='age_years', scatter_kws={'s': 10}, color='#343837')
axes[0, 1].set_title('Frontal fEI\n(n=118)', fontweight='bold')
axes[0, 1].set_ylabel('frontal fEI')
axes[0, 1].set_xlabel('Age in years')

# Add text box and annotation
axes[0, 1].text(0.71, 0.97, "R2=0.09; p=0.003", transform=axes[0, 1].transAxes,
                fontsize=11, verticalalignment='top', bbox=dict(facecolor='white', edgecolor='black'))

# Plot third boxplot
sns.regplot(ax=axes[0, 2], data=df_filtered, y='parietal_fEI', x='age_years', scatter_kws={'s': 10}, color='#343837')
axes[0, 2].set_title('Parietal EI\n(n=118)', fontweight='bold')
axes[0, 2].set_ylabel('parietal fEI')
axes[0, 2].set_xlabel('Age in years')

# Add text box and annotation
axes[0, 2].text(0.71, 0.97, "R2=0.06; p=0.01", transform=axes[0, 2].transAxes,
                fontsize=11, verticalalignment='top', bbox=dict(facecolor='white', edgecolor='black'))

plt.delaxes(axes[1,0])
plt.delaxes(axes[1,1])
plt.delaxes(axes[1,2])
plt.tight_layout()
plt.show()

                      
#SRS total et awareness 
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14,8))
background_color = 'white'
for ax in axes.flatten():
    ax.set_facecolor(background_color)
# Plot first boxplot
sns.regplot(ax=axes[0, 0], data=df_filtered, x='parietal_fEI', y='srs_total_t', scatter_kws={'s': 10}, scatter=True, color='#343837')
axes[0, 0].set_title('Parietal fEI\n(n=105)', fontweight='bold')
axes[0, 0].set_xlabel('parietal fEI')
axes[0, 0].set_ylabel('SRS total')

# Add text box and annotation
axes[0, 0].text(0.71, 0.97, "R2=0.04; p=0.04; FDR=0.1", transform=axes[0, 0].transAxes, fontsize=11, verticalalignment='top', bbox=dict(facecolor='white', edgecolor='black'))

# Plot second boxplot
sns.regplot(ax=axes[0, 1], data=df_filtered, x='occipital_fEI', y='srs_total_t', scatter_kws={'s': 10}, color='#343837')
axes[0, 1].set_title('Occipital fEI\n(n=105)', fontweight='bold')
axes[0, 1].set_xlabel('occipital fEI')
axes[0, 1].set_ylabel('SRS total')

# Add text box and annotation
axes[0, 1].text(0.71, 0.97, "R2=0.03; p=0.06; FDR=0.1", transform=axes[0, 1].transAxes,
                fontsize=11, verticalalignment='top', bbox=dict(facecolor='white', edgecolor='black'))

# Plot third boxplot
sns.regplot(ax=axes[1, 0], data=df_filtered, x='parietal_fEI', y='srs_social_awareness_t', scatter_kws={'s': 10}, color='#343837')
axes[1, 0].set_title('Parietal EI\n(n=102)', fontweight='bold')
axes[1, 0].set_xlabel('parietal fEI')
axes[1, 0].set_ylabel('SRS social awareness')

# Add text box and annotation
axes[1, 0].text(0.71, 0.97, "R2=0.03; p=0.06; FDR=0.1", transform=axes[1, 0].transAxes,
                fontsize=11, verticalalignment='top', bbox=dict(facecolor='white', edgecolor='black'))

# Plot fourth boxplot
sns.regplot(ax=axes[1, 1], data=df_filtered, x='occipital_fEI', y='srs_social_awareness_t', scatter_kws={'s': 10}, color='#343837')
axes[1, 1].set_title('Occipital fEI\n(n=102)', fontweight='bold')
axes[1, 1].set_xlabel('occipital fEI')
axes[1, 1].set_ylabel('SRS social awareness')

axes[1, 1].text(0.71, 0.97, "R2=0.05; p=0.02; FDR=0.1", transform=axes[1, 1].transAxes,
                fontsize=11, verticalalignment='top', bbox=dict(facecolor='white', edgecolor='black'))
plt.tight_layout()
plt.show()




# Hyper (polynomial (order =2))
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

# Set the background color to light gray for all subplots
background_color = 'white'
for ax in axes.flatten():
    ax.set_facecolor(background_color)

# Plot first boxplot
sns.regplot(ax=axes[0], data=df_filtered, x='frontal_fEI', y='hyper', scatter_kws={'s': 10}, scatter=True, color='#343837', order=2)
axes[0].set_title('Frontal fEI\n(n=50)', fontweight='bold')
axes[0].set_xlabel('frontal fEI')
axes[0].set_ylabel('Hypersensitivity')

# Add text box and annotation
axes[0].text(0.75, 0.97, "R2=0.08; p=0.03", transform=axes[0].transAxes, fontsize=11, verticalalignment='top', bbox=dict(facecolor='white', edgecolor='black'))



sns.regplot(ax=axes[1], data=df_filtered, x='r_temporal_fEI', y='hyper', scatter_kws={'s': 10}, color='#343837', order=2)
axes[1].set_title('Right Temporal fEI\n(n=50)', fontweight='bold')
axes[1].set_xlabel('right temporal fEI')
axes[1].set_ylabel('Hypersensitivity')

# Add text box and annotation
axes[1].text(0.75, 0.97, "R2=0.09; p=0.03", transform=axes[1].transAxes,
             fontsize=11, verticalalignment='top', bbox=dict(facecolor='white', edgecolor='black'))

#plt.subplots_adjust(top=0.855, bottom=0.39, left=0.045, right=0.96, hspace=0.2, wspace=0.2)
plt.tight_layout()
plt.show()

                      
           
# Permutation t-test (ttest on each electrode)
##  with ttest pg
asd_ei = df.loc[df['group'] == 'ASD', df.columns[3:124]]
controls_ei = df.loc[df['group'] == 'Controls', df.columns[3:124]]

p_values = []
t_values = []

for column in asd_ei.columns:
    result = pg.ttest(asd_ei[column], controls_ei[column])
    p_value = result['p-val'].values[0]
    p_values.append(p_value)
    t_value = result['T'].values[0]
    t_values.append(t_value)

# Create a DataFrame to store the p-values for each column
results_df = pd.DataFrame({'Column': asd_ei.columns, 'P-value': p_values,'t_val':t_values})
results_df['p-value_t'] = results_df['P-value'].apply(lambda x: 1 if x < 0.05 else 0)

significant_p_values = results_df[results_df['P-value'] < 0.05]
count = len(significant_p_values)
print("Number of p-values < 0.05:", count)


ei_pval= results_df['p-value_t']

min_range_r, max_range_r = 0,1

fig, ax = plt.subplots()
im, _ = mne.viz.plot_topomap(ei_pval, pos=xy_pos, vlim=(min_range_r, max_range_r),  cmap='bwr'  ,contours=0,axes=ax)
cbar = plt.colorbar(im, ax=ax)
plt.legend(loc='lower center') #loc permet de choisir emplacement légende
plt.gcf().set_size_inches(7, 6)
plt.subplots_adjust(top=0.94,
bottom=0.048,
left=0.053,
right=0.985,
hspace=0.2,
wspace=0.2)
ax.set_title('significative t-test between ASD and controls', fontsize= 18, fontweight='bold')
plt.show()


## with ttest scipy
import pandas as pd
from scipy import stats

asd_ei = df.loc[df['group'] == 'ASD', df.columns[3:124]]
controls_ei = df.loc[df['group'] == 'Controls', df.columns[3:124]]

p_values = []

for column in asd_ei.columns:
    t_statistic, p_value = stats.ttest_ind(asd_ei[column], controls_ei[column], equal_var=False, nan_policy='omit')
    p_values.append(p_value)

# Create a DataFrame to store the p-values for each column
results_df = pd.DataFrame({'Column': asd_ei.columns, 'P-value': p_values})

significant_p_values = results_df[results_df['P-value'] < 0.05]
count = len(significant_p_values)

print("Number of p-values < 0.05:", count)
