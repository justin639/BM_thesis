# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 03:29:13 2022

@author: 82108
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_filter import getOffset, WIN_SIZE_EMG
from scipy.signal import butter, convolve, filtfilt, savgol_filter

file_path = 'GPData/EMG/'
activity = ['abduction', 'static', 'flexion', 'drink']
trial = ['01_', '02_']
member = ['KJS', 'KTW', 'LMS', 'PJM']

emg = pd.read_excel(file_path + 'abduction01_KTW_EMG.xlsx', engine='openpyxl')

deltoid = emg.values[..., 0]
bicep = emg.values[..., 1]
tricep = emg.values[..., 2]

offset_del = getOffset(deltoid)
offset_bi = getOffset(bicep)
offset_tri = getOffset(tricep)
core = np.full(WIN_SIZE_EMG, 1/WIN_SIZE_EMG)
del_avg = convolve(deltoid, core, mode='same') - offset_del
bicep_avg = convolve(bicep, core, mode='same') - offset_bi
tricep_avg = convolve(tricep, core, mode='same') - offset_tri

low_pass=20
sfreq=500
high_band=200
low_band=20

high_cut = high_band / (sfreq / 2)
low_cut = low_band / (sfreq / 2)

# create bandpass filter for EMG
b1, a1 = butter(N=4, Wn=[low_cut, high_cut], btype='bandpass')

# process EMG signal: filter EMG
bi_filtered = filtfilt(b1, a1, bicep_avg)

# process EMG signal: rectify
bi_rectified = abs(bi_filtered)

bi_smooth = savgol_filter(bi_rectified, 101, 3)

# # create lowpass filter and apply to rectified signal to get EMG envelope
# low_pass = low_pass / (sfreq / 2)
# b2, a2 = butter(4, low_pass, btype='lowpass')
# emg_envelope = filtfilt(b2, a2, emg_rectified)

# data_label = []
# for i in range(0, len(emg), 1):
#     data_label.append(i)
# np.array(data_label)
#
# fig = plt.figure()
# plt.plot(data_label, emg)
# plt.xlabel('Data Label')
# plt.ylabel('EMG (a.u.)')
#
# fig_name = 'fig_EMG.png'
# fig.set_size_inches(w=16, h=9)
# fig.savefig(fig_name)
#
# emg_correctmean = emg - np.mean(emg)


def filteremg(data_label, emg, low_pass=20, sfreq=500, high_band=200, low_band=20):
    # normalise cut-off frequencies to sampling frequency
    high_band = high_band / (sfreq / 2)
    low_band = low_band / (sfreq / 2)

    # create bandpass filter for EMG
    b1, a1 = butter(4, [low_band, high_band], btype='bandstop')

    # process EMG signal: filter EMG
    emg_filtered = filtfilt(b1, a1, emg)

    # process EMG signal: rectify
    emg_rectified = abs(emg_filtered)

    # # create lowpass filter and apply to rectified signal to get EMG envelope
    # low_pass = low_pass / (sfreq / 2)
    # b2, a2 = butter(4, low_pass, btype='lowpass')
    # emg_envelope = filtfilt(b2, a2, emg_rectified)

    # # plot graphs
    # fig = plt.figure()
    # plt.subplot(1, 4, 1)
    # plt.subplot(1, 4, 1).set_title('EMG signal')
    # plt.plot(data_label, emg)
    # plt.locator_params(axis='x', nbins=4)
    # plt.locator_params(axis='y', nbins=4)
    # plt.xlabel('Data Label')
    # plt.ylabel('EMG (a.u.)')
    #
    # plt.subplot(1, 4, 2)
    # plt.subplot(1, 4, 2).set_title('Bandpass Butterworth')
    # plt.plot(data_label, emg_rectified)
    # plt.locator_params(axis='x', nbins=4)
    # plt.locator_params(axis='y', nbins=4)
    # plt.xlabel('Data Label')
    #
    # plt.subplot(1, 4, 3)
    # plt.subplot(1, 4, 3).set_title('lowpass')
    # plt.plot(data_label, emg_envelope)
    # plt.locator_params(axis='x', nbins=4)
    # plt.locator_params(axis='y', nbins=4)
    # plt.xlabel('Data Label')
    #
    # # 1cycle 기준으로 돌려봄
    # plt.subplot(1, 4, 4)
    # plt.subplot(1, 4, 4).set_title('Focussed region')
    # plt.plot(data_label[1012:1700], emg_envelope[1012:1700])
    # plt.locator_params(axis='x', nbins=4)
    # plt.locator_params(axis='y', nbins=4)
    # plt.xlabel('Data Label')
    #
    # fig_name = 'fig_bandworthEMG.png'
    # fig.set_size_inches(w=16, h=9)
    # fig.savefig(fig_name)

    return emg_rectified


# emg_envelop = filteremg(data_label, emg_correctmean, low_pass=20)
#
# fig = plt.figure()


# normalization
def normalized(origin_signal):
    k = origin_signal - np.mean(origin_signal)  # Eliminate DC component
    k = k / np.max(np.abs(k))  # Amplitude normalization
    return k


normalized_arr = normalized(bi_smooth)

plt.figure(figsize=(36, 16))
plt.subplot(2, 3, 1)
plt.plot(bicep, color='darkseagreen', label='Bicep')
plt.plot(bicep_avg, color='cornflowerblue', label='Bicep_AVG')
plt.legend(loc='upper right')
plt.title('Bicep_AVG')
plt.subplot(2, 3, 2)
plt.plot(bicep_avg, color='cornflowerblue', label='Bicep_AVG')
plt.plot(bi_filtered, color='moccasin', label='Bicep_Butter')
plt.title('Bicep_Butter')
plt.legend(loc='upper right')
plt.subplot(2, 3, 3)
plt.plot(bi_rectified, color='lightseagreen', label='Bicep_Rectified')
plt.title('Bicep_Rectified')
plt.subplot(2, 3, 4)
plt.plot(bi_rectified,  color='lightseagreen', marker='o', linestyle='None',label='Bicep_Rectified')
plt.plot(bi_smooth, 'orangered', label='Bicep_Smooth')
plt.legend(loc='upper right')
plt.title('Bicep_Smooth')
plt.subplot(2, 3, 5)
plt.plot(normalized_arr, 'orangered', label='Bicep_Norm')
plt.title("Normalization")
plt.show()
