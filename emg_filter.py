import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_filter import getOffset, WIN_SIZE_EMG, data_shape, total_matrix
from scipy.signal import butter, convolve, filtfilt, savgol_filter


# normalization
def normalized(origin_signal):
    k = origin_signal - np.mean(origin_signal)  # Eliminate DC component
    k = k / np.max(np.abs(k))  # Amplitude normalization
    return k


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
core = np.full(WIN_SIZE_EMG, 1 / WIN_SIZE_EMG)
del_avg = convolve(deltoid, core, mode='same') - offset_del
bicep_avg = convolve(bicep, core, mode='same') - offset_bi
tricep_avg = convolve(tricep, core, mode='same') - offset_tri

low_pass = 20
sfreq = 500
high_band = 200
low_band = 20

high_cut = high_band / (sfreq / 2)
low_cut = low_band / (sfreq / 2)

# create bandpass filter for EMG
b1, a1 = butter(N=4, Wn=[low_cut, high_cut], btype='bandpass')

# process EMG signal: filter EMG
del_filtered = filtfilt(b1, a1, del_avg)
del_rectified = abs(del_filtered)
del_smooth = savgol_filter(del_rectified, 101, 3)
del_normalized = normalized(del_smooth)

bi_filtered = filtfilt(b1, a1, bicep_avg)
bi_rectified = abs(bi_filtered)
bi_smooth = savgol_filter(bi_rectified, 101, 3)
bi_normalized = normalized(bi_smooth)

tri_filtered = filtfilt(b1, a1, tricep_avg)
tri_rectified = abs(tri_filtered)
tri_smooth = savgol_filter(tri_rectified, 101, 3)
tri_normalized = normalized(tri_smooth)

del_resized = np.resize(del_normalized, data_shape)
bi_resized = np.resize(bi_normalized, data_shape)
tri_resized = np.resize(tri_normalized, data_shape)

# show bi emg filter result
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
plt.plot(bi_rectified, color='lightseagreen', marker='o', linestyle='None', label='Bicep_Rectified')
plt.plot(bi_smooth, 'orangered', label='Bicep_Smooth')
plt.legend(loc='upper right')
plt.title('Bicep_Smooth')
plt.subplot(2, 3, 5)
plt.plot(bi_normalized, 'orangered', label='Bicep_Norm')
plt.title("Normalization")
plt.show()


