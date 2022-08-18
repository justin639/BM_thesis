import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import openpyxl
import os
from scipy.signal import convolve

folder_path = "GPData/MECH/3"

action = ["abduction", "drink", "flexion", "static"]
setnum = ["01", "02"]
name = ["KJS", "KTW", "LMS", "PJM", "1", "2", "3"]
format = ".xlsx"

file_name = "static_03.xlsx"

# read data from excel
df = pd.read_excel(os.path.join(folder_path, file_name), engine='openpyxl')


# calculate offset [0:100]/100
def getOffset(data):
    ofst = 0
    for i in range(100):
        ofst = ofst + data[i]

    ofst = ofst / 100
    return ofst


WIN_SIZE = 50
WIN_SIZE_ACC = 25
WIN_SIZE_EMG = 10

# FSR(1:2) | ANG(3)
# average filter window size : 50, offset = data[0:100]/100
fsr_1 = df.values[..., 1]
fsr_2 = df.values[..., 2]
# average filter
offset_1 = getOffset(fsr_1)
offset_2 = getOffset(fsr_2)
core = np.full(WIN_SIZE, 1 / WIN_SIZE)
fsr_1_avg = convolve(fsr_1, core, mode='same') - offset_1
fsr_2_avg = convolve(fsr_2, core, mode='same') - offset_2
# show result
plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(fsr_1, label='FSR_1')
plt.title('FSR_1')
plt.subplot(2, 1, 2)
plt.plot(fsr_1_avg, label='FSR_1_AVG')
plt.title('FSR_1_AVG')
plt.show()

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(fsr_2, label='FSR_2')
plt.title('FSR_2')
plt.subplot(2, 1, 2)
plt.plot(fsr_2_avg, label='FSR_2_AVG')
plt.title('FSR_2_AVG')
plt.show()
# ACC(4:6) | GYR(7:9)
acc_1 = df.values[..., 4]
acc_2 = df.values[..., 5]
acc_3 = df.values[..., 6]
# average filter
core = np.full(WIN_SIZE_ACC, 1 / WIN_SIZE_ACC)
acc_1_avg = convolve(acc_1, core, mode='same')
acc_2_avg = convolve(acc_2, core, mode='same')
acc_3_avg = convolve(acc_3, core, mode='same')
# show result
plt.figure(figsize=(16, 8))
plt.subplot(2, 1, 1)
plt.plot(acc_1, label='ACC_1')
plt.plot(acc_2, label='ACC_2')
plt.plot(acc_3, label='ACC_3')
plt.legend(loc='lower right')
plt.title('ACC')
plt.subplot(2, 1, 2)
plt.plot(acc_1_avg, label='ACC_1_AVG')
plt.plot(acc_2_avg, label='ACC_2_AVG')
plt.plot(acc_3_avg, label='ACC_3_AVG')
plt.legend(loc='lower right')
plt.title('ACC_AVG')
plt.show()
# save result as excel
save = pd.DataFrame([acc_1_avg, acc_2_avg, acc_3_avg]).T
save.to_excel(os.path.join(folder_path, "filtered_"+file_name), index=False)
# average filter window size : 50, offset = 0
# EMG
# average filter window size : 10, offset = data[0:100]/100
# butter

