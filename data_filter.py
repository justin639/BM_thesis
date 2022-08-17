import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import openpyxl
from scipy.signal import convolve

folder_path = "GPData/MECH/KJS/"

action = ["abduction", "drink", "flexion", "static"]
setnum = ["01", "02"]
name = ["KJS", "KTW", "LMS", "PJM", "1", "2", "3"]
format = ".xlsx"

file_name = "abduction01_KJS.xlsx"

# read data from excel
df = pd.read_excel(folder_path + file_name, engine='openpyxl')


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
# ACC(4:6) | GYR(7:9)
# average filter window size : 50, offset = 0
# EMG
# average filter window size : 10, offset = data[0:100]/100
# butter
target = df.values[..., 1]
offset = getOffset(target)
core = np.full(WIN_SIZE, 1 / WIN_SIZE)

# average filter(win=50) - offset
filtered_data = convolve(target, core, mode='same') - offset

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(target, label='FSR_1')
plt.title('FSR_1')
plt.subplot(2, 1, 2)
plt.plot(filtered_data, label='FSR_1_AVG')
plt.title('FSR_1_AVG')
plt.show()


