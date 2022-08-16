import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import openpyxl

folder_path = "GPData/MECH/"

action = ["abduction", "drink", "flexion", "static"]
setnum = ["01", "02"]
name = ["KJS", "KTW", "LMS", "PJM"]
format = ".xlsx"

file_name = "abduction01_KJS.xlsx"

df = pd.read_excel(folder_path+file_name, engine='openpyxl')

print(df.head(3))

plt.show(df)

