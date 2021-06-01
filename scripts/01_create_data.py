# -------------------------------
# 1. Create data
# -------------------------------

# ---- imports 

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----- path

project_folder = "../"
data_folder = project_folder + "/data/"
report_folder = project_folder + "/report/"

# ---- data simulation

n = 100
X = np.random.normal(loc = 2, scale = 0.4, size=n)
error = np.random.normal(loc = 0, scale = 0.1, size=n)
intercept = 3
slope = -1
Y = intercept + X*slope + error


# ---- data export

data_dict = {"X": X,
			 "Y": Y}

data_df = pd.DataFrame.from_dict(data_dict)

data_df.to_csv(data_folder + "/data.csv", sep = ";", index=False)


# --- finish
print("done")