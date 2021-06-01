# -------------------------------
# 2. Data analysis
# -------------------------------

# ---- imports

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# ----- path

project_folder = "../"
data_folder = project_folder + "/data/"
report_folder = project_folder + "/report/"

# ---- data import

df = pd.read_csv(data_folder + "data.csv", sep = ";")

# ---- exploratory analysis



fig, ax = plt.subplots(1,2,figsize = (17,8))

ax[0].hist(df["Y"], bins = 20)
ax[0].set_xlabel("Y")
ax[0].set_ylabel("counts")
ax[0].grid()

ax[1].scatter(df["X"], df["Y"], c = "purple", s = 90, marker = "*")
ax[1].grid()
ax[1].set_ylabel("Y")
ax[1].set_xlabel("X")

fig.suptitle("Exploratory analysis")

plt.savefig(report_folder + "/figures/fig_exploratory_analysis.pdf")


# --- linear regression

X = df["X"].to_numpy()
X.shape = (X.shape[0], 1)
y = df["Y"].to_numpy()
reg = LinearRegression().fit(X, y)
reg_equation = "y = {:.2f} + {:.2f}X \n".format(reg.intercept_,reg.coef_[0])

print("slope: {} \nintercept: {}".format(reg.coef_, reg.intercept_))

# --- fitted values

y_fitted = reg.predict(X)

# --- plot linear regression

fig, ax = plt.subplots(1,2,figsize = (17,8))


ax[0].scatter(X[:,0],y, c = "purple", s = 90, marker = "*", label = "observed data")
ax[0].plot(X[:,0], y_fitted, c = "green", linewidth = 1, label = reg_equation)
ax[0].grid()
ax[0].set_ylabel("Y")
ax[0].set_xlabel("X")
ax[0].legend(fontsize = 12)
ax[0].set_title("Regression model")


ax[1].scatter(y,y_fitted, c = "red", s = 50, marker = "D")
ax[1].grid()
ax[1].set_ylabel("Y fitted")
ax[1].set_xlabel("Y")
ax[1].set_title("Model adequacy")



plt.savefig(report_folder + "/figures/fig_linear_regression.pdf")



# --- finish
print("done")