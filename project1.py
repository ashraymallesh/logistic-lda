import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import data into a pandas dataframe
winedf = pd.read_csv("winequality-red.csv", sep=';')

#checking for missing values
print(winedf.isnull().sum())

wineData = winedf.to_numpy()
print(wineData.head)
#import wine data into a numpy ndarray
wineData = np.genfromtxt("winequality-red.csv", delimiter=";", skip_header=1)
#checking for missing values
n = np.isnan(wineData).astype(int).sum()


#create a quality column with 1/0 labels for wines with a rating of 6 or higher
quality = (wineData[:,11]>=6).astype(int)

#append the column to the end of data array
wineData = np.c_[wineData, quality]

#import breast cancer data into a numpy ndarray
bcData = np.genfromtxt("breast-cancer-wisconsin.data", delimiter=",")

#checking for missing values
n = np.isnan(bcData).astype(int).sum()



"""
# Check for null data
null_data = winedf[winedf.isnull().any(axis=1)]
print(null_data)
"""

def label_binary_quality(row):
    if row["quality"] <= 0.5:
        return 0
    else:
        return 1
#winedf["binary_quality"] = winedf.apply(lambda row: label_binary_quality(row), axis=1)

