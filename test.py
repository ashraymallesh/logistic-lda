import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import data into a pandas dataframe
winedf = pd.read_csv("winequality-red.csv", sep=';')

#checking for missing values
#print(winedf.isnull().sum())

wineData = winedf.to_numpy()

#create a quality column with 1/0 labels for wines with a rating of 6 or higher
quality = (wineData[:,11]>=6).astype(int)

#append the column to the end of data array
wineData = np.c_[wineData, quality]

#import breast cancer data into a numpy ndarray
bcData = np.genfromtxt("breast-cancer-wisconsin.data", delimiter=",")



bcdf = pd.read_csv("breast-cancer-wisconsin.data", sep=',', header=None)
#bcData = bcdf.to_numpy()
print(np.isnan(bcData).astype(int).sum())
#bcdf.replace('?', np.NaN, inplace=True)

#bcdf.apply(pd.to_numeric, errors='coerce')