import pandas as pd
import numpy as np

#Data Cleaning
winedf = pd.read_csv("winequality-red.csv", sep=';')
wineData = winedf.to_numpy()
wineData[:,-1] = (wineData[:,-1]>=6).astype(int) #convert 6+ to 1 and <5 to 0
y = wineData[:,-1][:, np.newaxis]