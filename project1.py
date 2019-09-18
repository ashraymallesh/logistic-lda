import pandas as pd
import numpy as np

# Import data into pandas dataframe
#winedf = pd.read_csv("winequality-red.csv") #wine dataframe

#importing data into a numpy ndarray
wineData = np.genfromtxt("winequality-red.csv", delimiter=";", skip_header=1)
print(wineData)

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

