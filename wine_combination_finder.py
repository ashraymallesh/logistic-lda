import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit
import math
import itertools
import utils

#Data Cleaning
winedf = pd.read_csv("datasets/winequality-red.csv", sep=';')
wineData = winedf.to_numpy()
wineData[:,-1] = (wineData[:,-1]>=6).astype(int) #convert 6+ to 1 and <5 to 0
wineData = wineData/wineData.max(axis=0) #normalize data
bcdf = pd.read_csv("datasets/breast-cancer-wisconsin.data", sep=',', header=None)
bcdf.columns = ['Sample Code Number', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape',
                'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin',
                'Normal Nucleoli', 'Mitoses', 'Class']
bcdf.replace('?', np.NaN, inplace=True)
bcdf.dropna(inplace=True)
bcdf.drop(['Sample Code Number'], axis=1, inplace=True)
bcData = bcdf.to_numpy().astype(float)
bcData[:,-1] = (bcData[:,-1]>3).astype(int) #change 2/4 last column to 0/1 labels
bcData = bcData/bcData.max(axis=0) #normalize data


def train_test_split(dataset, ratio=0.2):
    """
    split dataset into training and test subsets
    test set size will be ratio * dataset size (default = 20% of total size)
    returns data_train, data_test
    """
    dataset_size = dataset.shape[0]
    test_set_size = math.floor(ratio * dataset_size)
    training_set_size = dataset_size - test_set_size
    data_train = dataset[0:training_set_size, :]
    data_test = dataset[test_set_size:, :]
    return data_train, data_test



# Try every combination of 3 features in wine dataset
maxAcc = 0
onFeatures = []
for nFeatures in range(1, 11):
    combs = itertools.combinations([x for x in range(11)], nFeatures)
    for com in combs:
        wineSubset = wineData[:, list(com) + [11]]
        model = LogisticRegression(data=wineSubset)
        model.fit(steps=1000)
        X_test = wineSubset[:,:-1]
        y_test = wineSubset[:,-1][:, np.newaxis]
        y_predicted = model.predict(X_test=X_test)
        wineAccuracy = model.evaluate_acc(y_predict=y_predicted, y_test=y_test)
        if wineAccuracy > maxAcc:
            maxAcc = wineAccuracy
            onFeatures = list(com)

print("----- vs baseline -----")
wineSubset = wineData
model = LogisticRegression(data=wineSubset)
model.fit(steps=5000)
X_test = wineSubset[:,:-1]
y_test = wineSubset[:,-1][:, np.newaxis]
y_predicted = model.predict(X_test=X_test)
wineAccuracy = model.evaluate_acc(y_predict=y_predicted, y_test=y_test)
print(maxAcc, "vs", wineAccuracy)
print(onFeatures)
