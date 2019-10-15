import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from LDA import LDA
from LogisticRegression import LogisticRegression
import utils, kfold

#Data cleaning and preprocessing
winedf = pd.read_csv("datasets/winequality-red.csv", sep=';')
del winedf['citric acid']
del winedf['fixed acidity']
del winedf['free sulfur dioxide']
wineData = winedf.to_numpy() #create numpy array
wineData[:,-1] = (wineData[:,-1]>=6).astype(int) #convert 6+ to 1 and <5 to 0
wineData = wineData/wineData.max(axis=0) #normalize data

bcdf = pd.read_csv("datasets/breast-cancer-wisconsin.data", sep=',', header=None)
bcdf.columns = ['Sample Code Number', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape',
                'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin',
                'Normal Nucleoli', 'Mitoses', 'Class']
bcdf.replace('?', np.NaN, inplace=True)
bcdf.dropna(inplace=True) #drop columns with NaN values
bcdf.drop(['Sample Code Number'], axis=1, inplace=True)
bcData = bcdf.to_numpy().astype(float)
bcData[:,-1] = (bcData[:,-1]>3).astype(int) #change 2/4 last column to 0/1 labels
bcData = bcData/bcData.max(axis=0) #normalize data


#test accuracy
avg_accuracy = kfold.kFoldCrossValidate(dataset=bcData, classificationModel='LR')
print(avg_accuracy)

def test_learning_rates():
    learning_rates = [0.000001, 0.00001, 0.0001, 0.001, 0.1, 1]
    learning_rates2 = np.arange(start=0.0001,stop= 0.001, step=0.0001)
    accuracies = []; accuracies2 = []
    for alpha in learning_rates:
        acc = kfold.kFoldCrossValidate(dataset=wineData, classificationModel='LR', alpha=alpha)
        accuracies.append(acc)
    for alpha in learning_rates:
        acc = kfold.kFoldCrossValidate(dataset=wineData, classificationModel='LR', alpha=alpha)
        accuracies2.append(acc)
    print(accuracies)

def test_number_of_steps():
    stepsList = [10, 100, 1000, 10000, 100000, 1000000]
    stepsList2 = [x for x in range(10000, 100000, 10000)]
    accuracies = []; accuracies2 = []
    for steps in stepsList2:
        acc = kfold.kFoldCrossValidate(dataset=wineData, classificationModel='LR', steps=steps)
        accuracies.append(acc)
    print(accuracies)
