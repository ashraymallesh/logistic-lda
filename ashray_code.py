import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
"""
def isnumber(x):
    try:
        float(x)
        return True
    except:
        return False

bcdf[bcdf.applymap(isnumber)]

"""
#checking for missing values
#print(winedf.isnull().sum())

# Import data into a pandas dataframe
winedf = pd.read_csv("winequality-red.csv", sep=';')



wineData = winedf.to_numpy()

#create a quality column with 1/0 labels for wines with a rating of 6 or higher
quality = (wineData[:,11]>=6).astype(int)

#append the column to the end of data array
wineData = np.c_[wineData, quality]

#import breast cancer data into a numpy ndarray
bcData = np.genfromtxt("breast-cancer-wisconsin.data", delimiter=",")
bcdf = pd.read_csv("breast-cancer-wisconsin.data", sep=',', header=None)
bcdf.replace('?', np.NaN, inplace=True)
bcdf = bcdf.dropna()

class LogisticRegression:
    """Implementing Logistic Regression without sklearn"""

    learningRate = 0.001

    def __init__(self, y):
        """ Contstructor to create a new object """
        self.y = y
        
    def fit(self, X, y):
        """ Fit training data """
        return self.X, self.y

    def predict(self, X_new):
        """ Given a trained model, predict labels for a new set of data"""
        y_new = np.zeros((X_shape[0], 1))
        return self.y_new

    def evaluate_acc(self, y, y_new):
        """ Change some information """
        return self.y - self.y_new

    def sigmoid(z):
    	return 1 / (1 + np.exp(-z))



