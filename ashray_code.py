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
wine_quality = (wineData[:,11]>=6).astype(int)

#append the column to the end of data array
wineData = np.c_[wineData, wine_quality]

#import breast cancer data into a numpy ndarray
bcdf = pd.read_csv("breast-cancer-wisconsin.data", sep=',', header=None)
bcdf.replace('?', np.NaN, inplace=True)
bcdf = bcdf.dropna()
bcData = bcdf.to_numpy().astype(float)

#weights=self.weights, alpha=self.alpha, X=self.X, y=self.y

class LogisticRegression:
    """Implementing a Logistic Regression model without sklearn"""

    def __init__(self, data, alpha=0.001):
        """

        Constructor to create a new LogisticRegression instance

        data        = matrix (numpy ndarray) of dataset with labels as the last column
        X           = matrix (numpy ndarray) of dataset with labels (last column) removed
        y           = vector (numpy ndarray) of labels (last column of data)
        m           = number of training examples (= number of rows of X)
        n           = number of features (= number of columns of X)
        weights     = vector (numpy ndarray) of weights for gradient descent
        alpha       = learning rate that controls how quickly gradient descent will converge

        """
        self.data = data
        self.X = data[:,:-1]
        self.m = X.shape[0]
        self.y = data[:,-1]
        X = np.c_[np.ones(shape=(m,1)), X] #add bias column of ones to feature matrix
        self.n = X.shape[1]
        self.weights = np.zeros(shape=(n, 1)) #set initial weights vector to n-dim vector of zeros
        self.alpha = alpha


    # Some helper functions:
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def label(v):
        """ 
        v is a vector (numpy ndarray)
        this function updates each row to 1 if >= 0.5 and 0 if <0.5
        """
        v[v>=0.5] = 1
        v[v<0.5] = 0
        return v


    # Model implementation begins below:     
    def fit(self, steps):
        """
        Fit training data using gradient descent (GD update perfomed 'steps' number of times)
        Calculates and updates optimal weights for the model after "training" with data.
        """

        for i in range(steps):
            self.weights = self.weights + ( self.alpha * self.X.T.dot(self.y - sigmoid(self.X @ self.weights)) )
            # X @ weights ==> matrix multiplication of mxn and nx1 produces a mx1 vector
            # then multiplying by X.T is a nxm @ mx1 produces a nx1 vector
            # this numpy vectorized implementation of the gradient descent update is a lot more efficient than
            # manually updating each weight using a python for-loop to calculate summmations
        #return self.weights

    def predict(self, X_test):
        """
        Given a trained model, predict labels for new data X_test (which is a mxn matrix),
        mxn @ nx1 gives a mx1 vector of predicted 0/1 labels. Sigmoid function calculates
        a vector of probabilities where each row is the probablity of being classified positive (1)

        This vector is passed into a "label" function which outputs 1 if probability>=0.5
        and 0 if probability<0.5

        predict returns a m-dimensional vector of 0/1 predicted labels
        """
        predicted_labels = label(sigmoid(X_test @ self.weights))
        return predicted_labels

    def evaluate_acc(self, y_predicted, y_test):
        """
        Check the accuracy of the predictions calculated by the predict method of the model
        Returns a percentage accuracy (float)
        """

        if y_predicted.shape != y_test: return None # shapes aren't equal

        test_set_size = y_predicted.shape[0]
        numErrors = np.sum(np.abs(y_predicted - y_test), dtype=float) #output float to force float division
        accuracy = 100 - (numErrors/test_set_size)
        return accuracy

    



