import numpy as np
from utils import sigmoid, classifyLogistic

class LogisticRegression:
    """Implementing a Logistic Regression model without sklearn"""

    def __init__(self, data):
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
        self.y = self.data[:,-1][:, np.newaxis] #np.newaxis to turn this into an (n,1)-shaped array
        self.m = self.X.shape[0]
        X = np.c_[np.ones(shape=(self.m,1)), self.X] #add bias column of ones to feature matrix
        self.n = self.X.shape[1]
        self.weights = np.zeros(shape=(self.n, 1)) #set initial weights vector to n-dim vector of zeros
        

    # Model implementation begins below:     
    def fit(self, steps=30000, alpha=0.0001):
        """
        Fit training data using gradient descent (GD update perfomed 'steps' number of times)
        Calculates and updates optimal weights for the model after "training" with data.
        """

        for i in range(steps):
            self.weights = self.weights + ( alpha * (self.X.T @ (self.y - sigmoid(self.X @ self.weights))) )
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

        This vector is passed into a "label" function which outputs 1 if probability >=0.5
        and 0 if probability<0.5

        predict returns a m-dimensional vector of 0/1 predicted labels
        """
        predicted_labels = classifyLogistic(sigmoid(X_test @ self.weights))
        return predicted_labels