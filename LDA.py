import numpy as np
from utils import classifyLDA

class LDA:
    """Implementing an LDA (Linear Discriminant Analysis) model from scratch"""

    def __init__(self, data):
        """
        Constructor to create a new LDA instance

        data        = matrix (numpy ndarray) of dataset with labels as the last column
        X           = matrix (numpy ndarray) of dataset with labels (last column) removed
        X1          = (numpy ndarray) rows of X that belong to class 1
        X0          = (numpy ndarray) rows of X that belong to class 0

        """
        self.X = data[:,:-1]
        self.X1 = self.X[data[:,-1]==1]
        self.X0 = self.X[data[:,-1]==0]

    # Model implementation begins below:     
    def fit(self):
        """
        Fit training data
        Calculates mu1 and mu0 (mean vectors) and sigma (shared covariance matrix) using training data.

        num0        = number of training examples in class 0 (number of rows of X0)
        num1        = number of training examples in class 1 (number of rows of X1)
        mu1         = array of mean of columns in X1
        mu0         = array of mean of columns in X0
        m           = total number of training examples (= number of rows of X)
        sigma       = shared covariance matrix for the dataset

        """

        self.num1 = self.X1.shape[0] 
        self.num0 = self.X0.shape[0]
        self.mu1 = np.mean(self.X1, axis=0)[:,np.newaxis]
        self.mu0 = np.mean(self.X0, axis=0)[:,np.newaxis]
        m = self.X.shape[0]
        self.sigma = (np.cov(self.X1.T) + np.cov(self.X0.T)) / (m - 2)


    def predict(self, X_test):
        """
        Given a trained model, predict labels for new data X_test (which is a mxn matrix),
        predict returns a m-dimensional vector of 0/1 predicted labels

        calculated using the formula on lecture 5, slide 26, comp 551 fall 2019
        from https://cs.mcgill.ca/~wlh/comp551/slides/05-linear_classification_cont.pdf


        logodds_vector = vector of predicted log odds for every row in X_test
        """
        sigma_inv = np.linalg.inv(self.sigma)
        term1 = np.log(self.num1/self.num0) #no need to include the denominators as they cancel out
        term2 = 0.5*(self.mu1.T @ sigma_inv @ self.mu1)
        term3 = 0.5*(self.mu0.T @ sigma_inv @ self.mu0)
        term4 = X_test @ sigma_inv @ (self.mu1-self.mu0)

        logodds_vector = term1 - term2 + term3 + term4
        predicted_labels = classifyLDA(logodds_vector)
        return predicted_labels