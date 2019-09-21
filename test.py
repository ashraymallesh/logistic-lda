import numpy as np
import math

# Import data into pandas dataframe
# winedf = pd.read_csv("winequality-red.csv") #wine dataframe
# print(winedf)


# Logistic function
def logistic(a):
    return 1 / (1 + np.exp(-a))

class LogisticRegressionModel:
    def __init__(numFeatures):
        # Initialize a weights matrix
        self.weights = np.ones((numFeatures, 1))

    
    def fit(X, Y, learningRate, ):
        """
        Pseudo code:
        new_weights = self.weights + learningRate * (sum from i = 1 to n of x_i * (y_i - logistic(self.weights^Transpose * X_i))
        """

        loss = 0
        # Iterate over every feature 
        for i in range(0, X.shape[1] + 1):
            X[i] * (Y[i] - y[i]
    
    """
    Takes a feature vector "X" and a binary 0/1 classification
    based on trained weights
    """
    def predict(X):
        #

        return logistic(X @ self.weights)

    


# Import csv file into 
wine_data = np.genfromtxt('winequality-red.csv', delimiter=';')

# Remove first row from data (row that has column labels)
wine_data = wine_data[1:, :]

# Collapse quality ranking into binary classification problem
# If wine's quality score is greater than or equal to 6 it is good wine (now represented by 1)
# otherwise it is of bad quality (now represented by 0)
quality_data = np.apply_along_axis(lambda x: x >= 6, 0, wine_data[:,11])
print(wine_data)
print(quality_data)


"""
# Check for null data
null_data = winedf[winedf.isnull().any(axis=1)]
print(null_data)

def label_binary_quality(row):
    if row["quality"] <= 0.5:
        return 0
    else:
        return 1
winedf["binary_quality"] = winedf.apply(lambda row: label_binary_quality(row), axis=1)
"""