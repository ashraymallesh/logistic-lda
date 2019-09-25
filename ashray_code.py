import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit
import math

# Some helper functions:
def sigmoid(z):
    #return 1 / (1 + np.exp(-z))
    return expit(z)

def label(v):
    """ 
    v is a vector (numpy ndarray)
    this function updates each row to 1 if >= 0.5 and 0 if <0.5
    """
    v[v>=0.5] = 1
    v[v<0.5] = 0
    return v
    
def normalize(dataset):
    return dataset/dataset.max(axis=0)

#Data Cleaning
winedf = pd.read_csv("winequality-red.csv", sep=';')
wineData = winedf.to_numpy()
normalize(wineData)
wineData[:,-1] = (wineData[:,-1]>=6).astype(int) #convert 6+ to 1 and <5 to 0

bcdf = pd.read_csv("breast-cancer-wisconsin.data", sep=',', header=None)
bcdf.columns = ['Sample Code Number', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape',
                'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin',
                'Normal Nucleoli', 'Mitoses', 'Class']
bcdf.replace('?', np.NaN, inplace=True)
bcdf.dropna(inplace=True)
bcdf.drop(['Sample Code Number'], axis=1, inplace=True)
bcData = bcdf.to_numpy().astype(float)
bcData[:,-1] = (bcData[:,-1]>3).astype(int) #change 2/4 last column to 0/1 labels

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


def k_fold_split(dataset, num_folds=5):
    """
    Split the data into "num_folds" equal sections. Return
    every combination of them in which all but 1 are
    used for training and the remaining one is used 
    for validation
    """
    # Determine the number of examples in the dataset
    dataset_size = dataset.shape[0]

    # Determine the number of examples in a single fold
    fold_size = math.floor(1 / num_folds * dataset_size)

    # Break the examples up into "folds" number of folds
    folds = []
    for i in range(num_folds):
        folds.append(dataset[(i * fold_size):((i + 1) * fold_size), :])
    
    # Return every combination of total_folds
    fold_combinations = []
    for i in range(num_folds):
        validation_data = folds[i]
        training_flag = False
        training_data = None
        for j in range(num_folds):
            if i != j:
                if not training_flag:
                    training_flag = True
                    training_data = folds[i]
                else:
                    training_data = np.vstack((training_data, folds[i]))
        fold_combinations.append({"validation": validation_data, "training": training_data})
    
    return fold_combinations
    
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
        self.m = self.X.shape[0]
        self.y = self.data[:,-1][:, np.newaxis]
        X = np.c_[np.ones(shape=(self.m,1)), self.X] #add bias column of ones to feature matrix
        self.n = self.X.shape[1]
        self.weights = np.zeros(shape=(self.n, 1)) #set initial weights vector to n-dim vector of zeros
        

    # Model implementation begins below:     
    def fit(self, steps, alpha=0.0001):
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
        predicted_labels = label( sigmoid(X_test @ self.weights) )
        return predicted_labels

    def evaluate_acc(self, y_predicted, y_test):
        """
        Check the accuracy of the predictions calculated by the predict method of the model
        Returns a percentage accuracy (float)
        """

        test_set_size = y_test.shape[0]
        numErrors = np.sum( np.abs(y_predicted - y_test) )
        errorP = (numErrors/test_set_size) * 100
        accuracy = 100 - errorP
        return accuracy


#Training on wine data
num_folds = 5
wine_folds = k_fold_split(wineData, num_folds)
wine_train, wine_test = train_test_split(dataset=wineData)

# Training wine data using "num_folds" number of folds
"""
#####################
# BASIC K-FOLD CODE #
#####################
accSum = 0
for i in range(num_folds):
    training_data = wine_folds[i]["training"]
    validation_data = wine_folds[i]["validation"]
    model = LogisticRegression(data=training_data)
    model.fit(steps=10000, alpha=0.00005)
    X_test = validation_data[:, :-1] #delete labels column of wine_test
    y_test = validation_data[:,-1][:, np.newaxis] #take last column of test set as y_test
    y_predicted = model.predict(X_test=X_test) #using the test set to test accuracy
    wineAccuracy = model.evaluate_acc(y_predicted=y_predicted, y_test=y_test)
    print("k-fold", i + 1, "accuracy is", wineAccuracy, "%")
    accSum += wineAccuracy
print("---------------")
print("Average accuracy is", accSum / 5, "%")
"""

####################################
# TESTING DIFFERENT LEARNING RATES #
####################################
alphas = []
accuracies = []
for alpha in np.arange(0.00005, 0.001, 0.00005):
    accSum = 0
    """
    # k-fold
    for i in range(num_folds):
        training_data = wine_folds[i]["training"]
        validation_data = wine_folds[i]["validation"]
        model = LogisticRegression(data=training_data)
        model.fit(steps=2000, alpha=alpha)
        X_test = validation_data[:, :-1] #delete labels column of wine_test
        y_test = validation_data[:,-1][:, np.newaxis] #take last column of test set as y_test
        y_predicted = model.predict(X_test=X_test) #using the test set to test accuracy
        wineAccuracy = model.evaluate_acc(y_predicted=y_predicted, y_test=y_test)
        accSum += wineAccuracy
    accSum /= 5
    """
    
    # k-fold
    training_data = wineData
    validation_data = wineData
    model = LogisticRegression(data=training_data)
    model.fit(steps=2000, alpha=alpha)
    X_test = validation_data[:, :-1] #delete labels column of wine_test
    y_test = validation_data[:,-1][:, np.newaxis] #take last column of test set as y_test
    y_predicted = model.predict(X_test=X_test) #using the test set to test accuracy
    wineAccuracy = model.evaluate_acc(y_predicted=y_predicted, y_test=y_test)
    accSum = wineAccuracy
    alphas.append(alpha)
    accuracies.append(accSum)

# Plot
plt.scatter(alphas, accuracies, c=(0,0,0), alpha=0.5)
plt.title('Alpha vs Accuracy')
plt.xlabel('Alpha')
plt.ylabel('Accuracy')
plt.show()


###########################################################
# Test different subsets of features on the winedata set  #
# to see if remove some improves accuracy                 #
###########################################################
# Select 3 features from the wine data to train with
# respect to and discard the reset
# ---------------------------------------------------------
# 12 choose 3 results in sample space of 220






"""
#Training on BC data
bc_train, bc_test = train_test_split(bcData)
model = LogisticRegression(data=bc_train)
model.fit(steps=10000)
X_test = bc_test[:, :-1] #delete labels column of wine_test
y_test = bc_test[:,-1][:, np.newaxis] #take last column of test set as y_test
y_predicted = model.predict(X_test=X_test) #using the test set to test accuracy
bcAccuracy = model.evaluate_acc(y_predicted=y_predicted, y_test=y_test)

print("breast cancer model accuracy is")
print(bcAccuracy)
"""