from scipy.special import expit
import numpy as np
from math import floor

#Function for Task 2
def evaluate_acc(y_predict, y_test):
        """
        y_predict and y_test are vectors
        Check the accuracy of y_predict by comparing it to y_test
        Returns a percentage accuracy (float)
        """

        test_set_size = y_test.shape[0]
        numErrors = np.sum( np.abs(y_predict - y_test) )
        errorPercentage = (numErrors/test_set_size) * 100
        accuracy = 100 - errorPercentage
        return accuracy

#Helper functions:
def sigmoid(z):
    #using the numerically stable expit function from scipy to avoid overflow errors
    return expit(z)

def classifyLogistic(v):
    """ 
    v is a vector (numpy ndarray) of probabilities
    classify each example (row) 1 if >=0.5 and 0 otherwise
    """
    v[v>=0.5] = 1
    v[v<0.5] = 0
    return v

def classifyLDA(v):
    """ 
    v is a vector (numpy ndarray) of log-odds ratios
    classify each example (row) 1 if >0 and 0 otherwise
    """
    v[v>0] = 1
    v[v<=0] = 0
    return v

def train_test_split(dataset, ratio=0.2):
    """
    split dataset into training and test subsets
    test set size will be ratio * dataset size (default = 20% of total size)
    returns data_train, data_test
    """
    dataset_size = dataset.shape[0]
    test_set_size = floor(ratio * dataset_size)
    training_set_size = dataset_size - test_set_size
    data_train = dataset[0:training_set_size, :]
    data_test = dataset[test_set_size:, :]
    return data_train, data_test