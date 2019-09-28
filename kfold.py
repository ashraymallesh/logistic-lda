import numpy as np
from math import floor
from LogisticRegression import LogisticRegression
from LDA import LDA
import utils

def kFoldCrossValidation(dataset, model, numFolds=5, shuffle='off'):
    """
    numFolder   = number of folds (default: 5)
    dataset		= numpy array of preprocessed dataset with last column = labels
    model		= "LDA" or "LR"
    shuffle		= 'on' or 'off' - if shuffle is 'on', shuffle rows using np.random.shuffle
    Split dataset into numFolds (default: 5) equal sections, train model on the other
    k - 1 (default: 4) folds and return average accuracy (as a float decimal) over 5 folds.

    """
    if shuffle=='on':
    	np.random.shuffle(dataset)

    totalAccuracy = 0

    foldsList = np.array_split(dataset, numFolds) #np.array_split splits dataset into a list of numFolds number of folds
    
    for foldIndex in range(numFolds):

    	validationData = foldsList[foldIndex] #assign current fold to validationData
    	del foldsList[foldIndex] #remove the current fold from the list...
    	trainingData = np.vstack(foldsList)
    	# vertically stack the remaining elements in the list creating a matrix of
    	# the dataset with validation data removed --> creating the training set
    	foldsList.insert(foldIndex, validationData) # add it back at the same index

    	if model=="LDA":
    		LDAmodel = LDA(data=trainingData)
    		LDAmodel.fit()
    		X_test = validationData[:,:-1] #remove last col of validationData
    		y_predict = LDAmodel.predict(X_test)
    	elif model=="LR":
    		LRmodel = LogisticRegression(data=trainingData)
    		LRmodel.fit(steps=10000)
    		X_test = validationData[:,:-1] #remove last col of validationData
    		y_predict = LRmodel.predict(X_test)

    	y_test = validationData[:,-1][:,np.newaxis] #last col of validationData
    	totalAccuracy += utils.evaluate_acc(y_predict, y_test)

    return totalAccuracy / numFolds
