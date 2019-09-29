import numpy as np
from math import floor
from LogisticRegression import LogisticRegression
from LDA import LDA
import utils

def kFoldCrossValidate(dataset, classificationModel, numFolds=5, shuffle='off', alpha=0.0002, steps=10000):
    """
    numFolder               = number of folds (default: 5)
    dataset                 = numpy array of preprocessed dataset with last column = labels
    classificationModel     = "LDA" or "LR"
    shuffle                 = 'on' or 'off' - if shuffle is 'on', shuffle rows using np.random.shuffle
    alpha                   = learning rate for logistic regression
    Split dataset into numFolds (default: 5) equal sections, train model on the other
    numFolds - 1 (default: 4) folds and return average accuracy (as a float decimal) over numFolds folds.

    """
    if classificationModel!='LDA' and classificationModel!='LR':
        return -1 #error

    if shuffle=='on':
        np.random.shuffle(dataset)

    foldsList = np.array_split(dataset, numFolds) #np.array_split splits dataset into a list of numFolds number of folds
    totalAccuracy = 0

    for currentFoldIndex in range(numFolds):

        validationData = foldsList[currentFoldIndex] #assign current fold to validationData
        del foldsList[currentFoldIndex] #remove the current fold from the list...
        trainingData = np.vstack(foldsList)
        # vertically stack the remaining elements in the list creating a matrix of
        # the dataset with validation data removed --> creating the training set
        foldsList.insert(currentFoldIndex, validationData) # add it back at the same index to leave the list unchanged

        if classificationModel=='LDA':
            model = LDA(data=trainingData)
            model.fit()
        elif classificationModel=='LR':
            model = LogisticRegression(data=trainingData)
            model.fit(steps=steps, alpha=alpha)
        X_test = validationData[:,:-1] #remove last col of validationData
        y_predict = model.predict(X_test=X_test)
        y_test = validationData[:,-1][:,np.newaxis] #last col of validationData
        totalAccuracy += utils.evaluate_acc(y_predict=y_predict, y_test=y_test)

    return totalAccuracy / numFolds
