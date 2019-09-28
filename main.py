import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from LDA import LDA
from LogisticRegression import LogisticRegression
import utils, kfold

#Data cleaning and preprocessing
winedf = pd.read_csv("winequality-red.csv", sep=';')
wineData = winedf.to_numpy()
wineData[:,-1] = (wineData[:,-1]>=6).astype(int) #convert 6+ to 1 and <5 to 0
wineData = wineData/wineData.max(axis=0) #normalize data

bcdf = pd.read_csv("breast-cancer-wisconsin.data", sep=',', header=None)
bcdf.columns = ['Sample Code Number', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape',
                'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin',
                'Normal Nucleoli', 'Mitoses', 'Class']
bcdf.replace('?', np.NaN, inplace=True)
bcdf.dropna(inplace=True)
bcdf.drop(['Sample Code Number'], axis=1, inplace=True)
bcData = bcdf.to_numpy().astype(float)
bcData[:,-1] = (bcData[:,-1]>3).astype(int) #change 2/4 last column to 0/1 labels
bcData = bcData/bcData.max(axis=0) #normalize data

model = LDA(bcData)
model.fit()
X_test = bcData[:,:-1]
y_predict = model.predict(X_test)
y_test = bcData[:,-1][:,np.newaxis]
acc = utils.evaluate_acc(y_predict, y_test)
print(acc)

"""
#testing on the whole dataset
model = LogisticRegression(data=wineData)
model.fit(steps=10000)
X_test = wineData[:,:-1]
y_test = wineData[:,-1][:, np.newaxis]
"""

"""
#split data into train and test sets
data_train, data_test = train_test_split(wineData)
model = LogisticRegression(data=data_train)
model.fit(steps=10000)
X_test = data_test[:,:-1]
y_test = data_test[:,-1][:,np.newaxis]


y_predict = model.predict(X_test=X_test)
acc = model.evaluate_acc(y_predict=y_predict, y_test=y_test)
print("wine accuracy is")
print(acc)
"""

"""
#Training on wine data
num_folds = 5
wine_folds = k_fold_split(wineData, num_folds)
wine_train, wine_test = train_test_split(dataset=wineData)

# Training wine data using "num_folds" number of folds

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
"""
####################################
# TESTING DIFFERENT LEARNING RATES #
####################################
alphas = []
accuracies = []
for alpha in np.arange(0, 0.001, 0.00005):
    accSum = 0
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
    alphas.append(alpha)
    accuracies.append(accSum)

# Plot

plt.scatter(alphas, accuracies, c=(0,0,0), alpha=0.5)
plt.title('Alpha vs Accuracy')
plt.xlabel('Alpha')
plt.ylabel('Accuracy')
plt.show()

"""


###########################################################
# Test different subsets of features on the winedata set  #
# to see if remove some improves accuracy                 #
###########################################################
# Select 3 features from the wine data to train with
# respect to and discard the reset
# ---------------------------------------------------------
# 12 choose 3 results in sample space of 220






"""
#Training on entire BC data
model = LogisticRegression(data=bcData)
model.fit(steps=10000)
X_test = bcData[:, :-1] #delete labels column of bcData
y_test = bcData[:,-1][:, np.newaxis] #take last column of bcData set as y_test
y_predicted = model.predict(X_test=X_test) #using the test set to test accuracy
bcAccuracy = model.evaluate_acc(y_predict=y_predicted, y_test=y_test)

print("breast cancer model accuracy is")
print(bcAccuracy)

"""