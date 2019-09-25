import numpy as np
import pandas as pd
import math


# Logistic function
def logistic(a):
    return 1 / (1 + np.exp(-a))


class LogisticRegressionModel:
    def __init__(self, numFeatures):
        # Save the number of features this model has
        self.numFeatures = numFeatures

        # Initialize a weights matrix
        self.weights = np.zeros((numFeatures, 1))

    
    def fit(self, X, Y, learningRate, steps):
        # Perform gradient descent "steps" number of times
        for s in range(steps):
            # Iterate through each row in the features and category matrixes add
            # them to sum (which is porportional to how weights get adjusted)
            total = np.zeros((1, self.numFeatures))
            for i in range(X.shape[0]):
                total += X[i] * float(Y[i] - logistic(self.weights.T @ X[i]))
            
            # Adjust the weights by the sum multipled by the learning weight
            self.weights = self.weights + (learningRate * total).T
    
    """
    Takes a feature vector "X" and a binary 0/1 classification
    based on trained weights
    """
    def predict(self, X):
        # Use the weights trained in the fit method to predict what class
        # an object with a certain feature vector "X" will be from
        return logistic(X @ self.weights)


# Import csv file into 
wine_data = np.genfromtxt('winequality-red.csv', delimiter=';')
cancer_data = pd.read_csv("breast-cancer-wisconsin.data", sep=',', header=None)
cancer_data.columns = ['Sample code', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape',
                'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin',
                'Normal Nucleoli', 'Mitoses', 'Class']
cancer_data.replace('?', np.NaN, inplace=True)
cancer_data = cancer_data.dropna().to_numpy().astype(float)

# Remove first row from data (row that has column labels)
wine_data = wine_data[1:, :]

# Collapse quality ranking into binary classification problem
# If wine's quality score is greater than or equal to 6 it is good wine (now represented by 1)
# otherwise it is of bad quality (now represented by 0)
quality_data = np.apply_along_axis(lambda x: x >= 6, 0, wine_data[:,11])
quality_data = quality_data.astype(int)

# Determine class vector for tumors
cancer_class_data = np.apply_along_axis(lambda x: x == 4, 0, cancer_data[:,10])
cancer_class_data = cancer_class_data.astype(int)

# Remove quality column
wine_data = wine_data[:, :11]
cancer_data = cancer_data[:, 1:10]

# Normalize data
wine_data = wine_data / wine_data.max(axis=0)
cancer_data = cancer_data / cancer_data.max(axis=0)

# Add add column of ones to the end as bias term
bias_column = np.array(np.ones((wine_data.shape[0], 1)))
wine_data = np.append(wine_data, bias_column, axis=1)
bias_column = np.array(np.ones((cancer_data.shape[0], 1)))
cancer_data = np.append(cancer_data, bias_column, axis=1)

# Apply model
model = LogisticRegressionModel(cancer_data.shape[1])
model.fit(cancer_data, cancer_class_data, 0.0001, 1000)
errors = 0
for i in range(cancer_data.shape[0]):
    if (cancer_class_data[i] == 1 and model.predict(cancer_data[i]) < 0.5):
        errors += 1
    elif (cancer_class_data[i] == 0 and model.predict(cancer_data[i]) > 0.5):
        errors += 1
    print(cancer_class_data[i], "vs", model.predict(cancer_data[i]))
print("%0.2f percent accuracy" % ((1 - errors / cancer_data.shape[0]) * 100))
