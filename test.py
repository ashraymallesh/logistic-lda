import numpy as np
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
            self.weights += (learningRate * total).T
    
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

# Remove first row from data (row that has column labels)
wine_data = wine_data[1:, :]

# Collapse quality ranking into binary classification problem
# If wine's quality score is greater than or equal to 6 it is good wine (now represented by 1)
# otherwise it is of bad quality (now represented by 0)
quality_data = np.apply_along_axis(lambda x: x >= 6, 0, wine_data[:,11])
quality_data = quality_data.astype(int)

# Remove quality column
wine_data = wine_data[:, :11]

# Normalize wine_data
#wine_data = wine_data / wine_data.max(axis=0)
#print(wine_data)

# Add add column of ones to the end as bias term
bias_column = np.array(np.ones((wine_data.shape[0], 1)))
wine_data = np.append(wine_data, bias_column, axis=1)

# Apply model
model = LogisticRegressionModel(wine_data.shape[1])
model.fit(wine_data, quality_data, 0.01, 100)
errors = 0
for i in range(wine_data.shape[0]):
    if (quality_data[i] == 1 and model.predict(wine_data[i]) < 0.5):
        errors += 1
    elif (quality_data[i] == 0 and model.predict(wine_data[i]) > 0.5):
        errors += 1
    print(quality_data[i], "vs", model.predict(wine_data[i]))
print("%0.2f percent accuracy" % (errors / wine_data.shape[0] * 100))

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