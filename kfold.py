import numpy as np
from math import floor

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
    fold_size = floor(dataset_size/num_folds)

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

def kFoldSplit(dataset, numFolds=5):
    """
    Split dataset into numFolds equal sections. Return a list 

    """
