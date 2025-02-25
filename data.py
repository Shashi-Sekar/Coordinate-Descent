import numpy as np
import sklearn.datasets as datasets

class WineData:
    def __init__(self, dataset):
        self.data = dataset()

    def load_data(self, classes = (0,1), train_size = 0.8):
        # Load the data

        X = self.data.data
        y = self.data.target

        #Retrieve only the required classes
        reduced_X = []
        reduced_Y = []
        for class_name in classes:
            indices = np.where(y == class_name)
            reduced_X.append(X[indices])
            reduced_Y.append(y[indices])

        # Convert the list to a numpy array
        reduced_X = np.concatenate(reduced_X)
        reduced_Y = np.concatenate(reduced_Y)

        # Normalize the data
        reduced_X = (reduced_X - reduced_X.min(axis=0)) / (reduced_X.max(axis=0) - reduced_X.min(axis=0))

        # Get the number of samples and features
        num_samples, num_features = reduced_X.shape

        return reduced_X, reduced_Y