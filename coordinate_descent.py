import numpy as np
import matplotlib.pyplot as plt

from data import WineData

from sklearn.datasets import load_wine
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

import scipy.stats as stats

def compute_confidence_interval(data, confidence=0.95):
    '''
    Compute the confidence interval of the data
    Parameters:
        data: list
        confidence: confidence level
    Returns:
        mean: mean of the data
        confidence_interval: confidence interval
    '''
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), stats.sem(a)
    h = se * stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h

def accuracy(y_true, y_pred):
    '''
    Compute the accuracy of the model
    Parameters:
        y_true: (N, 1) true labels
        y_pred: (N, 1) predicted labels
    Returns:
        accuracy: accuracy of the model
    '''
    return np.mean(y_true == y_pred)

def cross_entropy_loss(y_true, y_pred):
    '''
    Compute the cross-entropy loss between the true labels and the predicted probabilities
    Parameters:
        y_true: (N, 1) true labels
        y_pred: (N, 1) predicted probabilities (model.predict_proba)
    Returns:
        loss: cross-entropy loss
    '''
    return -np.mean( y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred) )

def sigmoid(x):
    '''
    Sigmoid Function
    Parameters:
        x: input
    Returns:
        sigmoid(x)
    '''
    return 1 / (1 + np.exp(-x))

class SteepestCoordinateDescent:
    def __init__(self, num_features, lr=0.01, max_iter=1000000):
        '''
        Initialize the weights to zero
        Parameters:
            num_features: number of features in the data
            lr: learning rate
        '''
        self.max_iter = max_iter
        self.lr = lr
        self.w = np.zeros(num_features).reshape(-1, 1)
        self.bias = 0
        self.losses = []

    def compute_gradient(self, x, y, pred_proba):
        '''
        Compute the gradient of the cross-entropy loss with respect to the weights
        Parameters:
            x: (N, D) data
            y: (N, 1) true labels
            pred_proba: (N, 1) predicted probabilities
        Returns:
            gradient: gradient of the loss with respect to the weights
        '''
        gradient_w = np.dot(x.T, pred_proba - y)
        gradient_b = np.sum(pred_proba - y)
        return gradient_w, gradient_b

    def update(self, weight_index, gradient_w, gradient_b):
        '''
        steepest Update - Update the weight with the steepest gradient
        Parameters:
            weight: index of the weight to update
            gradient: gradient of the loss with respect to the weight
        Returns:
            updated_weight: updated weight
        '''
        self.w[weight_index] = self.w[weight_index] - self.lr * gradient_w
        self.bias = self.bias - self.lr * gradient_b

    def fit(self, X, Y):
        
        losses = []
        num_iter = 0
  
        #for i in range(self.max_iter):
        while((len(losses) == 0 or losses[-1] > 0.0000023) and num_iter <= self.max_iter):
            # Compute the predicted probabilities
            pred_proba = sigmoid(np.dot(X, self.w) + self.bias)

            # Compute the loss
            loss = cross_entropy_loss(Y, pred_proba)
            self.losses.append(loss)

            # Compute the gradients
            gradients_w, gradients_b = self.compute_gradient(X, Y, pred_proba)

            # Find the steepest gradient
            largest = np.argmax(np.abs(gradients_w))

            # Pick the coorindate with the largest gradient
            self.update(largest, gradients_w[largest], gradients_b)

            num_iter += 1
        
        print(f"Number of iterations to converge to L*: {num_iter}")
        return self.losses
    
    def predict_proba(self, X):
        '''
        Predict the probabilities of the test data
        Parameters:
            X: (N, D) data
        Returns:
            pred_proba: (N, 1) predicted probabilities
        '''
        return sigmoid(np.dot(X, self.w) + self.bias)
    
    def plot_loss(self):
        '''
        Plot the loss curve
        '''
        plt.plot(self.losses)
        plt.xlabel(f"Number of iterations {self.max_iter}")
        plt.ylabel(f"Loss")
        plt.title("Loss Curve for the steepest Coordinate Descent")
        plt.show()

class Baseline:
    def __init__(self, num_features, lr=0.01, max_iter=1000000):
        '''
        Initialize the weights to zero
        Parameters:
            num_features: number of features in the data
            lr: learning rate
        '''
        self.max_iter = max_iter
        self.lr = lr
        self.w = np.zeros(num_features).reshape(-1, 1)
        self.bias = 0
        self.losses = []

    def compute_gradient(self, x, y, pred_proba):
        '''
        Compute the gradient of the cross-entropy loss with respect to the weights
        Parameters:
            x: (N, D) data
            y: (N, 1) true labels
            pred_proba: (N, 1) predicted probabilities
        Returns:
            gradient: gradient of the loss with respect to the weights
        '''
        gradient_w = np.dot(x.T, pred_proba - y)
        gradient_b = np.sum(pred_proba - y)
        return gradient_w, gradient_b

    def update(self, weight_index, gradient_w, gradient_b):
        '''
        Random Update - Update a random weight
        Parameters:
            weight: index of the weight to update
            gradient: gradient of the loss with respect to the weight
        Returns:
            updated_weight: updated weight
        '''
        self.w[weight_index] = self.w[weight_index] - self.lr * gradient_w
        self.bias = self.bias - self.lr * gradient_b

    def fit(self, X, Y):
        
        losses = []
        num_iter = 0

        while((len(losses) == 0 or losses[-1] > 0.0000023) and num_iter <= self.max_iter):
            # Compute the predicted probabilities
            pred_proba = sigmoid(np.dot(X, self.w) + self.bias)

            # Compute the loss
            loss = cross_entropy_loss(Y, pred_proba)
            self.losses.append(loss)

            # Compute the gradients
            gradients_w, gradient_b = self.compute_gradient(X, Y, pred_proba)
            
            # Find a random weight to update
            random_index = np.random.randint(0, self.w.shape[0])

            # Update a random weight
            self.update(random_index, gradients_w[random_index], gradient_b)

            num_iter += 1
        
        print(f"Number of iterations to converge to L*: {num_iter}")
        return self.losses
    
    def predict_proba(self, X):
        '''
        Predict the probabilities of the test data
        Parameters:
            X: (N, D) data
        Returns:
            pred_proba: (N, ) predicted probabilities
        '''
        return sigmoid(np.dot(X, self.w) + self.bias)
    
    def plot_loss(self):
        '''
        Plot the loss curve
        '''
        plt.plot(self.losses)
        plt.xlabel(f"Number of iterations {self.max_iter}")
        plt.ylabel(f"Loss")
        plt.title("Loss Curve for the Baseline")
        plt.show()


if __name__ == "__main__":
    print("->Loading the Wine Data....")
    # Load the data
    WineData = WineData(load_wine)

    print("->Reducing the data to a binary classification problem....")
    # Retrieve the training and testing data
    reduced_X, reduced_Y = WineData.load_data()

    print("\n--------\n->Logistic Regression Model")
    print("->Initializing the model....")
    #Logistic Regression Model
    logistic_regression = LogisticRegression(penalty=None, max_iter=100000)

    print("->Train the model....")
    # Fit the model
    logistic_regression.fit(reduced_X, reduced_Y)
    
    print("->Predicting the probabilities....")
    # Predict the training and testing data
    y_pred = logistic_regression.predict_proba(reduced_X)

    print("->Computing the loss....")
    # Final loss value
    l_star = log_loss(reduced_Y, y_pred)
    print("Logistic Regression Loss: {:.8f}".format(l_star))
    
    # Reshape the data to (N, 1)
    reduced_Y = reduced_Y.reshape(-1, 1)
    
    print("\n--------\n->Baseline Coordinate Descent")
    print("-> Baseline is run for 10 times to account for randomness and compute the confidence interval")
    baseline_final_losses = []

    for i in range(10):  
        print("->Initializing the model....")
        # Baseline Coordinate Descent
        baseline = Baseline(reduced_X.shape[1])

        print("->Train the model....")
        baseline_losses = baseline.fit(reduced_X, reduced_Y)

        print("->Predict the probabilities....")
        baseline_pred_proba = baseline.predict_proba(reduced_X)
        baseline_pred = np.where(baseline_pred_proba >= 0.5, 1, 0)

        print("->Compute the loss....")
        baseline_loss = cross_entropy_loss(reduced_Y, baseline_pred_proba)
        print("Baseline Loss: {:.8f}".format(baseline_loss))
        print("Baseline Accuracy: {:.8f}".format(accuracy(reduced_Y, baseline_pred)))

        baseline_final_losses.append(baseline_loss)

    mean, confidence_interval = compute_confidence_interval(baseline_final_losses)
    print(f"Mean Loss: {mean}")
    print(f"Confidence Interval: {confidence_interval}")

    #baseline.plot_loss()
    
    # steepest Coordinate Descent
    print("\n--------\n->steepest Coordinate Descent")
    print("->Initializing the model....")
    steepest = SteepestCoordinateDescent(reduced_X.shape[1])

    print("->Train the model....")
    steepest_losses = steepest.fit(reduced_X, reduced_Y)

    print("->Predict the probabilities....")
    steepest_pred_prob = steepest.predict_proba(reduced_X)
    steepest_pred = np.where(steepest_pred_prob >= 0.5, 1, 0)

    print("->Compute the loss....")
    steepest_loss = cross_entropy_loss(reduced_Y, steepest_pred_prob)
    print("steepest Coordinate Descent Loss: {:.8f}".format(steepest_loss))
    print("steepest Coordinate Descent Accuracy: {:.8f}".format(accuracy(reduced_Y,steepest_pred)))

    #steepest.plot_loss()

    # Plot all the loss curves
    plt.plot(baseline_losses, label="Baseline")
    plt.plot(steepest_losses, label="steepest")
    plt.plot([l_star]*len(baseline_losses), label="Logistic Regression")
    plt.xlabel("Number of iterations")
    plt.ylabel("Loss")
    plt.title("Loss Curve Comparisons")
    plt.legend()
    plt.show()