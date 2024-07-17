import numpy as np

class LinearRegression:
    def __init__(self):
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        
        X = np.hstack((np.ones((X.shape[0],1)), X))
        
        self.weights = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
        
        self.bias = self.weights[0]
        self.weights = self.weights[1:]
        
    def predict(self, X):
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        
        y_pred = X.dot(self.weights) + self.bias
        
        return y_pred