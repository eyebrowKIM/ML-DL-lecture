import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self, X, y, regularization=None, alpha=0.1, lr=3e-4):
        self.X = np.hstack((np.ones((X.shape[0],1)), X))
        self.y = y
        
        self.regularization = regularization
        self.alpha = alpha
        self.lr = lr
    
    def _compute_loss(self, X, y):
        m = y.shape[0]
        
        h = np.dot(X, self.weights)
        
        loss = 1/(2*m) * np.sum(np.square(h-y))
        
        return loss
    
    def _compute_loss_l1(self, X, y):
        m = y.shape[0]
        
        h = np.dot(X, self.weights)
        
        loss = 1/(2*m) * np.sum(np.square(h-y)) + self.alpha * np.sum(np.abs(self.weights))
        
        return loss
    
    def _compute_loss_l2(self, X, y):
        m = y.shape[0]
        
        h = np.dot(X, self.weights)
        
        loss = 1/(2*m) * np.sum(np.square(h-y)) + self.alpha * np.sum(np.square(self.weights))
            
        return loss
    
    def _compute_grad(self, X, y):
        m = y.shape[0]
        
        h = np.dot(X, self.weights)
        
        grad = 1/m * np.dot(X.T, h-y)
        
        return grad
    
    def _compute_grad_l1(self, X, y):
        m = y.shape[0]
        
        h = np.dot(X, self.weights)
        
        grad = 1/m * np.dot(X.T, h-y) + self.alpha * np.sign(self.weights)
        
        return grad
    
    def _compute_grad_l2(self, X, y):
        m = y.shape[0]
        
        h = np.dot(X, self.weights)
        
        grad = 1/m * np.dot(X.T, h-y) + self.alpha * self.weights
        
        return grad
    
    # Compute cost function
    def compute_loss(self, X, y):
        if self.regularization == 'l1':
            loss = self._compute_loss_l1(X, y)
        elif self.regularization == 'l2':
            loss = self._compute_loss_l2(X, y)
        else:
            loss = self._compute_loss(X, y)
        return loss
    
    # Compute gradient
    def compute_gradient(self, X, y):
        if self.regularization == 'l1':
            grad = self._compute_grad_l1(X, y)
        elif self.regularization == 'l2':
            grad = self._compute_grad_l2(X, y)
        else:
            grad = self._compute_grad(X, y)
        return grad
    
    # Fitting
    def fit(self):
        self.weights = np.random.rand(self.X.shape[1])
        self.bias = self.weights[0]
        
        for i in range(10000):
            grad = self.compute_gradient(self.X, self.y)
            
            self.weights -= self.lr * grad
            
            if i == len(range(10000)) - 1:
                loss = self.compute_loss(self.X, self.y)
                print(f'Loss at iteration {i}: {loss}')
                print(f'Weights: {self.weights}')
    
    def fit_analytic(self):
        self.weights = np.linalg.inv(self.X.T.dot(self.X)).dot(self.X.T).dot(self.y)
        self.bias = self.weights[0]
        
        loss = self.compute_loss(self.X, self.y)
        print(f'Loss: {loss}')
        print(f'Weights: {self.weights}')
        
        
    def predict(self, X):
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        
        y_pred = X.dot(self.weights)  # Remove self.bias here
        
        return y_pred

def preprocessing_student_data(data: pd.DataFrame):
    # Yes,No -> 1, 0
    data['Extracurricular Activities'] = data['Extracurricular Activities'].map({'Yes': 1, 'No': 0})
    
    # Last column : target
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    
    return X, y

def main():
    data_path = 'data/Student_Performance.csv'
    
    data = pd.read_csv(data_path)

    X, y = preprocessing_student_data(data)
    
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2)

    model_lr = LinearRegression(X_tr, y_tr)
    model_lr_l1 = LinearRegression(X_tr, y_tr, regularization='l1', alpha=0.1)
    model_lr_l2 = LinearRegression(X_tr, y_tr, regularization='l2', alpha=0.1)
    model_lr_a = LinearRegression(X_tr, y_tr)
    
    print("Fitting model without regularization:")
    model_lr.fit()
    
    print("\nFitting model with L1 regularization:")
    model_lr_l1.fit()
    
    print("\nFitting model with L2 regularization:")
    model_lr_l2.fit()
    
    # analytic linear regression
    print("\nFitting model without regularization (analytic):")
    model_lr_a.fit_analytic()
    
    # validation
    y_pred = model_lr.predict(X_val)
    y_pred_l1 = model_lr_l1.predict(X_val)
    y_pred_l2 = model_lr_l2.predict(X_val)
    y_pred_a = model_lr_a.predict(X_val)
    
    # subplot
    fig, ax = plt.subplots(1, 4, figsize=(15, 5))
    
    ax[0].scatter(y_val, y_pred)
    ax[0].set_title('Without Regularization')
    
    ax[1].scatter(y_val, y_pred_l1)
    ax[1].set_title('L1 Regularization')
    
    ax[2].scatter(y_val, y_pred_l2)
    ax[2].set_title('L2 Regularization')
    
    ax[3].scatter(y_val, y_pred_a)
    ax[3].set_title('Analytic Linear Regression')
    
    plt.savefig("result/mygraph.png")

    
    # evaluation
    mse = mean_squared_error(y_val, y_pred)
    mse_l1 = mean_squared_error(y_val, y_pred_l1)
    mse_l2 = mean_squared_error(y_val, y_pred_l2)
    mse_a = mean_squared_error(y_val, y_pred_a)
    
    print(f'MSE without regularization: {mse}')
    print(f'MSE with L1 regularization: {mse_l1}')
    print(f'MSE with L2 regularization: {mse_l2}')
    print(f'MSE with analytic linear regression: {mse_a}')

if __name__ == '__main__':
    main()