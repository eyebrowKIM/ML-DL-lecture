import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

def load_data():
    data_path = 'data/classification.csv'
    df = pd.read_csv(data_path)
    df = df.dropna()
    df = df.drop(['User ID','Gender'], axis=1)
    
    return df

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

class LogisticRegresson:
    def __init__(self, X, y):
        self.X = np.hstack((np.ones((X.shape[0],1)), X))
        self.y = y
        self.W = np.random.rand(self.X.shape[1])

    
    def _compute_cost(self):
        m = self.y.shape[0]
        
        h = sigmoid(z=np.dot(self.X, self.W))
        
        h = h.reshape(-1, 1)
        
        loss = -1/m * np.sum(self.y * np.log(h) + (1-self.y) * np.log(1-h))
        
        return loss
    
    def _compute_grad(self):
        m = self.y.shape[0]
        
        h = sigmoid(np.dot(self.X, self.W))
        
        h = h.reshape(-1, 1)
        
        grad = (1/m * np.dot(self.X.T, h - self.y)).reshape(-1,)
        
        
        return grad
        
    def fit(self, lr=1e-3, threshold=1e-4, max_iter=100000):
        for i in range(max_iter):
            cost = self._compute_cost()
            
            if cost < threshold:
                break
            self.W -= lr * self._compute_grad()
        
            if i % 100 == 0:
                print(f'Cost at iteration {i}: {cost}')
                print(f'W: {self.W}')
            
    def predict(self, X):
        X = np.hstack((np.ones((X.shape[0],1)), X))
        
        y_pred = sigmoid(np.dot(X, self.W))
        
        return y_pred
    
    def eval(self, X, y):
        y_pred = self.predict(X)
        
        y_pred = np.where(y_pred > 0.5, 1, 0)
        
        acc = np.sum(y_pred == y) / y.shape[0]
        
        return acc
    
def visualize_3d(X, y):
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    X_class0 = X[y.flatten() == 0]
    X_class1 = X[y.flatten() == 1]
    y_class0 = y[y.flatten() == 0]
    y_class1 = y[y.flatten() == 1]
    
    ax.scatter(X_class0[:,0], X_class0[:,1], y_class0, c='r', marker='o', label='class 0')
    ax.scatter(X_class1[:,0], X_class1[:,1], y_class1, c='b', marker='x', label='class 1')
    
    ax.set_xlabel('Age')
    ax.set_ylabel('EstimatedSalary')
    ax.set_zlabel('Purchased')
    
    ax.legend()
    
    plt.show()
        
def main():
    df = load_data()
    X_columns = ['Age', 'EstimatedSalary']
    y_columns = ['Purchased']
    
    df['EstimatedSalary'] = StandardScaler().fit_transform(df['EstimatedSalary'].values.reshape(-1, 1))
    
    X = df[X_columns].values # (400, 2)
    y = df[y_columns].values # (400, 1)
    

    X_tr, X_val = X[:300], X[300:]
    y_tr, y_val = y[:300], y[300:]

    model_lr = LogisticRegresson(X_tr, y_tr)
    
    model_lr.fit()
    
    print(f'Accuracy: {model_lr.eval(X_val, y_val)}')
    
    visualize_3d(X_val, y_val)
    

if __name__ == '__main__':
    main()