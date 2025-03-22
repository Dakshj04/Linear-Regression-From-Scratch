import numpy as np
from sklearn.preprocessing import StandardScaler

class LinearRegression:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None
        self.scaler = StandardScaler()  # Feature Scaling

    def fit(self, X, y):
        # Standardize features
        X = self.scaler.fit_transform(X)

        m, n = X.shape
        self.weights = np.zeros((n, 1))
        self.bias = 0

        for i in range(self.iterations):
            y_pred = X @ self.weights + self.bias
            error = y_pred - y

            # Compute gradients
            dw = (1 / m) * (X.T @ error)
            db = (1 / m) * np.sum(error)

            # Update weights
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # Print cost every 100 iterations
            if i % 100 == 0:
                cost = (1 / (2 * m)) * np.sum(error ** 2)
                print(f"Iteration {i}: Cost {cost}")

    def predict(self, X):
        X = self.scaler.transform(X)  # Apply same scaling to test data
        return X @ self.weights + self.bias
