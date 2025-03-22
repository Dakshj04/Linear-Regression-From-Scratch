# File: tests/test_linear_regression.py

import numpy as np
import unittest
from models.linear_regression import LinearRegression

class TestLinearRegression(unittest.TestCase):

    def test_gradient_descent(self):
        X = np.array([[1], [2], [3]])
        y = np.array([[2], [3], [4]])
        model = LinearRegression(learning_rate=0.1, epochs=1000)
        model.fit(X, y)
        predictions = model.predict(X)
        self.assertAlmostEqual(predictions[0][0], 2, places=1)

    def test_normal_equation(self):
        X = np.array([[1], [2], [3]])
        y = np.array([[2], [3], [4]])
        model = LinearRegression(method="normal_equation")
        model.fit(X, y)
        predictions = model.predict(X)
        self.assertAlmostEqual(predictions[0][0], 2, places=1)

if __name__ == "__main__":
    unittest.main()
