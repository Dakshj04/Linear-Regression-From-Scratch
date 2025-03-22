# File: models/utils.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_data(file_path):
    """Load dataset from CSV file"""
    df = pd.read_csv(file_path)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values.reshape(-1, 1)
    return X, y

def plot_cost_history(cost_history):
    """Plot cost history for gradient descent"""
    plt.plot(cost_history)
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.title("Cost Function Convergence")
    plt.show()
