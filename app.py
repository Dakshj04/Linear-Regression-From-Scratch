import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Define the Linear Regression model with Gradient Descent and Normal Equation
class LinearRegression:
    def __init__(self, learning_rate=0.01, iterations=1000, method="gradient_descent"):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.method = method
        self.weights = None
        self.bias = None
        self.scaler = StandardScaler()  # for feature scaling
        self.cost_history = []  # to store cost during training

    def fit(self, X, y):
        # Standardize features
        X = self.scaler.fit_transform(X)
        m, n = X.shape
        # Initialize parameters
        self.weights = np.zeros((n, 1))
        self.bias = 0
        self.cost_history = []

        if self.method == "gradient_descent":
            for i in range(self.iterations):
                y_pred = X @ self.weights + self.bias
                error = y_pred - y
                # Compute gradients
                dw = (1 / m) * (X.T @ error)
                db = (1 / m) * np.sum(error)
                # Update parameters
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db
                # Compute and record cost
                cost = (1 / (2 * m)) * np.sum(error ** 2)
                self.cost_history.append(cost)
        elif self.method == "normal_equation":
            # Add bias term directly to X
            X_b = np.c_[np.ones((m, 1)), X]
            # Normal Equation: w = (X^T X)^{-1} X^T y
            params = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
            self.bias = params[0, 0]
            self.weights = params[1:]
            # For consistency, we store a single cost value
            y_pred = X_b.dot(params)
            error = y_pred - y
            cost = (1 / (2 * m)) * np.sum(error ** 2)
            self.cost_history.append(cost)

    def predict(self, X):
        X = self.scaler.transform(X)  # use the same scaling as training
        return X @ self.weights + self.bias

def load_data(source):
    try:
        # If source is a file-like object (from Streamlit uploader), reset its pointer
        if hasattr(source, 'read'):
            source.seek(0)
        df = pd.read_csv(source)
        if df.empty:
            st.error("The CSV file is empty. Please upload a valid file with data.")
        return df.iloc[:, :-1].values, df.iloc[:, -1].values.reshape(-1, 1)
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        return None, None

# Streamlit App
st.title("Linear Regression Model Demonstration")
st.markdown(
    """
This demo showcases a Linear Regression model built from scratch using Gradient Descent 
and the Normal Equation. Adjust the parameters in the sidebar and click **Train Model** to begin.
"""
)

# Sidebar controls
st.sidebar.header("Model Parameters")
learning_rate = st.sidebar.number_input("Learning Rate", value=0.01, format="%.3f")
iterations = st.sidebar.number_input("Iterations", value=1000, step=100)
method = st.sidebar.selectbox("Method", ["gradient_descent", "normal_equation"])

# Option to upload a custom CSV file or use default synthetic data
data_file = st.sidebar.file_uploader("Upload your CSV data", type=["csv"])
if data_file is not None:
    df = pd.read_csv(data_file)
    st.write("Using Uploaded Dataset")
else:
    # Load default synthetic data
    df = pd.read_csv("data/synthetic_data.csv")
    st.write("Using Default Synthetic Dataset")

st.subheader("Dataset Preview")
st.write(df.head())

# Prepare data
X, y = load_data("data/synthetic_data.csv") if data_file is None else load_data(data_file)

# Train the model when the button is clicked
if st.button("Train Model"):
    model = LinearRegression(learning_rate=learning_rate, iterations=iterations, method=method)
    model.fit(X, y)
    
    st.success("Model training completed!")
    st.write(f"**Final Weights:** {model.weights.flatten()}")
    st.write(f"**Final Bias:** {model.bias:.4f}")
    
    # Plot cost history if using gradient descent
    if method == "gradient_descent":
        fig, ax = plt.subplots()
        ax.plot(model.cost_history)
        ax.set_xlabel("Iterations")
        ax.set_ylabel("Cost")
        ax.set_title("Cost Function Convergence")
        st.pyplot(fig)
    else:
        st.write(f"Cost computed using Normal Equation: {model.cost_history[0]:.4f}")
    
    # Make predictions on test data
    st.subheader("Make Predictions")
    test_input = st.text_input("Enter test data (comma-separated values for each feature)", "5,3,4")
    try:
        test_features = np.array([float(x.strip()) for x in test_input.split(",")]).reshape(1, -1)
        prediction = model.predict(test_features)
        st.write(f"**Prediction:** {prediction[0,0]:.4f}")
    except Exception as e:
        st.error("Invalid input. Please ensure you enter numeric values separated by commas.")
