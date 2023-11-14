import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer # Trasform Column 
from sklearn.preprocessing import OneHotEncoder # Kind of Encoding (Transformation)
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

dataset = pd.read_csv('C:/Users/User/Desktop/ML_dataset/loan_old.csv')
missing_values = dataset.isnull().sum()
print("Missing Values:\n", missing_values)
feature_types = dataset.dtypes
print("\nFeature Types:\n", feature_types)
numerical_features = dataset.select_dtypes(include=['int64', 'float64'])
feature_scales_std = numerical_features.std()
print("\nFeature Scales (Standard Deviation):\n", feature_scales_std)
dataset.shape
dataset.dropna(inplace=True)
dataset
missing_values = dataset.isnull().sum()
print("Missing Values:\n", missing_values)
X =  dataset.iloc[: , 1: -2].values #dataset.iloc[: , [0]].values  pandas series 
Y1 = dataset.iloc[:, -2].values
Y2 = dataset.iloc[:, -1].values
X_train, X_test, Y1_train, Y1_test, Y2_train, Y2_test = train_test_split(X, Y1, Y2, test_size=0.2, random_state=1)
le = LabelEncoder()

X_train[:,0] = le.fit_transform(X_train[:,0]) # 1/0 -- Male/Female
X_train[:,1] = le.fit_transform(X_train[:,1]) # 1/0 Yes/No
X_train[:,2] = le.fit_transform(X_train[:,2]) 
X_train[:,3] = le.fit_transform(X_train[:,3]) # 1/0 Not Graduate/Graduate


X_test[:,0] = le.fit_transform(X_test[:,0]) # 1/0 -- Male/Female
X_test[:,1] = le.fit_transform(X_test[:,1]) # 1/0 Yes/No
X_test[:,2] = le.fit_transform(X_test[:,2]) 
X_test[:,3] = le.fit_transform(X_test[:,3]) # 1/0 Not Graduate/Graduate


# kind of transformation, Encoder algo, idx of column, remainder of columns passthrough without any transformations
ct = ColumnTransformer( transformers = [('encoder', OneHotEncoder(), [8])], remainder='passthrough') 
X_train = np.array(ct.fit_transform(X_train)) # fit_transform doesn't return Numpy array 
X_test = np.array(ct.fit_transform(X_test)) # fit_transform doesn't return Numpy array 
Y2_train = le.fit_transform(Y2_train) # 1/0 Y/N
Y2_test = le.fit_transform(Y2_test) # 1/0 Y/N
nf1 = X_train[:, 7:10]  
nf2 = X_test[:, 7:10]  

# Initialize the StandardScaler
scaler = StandardScaler()

# Fit and transform the numerical features
nf_standardized1 = scaler.fit_transform(nf1)
nf_standardized2 = scaler.fit_transform(nf2)

# Replace the original numerical features with the standardized ones in X
X_train[:, 7:10]   = nf_standardized1
X_test[:, 7:10]   = nf_standardized2


import numpy as np




def sigmoid(z):
    """Sigmoid activation function"""
    return 1 / (1 + np.exp(-z))

def initialize_parameters(dim):
    """Initialize weights and bias to zeros"""
    w = np.zeros((dim, 1))
    b = 0
    return w, b

def forward_propagation(X, w, b):
    """Forward propagation to calculate the predicted values"""
    z = np.dot(X, w) + b
    A = sigmoid(z)
    return A

def compute_cost(A, Y):
    """Compute the binary cross-entropy cost"""
    m = len(Y)
    cost = -1/m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))
    return cost

def backward_propagation(X, A, Y):
    """Backward propagation to compute gradients"""
    m = len(Y)
    dz = A - Y
    dw = 1/m * np.dot(X.T, dz)
    db = 1/m * np.sum(dz)
    return dw, db

def update_parameters(w, b, dw, db, learning_rate):
    """Update weights and bias using gradient descent"""
    w = w - learning_rate * dw
    b = b - learning_rate * db
    return w, b

def logistic_regression(X_train, Y_train, num_iterations=1000, learning_rate=0.01):
    """Train a logistic regression model from scratch using gradient descent"""
    m, n = X_train.shape
    w, b = initialize_parameters(n)

    # Gradient Descent
    for i in range(num_iterations):
        # Forward Propagation
        A_train = forward_propagation(X_train, w, b)

        # Compute Cost
        cost_train = compute_cost(A_train, Y_train)

        # Backward Propagation
        dw, db = backward_propagation(X_train, A_train, Y_train)

        # Update Parameters
        w, b = update_parameters(w, b, dw, db, learning_rate)

        # Print cost every 100 iterations
        if i % 100 == 0:
            print(f"Iteration {i}, Training Cost: {cost_train}")

    return w, b

# Training the logistic regression model
num_iterations = 1000
learning_rate = 0.01
w, b = logistic_regression(X_train, Y2_train, num_iterations, learning_rate)


# Now, you can test the model on the new data
# Ensure that you preprocess the new data in the same way as the training data before testing.

# Preprocess the new data
new_data = pd.read_csv('C:/Users/User/Desktop/ML_dataset/loan_new.csv')  # Replace 'path_to_new_data.csv' with the actual path
new_data.Gender = lbl.fit_transform(new_data.Gender)
new_data.Married = lbl.fit_transform(new_data.Married)
new_data.Dependents = lbl.fit_transform(new_data.Dependents)
new_data.Education = lbl.fit_transform(new_data.Education)
new_data.Property_Area = lbl.fit_transform(new_data.Property_Area)

# Standardize numerical features
new_data[numerical_feature_columns] = scaler.transform(new_data[numerical_feature_columns])

# Extract features for testing
X_new_test = new_data.iloc[:, 1:-2].values

# Make predictions on the new data
predictions = forward_propagation(X_new_test, w, b)

# Convert predicted probabilities to binary predictions (0 or 1)
predictions_binary = (predictions >= 0.5).astype(int)

# Display the predictions
print("Predictions on the new data:\n", predictions_binary)


# Save predictions to a text file
output_file_path = 'C:/Users/User/Desktop/ML_dataset/Predictions.txt'

with open(output_file_path, 'w') as file:
    for prediction in predictions_binary:
        file.write(f'{prediction}\n')

print(f'Predictions saved to {output_file_path}')