# %% [markdown]
# # Assignment-1 Abdelrahman Sayed 

# %% [markdown]
# ## Import Libraries

# %%
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

# %% [markdown]
# ## Import Data From Dataset

# %%
dataset = pd.read_csv('loan_old.csv')


# %% [markdown]
# ## Analyze some data

# %% [markdown]
# ### Checking Missing Values
# 

# %%
missing_values = dataset.isnull().sum()
print("Missing Values:\n", missing_values)

# %% [markdown]
# ### Type of Features

# %%
feature_types = dataset.dtypes
print("\nFeature Types:\n", feature_types)

# %% [markdown]
# ### Check The Scale

# %%
numerical_features = dataset.select_dtypes(include=['int64', 'float64'])
feature_scales_std = numerical_features.std()
print("\nFeature Scales (Standard Deviation):\n", feature_scales_std)

# %% [markdown]
# ### Visualize pairplot between numerical values
# 

# %%
sns.pairplot(numerical_features)
plt.show()

# %% [markdown]
# # Assignment-1 Seif El Din Mohamed

# %% [markdown]
# ### Dataset Size before removing records

# %%
dataset.shape

# %% [markdown]
# ### Drop Records with empty values

# %%
dataset.dropna(inplace=True)
dataset

# %% [markdown]
# ### Check Missing Values after Deleting empty value records
# 

# %%
missing_values = dataset.isnull().sum()
print("Missing Values:\n", missing_values)

# %% [markdown]
# ### Dataset Size after removing records

# %%
dataset.shape

# %% [markdown]
# ## Separate Dataset

# %%
X =  dataset.iloc[: , 1: -2].values #dataset.iloc[: , [0]].values  pandas series 
Y1 = dataset.iloc[:, -2].values
Y2 = dataset.iloc[:, -1].values


# %% [markdown]
# ## Train & Test Split

# %%
X_train, X_test, Y1_train, Y1_test, Y2_train, Y2_test = train_test_split(X, Y1, Y2, test_size=0.2, random_state=1)

# %%
print(X_train)

# %%
print(Y1_train)


# %%
print(Y2_train)

# %% [markdown]
# ## Encoding Features

# %%

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



# %% [markdown]
# ## Encoding Targets

# %%
Y2_train = le.fit_transform(Y2_train) # 1/0 Y/N 
Y2_test = le.fit_transform(Y2_test) # 1/0 Y/N


# %% [markdown]
# ##  numerical features are standardized
# 

# %%
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

# %%
X_train[:,7:10]

# %% [markdown]
# ## Linear Regression
# 

# %% [markdown]
# ### Train & Fit The Model

# %%
linear_model = LinearRegression()
linear_model.fit(X_train, Y1_train)

# %% [markdown]
# ### Predict New Values

# %%
Y_predict  = linear_model.predict(X_test)

# %% [markdown]
# ### R-Squared Error Evaluation

# %%
r2 = r2_score(Y1_test, Y_predict)
print(f'R-squared score: {r2}')

# %%
print(X_train[:,7:])

# %% [markdown]
# ## Logistic Regression From Scratch

# %%
def sigmoid(z):
    """Sigmoid activation function"""
    return 1 / (1 + np.exp(-z))

def initialize_parameters(dim):
    """Initialize weights and bias to zeros"""
    w = np.zeros((dim, 1))
    b = np.zeros((X_train.shape[0], 1))  # Ensure the correct shape
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

# %% [markdown]
# ## Train Logistic Regression Model

# %%
num_iterations = 1000
learning_rate = 0.01
w, b = logistic_regression(np.array(X_train), np.array(Y2_train), num_iterations, learning_rate)


