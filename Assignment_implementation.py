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

# %% [markdown]
# ## Encoding Features

# %%

le = LabelEncoder()

X_train[:,0] = np.array(le.fit_transform(X_train[:,0])) # 1/0 -- Male/Female
X_train[:,1] = np.array(le.fit_transform(X_train[:,1])) # 1/0 Yes/No
X_train[:,2] = np.array(le.fit_transform(X_train[:,2]))
X_train[:,3] = np.array(le.fit_transform(X_train[:,3])) # 1/0 Not Graduate/Graduate
# X_train[:,8] = le.fit_transform(X_train[:,8]) # 1/0 Not Graduate/Graduate


X_test[:,0] = np.array(le.fit_transform(X_test[:,0])) # 1/0 -- Male/Female
X_test[:,1] = np.array(le.fit_transform(X_test[:,1])) # 1/0 Yes/No
X_test[:,2] = np.array(le.fit_transform(X_test[:,2])) 
X_test[:,3] = np.array(le.fit_transform(X_test[:,3])) # 1/0 Not Graduate/Graduate
# X_test[:,8] = le.fit_transform(X_test[:,8]) # 1/0 Not Graduate/Graduate


# kind of transformation, Encoder algo, idx of column, remainder of columns passthrough without any transformations
ct = ColumnTransformer( transformers = [('encoder', OneHotEncoder(), [8])], remainder='passthrough') 
X_train = np.array(ct.fit_transform(X_train)) # fit_transform doesn't return Numpy array 
X_test = np.array(ct.fit_transform(X_test)) # fit_transform doesn't return Numpy array 



# %% [markdown]
# ## Encoding Targets

# %%
Y2_train = np.array(le.fit_transform(Y2_train)) # 1/0 Y/N 
Y2_test = np.array(le.fit_transform(Y2_test)) # 1/0 Y/N


# %% [markdown]
# ##  numerical features are standardized
# 

# %%
nf1 = X_train[:, 7:10]  
nf2 = X_test[:, 7:10]  

# Initialize the StandardScaler
scaler = StandardScaler()

# Fit and transform the numerical features
nf_standardized1 = np.array(scaler.fit_transform(nf1))
nf_standardized2 = np.array(scaler.fit_transform(nf2))

# Replace the original numerical features with the standardized ones in X
X_train[:, 7:10]   = np.array(nf_standardized1)
X_test[:, 7:10]   = np.array(nf_standardized2)

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

# %% [markdown]
# ## Logistic Regression From Scratch

# %%
def sigmoid(x):
    return 1/(1+np.exp(-x))

class LogisticRegression():

    def __init__(self, lr=0.001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            linear_pred = np.array(np.dot(X, self.weights),dtype=np.float32) + self.bias
            predictions = sigmoid(linear_pred)

            dw = (1/n_samples) * np.array(np.dot(X.T, (predictions - y)),dtype=np.float32)
            db = (1/n_samples) * np.sum(predictions-y)

            self.weights = self.weights - self.lr*dw
            self.bias = self.bias - self.lr*db


    def predict(self, X):
        linear_pred = np.array(np.dot(X, self.weights), dtype=np.float32) + self.bias
        y_pred = sigmoid(linear_pred)
        class_pred = [0 if y<=0.5 else 1 for y in y_pred]
        return class_pred

# %% [markdown]
# ## Train Logistic Regression Model

# %%
clf = LogisticRegression(lr=0.15)
clf.fit(X_train,Y2_train)
y_pred = clf.predict(X_test)

def accuracy(y_pred, y_test):
    return np.sum(y_pred==y_test)/len(y_test)

acc = accuracy(y_pred, Y2_test)
print(acc)

# %% [markdown]
# # Load Newloan Data

# %%
new_dataset = pd.read_csv('loan_new.csv')
new_dataset.dropna(inplace=True)


# %% [markdown]
# ## Preprocessing New Data

# %%
new_X =  new_dataset.iloc[: , 1: ].values #dataset.iloc[: , [0]].values  pandas series

# %% [markdown]
# ### Encoding

# %%
lbl = LabelEncoder()

new_X[:,0] = np.array(le.fit_transform(new_X[:,0])) # 1/0 -- Male/Female
new_X[:,1] = np.array(le.fit_transform(new_X[:,1])) # 1/0 Yes/No
new_X[:,2] = np.array(le.fit_transform(new_X[:,2]))
new_X[:,3] = np.array(le.fit_transform(new_X[:,3])) # 1/0 Not Graduate/Graduate

# kind of transformation, Encoder algo, idx of column, remainder of columns passthrough without any transformations
new_ct = ColumnTransformer( transformers = [('encoder', OneHotEncoder(), [8])], remainder='passthrough') 
new_X = np.array(new_ct.fit_transform(new_X)) # fit_transform doesn't return Numpy array 

# %% [markdown]
# ### Standardization

# %%
new_nf = new_X[:, 7:10]

# Initialize the StandardScaler
new_scaler = StandardScaler()

# Fit and transform the numerical features
new_nf_standardized = np.array(new_scaler.fit_transform(new_nf))

# Replace the original numerical features with the standardized ones in X
new_X[:, 7:10]   = np.array(new_nf_standardized)


# %% [markdown]
# # Predict New Values

# %%
new_Y2_pred = clf.predict(new_X)
new_Y1_pred = linear_model.predict(new_X)

# %% [markdown]
# # Write Predicted Values on new Excel File

# %%
df = pd.DataFrame(new_dataset)
df['Max Loan Amount'] = new_Y1_pred
df['Loan Status'] = new_Y2_pred
df.to_csv('D:\\4th_1st_Sem\\Machine Learning\\Assignment 1\\PREDICTED_LOAN.csv', index=False)
print(df)



