# %% [markdown]
# # Assignment-2:
# # Abdelrahman Sayed_______20201114
# # Seif El-Din Mohamed_____20200239
# # Shahd Fekry ali_________20201101
# # Mariam Alaa Eldeen______20200525
# # Wessam Fawzy____________20201215
# 

# %% [markdown]
# # Problem-1
# 

# %% [markdown]
# ## Import Libraries

# %%
import pandas as pd  # Pandas for data manipulation
from sklearn.model_selection import train_test_split  # Train-test split for model evaluation
from sklearn.tree import DecisionTreeClassifier  # Decision tree classifier for modeling
from sklearn.metrics import accuracy_score  # Metric for evaluating model performance
from sklearn.compose import ColumnTransformer  # Used for transforming specific columns
from sklearn.preprocessing import OneHotEncoder  # One-hot encoding for categorical variables
import numpy as np  # NumPy for numerical operations
from sklearn.preprocessing import LabelEncoder  # Label encoding for categorical variables
import matplotlib.pyplot as plt  # Matplotlib for plotting

# %% [markdown]
# ## import Data from Dataset

# %%
dataset = pd.read_csv('drug.csv') # Read the dataset from a CSV file named 'drug.csv' using Pandas


# %% [markdown]
# ## Data Preprocessing

# %% [markdown]
# ### Count Missing Values Occurs

# %%
missing_values = dataset.isnull().sum() # Check for missing values in the dataset
print("Missing Values:\n", missing_values) # Print the count of missing values for each column

# %% [markdown]
# ### Handling Missing Data Features

# %%
# We Will Drop Records which have no BP and cholestrol Features
dataset.dropna(subset=['BP', 'Cholesterol'], inplace=True)

# Then We Will Fill Records has no Na_to_K Feature with Average of Na_to_K Values
dataset['Na_to_K'].fillna(dataset['Na_to_K'].mean(), inplace=True)


# %% [markdown]
# ### Double-Check Missing Features
# 

# %%
missing_values = dataset.isnull().sum() # Check for missing values in the dataset
print("Missing Values:\n", missing_values) # Print the count of missing values for each column

# %% [markdown]
# ### Split Dataset Into Feature set and Target set

# %%
# Extract the feature matrix (X) and target variable (Y) from the dataset
X = dataset.iloc[:, :-1].values  # Features (all columns except the last one)
Y = dataset.iloc[:, -1].values   # Target variable (last column)

# Display the feature matrix (X)
print("Feature Matrix (X):\n", X)


# %% [markdown]
# ### Encode Categorical Data in Features (One-Hot Encoding)

# %%
# Define a ColumnTransformer for applying transformations to specific columns
# - 'encoder': OneHotEncoder is used for one-hot encoding
# - [1, 2, 3]: Columns 1, 2, and 3 are one-hot encoded, specified by their indices
# - 'remainder': 'passthrough' indicates that the remaining columns are passed through without any transformations
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1, 2, 3])], remainder='passthrough')

# Apply the transformations defined by ColumnTransformer to the feature matrix (X)
X = np.array(ct.fit_transform(X))


# %% [markdown]
# ### Encode Target Data (Label Encoding)

# %%
le = LabelEncoder() # Use LabelEncoder to encode the target variable (Y)


Y = np.array(le.fit_transform(Y)) # Transform and overwrite the original target variable (Y) with encoded values

# Display the Target matrix (Y)
Y

# %% [markdown]
# ## 1st Experiment

# %%
# Define a list of random states which have high variance to ensure Randomization Facotr
random_states = [1736,0, 123, 789, 987, 654]

# Intialize Indicator Variables with -ve values to detect the maximum Accuracy, best experiment, Tree size of the best experiment and highst accuracy 
max_accuracy = -100
best_experiment = 0
best_treesize = -1

# Loop Over Random States
for i in range(5):
    
    # Split the dataset into training and testing sets using a random state
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=random_states[i])
    
    # Create and train a Decision Tree Classifier model
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Calculate and print the accuracy of the current experiment
    accuracy = accuracy_score(y_test, y_pred)
    
    # Update variables if the current experiment has higher accuracy
    if accuracy >= max_accuracy:
        max_accuracy = accuracy
        best_experiment = i+1
        best_treesize = model.tree_.node_count

    # Print Each Experiment with It's Accuracy and Tree Size
    print(f"Experiment {i+1} with Tree Size: {model.tree_.node_count} and Accuracy: {accuracy}")

print("\n")
# Print Details of Experiment hasing best accuracy
print(f"Best Experiment: {best_experiment}\nAccuracy: {max_accuracy}\nTree Size: {best_treesize}")


# %%
def run_decision_tree_experiment(X, Y, random_states, trn_size):
    
    accuracies = []
    tree_sizes = []

    for i in range(len(random_states)):
        # Split the dataset into training and testing sets using a random state
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 1-trn_size, random_state=random_states[i])

        # Create and train a Decision Tree Classifier model
        model = DecisionTreeClassifier()
        model.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = model.predict(X_test)

        # Calculate accuracy of the current experiment
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)
        tree_sizes.append(model.tree_.node_count)

    # Return details of the best experiment as a tuple
    return np.mean(accuracies), np.max(accuracies) ,np.min(accuracies),np.mean(tree_sizes),np.max(tree_sizes),np.min(tree_sizes)

# %% [markdown]
# ## 2nd Experiment
# 

# %%
train_size = 0.3

# Initialize lists to store statistics
mean_accuracies = []
max_accuracies = []
min_accuracies = []
mean_tree_sizes = []
max_tree_sizes = []
min_tree_sizes = []

# Loop through different training set sizes using while loop
while train_size <= 0.7:
    
    a_mean, a_max, a_min, t_mean, t_max, t_min = run_decision_tree_experiment(X,Y,random_states,train_size)
    
    # Calculate mean, max, and min statistics for the current training set size
    mean_accuracies.append(a_mean) # Append Mean of each experiment with specific train_size
    max_accuracies.append(a_max)
    min_accuracies.append(a_min)
    mean_tree_sizes.append(t_mean)
    max_tree_sizes.append(t_max)
    min_tree_sizes.append(t_min)

    # Increment training_size
    train_size += 0.1

# Display the statistics
report = pd.DataFrame({
    'Training Set Size': [0.3, 0.4, 0.5, 0.6, 0.7],
    'Mean Accuracy': mean_accuracies,
    'Max Accuracy': max_accuracies,
    'Min Accuracy': min_accuracies,
    'Mean Tree Size': mean_tree_sizes,
    'Max Tree Size': max_tree_sizes,
    'Min Tree Size': min_tree_sizes
})

print(report)

# Define the size of the first plot (accuracy plot) to be 9 units in width and 5 units in height
plt.figure(figsize=(9, 5))

# Create a line plot for mean accuracy with black color and add a legend label
plt.plot([0.3, 0.4, 0.5, 0.6, 0.7], mean_accuracies, label='Mean Accuracy', color='black')

# Fill the area between the minimum and maximum accuracy curves with blue color and transparency, and add a legend label
plt.fill_between([0.3, 0.4, 0.5, 0.6, 0.7], min_accuracies, max_accuracies, alpha=0.3, color='blue', label='Accuracy Range')

# Set the title of the accuracy plot
plt.title('Accuracy vs Training Set Size')

# Label the x-axis as 'Training Set Size'
plt.xlabel('Training Set Size')

# Label the y-axis as 'Accuracy'
plt.ylabel('Accuracy')

# Display the legend in the plot
plt.legend()

# Show the accuracy plot
plt.show()

# Define the size of the second plot (tree size plot) to be 9 units in width and 5 units in height
plt.figure(figsize=(9, 5))

# Create a line plot for mean tree size with blue color and add a legend label
plt.plot([0.3, 0.4, 0.5, 0.6, 0.7], mean_tree_sizes, label='Mean Tree Size', color='blue')

# Fill the area between the minimum and maximum tree size curves with black color and transparency, and add a legend label
plt.fill_between([0.3, 0.4, 0.5, 0.6, 0.7], min_tree_sizes, max_tree_sizes, alpha=0.3, color='black', label='Tree Size Range')

# Set the title of the tree size plot
plt.title('Tree Size vs Training Set Size')

# Label the x-axis as 'Training Set Size'
plt.xlabel('Training Set Size')

# Label the y-axis as 'Tree Size'
plt.ylabel('Tree Size')

# Display the legend in the plot
plt.legend()

# Show the tree size plot
plt.show()



