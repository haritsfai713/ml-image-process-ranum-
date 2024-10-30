# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Step 1: Load the data
# Assuming you have a CSV file for the dataset
# Replace 'your_data.csv' with your actual file path or dataframe if already loaded
data = pd.read_csv('color_data_with_clusters.csv')

# Step 2: Inspect the data
print("Data preview:")
print(data.head())

# Step 3: Split the data into features (L, a, b, aspect ratio, firmness) and target (sugar content)
# Adjust the column names based on your actual dataset
X = data[['L', 'a', 'b']]  # Features
y = data['TAT']  # Target

# Step 4: Normalize the feature data
scaler = MinMaxScaler()  # You can also use StandardScaler for Z-score normalization
X_normalized = scaler.fit_transform(X)

# Convert the normalized data back to a DataFrame for easy viewing
X_normalized_df = pd.DataFrame(X_normalized, columns=X.columns)

print("Normalized feature data:")
print(X_normalized_df.head())

# Step 5: Split the data into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_normalized_df, y, test_size=0.2, random_state=42)

# Optional: Check the shape of the training and test sets
print("Training set size:", X_train.shape)
print("Test set size:", X_test.shape)

# Import necessary libraries
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

# # Step 1: Train a Random Forest Regressor
# # Initialize the model
# rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# # Train the model using the training data
# rf_model.fit(X_train, y_train)

# # Step 2: Evaluate the model on the test set
# # Predict sugar content using the test set
# y_pred = rf_model.predict(X_test)

# # Calculate performance metrics
# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)

# print(f"Mean Squared Error (MSE): {mse}")
# print(f"R-squared (R2): {r2}")

# # Step 3: Extract and visualize feature importance
# # Get feature importance scores
# importances = rf_model.feature_importances_

# # Create a bar plot of feature importance
# features = X_train.columns
# indices = np.argsort(importances)

# plt.figure(figsize=(10, 6))
# plt.title("Feature Importance")
# plt.barh(range(len(indices)), importances[indices], color='b', align='center')
# plt.yticks(range(len(indices)), [features[i] for i in indices])
# plt.xlabel("Relative Importance")
# plt.show()

# # Step 4: Print feature importance values for easier reading
# importance_df = pd.DataFrame({
#     'Feature': features,
#     'Importance': importances
# }).sort_values(by='Importance', ascending=False)

# print("Feature importance:")
# print(importance_df)

# Import necessary libraries
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

# Step 1: Define the parameter grid
param_grid = {
    'n_estimators': [50, 100, 200,500],
    'max_depth': [None, 10, 20, 30, 40],
    'min_samples_split': [2, 5, 10, 15],
    'min_samples_leaf': [1, 2, 4, 5],
    'max_features': ['auto', 'sqrt', 'log2']
}

# Step 2: Initialize the Random Forest Regressor
rf_model = RandomForestRegressor(random_state=42)

# Step 3: Use GridSearchCV to search for the best parameters
# 5-fold cross-validation will be used to assess the performance
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, 
                           cv=5, n_jobs=-1, verbose=2, scoring='r2')

# Step 4: Fit the model on the training data
grid_search.fit(X_train, y_train)

# Step 5: Print the best parameters and the corresponding R-squared value
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best R-squared value: {grid_search.best_score_}")

# Step 6: Evaluate the model with the best parameters on the test set
best_rf_model = grid_search.best_estimator_
y_pred = best_rf_model.predict(X_test)


# Calculate performance metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Test Mean Squared Error (MSE): {mse}")
print(f"Test R-squared (R2): {r2}")

rf_model = best_rf_model

# Step 3: Extract and visualize feature importance
# Get feature importance scores
importances = rf_model.feature_importances_

# Create a bar plot of feature importance
features = X_train.columns
indices = np.argsort(importances)

plt.figure(figsize=(10, 6))
plt.title("Feature Importance")
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel("Relative Importance")
plt.show()

# Step 4: Print feature importance values for easier reading
importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print("Feature importance:")
print(importance_df)

import pickle

with open('random_forest_tat.pkl', 'wb') as file:
    pickle.dump(rf_model, file)

print("Model saved successfully!")

# Export the first three decision trees from the forest

# from sklearn.tree import export_graphviz
# import graphviz

# # Visualize the first 3 trees
# for i in range(3):
#     # Extract the tree
#     tree = rf_model.estimators_[i]
    
#     # Export the tree structure to Graphviz .dot format
#     dot_data = export_graphviz(tree, out_file=None, feature_names=X_train.columns,
#                                filled=True, rounded=True, special_characters=True)
    
#     # Use graphviz to visualize the tree inline (if you're in Jupyter) or save it
#     graph = graphviz.Source(dot_data)
#     graph.render(f"tree_{i}")  # Saves each tree as 'tree_0.pdf', 'tree_1.pdf', etc.
#     display(graph)  # Display inline if you're in a Jupyter Notebook or compatible environment