# Data Processing
import pandas as pd
import numpy as np

# Modelling
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint


# Tree Visualisation
from sklearn.tree import export_graphviz
from IPython.display import Image
from IPython.display import display
import graphviz
import matplotlib.pyplot as plt

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
y = data['Cluster']  # Target

# Step 5: Split the data into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(n_estimators=146,max_depth=17)
rf.fit(X_train, y_train)

import pickle

with open('random_forest_classifier.pkl', 'wb') as file:
    pickle.dump(rf, file)

print("Model saved successfully!")

y_pred = rf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Generate predictions with the best model
y_pred = rf.predict(X_test)

# Create the confusion matrix
cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm).plot();
disp.plot()
plt.show()

# Create a series containing feature importances from the model and feature names from the training data
feature_importances = pd.Series(rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)

# Plot a simple bar chart
disp = feature_importances.plot.bar();
disp.plot()
plt.show()

# Export the first three decision trees from the forest

# for i in range(3):
#     tree = rf.estimators_[i]
#     dot_data = export_graphviz(tree,
#                                feature_names=X_train.columns,  
#                                filled=True,  
#                                max_depth=2, 
#                                impurity=False, 
#                                proportion=True)
#     graph = graphviz.Source(dot_data)
#     graph.render(f"tree_{i}")  # Saves each tree as 'tree_0.pdf', 'tree_1.pdf', etc.
#     display(graph)

# param_dist = {'n_estimators': randint(50,500),
#               'max_depth': randint(1,20)}

# # Create a random forest classifier
# rf = RandomForestClassifier()

# # Use random search to find the best hyperparameters
# rand_search = RandomizedSearchCV(rf, 
#                                  param_distributions = param_dist, 
#                                  n_iter=5, 
#                                  cv=5)

# # Fit the random search object to the data
# rand_search.fit(X_train, y_train)

# # Create a variable for the best model
# best_rf = rand_search.best_estimator_

# # Print the best hyperparameters
# print('Best hyperparameters:',  rand_search.best_params_)