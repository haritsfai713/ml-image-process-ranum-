from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Example dataset (replace with real data)
# Each entry should be the cluster proportions (features) and ripeness stage (label)
X = []  # List of cluster proportions (features)
y = []  # List of ripeness stages (labels)

# You can loop over your dataset and fill X and y
# For example, extract cluster proportions and label for each mango image
# X.append(proportions), y.append(ripeness_stage)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest classifier
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Predict on the test set
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
