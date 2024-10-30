# Import necessary libraries
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the color data from CSV file
data = pd.read_csv('mangodata.csv')  # Update the file name if needed

# Assuming your CSV has columns 'L', 'a', and 'b'
L = data['L']
a = data['a']
b = data['b']

# Prepare the data for clustering
color_data = data[['L', 'a', 'b']]

# Apply K-Means clustering
kmeans = KMeans(n_clusters=5, random_state=42)  # 5 clusters
kmeans.fit(color_data)

# Add the cluster labels to the original data
data['Cluster'] = kmeans.labels_

# Save the data with clusters to a new CSV file
data.to_csv('color_data_with_clusters.csv', index=False)

# Calculate and display the average color of each cluster
average_colors = data.groupby('Cluster')[['L', 'a', 'b']].mean()

print("Average color of each cluster (L, a, b values):")
print(average_colors)

# Visualizing the clusters in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(data['L'], data['a'], data['b'], c=data['Cluster'], cmap='viridis')

# Add color bar
plt.colorbar(scatter, ax=ax)

ax.set_xlabel('L value')
ax.set_ylabel('a value')
ax.set_zlabel('b value')
plt.show()

['#786720', '#89681f', '#9a681e', '#aa671c', '#bb661b']