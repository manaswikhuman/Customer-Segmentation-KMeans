import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# 1. Create a simple dataset (or load your CSV)
data = {
    'CustomerID': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Annual_Income': [15, 16, 17, 70, 72, 75, 20, 80, 85, 22],
    'Spending_Score': [39, 81, 6, 77, 40, 78, 90, 10, 15, 5]
}
df = pd.DataFrame(data)

# 2. Select features for clustering
# We use columns 1 and 2 (Income and Score)
X = df.iloc[:, [1, 2]].values

# 3. Apply K-Means
# Let's start with 3 clusters for simplicity
model = KMeans(n_clusters=3, random_state=42)
clusters = model.fit_predict(X)

# 4. Add the cluster labels back to our data
df['Segment'] = clusters
print(df)

# 5. Visualizing the results
plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap='rainbow')
plt.title('Customer Segments')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.show()
