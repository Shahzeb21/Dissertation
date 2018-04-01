from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

t_X = pd.read_csv(r'''C:\Users\shahz\Documents\Research\electricitydataset.csv''')

X = t_X[['kwh','athome']]

print(X)
kmean = KMeans(n_clusters=4, random_state=0).fit(X)

Cluster = kmean.labels_


centroids = kmean.cluster_centers_

print(centroids)
print(Cluster)


plt.scatter(X['kwh'], X['athome'], c='black', s=10, cmap='viridis')

for i in range(3):
    # select only data observations with cluster label == i
    ds = X[np.where(Cluster==i)]
    # plot the data observations
    plt.plot(X.loc('kwh','athome'),'o')
    # plot the centroids
    lines = plt.plot(centroids[i,0],centroids[i,1],1)
    # make the centroid x's bigger
    plt.setp(lines,ms=15.0)
    plt.setp(lines,mew=2.0)
plt.show()

