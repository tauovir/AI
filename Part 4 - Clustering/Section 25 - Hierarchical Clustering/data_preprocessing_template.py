# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:,[3,4]].values

# Using Dendrogram to find the optimal number of cluster
import scipy.cluster.hierarchy as sch
dendogram = sch.dendrogram(sch.linkage(X,method='ward'))    #Method:ward, means to minimize variance in each cluster
plt.title("Dendrogram")
plt.xlabel("Customers")
plt.ylabel("Euclidean Distances")
plt.show()
#Here we can see that longest vertical that cross 5 line, so optimal num er of cluster is 5

#Fitting Hierarchical clustering to mall dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=5, affinity='euclidean',linkage='ward')
y_hc = hc.fit_predict(X)

#Visualize the predicted cluster
plt.scatter(X[y_hc ==0,0],X[y_hc ==0,1], s=100,marker = '$*$',c='red',label = 'Cluster1')
#if cluster ==0 then take X 0 position
plt.scatter(X[y_hc ==1,0],X[y_hc ==1,1], s=100,c='blue',label = 'Cluster2',marker = '$m$')
plt.scatter(X[y_hc ==2,0],X[y_hc ==2,1], s=100,c='green',label = 'Cluster3',marker = '$n$')
plt.scatter(X[y_hc ==3,0],X[y_hc ==3,1], s=100,c='cyan',label = 'Cluster4',marker = '$s$')
plt.scatter(X[y_hc ==4,0],X[y_hc ==4,1], s=100,c='magenta',label = 'Cluster5',marker = '$y$')
#Lets marks Centroid points
#plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,c= 'yellow',label = 'centroid')
plt.title("Cluster of Client")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.legend()
plt.show()

