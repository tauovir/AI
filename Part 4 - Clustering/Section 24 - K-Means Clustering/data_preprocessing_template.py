# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3,4]].values
#Using the Elbow Method to find optimal number of cluster
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    #n_clusters: number of cluster, init: Initialize method,max_iter: Maximum Iteration, default:300
    #n_init : which number of kmeans algo run with diffrend number of centroid while K means alfo will run, default = 10
    kmeans = KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_) # cluster sum of square
plt.plot(range(1,11), wcss)
plt.title("Elbow Method")
plt.xlabel("Number of Cluster")
plt.ylabel("WCSS")
plt.show()
#Now  set optimal cluster
kmeans = KMeans(n_clusters=5,init='k-means++',max_iter=300,n_init=10, random_state=0)
y_kmeans = kmeans.fit_predict(X)
print("******************************************")

#Visualising Data
print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
print(X)
print('========================================')
print(X[y_kmeans ==0,1])

plt.scatter(X[y_kmeans ==0,0],X[y_kmeans ==0,1], s=100,c='red',label = 'Cluster1')
#if cluster ==0 then take X 0 position
plt.scatter(X[y_kmeans ==1,0],X[y_kmeans ==1,1], s=100,c='blue',label = 'Cluster2')
plt.scatter(X[y_kmeans ==2,0],X[y_kmeans ==2,1], s=100,c='green',label = 'Cluster3')
plt.scatter(X[y_kmeans ==3,0],X[y_kmeans ==3,1], s=100,c='cyan',label = 'Cluster4')
plt.scatter(X[y_kmeans ==4,0],X[y_kmeans ==4,1], s=100,c='magenta',label = 'Cluster5')
#Lets marks Centroid points
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,c= 'yellow',label = 'centroid')
plt.title("Cluster of Client")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.legend()
plt.show()
