#-------------------------------------------------------------------------
# AUTHOR: Josephine Nguyen
# FILENAME: clustering.py
# SPECIFICATION: This program runs kmeans with k values between 2-20 to find the one that results in the best silhouette coefficient, plots them, then compares to AgglomerativeClustering
# FOR: CS 4210- Assignment #5
# TIME SPENT: 30 minutes
#-----------------------------------------------------------*/

#importing some Python libraries
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn import metrics

df = pd.read_csv('training_data.csv', sep=',', header=None) #reading the data by using Pandas library

#assign your training data to X_training feature matrix
X_training = df

best_k = -1
best_coeff = -1
#run kmeans testing different k values from 2 until 20 clusters
     #Use:  kmeans = KMeans(n_clusters=k, random_state=0)
     #      kmeans.fit(X_training)

for k in range(2,21):
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(X_training)
    curr_coeff = silhouette_score(X_training, kmeans.labels_)
    if curr_coeff > best_coeff:
        best_coeff = curr_coeff
        best_k = k

    #plot the value of the silhouette_coefficient for each k value of kmeans so that we can see the best k
    plt.scatter(k, curr_coeff)

plt.show()

#reading the validation data (clusters) by using Pandas library
df = pd.read_csv('testing_data.csv', sep=',', header=None)

#assign your data labels to vector labels (you might need to reshape the row vector to a column vector)
labels = np.array(df.values).reshape(1,len(df))[0]

#Calculate and print the Homogeneity of this kmeans clustering
print("K-Means Homogeneity Score = " + metrics.homogeneity_score(labels, kmeans.labels_).__str__())

#rung agglomerative clustering now by using the best value o k calculated before by kmeans
agg = AgglomerativeClustering(n_clusters=best_k, linkage='ward')
agg.fit(X_training)

#Calculate and print the Homogeneity of this agglomerative clustering
print("Agglomerative Clustering Homogeneity Score = " + metrics.homogeneity_score(labels, agg.labels_).__str__())
