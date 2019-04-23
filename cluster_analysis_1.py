import numpy as np
import random
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import kurtosis, skew
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet, fcluster
from scipy.spatial.distance import pdist
from itertools import groupby
from operator import itemgetter
import csv
import analysis_data

np.set_printoptions(threshold=np.nan)

sFlow = analysis_data.sFlow
#print (np.shape(sFlow))

def PlotDendogram(Z):
    plt.figure(figsize=(25, 10))
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('sample index')
    plt.ylabel('distance')
    dendrogram(
    Z,
    leaf_rotation=90.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
    )
    plt.show()
    
def PlotClusters(X, clusters):
    plt.scatter(X, X, c=clusters, cmap='prism')  # plot points with cluster dependent colors
    plt.show()
    
def CollectClusters(X, clusters):
    groups = []
    uniquekeys = []
    #print (clusters)
    #print(list(zip(X, clusters)))
    elements = sorted(list(zip(X, clusters)), key= lambda l: l[1])
    #print (elements)
    for i in range (0, len(elements)):
        #print(np.asscalar(elements[i][0]))
        elements[i] = list(elements[i])
        elements[i][0] = round(np.asscalar(elements[i][0]), 2)
    for k, g in groupby(elements, key=lambda l: l[1]):
        groups.append(list(g))      # Store group iterator as a list
        uniquekeys.append(k)
    #print (groups)
    c = [zip(*i)[0] for i in groups]
    #print (c)
    return c

def FormCluster(X):
    Z = linkage(X, 'centroid')
    #print (Z)
    #print(Z[:, 2])
    c, coph_dists = cophenet(Z, pdist(X))
    #plt.plot(Z[:, 2])
    #plt.show()
    #print (c)
    #PlotDendogram(Z)
    clusters = fcluster(Z, 10, criterion='distance')
    #print (clusters)
    #print(len(clusters))
    c = CollectClusters(X, clusters)
    #PlotClusters(X, clusters)
    print("#############")
    print(len(c))
    print (sorted(c))
    return c

X = []
def RunClusterAnalysis(sFlow, tWindow):
    clusters = []
    for i in range (0,  100): 
        x = sFlow[i, 0:7, :].flatten()
        np.reshape(x, (1, 2016))
        print(np.shape(x))
        X.append(x)

clusters = RunClusterAnalysis(sFlow, tWindow=3)

print(np.shape(X))
np.reshape(X, (100, 2016))

Z = linkage(X, 'ward')
c, coph_dists = cophenet(Z, pdist(X))

print (c)
PlotDendogram(Z)
clusters = fcluster(Z, 5000, criterion='distance')
print(clusters)

plt.plot(sFlow[0, 0:7, :].flatten())
plt.show()

plt.plot(sFlow[4, 0:7, :].flatten())
plt.show()







        
        
        
