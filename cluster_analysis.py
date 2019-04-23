import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy import stats
from scipy.stats import kurtosis, skew
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet, fcluster
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA
from itertools import groupby
from operator import itemgetter
import csv
import analysis_data

np.set_printoptions(threshold=np.nan)

sFlow = analysis_data.sFlow
#print (np.shape(sFlow))

def Clusters(locations, clusters):
    elements = sorted(list(zip(clusters, locations)), key= lambda l: l[1])
    groups = []
    uniquekeys = []
    c = [zip(*i)[0] for i in groups]
    
    for k, g in groupby(elements, key=lambda l: l[0]):
        groups.append(list(g))      # Store group iterator as a list
        uniquekeys.append(k)
        
    c = [zip(*i)[1] for i in groups]
    
    return c
    
    
def PlotDendogram(Z):
    plt.figure(figsize=(25, 10))
    #plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Sample index', fontsize=32)
    plt.ylabel('Distance', fontsize=32)
    dendrogram(
    Z,
    leaf_rotation=90.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
    )
    plt.savefig('dendogram.eps')
    plt.show()
    
    
def TimeSeries(sFlow):
    X = []
    for i in range (0,  100): 
        x = sFlow[i, 0:7, :].flatten()
        np.reshape(x, (1, 2016))
        X.append(x)
    return X
    
X = TimeSeries(sFlow)



#pca = PCA(100)
#pca.fit(X)

#print(pca.components_)
#print(pca.explained_variance_)
pca = PCA(0.99).fit(X)
print(pca.n_components_)

X_pca = pca.transform(X)

X = X_pca
np.reshape(X, (100, 33))
Z = linkage(X, 'centroid')

c, coph_dists = cophenet(Z, pdist(X))

print (c)

plt.plot(Z[:, 2])
ax = plt.axes()
ax.minorticks_on()
ax.grid(which='major')
ax.grid(which='minor')
plt.xlabel('Locations')
plt.ylabel('Euclidean distance')
plt.show()

PlotDendogram(Z)
clusters = fcluster(Z, 3500, criterion='distance')
print(clusters)

locations = range(0, 100)
groups = Clusters(locations, clusters)

print (groups)

plt.subplot(211)
plt.plot(sFlow[0, 0:7, :].flatten())
plt.xlabel("Time slot", fontsize=24)
plt.ylabel("Traffic", fontsize=24)
plt.axis([0, 2200, 0, 400])

plt.subplot(212)
plt.plot(sFlow[4, 0:7, :].flatten())
plt.xlabel("Time slot", fontsize=24)
plt.ylabel("Traffic", fontsize=24)
plt.axis([0, 2200, 0, 400])

plt.subplots_adjust(left=None, wspace=0.3, hspace=0.4, top=None)

plt.show()







        
        
        
