import numpy as np
import random
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import kurtosis, skew
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet, fcluster
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA
from itertools import groupby
from operator import itemgetter
import analysis_data

np.set_printoptions(threshold=np.nan)

sFlow = analysis_data.sFlow


cl1 = []
cl2 = []
cl3 = []
cl4 = []
cl5 = []
cl6 = []
cl7 = []

def FormCluster(X):
    Z = linkage(X, 'single')
    c1, coph_dists = cophenet(Z, pdist(X))
    cl1.append(c1)
    
    Z = linkage(X, 'complete')
    c2, coph_dists = cophenet(Z, pdist(X))
    cl2.append(c2)
    
    Z = linkage(X, 'average')
    c3, coph_dists = cophenet(Z, pdist(X))
    cl3.append(c3)
    
    Z = linkage(X, 'weighted')
    c4, coph_dists = cophenet(Z, pdist(X))
    cl4.append(c4)
    
    Z = linkage(X, 'centroid')
    c5, coph_dists = cophenet(Z, pdist(X))
    cl5.append(c5)
    
    Z = linkage(X, 'median')
    c6, coph_dists = cophenet(Z, pdist(X))
    cl6.append(c6)
    
    Z = linkage(X, 'ward')
    c7, coph_dists = cophenet(Z, pdist(X))
    cl7.append(c7)
    
    
    #PlotDendogram(Z)
    #clusters = fcluster(Z, 50, criterion='distance')
    #print (clusters)
    #print(len(clusters))
    #CollectClusters(X, clusters)
    #PlotClusters(X, clusters)
    

def TimeSeries(sFlow):
    X = []
    for i in range (0,  100): 
        x = sFlow[i, 0:7, :].flatten()
        np.reshape(x, (1, 2016))
        print(np.shape(x))
        X.append(x)
    return X

X = TimeSeries(sFlow)
FormCluster(X)
        
        
print (np.mean(cl1))  
print (np.mean(cl2))        
print (np.mean(cl3))        
print (np.mean(cl4))        
print (np.mean(cl5))        
print (np.mean(cl6))        
print (np.mean(cl7))              
        
