import numpy as np
import random
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import kurtosis, skew
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet, fcluster
from scipy.spatial.distance import pdist
import statsmodels.api as sm
from itertools import groupby
from operator import itemgetter
import csv
import analysis_data

np.set_printoptions(threshold=np.nan)

sFlow = analysis_data.sFlow
#print (np.shape(sFlow))

segment_size = 12

def Deviation(array, mu):
    #print ("########")
    dList = [round(abs(mu-i), 2) for i in array]
    deviation = np.max(dList)
    #print (array)
    #print (mu)
    #print (sorted(dList))
    return deviation

def Mestimator(X):
    #parameters = []
    #array = X[i]
    MAD = sm.robust.scale.mad(X)
    #print (array)
    #print(np.mean(array))
    #print(np.median(array))
    #print (MAD)
    if (MAD == 0):
        mu = np.median(X)
    else:
        mu = sm.robust.norms.estimate_location(X, MAD, norm=None, axis=0, initial=None, maxiter=30, tol=1e-06)
    #print (mu)
    deviation = Deviation(X, mu)
    #parameters.append((mu, deviation))
    parameters = (mu, deviation) 
    #print (parameters) 
    #print(max([round(abs(mu-i), 2) for i in X]))
    return parameters 


X = []
Y = []
    
def RunClusterAnalysis(sFlow, tWindow):
    clusters = []
    for i in range (0,  1): 
        series = sFlow[i, 0:7, :].flatten()
        for j in range(0, len(series) - segment_size):
            x = series[j:j+segment_size]
            y = Mestimator(x)[1]
            X.append(sorted(x)[segment_size/4:len(x) - segment_size/4])
            print("##############")
            print(Mestimator(x))
            print(Mestimator(x[4:8]))
            print(np.median(x))
            print(np.median(x[4:8]))
            Y.append(y)
            
clusters = RunClusterAnalysis(sFlow, tWindow=3)

with open("segments.csv", mode = "w" ) as segment_file:
    writer = csv.writer(segment_file)
    for row in X:
        writer.writerow(row)

with open("deviation.csv", mode = "w", ) as cluster_file:
    writer = csv.writer(cluster_file)
    writer.writerow(Y)


        
        
