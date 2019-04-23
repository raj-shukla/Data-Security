import numpy as np
import random
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import kurtosis, skew
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet, fcluster
from scipy.spatial.distance import pdist
from scipy.stats import t as student_t
import statsmodels.api as sm
from itertools import groupby
from operator import itemgetter
import cluster_analysis
import clusters

l = []

clusters = clusters.clusters
clusters = cluster_analysis.clusters
def Deviation(array, mu):
    #print ("########")
    dList = [round(abs(mu-i), 2) for i in array]
    deviation = 1.48*np.median(dList)
    #print (array)
    #print (mu)
    #print (sorted(dList))
    #print (deviation)
    return deviation
    
'''
def Mestimator(X):
    parameters = []
    for i in range(0, len(X)):
        array = X[i]
        MAD = sm.robust.scale.mad(array)
        #print (array)
        #print(np.mean(array))
        #print(np.median(array))
        #print (MAD)
        if (MAD == 0):
            mu = np.median(array)
        else:
            mu = sm.robust.norms.estimate_location(array, MAD, norm=None, axis=0, initial=None, maxiter=30, tol=1e-06)
        #print (mu)
        deviation = Deviation(array, mu)
        parameters.append((mu, deviation))
        
    return parameters 
'''

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
    deviation = 2*Deviation(X, mu)
    #parameters.append((mu, deviation))
    parameters = (mu, deviation)  
    return parameters 


def MeanStatistics(X):
    mu = np.mean(X)
    #deviation = np.std(X)
    deviation =np.sqrt(np.sum([(i-mu)*(i-mu) for i in X])/(len(X) - 1))
    
    return (mu, deviation)
    

def RobustStatistics(X):
    mu = np.median(X)
    #deviation = np.std(X)
    deviation =  1.5*sm.robust.scale.mad(X)
    
    return (mu, deviation)

    
def FindScore(X, parameters):
    #score =  (max([abs (i - parameters[0]) for i in X]))/parameters[1]
    return (([abs(i - parameters[0]) for i in X]))/parameters[1]
    
def RemoveValue(X, parameters):
    #md = max([i - parameters[0] for i in X])
    d = [abs(i - parameters[0]) for i in X]
    elements = sorted(list(zip(X, d)), key= lambda l: l[1])
    mdi = X.index(elements[-1][0])
    X = X[:mdi] + X[mdi + 1:]
    return X
    
def FindLambda(alpha, n, i):
    i = i+1
    p = 1 - alpha/(2*(n-i+1))
    t = student_t.ppf(p, (n - i - 1))
    lam = t * (n - i) / float(np.sqrt((n - i - 1 + t**2) * (n - i + 1)))
   
    return lam



def GESD (X, alpha, k):
    #print ("################")
    #print (len(X))
    #print (k*len(X))
    #print (round(k*len(X)))
    mn_o = int(round(k*len(X)))
    #print (mn_o)
    R = []
    lamda = []
    n = len(X)
    print (X)
    #plt.scatter (X, X)
    #plt.show()
    #for i in range (0, mn_o):
    for i in range (0, 1):
        #parameters = Mestimator (X)
        parameters = Mestimator (X)
        score = FindScore (X, parameters)
        print (score)
        R.append(score)
        #print (parameters)
        #X = RemoveValue(X, parameters)
        #lam = FindLambda (alpha, n, i)
        #lamda.append(lam)
    #print (R)
    #print (lamdada
    #print (parameters)
    #diff = [j - k for j, k in zip (R, lamda)]
    #for l in range(0, mn_o):
        #print (R[l] - lamda[l])
    #P = [diff.index(k) for k in diff if k > 0]
    #if (P == []):
        #n_o = 0
    #else:
        #n_o = max(P) + 1
    #print (n_o)
    #l.append(n_o)
    
    #return n_o
'''
for i in range(0, len(custers)):
    print (clusters[i])
    for j in range(0, len(clusters[i])):
        parameters = Mestimator(clusters[i][j])
        print (parameters)
    GESD(clusters[i], parameters)
''' 

#array = (52.0, 54.33, 57.33, 27.67, 28.0, 28.0, 24.33, 24.67, 25.67, 35.33, 31.33, 27.0, 36.0, 35.33, 33.0, 52.33, 52.0, 42.0, 63.33, 55.33, 49.0, 56.33, 63.0, 66.0, 50.33, 46.33, 45.0, 29.33, 29.0, 31.67, 28.67, 29.67, 35.0, 29.67, 27.33, 28.0, 42.67, 43.0, 39.67, 54.0, 56.0, 63.67, 26.67, 23.0, 27.67, 35.33, 37.0, 37.33, 44.67, 46.67, 49.67, 32.67, 34.33, 37.0, 39.0, 43.33, 40.67, 50.0, 48.67, 46.0, 58.33, 58.67, 57.67, 35.67, 33.33, 30.0, 30.0, 32.67, 35.0, 34.67, 33.67, 84.67, 31.33, 34.67, 41.33, 39.33, 40.33, 36.67, 59.33, 52.67, 48.67, 61.33, 57.33, 55.0, 37.0, 32.33, 29.0, 30.33, 5.33, 30.0)

#print (clusters[0][1])
#GESD(clusters[0][1], alpha=0.05, mn_o = 5) 
#GESD(array, alpha=0.05, mn_o = 28) 

def OutlierAddition (X, magnitude):
    tmp_array = list(clusters[i][j])
    tmp = [ele + magnitude for ele in tmp_array[0:2]]
    tmp_array[0:2] = tmp
    array = tuple(tmp_array)
    
    return array
    

for i in range(0, 75):
    print ("##############################")
    for j in range(0, len(clusters[i])):
        print (np.std(clusters[i][j]))
        X = OutlierAddition(clusters[i][j], magnitude = 30)
        n_o = GESD(X, alpha=0.05, k = 0.30) 


print (l)
print (len([i for i in l if i==0]))
print (len([i for i in l if i!=0]))
print (len([i for i in l if i!=2]))
    
    
