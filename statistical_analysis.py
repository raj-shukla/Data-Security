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
import csv
import analysis_data


#clusters = clusters.clusters
sFlow = analysis_data.sFlow

def Deviation(array, mu):
    #print ("########")
    dList = [round(abs(mu-i), 2) for i in array]
    deviation = np.max(dList)
    #print (array)
    #print (mu)
    #print (sorted(dList))
    return 1.48*deviation
    
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
    deviation = Deviation(X, mu)
    #parameters.append((mu, deviation))
    parameters = (mu, deviation) 
    #print (parameters) 
    #print(max([round(abs(mu-i), 2) for i in X]))
    return parameters 


def MeanStatistics(X):
    mu = np.mean(X)
    #deviation = np.std(X)
    deviation =np.sqrt(np.sum([(i-mu)*(i-mu) for i in X])/(len(X) - 1))
    
    return (mu, deviation)
    

def RobustStatistics(X):
    mu = np.median(X)
    #deviation = np.std(X)
    deviation =  sm.robust.scale.mad(X)
    
    return (mu, deviation)

    
def FindScore(X, parameters):
    score =  (max([abs (i - parameters[0]) for i in X]))/parameters[1]
    return score
    
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
    #print (X)
    #plt.scatter (X, X)
    #plt.show()
    for i in range (0, mn_o):
        #parameters = Mestimator (X)
        parameters = Mestimator (X)
        #parameters = MeanStatistics(X)
        #parameters = RobustStatistics(X)
        score = FindScore (X, parameters)
        R.append(score)
        #print (parameters)
        X = RemoveValue(X, parameters)
        lam = FindLambda (alpha, n, i)
        lamda.append(lam)
    #print (R)
    #print (lamdada
    #print (parameters)
    diff = [j - k for j, k in zip (R, lamda)]
    #for l in range(0, mn_o):
        #print (R[l] - lamda[l])
    P = [diff.index(k) for k in diff if k > 0]
    if (P == []):
        n_o = 0
    else:
        n_o = max(P) + 1
    print (n_o)
    
    return n_o
'''
for i in range(0, len(custers)):
    print (clusters[i])
    for j in range(0, len(clusters[i])):
        parameters = Mestimator(clusters[i][j])
        print (parameters)
    GESD(clusters[i], parameters)
''' 
def OutlierAddition (x, outliers, magnitude):
    tmp_array = sorted(x)
    tmp_1 = [ele + magnitude for ele in tmp_array[-outliers['p']:]]
    tmp_array[-outliers['p']:] = tmp_1
    tmp_2 = [max((ele - magnitude), 0) for ele in tmp_array[0:outliers['n']]]
    tmp_array[0:outliers['n']]  = tmp_2
    array = tuple(tmp_array)
    return array
    
def SegmentForAnalysis(sFlow, segment_size, outliers,  magnitude):
    X = []
    series = sFlow[2, 0:7, :].flatten() 
    for j in range(0, len(series) - segment_size, segment_size):
        x_tmp = series[j:j+segment_size]
        x = OutlierAddition(x_tmp, outliers, magnitude)
        X.append(x)
    return X

segment_size = 12
X = SegmentForAnalysis(sFlow, segment_size, {'p': 2, 'n': 0}, 100.97)

print (np.shape(X))

for  x in X:
    n_o = GESD(x, alpha=0.05, k = 0.50) 
    print(n_o)
        
        
        
#print (total)
#print (correct)

    
    
