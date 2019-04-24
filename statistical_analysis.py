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


sFlow = analysis_data.sFlow

def Deviation(array, mu):
    dList = [round(abs(mu-i), 2) for i in array]
    deviation = np.median(dList)
    return deviation
    


def Mestimator(X):
    MAD = sm.robust.scale.mad(X)
    if (MAD == 0):
        mu = np.median(X)
    else:
        mu = sm.robust.norms.estimate_location(X, MAD, norm=None, axis=0, initial=None, maxiter=30, tol=1e-06)
    deviation = Deviation(X, mu)
    parameters = (mu, 1.4826*deviation) 
    return parameters 


def MeanStatistics(X):
    mu = np.mean(X)
    deviation =np.sqrt(np.sum([(i-mu)*(i-mu) for i in X])/(len(X) - 1))
    return (mu, deviation)
    

def RobustStatistics(X):
    mu = np.median(X)
    deviation =  sm.robust.scale.mad(X)
    
    return (mu, 1.4826*deviation)

    
def FindScore(X, parameters):
    score =  (max([abs (i - parameters[0]) for i in X]))/parameters[1]
    return score
    
def RemoveValue(X, parameters):
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

def FindParameters(X, method):
    #print(X)
    if (method == 'm_estimators'):
        parameters = Mestimator (X)
    if (method == 'robust_statistics'):
        parameters = RobustStatistics(X)
    if (method == 'mean_statistics'):
        parameters = MeanStatistics (X)
    
    return parameters
    

def GESD (X, method, alpha, k):
    mn_o = int(round(k*len(X)))
    R = []
    lamda = []
    n = len(X)
    for i in range (0, mn_o):
        parameters = FindParameters(X, method)
        score = FindScore (X, parameters)
        R.append(score)
        X = RemoveValue(X, parameters)
        lam = FindLambda (alpha, n, i)
        lamda.append(lam)
    diff = [j - k for j, k in zip (R, lamda)]
    P = [diff.index(k) for k in diff if k > 0]
    if (P == []):
        n_o = 0
    else:
        n_o = max(P) + 1
    
    return n_o


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


def FindAccuracy(X, outlier, method):
    tp=0
    fp=0
    fn=0
    
    N = max(outlier['p'], outlier['n'])
    for x in X:
        n_o = GESD(x, method, alpha=0.05, k = 0.50) 

        if (n_o >= N):
            tp = tp + N
            fp = fp + n_o - N
        
        if (n_o < N):
            tp = tp + n_o
            fn =  fn + N - n_o 
        
    P = round(float(tp)/(float(tp) + float(fp)), 2)
    R = round(float(tp)/(float(tp) + float(fn)), 2)
    F = round(2*((P*R)/(P+R)), 2)
    
    return (tp, fp, fn, P, R, F)


outliers_list = [{'p': 1, 'n': 0}, {'p': 2, 'n': 0}, {'p': 3, 'n': 0},
                 {'p': 0, 'n': 1}, {'p': 0, 'n': 2}, {'p': 0, 'n': 3}]

results = []
methods = ['mean_statistics', 'robust_statistics', 'm_estimators']
mean_std_deviation = 10.97


for method in methods:
    for outlier in outliers_list:
        magnitude = round(mean_std_deviation + mean_std_deviation/2, 2)
        for i in range(0, 12):
            X = SegmentForAnalysis(sFlow, segment_size, outlier, magnitude)
            accuracy = FindAccuracy(X, outlier, method)
            r = (method, outlier['p'], outlier['n'], magnitude) + accuracy
            if (i ==0):
                magnitude = round(magnitude + mean_std_deviation/2, 2)
            else:
                magnitude = round(magnitude + mean_std_deviation, 2)
            results.append(r)
            print (r)

print (results)

with open("results_statistics_1.csv", mode="w") as f:
        writer = csv.writer(f)
        for row in results:
            writer.writerow(row)



        
        

    
