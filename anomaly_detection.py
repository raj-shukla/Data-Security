import numpy as np
import random
import statsmodels.api as sm
import os
import csv
import analysis_data

np.set_printoptions(threshold=np.nan)

sFlow = analysis_data.sFlow
#print (np.shape(sFlow))


def Deviation(array, mu):
    dList = [round(abs(mu-i), 2) for i in array]
    deviation = np.max(dList)
    return deviation

def Mestimator(X):
    MAD = sm.robust.scale.mad(X)
    if (MAD == 0):
        mu = np.median(X)
    else:
        mu = sm.robust.norms.estimate_location(X, MAD, norm=None, axis=0, initial=None, maxiter=40, tol=1e-06)
    deviation = Deviation(X, mu)
    parameters = (mu, deviation) 
    return parameters 



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

def FindDeviations(X):
    deviations = []
    with open("data_with_outliers.csv", mode="w") as f:
        writer = csv.writer(f)
        for row in X:
            writer.writerow(row)
    os.system('./prediction.sh')
    #os.system('python LSTM_prediction.py')
    
    with open("predicted_deviations.csv", "r") as deviation_file:
        reader = csv.reader(deviation_file)
        for row in reader:
            deviations.append(float(row[0]))
    
    return deviations
    
    
def FindParameters(X):
    m_estimators = []
    deviations = []
    for data in X:
        m_estimator = Mestimator(data)[0]
        m_estimators.append(m_estimator)
     
    deviations = FindDeviations(X)
    
    return m_estimators, deviations
    
    

def FindAccuracy(X, m_estimators, deviations, outliers, param):
    tp=0
    fp=0
    fn=0
    for i in range(0, len(X)):
        m = m_estimators[i]
        d = deviations[i]
        p_o = sum(j > m+param*d  for j in X[i])
        n_o = sum(j < m-param*d  for j in X[i])
        t_o = p_o + n_o
        
        
        if(p_o >= outliers['p']):
            tp = tp + outliers['p']
            fp = fp + p_o - outliers['p']
        if(p_o < outliers['p']):
            tp =  tp + p_o
            fn = fn + outliers['p'] - p_o
                
        if(n_o >= outliers['n']):
            tp = tp + outliers['n']
            fp = fp + n_o - outliers['n']
        if(n_o < outliers['n']):
            tp = tp + n_o
            fn = fn + outliers['n'] - n_o
            
            
    P = round(float(tp)/(float(tp) + float(fp)), 2)
    R = round(float(tp)/(float(tp) + float(fn)), 2)
    F = round(2*((P*R)/(P+R)), 2)
        
    return (tp, fp, fn, P, R, F)
    
#outliers_list = [{'p': 1, 'n': 0}, {'p': 2, 'n': 0}, {'p': 3, 'n': 0},
                 #{'p': 0, 'n': 1}, {'p': 0, 'n': 2}, {'p': 0, 'n': 3},
                 #{'p': 1, 'n': 1}, {'p': 1, 'n': 2}, {'p': 2, 'n': 1},
                 #{'p': 2, 'n': 2}, {'p': 1, 'n': 3}, {'p': 3, 'n': 1},
                 #{'p': 2, 'n': 3}, {'p': 3, 'n': 2}, {'p': 3, 'n': 3}]
                 
outliers_list = [{'p': 1, 'n': 0}, {'p': 0, 'n': 1}, {'p': 2, 'n': 0},
                 {'p': 0, 'n': 2}, {'p': 3, 'n': 0}, {'p': 0, 'n': 3}]
segment_size = 12
results = []

mean_std_deviation = 10.97

'''
for outliers in outliers_list:
    magnitude = round(mean_std_deviation + mean_std_deviation/2, 2)
    for i in range(0, 12):
        X = SegmentForAnalysis(sFlow, segment_size, outliers, magnitude)
        m_estimators, deviations = FindParameters(X)
        accuracy = FindAccuracy(X, m_estimators, deviations, outliers, param=1.5)
        r = (outliers['p'], outliers['n'], magnitude) + accuracy
        if (i ==0):
            magnitude = round(magnitude + mean_std_deviation/2, 2)
        else:
            magnitude = round(magnitude + mean_std_deviation, 2)
        results.append(r)
        print (r)

print (results)

with open("results.csv", mode="w") as f:
        writer = csv.writer(f)
        for row in results:
            writer.writerow(row)
'''

'''
for outliers in outliers_list[9:10]:
    magnitude = 10.97 + 10.97 + 10.97 + 10.97 + 10.97
    for i in range(0, 1):
        param = 0.5
        for j in range(0, 15):
            X = SegmentForAnalysis(sFlow, segment_size, outliers, magnitude)
            m_estimators, deviations = FindParameters(X)
            accuracy = FindAccuracy(X, m_estimators, deviations, outliers, param)
            param = round(param+0.1, 2)
            r = (outliers['p'], outliers['n'], param, magnitude) + accuracy
            results.append(r)
            print (r)
        #magnitude = magnitude + 10

print (results)
'''


magnitude = round(mean_std_deviation + mean_std_deviation/2, 2)
for i in range(0, 12):
    for outliers in outliers_list:
        X = SegmentForAnalysis(sFlow, segment_size, outliers, magnitude)
        m_estimators, deviations = FindParameters(X)
        accuracy = FindAccuracy(X, m_estimators, deviations, outliers, param=1.5)
        r = (outliers['p'], outliers['n'], magnitude) + accuracy
        results.append(r)
        print (r)
    if (i ==0):
        magnitude = round(magnitude + mean_std_deviation/2, 2)
    else:
        magnitude = round(magnitude + mean_std_deviation, 2)

print (results)

with open("results_statistics_2.csv", mode="w") as f:
        writer = csv.writer(f)
        for row in results:
            writer.writerow(row)
'''
with open("results_parameter.csv", mode="a") as f:
        writer = csv.writer(f)
        for row in results:
            writer.writerow(row)
'''

