import numpy as np
import random
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import kurtosis, skew
from itertools import groupby
from operator import itemgetter
import readData
np.set_printoptions(threshold=np.nan)

days = readData.days
flow = np.array(readData.flow)
flowList = np.array(readData.flowList)
time = np.array(readData.time)
postMile = np.array(readData.postMile)
lanes = np.array(readData.lanes)

#flow = (flow - np.min(flow))/(np.max(flow) - np.min(flow))
flowArray = []

for i, val in enumerate(flow):
    flowArray.append(np.array(flow[i].reshape(24*12, 136)))
    

flowArray = np.asarray(flowArray)


threshhold_list = [[] for i in range(0, 136)]

def Difference(point):
    diff  = [[] for i in range(0, 12)]
    flowAtPoint = flowArray[:, :, point:point+1]

    for i in range(0, 288 - len(diff)):
        for j in range(0, len(diff)):
            diff[j].append( np.absolute(np.mean(flowAtPoint[:, i:i+j+1, :]) - np.mean(flowAtPoint[:, i+j+1:i+j+1 + j+1, :])) )
  
    threshhold_current = [np.mean(i) for i in diff]
    
    return threshhold_current
    
    
array = []   
for i in range (0, 136):
    threshhold_list[i] = Difference(i)

threshhold = np.mean(threshhold_list, axis=0)

print (threshhold)
