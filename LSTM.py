import numpy as np
import random
from scipy import stats
from scipy.stats import kurtosis, skew
#import clusters
import math
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import LSTM, Dropout
import csv
import pandas as pd

np.set_printoptions(threshold=np.nan)

np.random.seed(7)

#clusters = clusters.clusters
#clusters = cluster_analysis.clusters



clusters = []
with open("segments.csv", "r") as cluster_file:
    reader = csv.reader(cluster_file)
    for row in reader:
        print(row)
        cluster = [float(i)/957.0 for i in row]
        #print(cluster)
        index = int(len(cluster)/4)
        clusters.append(sorted(cluster))
        
with open("deviation.csv", "r") as deviation_file:
    reader = csv.reader(deviation_file)
    for row in reader:
        deviation = [float(i)/957.0 for i in row]



X = clusters
Y = deviation


maxLength = max([len(i) for i in X])  

for x in X :
    [x.append(0) for i in range (maxLength - len(x))]

    
X, Y = np.array(X), np.array(Y)

X = np.reshape(X, (X.shape[0], X.shape[1], 1))

Y = np.reshape(Y, (2004, 1))


print(X.shape)
#print(Y.shape)       


model = Sequential ()
model.add(LSTM(units=128, return_sequences=True, input_shape=(X.shape[1], 1))) 
model.add(Dropout(0.2))  
model.add(LSTM(units=50, return_sequences=True))  
model.add(Dropout(0.2))

model.add(LSTM(units=50, return_sequences=True))  
model.add(Dropout(0.2))

model.add(LSTM(units=50))  
model.add(Dropout(0.2))  
model.add(Dense(units = 1))  

#model.add(Flatten())

model.compile(optimizer = 'adam', loss = 'mean_squared_error')

model.fit(X, Y, epochs=10000)

model.save('model_10000.h5')

with open('model_architecture_10000.json', 'w') as f:
    f.write(model.to_json())


predictions = model.predict(X)

print(predictions.flatten())
print(len(predictions))
with open("predictions.csv", mode = "w", ) as prediction_file:
    writer = csv.writer(prediction_file)
    for row in predictions:
        writer.writerow(row)

   
        
