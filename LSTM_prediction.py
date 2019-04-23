import numpy as np
import random
from scipy import stats
from scipy.stats import kurtosis, skew
#import clusters
import math
from keras.models import Sequential
from keras.models import model_from_json
from keras.models import load_model
from keras.layers import Dense, Flatten
from keras.layers import LSTM, Dropout
import csv
import pandas as pd

np.random.seed(7)

X = []

with open("data_with_outliers.csv", "r") as f:
    reader = csv.reader(f)
    for row in reader:
        x = [float(i)/957.0 for i in row]
        index = int(len(x)/4)
        X.append(x[index:len(x)-index])

X = np.asarray(X)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))


model = load_model('model_10000.h5')
#model.summary()
model.get_weights()
predictions = model.predict(X)


for i in predictions:
    i[0] = i[0]*957

with open("predicted_deviations.csv", mode = "w", ) as prediction_file:
    writer = csv.writer(prediction_file)
    for row in predictions:
        writer.writerow(row)
