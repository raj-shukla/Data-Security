import numpy as np
import random
import matplotlib.pyplot as plt
import csv


results_1 = []
results_2 = []
with open("results_statistics.csv", "r") as r_file:
    reader = csv.reader(r_file)
    for row in reader:
        r = [float(i) for i in row[1:]]
        r.insert(0, row[0])
        results_1.append(r)
        


for result in results_1:
    print (result)
    
with open("results_statistics_2.csv", "r") as r_file:
    reader = csv.reader(r_file)
    for row in reader:
        r = [float(i) for i in row[1:]]
        #r.insert(0, row[0])
        results_2.append( r)
        

for result in results_2:
    print (result)

magnitude_list =  list(sorted(set([i[3]  for i in results_1])))

P_1, P_2, P_3, P_4, R_1, R_2, R_3, R_4, F_1, F_2, F_3, F_4 = [[] for i in range(12)]

for magnitude in magnitude_list:
    x_1 = [i[7]  for i in results_1 if ((i[0]=='mean_statistics') and (i[3])==magnitude)]
    x_2 = [i[8]  for i in results_1 if ((i[0]=='mean_statistics') and (i[3])==magnitude)]
    x_3 = [i[9]  for i in results_1 if ((i[0]=='mean_statistics') and (i[3])==magnitude)]
    
    P_1.append(round(np.mean(x_1), 2))
    R_1.append(round(np.mean(x_2), 2))
    F_1.append(round(np.mean(x_3), 2))
    
    x_4 = [i[7]  for i in results_1 if ((i[0]=='robust_statistics') and (i[3])==magnitude)]
    x_5 = [i[8]  for i in results_1 if ((i[0]=='robust_statistics') and (i[3])==magnitude)]
    x_6 = [i[9]  for i in results_1 if ((i[0]=='robust_statistics') and (i[3])==magnitude)]
    
    P_2.append(round(np.mean(x_4), 2))
    R_2.append(round(np.mean(x_5), 2))
    F_2.append(round(np.mean(x_6), 2))
    
    x_7 = [i[7]  for i in results_1 if ((i[0]=='m_estimators') and (i[3])==magnitude)]
    x_8 = [i[8]  for i in results_1 if ((i[0]=='m_estimators') and (i[3])==magnitude)]
    x_9 = [i[9]  for i in results_1 if ((i[0]=='m_estimators') and (i[3])==magnitude)]
    
    P_3.append(round(np.mean(x_7), 2))
    R_3.append(round(np.mean(x_8), 2))
    F_3.append(round(np.mean(x_9), 2))
    
    x_10 = [i[5]  for i in results_2 if (i[1]==magnitude) ]
    x_11 = [i[6]  for i in results_2 if (i[1]==magnitude) ]
    x_12 = [i[7]  for i in results_2 if (i[1]==magnitude) ]
    
    P_4.append(round(np.mean(x_10), 2))
    R_4.append(round(np.mean(x_11), 2))
    F_4.append(round(np.mean(x_12), 2))
 
print("#########")   
print (P_1)
print (P_2)
print (P_3)
print (P_4)
print("#########")   
print (R_1)
print (R_2)
print (R_3)
print (R_4)
print("#########")   
print (F_1)
print (F_2)
print (F_3)
print (F_4)

x = magnitude_list[1:] 
bar_width = 1
x1 = [i- 2.25*bar_width for i in x]
x2 = [i- 0.75*bar_width for i in x]
x3 = [i+ 0.75*bar_width for i in x]
x4 = [i+ 2.25*bar_width for i in x]
index = ['mean_std_2', 'mean_std_3', 'mean_std_4', 'mean_std_5',
         'mean_std_6', 'mean_std_7', 'mean_std_8', 'mean_std_9',
         'mean_std_10', 'mean_std_11', 'mean_std_12']
fig, ax = plt.subplots()
fig.set_size_inches(50, 10, forward=True)
ax.set_ylim(top=1.4)
ax.set_xlim([12, 140])
ax.bar(x1, P_3[1:], bar_width, label="G_mestimator", color= 'b')
ax.bar(x2, P_2[1:], bar_width, label="G_median", color= 'm')
ax.bar(x3, P_1[1:], bar_width, label="G_mean", color= 'c')
ax.bar(x4, P_4[1:], bar_width, label="LSTM_mestimator", color= 'g')
plt.xlabel('Magnitude', fontsize='24')
plt.ylabel('Precision', fontsize='24')
plt.xticks(x, index)
plt.legend(loc='best')
plt.show()

fig, ax = plt.subplots()
fig.set_size_inches(50, 10, forward=True)
ax.set_ylim(top=1.4)
ax.set_xlim([12, 140])
ax.bar(x1, R_3[1:], bar_width, label="G_mestimator", color= 'b')
ax.bar(x2, R_2[1:], bar_width, label="G_median", color= 'm')
ax.bar(x3, R_1[1:], bar_width, label="G_mean", color= 'c')
ax.bar(x4, R_4[1:], bar_width, label="LSTM_mestimator", color= 'g')
plt.xlabel('Magnitude', fontsize='24')
plt.ylabel('Recall', fontsize='24')
plt.xticks(x, index)
plt.legend(loc='best')
plt.show()

fig, ax = plt.subplots()
fig.set_size_inches(50, 10, forward=True)
ax.set_ylim(top=1.4)
ax.set_xlim([12, 140])
ax.bar(x1, F_3[1:], bar_width, label="G_mestimator", color= 'b')
ax.bar(x2, F_2[1:], bar_width, label="G_median", color= 'm')
ax.bar(x3, F_1[1:], bar_width, label="G_mean", color= 'c')
ax.bar(x4, F_4[1:], bar_width, label="LSTM_mestimator", color= 'g')
plt.xlabel('Magnitude', fontsize='24')
plt.ylabel('F-measure', fontsize='24')
plt.xticks(x, index)
plt.legend(loc='best')
plt.show()


    


    

