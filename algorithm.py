import time
import numpy as np
import matplotlib.pyplot as plt
import scipy
import time
import csv
import functions

#layers_dims = [5, 20, 7, 5, 1] 
#layers_dims = [20, 25, 25, 1] 

def L_layer_model(X, Y, layers_dims, learning_rate = 0.5, num_iterations = 10000, print_cost=False):#lr was 0.009

    np.random.seed(1)
    costs = []                  
    
    parameters = functions.initialize_parameters_deep(layers_dims)
    
    for i in range(0, num_iterations):

        AL, caches = functions.L_model_forward(X, parameters)
        
        cost = functions.compute_cost(AL, Y)
    
        grads = functions.L_model_backward(AL, Y, caches)
 
        parameters = functions.update_parameters(parameters, grads, learning_rate)
                
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)
            
    #plt.ylabel('cost')
    #plt.xlabel('iterations (per tens)')
    #plt.title("Learning rate =" + str(learning_rate))
    #plt.show()
    
    return parameters

layers_dims = [93, 40,  35,  30, 25, 20, 1] 




clusters = []
with open("clusters.csv", "r") as cluster_file:
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
    #print (len (x))
    
X, Y = np.array(X), np.array(Y)

print (X.shape)
print (Y.shape)
X = np.transpose(X)

Y = np.reshape(Y, (1, 512))

#for i in range(0, len(X)):
    #print("###########################")
    #print (X[i])

#print (Y)

parameters = L_layer_model(X, Y, layers_dims, num_iterations = 10000, print_cost = True)

pred_test, cost = functions.predict(X, Y, parameters)

print pred_test




