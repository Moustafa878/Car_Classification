import numpy as np
import pandas as pd
import operator
import math
#-----------------------------------(Convert features to int)------------------------------------------------------
def conv_to_int(y):
    y = np.asarray(y)
    for i in range(0, len(y)):
        if (y[i] == 'vhigh'):
            y[i] = 0
        elif (y[i] == 'high'):
            y[i] = 1
        elif (y[i] == 'med'):
            y[i] = 2
        elif (y[i] == 'low'):
            y[i] = 3
        elif (y[i] == '2'):
            y[i] = 0
        elif (y[i] == '3'):
            y[i] = 1
        elif (y[i] == '4'):
            y[i] = 2
        elif (y[i] == '5more'):
            y[i] = 3
        elif (y[i] == 'more'):
            y[i] = 2
        elif (y[i] == 'small'):
            y[i] = 0
        elif (y[i] == 'big'):
            y[i] = 2

        else:
            continue
    return y
#--------------------------------(spearate data into 75% & 25%)-----------------------------------------------------------
def sperate_data(dataset, trainingSet=[], testSet=[]):
    for x in range(len(dataset)):

        if x < 1727 * 0.75:
            trainingSet.append(dataset[x])
        else:
            testSet.append(dataset[x])
#--------------------------------(calcilate euclidean distance)---------------------------------------------------------------
def euclidean_Distance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)
#-----------------------------------(get nearest 5 neighbors)--------------------------------------------------------------
def get_K_Neighbors(trainingSet, testInstance,k):
    distances = []
    length = len(testInstance)-1
    for x in range(len(trainingSet)):
        dist = euclidean_Distance(testInstance, trainingSet[x], length)
        distances.append((trainingSet[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors
#-----------------------------------(classify car to class)----------------------------------------------------------------
def getResponse(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)

    return sortedVotes[0][0]
#------------------------------------(calculate accuracy)----------------------------------------------------------------
def calculate_Accuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] is predictions[x]:
            correct += 1
    return (correct/float(len(testSet))) * 100.0
#----------------------------------------------------------------------------------------------------
name = ['Bprice', 'MPrice', 'Ndoors', 'Capacity', 'sizeLB', 'safety', 'Class']
dataset = pd.read_csv("car.data.csv", names=name)

Bprice_encoded = conv_to_int(dataset["Bprice"])
MPrice_encoded = conv_to_int(dataset["MPrice"])
Ndoors_encoded = conv_to_int(dataset["Ndoors"])
Capacity_encoded = conv_to_int(dataset["Capacity"])
sizeLB_encoded = conv_to_int(dataset["sizeLB"])
safety_encoded = conv_to_int(dataset["safety"])
print('First 5 row of data ')
print('-------')
print(dataset.head())
Training_Set = []
Test_Set = []
prediction=[]
dataset_withoutHeader =dataset.iloc[1:,:].values

sperate_data(dataset_withoutHeader,Training_Set,Test_Set)

for i in range(len(Test_Set)):
     x=get_K_Neighbors(Training_Set,Test_Set[i],5)
     g=getResponse(x)
     prediction.append(g)
     print("nearest 5 neighbors : ")
     print(x)
     print('this car join class : ', g)
     print('***************************************************************************************************')


accuracy=calculate_Accuracy(Test_Set,prediction)
print('Accuracy = ', accuracy)