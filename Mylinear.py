import random

import interpolate as interpolate
import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
import spline as spline


def loadDataset(filename, split):
    dataset = pd.read_csv(filename)
    df = pd.DataFrame(dataset,dtype='float')
    # for index, row in df.iterrows():
    #     print(row["Age"],row['year'])
    df1 = df.sample(frac=split)
    rowlist = []
    for indexs in df1.index:
        rowlist.append(indexs)
    df2 = df.drop(rowlist, axis=0)
    return df1,df2


def getAccurate(testset, predictions):
    correct = 0
    for i in range(len(testset)):
        if int(testset.iat[i, -1]) == predictions[i]:
            correct += 1
    return (correct / float(len(testset)))


def getRandomWeight(weight, set):
    for i in range(set.shape[1]):
        weight.append(random.uniform(0.1, 0.9))


def getWeight(weights, traingsets, learningrate, threshold, labelpos, labelneg,testsets,allerrorrate,times):
    previouserror=[]
    newerror=[]
    for q in range(times):
        previouserror.append(0)
        newerror.append(1)
    mark=0
    t=0
    while True:
        for u in range(len(traingsets)):
            traingset=traingsets[u]
            testset=testsets[u]
            for i in range(len(traingset)):
                result = weights[u][0]
                list1 = traingset.iloc[i].values
                hlabel = 0
                clable = int(list1[-1])
                j = 0
                k = 1
                while j != len(list1) - 1 and k != len(weights[u]):
                    result = result + weights[u][k] * list1[j]
                    j += 1
                    k += 1
                if result < threshold:
                    hlabel = labelneg
                elif result > threshold:
                    hlabel = labelpos
                else:
                    r = random.randint(1, 2)
                    if r == 1:
                        hlabel = labelpos
                    else:
                        hlabel = labelneg
                if hlabel != clable:
                    for p in range(len(weights[u])):
                        if p == 0:
                            weights[u][p] = weights[u][p] + learningrate * (clable - hlabel) * 1
                        else:
                            weights[u][p] = weights[u][p] + learningrate * (clable - hlabel) * list1[p - 1]
                predictions = []
                for j in range(len(testset)):
                    row = testset.iloc[j].values
                    label = getlabel(row, weights[u], threshold, labelpos, labelneg)
                    predictions.append(label)
                    # print('predicted:', label, ',actual:', testset.iat[j,-1])
                newerror[u] = 1 - getAccurate(testset, predictions)
                allerrorrate[u].append(newerror[u])
            # print(abs(previouserror[u] - newerror[u]),previouserror[u],newerror[u])
        sumprevious=0
        sumnew=0
        for v in range(len(traingsets)):
            sumprevious+=previouserror[v]
            sumnew+=newerror[v]
        sumprevious/=len(traingsets)
        sumnew/=len(traingsets)
        if abs(sumprevious - sumnew) < 0.01:
            # print("errorrate", errorrate)
            break
            # mark+=1
        else:
            for v in range(len(traingsets)):
                previouserror[v]=newerror[v]
        # t+=1
        # print("t:",t)
        # if mark!=0:
        #     break
    # 对比错误率？
    return allerrorrate


def getlabel(instance, weight, threshold, labelpos, labelneg):
    result = weight[0]
    j = 0
    k = 1
    while j != len(instance) - 1 and k != len(weight):
        result = result + weight[k] * float(instance[j])
        j += 1
        k += 1
    if result < threshold:
        return labelneg
    else:
        return labelpos


def normalize(set):
    columns = list(set)
    length = len(set)-1
    for column in range(len(columns) - 1):
        # set.sort_values(by=columns[column], inplace=True, ascending=True)  # if it does not work , add inplace
        mymax=set.max(axis=0)
        # min = set.iat[0, column]
        # max = set.iat[length, column]
        mymin=set.min(axis=0)
        if mymax[columns[column]]<=1 and mymin[columns[column]]>=0:
            continue
        else:
            for i in range(len(set)):
                set.iat[i, column] = float(set.iat[i, column] - mymin[columns[column]]) / float(mymax[columns[column]] - mymin[columns[column]])

def main(filename, labelpos, labelneg,avgerror,allerrorrate):
    split = 0.7
    times=10
    learningrate = 0.5
    threshold = 0
    weight = []
    weights = []
    trainingsets=[]
    testsets=[]
    avgweight=[]
    for i in range(times):
        trainingset, testset = loadDataset(filename, split)
        print("id",i+1,"trainingset:", len(trainingset), "testset:", len(testset))
        if i == 0:
            getRandomWeight(weight, trainingset)
        weights.append(weight.copy())
        list3=[]
        allerrorrate.append(list3)
        normalize(trainingset)
        normalize(testset)
        trainingsets.append(trainingset)
        testsets.append(testset)
    # print(len(trainingsets))
    getWeight(weights, trainingsets, learningrate, threshold, labelpos, labelneg, testsets,allerrorrate,times)
    for k in range(len(weights[0])):
        sum=0
        for i in range(len(weights)):
            sum+=weights[i][k]
        avg=float(sum)/float(len(weights))
        avgweight.append(avg)
    print("avgweight: ",avgweight)
    trainingerrorrate=[]
    for i in range(len(trainingsets)):
        predictions=[]
        for j in range(len(trainingsets[i])):
            row = trainingsets[i].iloc[j].values
            label = getlabel(row, avgweight, threshold, labelpos, labelneg)
            predictions.append(label)
        trainingerrorrate.append(1 - getAccurate(trainingsets[i], predictions))
    print("trainingerrorrate: ",trainingerrorrate)
    plt.figure()  # Create a drawing object
    tx=[1,2,3,4,5,6,7,8,9,10]
    plt.xticks([x for x in range(max(tx) + 1) if x % 1 == 0])
    plt.plot(tx, trainingerrorrate, "r",linewidth=1)  # Draw on the current drawing object (X axis, Y axis, blue dotted line, line width)
    plt.xlabel("subset") #X label
    plt.ylabel("error rate")  #Y label
    plt.show()  #show the image
    for j in range(len(allerrorrate[0])):
        sum=0
        for i in range(len(allerrorrate)):
            sum+=allerrorrate[i][j]
        avg=float(sum)/float(len(allerrorrate))
        avgerror.append(avg)
    return len(trainingsets[0])
    # predictions = []
    # for i in range(len(testset)):
    #     row = testset.iloc[i].values
    #     label = getlabel(row, weight, threshold, labelpos, labelneg)
    #     predictions.append(label)
    #     print('predicted:', label, ',actual:', testset.iat[i, -1])
    # accuracy = getAccurate(testset, predictions)
    # print("error rate:", 1 - accuracy)
    # print("epochs:", epoch)



filename = 'test.csv'
print("positive label:")
# labelpos = int(input())
labelpos = 2
print("negative label:")
# labelneg = int(input())
labelneg = 0
avgerror = []
allerrorrate=[]
examples=[]
num=main(filename, labelpos, labelneg,avgerror,allerrorrate)
print("allerrorrate", allerrorrate)
print("avgerror",avgerror)
# for i in range(len(allerrorrate[0])):
#     examples.append(i+1)
#
# print("examples ",examples)
# plt.figure() #Create a drawing object
# plt.xticks([x for x in range(max(examples) + 1) if x % 10 == 0])
# plt.plot(examples,avgerror,"b--",linewidth=1)   #Draw on the current drawing object (X axis, Y axis, blue dotted line, line width)
# for i in range(len(avgerror)):
#     if (i+1)%(num)==0 :
#         plt.plot(examples[i],avgerror[i],"b--",linewidth=1,marker='o')
#     if i==9:
#         plt.plot(examples[i], avgerror[i], "r--", linewidth=1, marker='o')
#         plt.text(examples[i], round(avgerror[i],3), (examples[i], round(avgerror[i],3)), ha='center', va='bottom', fontsize=10)
# plt.xlabel("examples") #X label
# plt.ylabel("error rate")  #Y label
# plt.title("perceptron learning")
# plt.savefig('0and2_10.png')
# plt.show()  #show the image