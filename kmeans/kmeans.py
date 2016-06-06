import numpy as np
import random
import copy
import math
import pdb
import matplotlib.pyplot as plt


def getData():
    return np.genfromtxt('digit.txt')

def getLabel():
    return np.genfromtxt('labels.txt', dtype=int)

def selectK(k, rand, datas):
    index = []
    if rand:
        index = random.sample(range(len(datas)), k)
    else:
        index = range(k)
    kmeans = [0] * k
    for i in range(k):
        kmeans[i] = copy.deepcopy(datas[index[i]])
    return kmeans

def euclideanDist(x, c):
    distance = 0.0
    for i in range(len(c)):
        distance += math.pow(x[i] - c[i], 2)
    return distance

def assign2K(kmeans, datas):
    assignment = [0] * len(datas)
    for i in range(len(datas)):
        data = datas[i]
        mindist = float("infinity")
        for k in range(len(kmeans)):
            center = kmeans[k]
            tempdist = euclideanDist(data, center)
            if tempdist < mindist:
                assignment[i] = k
                mindist = tempdist
    return assignment

def recalcMeans(k, assignment, datas):
    kSum = np.zeros(shape = (k, len(datas[0])))
    kSize = [0] * k
    for i in range(len(datas)):
        kSum[assignment[i]] = np.add(kSum[assignment[i]], datas[i])
        kSize[assignment[i]] += 1
    for i in range(k):
        kSum[i] = np.divide(kSum[i], kSize[i])
    return kSum

def mistakeRate(k, numLabels, assignment, labels):
    kLabels = np.zeros(shape = (k, numLabels))
    for i in range(len(assignment)):
        kLabels[assignment[i]][(labels[i]-1)/2] += 1
    print kLabels
    mistake = 0.0
    for i in range(k):
        mistake += sum(kLabels[i]) - max(kLabels[i])
    return mistake / len(assignment)

def groupSumSquares(kmeans, assignment, datas):
    sumSquares = [0.0] * len(kmeans)
    for i in range(len(assignment)):
        sumSquares[assignment[i]] += euclideanDist(datas[i], kmeans[assignment[i]])
    return sumSquares

def kMeans(k, iterations, numLabels, rand, labels, datas):
    print "k =", k
    kmeans = selectK(k, rand, datas)
    i = 0
    prevAssignment = [0] * len(labels)
    newAssignment = [0] * len(labels)
    while i < iterations:
        #pdb.set_trace()
        newAssignment = assign2K(kmeans, datas)
        if cmp(prevAssignment, newAssignment) is 0:
            print "iteration", i+1
            break
        else:
            prevAssignment = newAssignment
        kmeans = recalcMeans(k, newAssignment, datas)
        i += 1
    SSE = sum(groupSumSquares(kmeans, newAssignment, datas))
    MR = mistakeRate(k, numLabels, newAssignment, labels)
    print "SSE", SSE
    print "MR", MR
    return (SSE, MR)

def plotK(k, iterations, numLabels, rand, labels, datas):
    SSEs = []
    MRs = []
    cluster = range(1, k + 1)
    for i in cluster:
        SSE, MR = kMeans(i, iterations, numLabels, rand, labels, datas)
        SSEs.append(SSE)
        MRs.append(MR)
    plt.figure("group sum of squares")
    plt.figure(figsize=(10,8))
    plt.plot(cluster, SSEs)
    plt.xlabel("k")
    plt.ylabel("SSE")
    plt.savefig("SSE.jpg")
    plt.figure("mistake rate")
    plt.figure(figsize=(10,8))
    plt.plot(cluster, MRs)
    plt.xlabel("k")
    plt.ylabel("MR")
    plt.savefig("MR.jpg")
if __name__ == '__main__':
    X = getData()
    Y = getLabel()
    #kMeans(2, 20, 4, False, Y, X)
    plotK(10, 20, 4, True, Y, X)