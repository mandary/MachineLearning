from math import exp
import random
import numpy as np
import pdb

# TODO: Calculate logistic
def logistic(x):
    return 1.0 / (1 + np.exp(-x))

# TODO: Calculate accuracy of predictions on data
def accuracy(data, predictions):
    correct = 0
    for i in range(len(data)):
        point = data[i]['label']
        prediction = predictions[i].item(0)
        if prediction >= 0.5 and point == 1:
            correct += 1
        if prediction < 0.5 and point == 0:
            correct += 1
    return correct * 1.0 / len(predictions)


class model:
    def __init__(self, structure):
        self.weights=[]
        self.bias = []
        for i in range(len(structure)-1):
            self.weights.append(np.random.normal(size=(structure[i], structure[i+1])))
            self.bias.append(np.random.normal(size=(1, structure[i+1])))
            
    # TODO: Calculate prediction based on model
    def predict(self, point):
        a = self.feedforward(point)
        return a[len(a) - 1]
        #return logistic(np.dot(point['features'], self.weights[0]) + self.bias[0])

    # TODO: Update model using learning rate and L2 regularization
    def update(self, a, delta, eta, lam):
        #pdb.set_trace()
        for i in range(len(self.weights)):
            self.weights[i] = self.weights[i] - eta * (lam * self.weights[i] - a[i].T * delta[i].T)
            self.bias[i] = self.bias[i] - eta * delta[i].T

    # TODO: Perform the forward step of backpropagation
    def feedforward(self, point):
        #pdb.set_trace()
        result = []
        temp = point['features']
        result.append(temp)
        for i in range(len(self.weights)):
            temp = logistic(np.dot(temp, self.weights[i]) + self.bias[i])
            result.append(temp)
        return result
    
    # TODO: Backpropagate errors
    def backpropagate(self, a, label):
        #pdb.set_trace()
        l = len(a) - 1
        error = []
        delta = label - a[l]
        error.insert(0, delta)
        l -= 1
        while l > 0:
            dadz = np.multiply(a[l], 1 - a[l])
            dldz = np.dot(delta, self.weights[l].T)
            delta = np.multiply(dldz, dadz).T
            error.insert(0, delta)
            l -= 1
        return error


    # TODO: Train your model
    def train(self, data, epochs, rate, lam):
        for i in range(epochs):
            for j in range(len(data)):
                select = random.randint(0, len(data) - 1)
                datapoint = data[select]
                a = self.feedforward(datapoint)
                error = self.backpropagate(a, datapoint['label'])
                self.update(a, error, rate, lam)

def logistic_regression(data, lam=0.0001):
    m = model([data[0]["features"].shape[1], 1])
    m.train(data, 100, 0.005, lam)#0.005
    return m
    
def neural_net(data, lam=0.001):
    m = model([data[0]["features"].shape[1], 15, 1])
    m.train(data, 500, 0.005, lam)
    return m
