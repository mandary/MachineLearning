from numpy import *
from scipy import *
import numpy as np
import matplotlib.pyplot as plt
import itertools
import pickle
import time
import pdb

##
def svm_binary(predicate, data, lamb=1e-2, eta=1e-2, w=None):
    (labels, features) = data
    if not w:
        w = zeros(features.shape[1] + 1)
    #TODO: do one pass of SGD over the dataset.
    for i in range(len(labels)):
    	#pdb.set_trace()
    	y = 2 * predicate(labels[i]) - 1
    	loss = 1 if ((1 - y * (np.dot(w[1:], features[i]) + w[0])) > 0) else 0
    	w[0] -= eta * (-y * loss)
    	w[1:] -= eta * (w[1:] * lamb - features[i] * y * loss)
    return w

def svm_accuracy(predicate, w, data):
    (labels, features) = data
    error = 0
    for i in xrange(labels.shape[0]):
        xi = features[i, :]; yi = 2 * predicate(labels[i]) - 1
        if (w[0] + w[1:].dot(xi)) * yi < 0:
            error += 1
    return 1 - error/(1.0 * labels.shape[0])

def svm(predicate, train, validation, params=[(1e-2, 1e-2)]):
    max_accuracy = 0.; max_param = None; max_w = None
    #TODO: iterate over @arg{params} and return that which attains the maximum
    #       accuracy on the validation set.
    for param in params:
    	w = svm_binary(predicate, train, param[0], param[1])
    	accuracy = svm_accuracy(predicate, w, validation)
    	if accuracy > max_accuracy:
    		max_accuracy = accuracy
    		max_param = param
    		max_w = w

    return (max_w, max_param, max_accuracy)

def svm_multiclass(train, validation, params=[(1e-2, 1e-2)]):
    (labels, features) = train
    label_set = np.unique(labels)
    weights = []
    #TODO: Compute one-vs-all classifiers by iterating over @arg{labels}
    #      return a set of tuples, weights = [*(l, w_l)]
    for label in label_set:
    	(w, param, accuracy) = svm(lambda ll: ll == label, train, validation, params)
    	weights.append((label, w))
    return weights

def svm_predict(x, svms):
    def svm_predict_kernel(x, svms):
        max_hh = 0; max_label = -1
        #TODO: return the predicted class of x using the one-versus-all scheme.
        # iterate over @arg{labels} and return a set of tuples, weights = [*(l, w_l)] 
        for svm in svms:
            (label, weights) = svm
            #pdb.set_trace()
            distance = np.dot(x, weights[1:]) + weights[0]
            if distance > max_hh:
                max_hh = distance
                max_label = label
        return max_label

    if len(x.shape) > 1:
        predict = zeros(x.shape[0], dtype=int64)
        for ii in xrange(x.shape[0]):
            predict[ii] = svm_predict_kernel(x[ii, :], svms)
        return predict
    else:
        return svm_predict_kernel(x, svms)
