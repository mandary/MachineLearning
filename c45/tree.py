from math import log
import pickle
import time

class Tree:
    leaf = True
    prediction = None
    feature = None
    threshold = None
    left = None
    right = None

def dump_model(model, fname):
    pickle.dump([time.time(), model], open(fname, 'wb'))

def load_model(fname):
    tstamp, model = tuple(pickle.load(open(fname, 'r')))
    return model

def predict(tree, point):
    if tree.leaf:
        return tree.prediction
    i = tree.feature
    if (point.values[i] < tree.threshold):
        return predict(tree.left, point)
    else:
        return predict(tree.right, point)

def most_likely_class(prediction):
    labels = list(prediction.keys())
    probs = list(prediction.values())
    return labels[probs.index(max(probs))]

def accuracy(data, predictions):
    total = 0
    correct = 0
    for i in range(len(data)):
        point = data[i]
        pred = predictions[i]
        total += 1
        guess = most_likely_class(pred)
        if guess == point.label:
            correct += 1
    return float(correct) / total

def split_data(data, feature, threshold):
    left = []
    right = []
    # TODO: split data into left and right by given feature.
    # left should contain points whose values are less than threshold
    # right should contain points with values greater than or equal to threshold
    for pt in data:
        if pt.values[feature] < threshold:
            left.append(pt)
        else:
            right.append(pt)

    return (left, right)

def count_labels(data):
    counts = {}
    # TODO: counts should count the labels in data
    # e.g. counts = {'spam': 10, 'ham': 4}
    for pt in data:
        counts[pt.label] = counts.get(pt.label, 0) + 1
    return counts

def counts_to_entropy(counts):
    entropy = 0.0
    # TODO: should convert a dictionary of counts into entropy
    total = sum(counts.values())
    for key, value in counts.iteritems():
        prob = (value * 1.0)/total
        if prob != 0:
            entropy += - prob * log(prob, 2)
    return entropy
    
def get_entropy(data):
    counts = count_labels(data)
    entropy = counts_to_entropy(counts)
    return entropy

# This is a correct but inefficient way to find the best threshold to maximize
# information gain.
def find_best_threshold(data, feature):
    entropy = get_entropy(data)
    best_gain = 0
    best_threshold = None
    for point in data:
        left, right = split_data(data, feature, point.values[feature])
        curr = (get_entropy(left)*len(left) + get_entropy(right)*len(right))/len(data)
        gain = entropy - curr
        if gain > best_gain:
            best_gain = gain
            best_threshold = point.values[feature]
    return (best_gain, best_threshold)

def find_best_threshold_fast(data, feature):
    entropy = get_entropy(data)
    best_gain = 0
    best_threshold = None
    # TODO: Write a more efficient method to find the best threshold.
    sortedData = sorted(data, key =lambda x : x.values[feature])
    #best_gain, best_threshold = find_best_threshold(sortedData, feature)
    left = []
    right = sortedData[:]
    rightc = count_labels(right)
    leftc = {}

    last_threshold = None

    for pt in sortedData:
        if pt.values[feature] is not last_threshold:
            con = (counts_to_entropy(leftc)*len(left) + counts_to_entropy(rightc)*len(right))/len(data)
            # calculate information gain
            cur_gain = entropy - con
            # update best gain and threshold when necessary
            if cur_gain > best_gain:
                best_gain = cur_gain
                best_threshold = pt.values[feature]

        # move the top element from right to left
        to_left = right.pop(0)
        left.append(to_left)
        last_threshold = to_left.values[feature]
        rightc[to_left.label] -= 1
        leftc[to_left.label] = leftc.get(to_left.label, 0) + 1
    return (best_gain, best_threshold)

def find_best_split(data):
    if len(data) < 2:
        return None, None
    best_feature = None
    best_threshold = None
    best_gain = 0
    # TODO: find the feature and threshold that maximize information gain.
    for feature in range(len(data[0].values)):
        cur_gain, cur_threshold = find_best_threshold_fast(data, feature)
        if cur_gain > best_gain:
            best_gain = cur_gain
            best_threshold = cur_threshold
            best_feature = feature
    return (best_feature, best_threshold)

def make_leaf(data):
    tree = Tree()   
    counts = count_labels(data)
    prediction = {}
    for label in counts:
        prediction[label] = float(counts[label])/len(data)
    tree.prediction = prediction
    return tree


def c45(data, max_levels):
    if max_levels <= 0:
        return make_leaf(data)
    # TODO: Construct a decision tree with the data and return it.
    # Your algorithm should return a leaf if the maximum level depth is reached
    # or if there is no split that gains information, otherwise it should greedily
    # choose an feature and threshold to split on and recurse on both partitions
    # of the data.
    # Do not change the API of this function.
    best_feature, best_threshold = find_best_split(data)
    # no information gain
    if best_feature is None or best_threshold is None:
        return make_leaf(data)
    # internal node
    leftpt, rightpt = split_data(data, best_feature, best_threshold)
    root = Tree()
    root.leaf = False
    root.feature = best_feature
    root.threshold = best_threshold
    root.left = c45(leftpt, max_levels - 1)
    root.right = c45(rightpt, max_levels - 1)
    return root

####Extra Credit: Enhancements to your classifier##############################
def classifier_final(data, max_levels):
    # TODO: Your classifier with arbitrary enhancements.
    # NOTE : You're free to change the API of this function.
    return c4(data, max_levels)

def predict_final(model, point):
    # TODO: The predictor corresponding to classifier_final with arbitrary enhancements.
    # NOTE: Don't change the API of this function.
    return predict(model, point)
###############################################################################
def submission(train, test):
    # TODO: Once your tests pass, make your submission as good as you can!
    if train.__class__ == Tree:
        tree = train
    else:
        tree = c45(train, 9)
        #tree = classifier_final(train, 4)
    #dump_model(tree, "tree.p")
    predictions = []
    for point in test:
        predictions.append(predict_final(tree, point))
    return predictions

# This might be useful for debugging.
def print_tree(tree):
    if tree.leaf:
        print "Leaf", tree.prediction
    else:
        print "Branch", tree.feature, tree.threshold
        print_tree(tree.left)
        print_tree(tree.right)
