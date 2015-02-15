#!/usr/bin/env python
import numpy as np
from collections import Counter

class Perceptron(object):
    '''
    Simple binary classifier
    '''

    def __init__(self):

        # keep track of how long a weight vector is sustained
        self.errors = list()
        self.alpha = 0.4 #learning rate
        self.epochs = 1000
        self.training_data = None # list of tuples (vector, 0 or 1)
        self.initial_weight = None
        self.weights = None
        self.weight_history = Counter()
        self.averaged_weights = None

    def initialize_weights(self):
        # a vector of zeros that is the length of our training samples
        return np.zeros(len(self.training_data[0][0]))

    def average_weights(self):
        # creates an averaged perceptron
        return np.mean([np.array(vector)*cnt for vector,cnt in self.weight_history.most_common()], axis=0)

    def threshold(self, label):
        # threshold is assumed to be 0
        return 0 if label < 0 else 1

    def train(self, training_data):

        self.training_data = training_data
        training_vectors, gold_labels = training_data

        self.initial_weights = self.initialize_weights()

        self.weights = self.initial_weights
        # for status at 10% increments
        status_increments = [i * int(self.epochs/float(10)) for i in range(1,11)]
        for epoch in range(1, self.epochs+1):
            # for status
            if epoch in status_increments:
                print "training {0}% complete...".format((status_increments.index(epoch)+1)*10)
            # training loop
            last = self.weights
            for training_instance in range(len(training_vectors)):
                # get vector and label
                training_vector = training_vectors[training_instance]
                gold_label = gold_labels[training_instance]
                # estimate label
                prediction = self.predict_label(training_vector, self.weights)
                # 0 if same...
                error = gold_label - prediction
                #self.errors[error] += 1
                if error != 0:
                    self.errors.append(error)
                    # update weights (only if predicted label is wrong)
                    bias = self.alpha * error * training_vector
                    self.weights += bias
                    self.weight_history[tuple(bias)] += 1

            # can we finish early?
            if np.dot(self.weights, last) == 1:
                self.averaged_weights = self.average_weights()
                return
        self.averaged_weights = self.average_weights()

    def predict(self, experimental_data, weights=None):
        weights = self.weights if weights==None else weights
        return [self.predict_label(vector, weights) for vector in experimental_data]

    def averaged_predict(self, experimental_data):
        return self.predict(experimental_data, weights=self.averaged_weights)

    def predict_label(self, vector, weights):
        raw_prediction = np.dot(weights, vector)
        return self.threshold(raw_prediction)

    def evaluate(self, gold_labels, predicted_labels):

        def calculate_accuracy():
            return sum([1 if gold_labels[i] == predicted_labels[i] else 0 for i in range(len(gold_labels))])/float(len(gold_labels))
            pass

        def calculate_recall(self):
            pass

        def calculate_f1(self):
            pass

        print "Accuracy:\t{0:.3}".format(calculate_accuracy())

        #return precision, recall, f1
        pass


letters = 'abcdefghijklmnopqrstuvwxyz'
from nltk.corpus import names

def mk_feature_vector(instance):
    instance = instance.lower()
    vector = []
    for letter in letters:
        vector.append(1 if instance.startswith(letter) else 0)
        vector.append(1 if instance.endswith(letter) else 0)
    return np.array(vector)

male_vectors = [mk_feature_vector(m) for m in names.words('male.txt')]
female_vectors = [mk_feature_vector(f) for f in names.words('female.txt')]

vectors = male_vectors + female_vectors
labels = [1 for i in xrange(len(male_vectors))] + [0 for i in xrange(len(female_vectors))]

with open("name_vectors.txt", "w") as out:
    for v in vectors:
        out.write(",".join(str(element) for element in v) + "\n")

with open("name_labels.txt", "w") as out:
    for label in labels:
        out.write("{0}\n".format(label))



#plt.scatter(*g1.T, color='r')
#plt.scatter(*g2.T, color='b')
#from sklearn import decomposition
#pca = decomposition.PCA(n_components=2)
#pca.fit(some_data)
#pca.transform(some_data)

#X = iris.data
#y = iris.target
#target_names = iris.target_names
#pca = PCA(n_components=2)
#X_r = pca.fit(X).transform(X)
# Percentage of variance explained for each components
#print('explained variance ratio (first two components): %s'
#      % str(pca.explained_variance_ratio_))

#plt.figure()
#for c, i, target_name in zip("rgb", [0, 1, 2], target_names):
#    plt.scatter(X_r[y == i, 0], X_r[y == i, 1], c=c, label=target_name)
#plt.legend()
#plt.title('PCA of IRIS dataset')

'''
training_data =  [(array([0,0,1]), 0),
                  (array([0,1,1]), 1),
                  (array([1,0,1]), 1),
                  (array([1,1,1]), 1)]
'''
