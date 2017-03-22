from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import time
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
from scipy.ndimage import filters
from scipy.spatial.distance import cosine
import urllib
from numpy import random
import os
import shutil
import string
import operator
from sets import Set
import tensorflow as tf

# ======================================================================================================================
# ========================================= Running the code ===========================================================
# ======================================================================================================================


run_part7  = False # Train LR model based on the wordvectors method
run_part8 = False # Similar words printed - examples from the report


# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================


def word2vec(path):
    '''
    Generates adjacent word tuple list from review database
    :param path: the location of the training set
    '''

    word_context = {'adjacent': []}
    # Go through the negative reviews
    for review in os.listdir(path + "/neg"):
        with open(os.getcwd()+"/"+path+"/neg/"+review) as f:
            text = f.read().split()
            for i in range(len(text)-1):
                word_context['adjacent'].append((text[i], text[i+1]))

    # Go through the positive reviews
    for review in os.listdir(path + "/pos"):
        with  open(os.getcwd()+"/"+path+"/pos/"+review)as f:
            text = f.read().split()
            for i in range(len(text)-1):
                word_context['adjacent'].append((text[i], text[i+1]))

    return word_context

def get_datasets(word_contexts, word_index_dictionary, embeddings, train_size, test_val_size):
    '''
    Generates matricies for the training set according to dictionaries specified
    Note that training examples for non-adjacent words are generated automatically
    ... the assumption is reasonable according to Piazza posts
    :param neg_dict: dictionary of negative reviews
    :param pos_dict: dictionary of positive reviews
    :return: x and y matrices prepared for classification
            X is a Mx256 vector, where M is the number of adjacent/non-adjacent words
                                     256 indicates 2 concatenated word vecotrs
            Y is a Mx1 vector, where M is the number of adjacent/non-adjacent words
    '''

    working_sets = {}
    random.seed(34)
    adjacent_indices = random.choice(len(word_contexts['adjacent']), train_size/2 + test_val_size, replace=False)
    random.seed(35)
    non_adjacent_indices1 = random.choice(len(embeddings), train_size/2 + test_val_size, replace=False)
    random.seed(36)
    non_adjacent_indices2 = random.choice(len(embeddings), train_size/2 + test_val_size, replace=False)

    x_train = zeros((train_size, 256))
    y_train = zeros((train_size, 2))  # one hot encoding
    x_test = zeros((test_val_size, 256))
    y_test = zeros((test_val_size, 2))  # one hot encoding
    x_validation = zeros((test_val_size, 256))
    y_validation = zeros((test_val_size, 2))  # one hot encoding

    # Fill up the first half - adjacent words
    curr_row = 0
    for i in range(0, train_size/2):
        index = adjacent_indices[i]
        word_tuple = (word_contexts['adjacent'])[index]
        a = embeddings[word_index_dictionary[word_tuple[0]]]
        b = embeddings[word_index_dictionary[word_tuple[1]]]
        x_train[curr_row, :] = np.concatenate([a, b])
        y_train[curr_row] = [1., 0.]
        curr_row += 1

    curr_row = 0
    for i in range(train_size / 2, train_size / 2 + test_val_size/2):
        index = adjacent_indices[i]
        word_tuple = (word_contexts['adjacent'])[index]
        a = embeddings[word_index_dictionary[word_tuple[0]]]
        b = embeddings[word_index_dictionary[word_tuple[1]]]
        x_test[curr_row, :] = np.concatenate([a, b])
        y_test[curr_row] = [1., 0.]
        curr_row += 1

    curr_row = 0
    for i in range(train_size / 2 + test_val_size / 2, train_size / 2 + test_val_size):
        index = adjacent_indices[i]
        word_tuple = (word_contexts['adjacent'])[index]
        a = embeddings[word_index_dictionary[word_tuple[0]]]
        b = embeddings[word_index_dictionary[word_tuple[1]]]
        x_validation[curr_row, :] = np.concatenate([a, b])
        y_validation[curr_row] = [1., 0.]
        curr_row += 1

    # Fill up the second half - non_adjacent words
    curr_row = train_size/2
    for i in range(0, train_size / 2):
        index1 = non_adjacent_indices1[i]
        index2 = non_adjacent_indices2[i]
        a = embeddings[index1]
        b = embeddings[index2]
        x_train[curr_row, :] = np.concatenate([a, b])
        y_train[curr_row] = [0., 1.]
        curr_row += 1

    curr_row = test_val_size/2
    for i in range(train_size / 2, train_size / 2 + test_val_size / 2):
        index1 = non_adjacent_indices1[i]
        index2 = non_adjacent_indices2[i]
        a = embeddings[index1]
        b = embeddings[index2]
        x_test[curr_row, :] = np.concatenate([a, b])
        y_test[curr_row] = [0., 1.]
        curr_row += 1

    curr_row = test_val_size/2
    for i in range(train_size / 2 + test_val_size / 2, train_size / 2 + test_val_size):
        index1 = non_adjacent_indices1[i]
        index2 = non_adjacent_indices2[i]
        a = embeddings[index1]
        b = embeddings[index2]
        x_validation[curr_row, :] = np.concatenate([a, b])
        y_validation[curr_row] = [0., 1.]
        curr_row += 1
    working_sets['test'] = (x_test, y_test)
    working_sets['train'] = (x_train, y_train)
    working_sets['validation'] = (x_validation, y_validation)
    print("Done constructing the data sets")
    return working_sets

def logistic_regression_part7(word_contexts, word_index_dictionary, embeddings, train_size, test_val_size, lam, total_iterations):
    '''
    Trains a LR model for prediciting adjacent words based on wordvectors
    :param word_contexts: array specifying tuples of adjacent words
    :param embeddings: list containing word vectors
    :param word_index_dictionary: dictionary specifying embedding indices
    :param train_size: number of adjacent/non-adjacent examples used in training
    :param test_val_size: number of adjacent/non-adjacent examples used in both testing and validation
    :param lam: regularization parameter
    :param total_iterations: number of training iterations
    :return: array of training results
    '''

    # Initialize Tensor Flow variables
    x = tf.placeholder(tf.float32, [None, 256])
    W0 = tf.Variable(tf.random_normal([256, 2], stddev=0.001))
    b0 = tf.Variable(tf.random_normal([2], stddev=0.001))

    y = tf.nn.softmax(tf.matmul(x, W0) + b0)
    y_ = tf.placeholder(tf.float32, [None, 2])

    decay_penalty = lam * tf.reduce_sum(tf.square(W0))
    reg_NLL = -tf.reduce_sum(y_ * tf.log(y)) + decay_penalty

    train_step = tf.train.AdamOptimizer(0.0005).minimize(reg_NLL)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    working_sets = get_datasets(word_contexts, word_index_dictionary, embeddings, train_size, test_val_size)
    test_x, test_y = working_sets['test']
    val_x, val_y = working_sets['validation']
    batch_xs, batch_ys = working_sets['train']


    # Run the TF, collect accuracies in separate arrays
    train_results = []
    test_results = []
    validation_results = []
    for i in range(total_iterations):
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

        if i % 10 == 0:
            test_accuracy = sess.run(accuracy, feed_dict={x: test_x, y_: test_y})
            validation_accuracy = sess.run(accuracy, feed_dict={x: val_x, y_: val_y})
            train_accuracy = sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys})

            print("i=", i)
            print("Train: " + str(train_accuracy))
            print("Validation: " + str(validation_accuracy))
            print("Test: "  + str(test_accuracy))
            print("Penalty: " + str(sess.run(decay_penalty)))
            train_results.append(train_accuracy)
            test_results.append(test_accuracy)
            validation_results.append(validation_accuracy)
    return (train_results, validation_results, test_results)

def find_similar(embeddings, word_index_dictionary, word, n):
    '''
    Finds and prints n words that have the closest wordvector to the one specified by 'word'
    :param embeddings: list containing word vectors
    :param word_index_dictionary: dictionary specifying embedding indices
    :param word: word specifying similaity
    :param n: number of words
    '''
    target_vector = embeddings[word_index_dictionary[word]]
    cosine_distances_dictionary = {}
    for similar_word in word_index_dictionary.keys():
        candidate_vector = embeddings[word_index_dictionary[similar_word]]
        cosine_distances_dictionary[similar_word] = cosine(target_vector, candidate_vector)

    top_n = sorted(cosine_distances_dictionary.items(), key=lambda x: x[1], reverse=False)[1:n+1]
    print(str(n)+" most similar words to \'"+word+"\' are:")
    for word in top_n:
        print(word[0])

    return True

if run_part7:
    # Optimal parameters
    train_size = 50000
    test_val_size = 5000
    iters = 1000
    lam = 0.001

    embeddings = load("embeddings.npz")["emb"]
    word_contexts = word2vec('training_set')
    word_indices = load("embeddings.npz")["word2ind"].flatten()[0]

    # Revert the word_indices list to a proper usable format
    word_index_dictionary = {}
    for i in range(len(word_indices)):
        word_index_dictionary[word_indices[i]] = i
    print len(word_contexts['adjacent'])
    # Sanity check - remove words, which don't have embeddings
    i = 0
    while i < len(word_contexts['adjacent']):
        word_tuple = (word_contexts['adjacent'])[i]
        if (word_tuple[0] in word_index_dictionary) and (word_tuple[1] in word_index_dictionary):
            i += 1
        else:
            del (word_contexts['adjacent'])[i]

    # Now, train the LR model
    results = logistic_regression_part7(word_contexts, word_index_dictionary, embeddings, train_size, test_val_size, lam, iters)

    train_results = results[0]
    validation_results = results[1]
    test_results = results[2]

    print("Final training accuracy: " + str(train_results[-1]))
    print("Final validation accuracy: " + str(validation_results[-1]))
    print("Final testing accuracy: " + str(test_results[-1]))

if run_part8:

    word_indices = load("embeddings.npz")["word2ind"].flatten()[0]
    embeddings = load("embeddings.npz")["emb"]

    # Revert the word_indices list to a proper usable format
    word_index_dictionary = {}
    for i in range(len(word_indices)):
        word_index_dictionary[word_indices[i]] = i

    find_similar(embeddings, word_index_dictionary, 'story', 10)
    find_similar(embeddings, word_index_dictionary, 'good', 10)

    find_similar(embeddings, word_index_dictionary, 'tv', 10)
    find_similar(embeddings, word_index_dictionary, 'space', 10)