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


run_part7  = False
run_part8 = True


# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================


def word2vec(path, size):
    '''




    Generates word occurence dictionaries for positive and negative reviews
    :param path: the location of the training set
    :return:
    '''
    total = 0
    total_non = 0
    word_context = {'adjacent': [], 'non_adjacent': []}
    # Go through the negative reviews
    for review in os.listdir(path + "/neg"):
        with open(os.getcwd()+"/"+path+"/neg/"+review) as f:
            text = f.read().split()
            for i in range(len(text)-1):
                total += 1
                word_context['adjacent'].append((text[i], text[i+1]))
                if total > size/2:
                    break
                for j in range(i+2, len(text)):
                    word_context['non_adjacent'].append((text[i], text[j]))
                    if total_non > size/2:
                        break
                    total_non += 1
                
    # Go through the positive reviews
    for review in os.listdir(path + "/pos"):
        with  open(os.getcwd()+"/"+path+"/pos/"+review)as f:
            text = f.read().split()
            for i in range(len(text)-1):
                total += 1
                word_context['adjacent'].append((text[i], text[i+1]))
                if total > size:
                    break
                for j in range(i+2, len(text)):
                    word_context['non_adjacent'].append((text[i], text[j]))
                    if total_non > size:
                        break
                    total_non += 1
    return word_context

def get_datasets(word_contexts, word_index_dictionary, embeddings, train_size, test_val_size):
    '''




    Generates matricies for the training set according to dictionaries specified
    :param neg_dict: dictionary of negative reviews
    :param pos_dict: dictionary of positive reviews
    :return: x and y matricies prepared for classificaiton
            X is a MxN vector, where M is the number of reviews
                                     N is the number of words
            Y is a Mx1 vector, where M is the number of reviews
    '''

    working_sets = {}
    random.seed(34)
    adjacent_indices = random.choice(len(word_contexts['adjacent']), train_size/2 + test_val_size, replace=False)
    random.seed(43)
    non_adjacent_indices = random.choice(len(word_contexts['non_adjacent']), train_size/2 + test_val_size, replace=False)

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
        y_train = [1, 0]
        curr_row += 1

    curr_row = 0
    for i in range(train_size / 2, train_size / 2 + test_val_size/2):
        index = adjacent_indices[i]
        word_tuple = (word_contexts['adjacent'])[index]
        a = embeddings[word_index_dictionary[word_tuple[0]]]
        b = embeddings[word_index_dictionary[word_tuple[1]]]
        x_test[curr_row, :] = np.concatenate([a, b])
        y_test = [1, 0]
        curr_row += 1

    curr_row = 0
    for i in range(train_size / 2 + test_val_size / 2, train_size / 2 + test_val_size):
        index = adjacent_indices[i]
        word_tuple = (word_contexts['adjacent'])[index]
        a = embeddings[word_index_dictionary[word_tuple[0]]]
        b = embeddings[word_index_dictionary[word_tuple[1]]]
        x_validation[curr_row, :] = np.concatenate([a, b])
        y_validation[curr_row] = [1, 0]
        curr_row += 1

    # Fill up the second half - non_adjacent words
    curr_row = train_size/2
    for i in range(0, train_size / 2):
        index = non_adjacent_indices[i]
        word_tuple = (word_contexts['non_adjacent'])[index]
        a = embeddings[word_index_dictionary[word_tuple[0]]]
        b = embeddings[word_index_dictionary[word_tuple[1]]]
        x_train[curr_row, :] = np.concatenate([a, b])
        y_train = [0, 1]
        curr_row += 1

    curr_row = test_val_size/2
    for i in range(train_size / 2, train_size / 2 + test_val_size / 2):
        index = non_adjacent_indices[i]
        word_tuple = (word_contexts['non_adjacent'])[index]
        a = embeddings[word_index_dictionary[word_tuple[0]]]
        b = embeddings[word_index_dictionary[word_tuple[1]]]
        x_test[curr_row, :] = np.concatenate([a, b])
        y_test = [0, 1]
        curr_row += 1

    curr_row = test_val_size/2
    for i in range(train_size / 2 + test_val_size / 2, train_size / 2 + test_val_size):
        index = non_adjacent_indices[i]
        word_tuple = (word_contexts['non_adjacent'])[index]
        a = embeddings[word_index_dictionary[word_tuple[0]]]
        b = embeddings[word_index_dictionary[word_tuple[1]]]
        x_validation[curr_row, :] = np.concatenate([a, b])
        y_validation[curr_row] = [0, 1]
        curr_row += 1
    working_sets['test'] = (x_test, y_test)
    working_sets['validation'] = (x_train, y_train)
    working_sets['train'] = (x_validation, y_validation)
    print("Done constructing the data sets")
    return working_sets

def logistic_regression_part7(word_contexts, word_index_dictionary, embeddings, train_size, test_val_size, lam, total_iterations):
    '''





    Performs logistic regression with Tensor Flow
    :param neg_dict: dictionary of negative reviews
    :param pos_dict: dictionary of positive reviews
    :param num_words: total number of words
    :param lam: regularizing parameter lambda
    :return: performance
    '''

    # Initialize Tensor Flow variables
    x = tf.placeholder(tf.float32, [None, 256])
    W0 = tf.Variable(tf.random_normal([256, 2], stddev=0.01))
    b0 = tf.Variable(tf.random_normal([2], stddev=0.01))

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
        # print i
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

        if i % 3 == 0:
            test_accuracy = sess.run(accuracy, feed_dict={x: test_x, y_: test_y})
            validation_accuracy = sess.run(accuracy, feed_dict={x: val_x, y_: val_y})
            train_accuracy = sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys})
            if print_iterations:
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
    train_size = 10000
    test_val_size = 1000

    embeddings = load("embeddings.npz")["emb"]
    word_contexts = word2vec('training_set', 20000)
    word_indices = load("embeddings.npz")["word2ind"].flatten()[0]

    # Revert the word_indices list to a proper usable format
    word_index_dictionary = {}
    for i in range(len(word_indices)):
        word_index_dictionary[word_indices[i]] = i

    print("Done constructing the index dictionary")

    # Sanity check
    i = 0
    while i < len(word_contexts['adjacent']):
        word_tuple = (word_contexts['adjacent'])[i]
        if (word_tuple[0] in word_index_dictionary) and (word_tuple[1] in word_index_dictionary):
            i += 1
        else:
            del (word_contexts['adjacent'])[i]

    i = 0
    while i < len(word_contexts['non_adjacent']):
        word_tuple = (word_contexts['non_adjacent'])[i]
        if (word_tuple[0] in word_index_dictionary) and (word_tuple[1] in word_index_dictionary):
            i += 1
        else:
            del (word_contexts['non_adjacent'])[i]

    print("Done sanity check")

    # Now, train the LR model
    iters = 100
    lam = 0
    results = logistic_regression_part7(word_contexts, word_index_dictionary, embeddings, train_size, test_val_size, lam, iters)

    train_results = results[0]
    validation_results = results[1]
    test_results = results[2]

    print(train_results[-1])
    print(validation_results[-1])
    print(test_results[-1])

if run_part8:

    word_indices = load("embeddings.npz")["word2ind"].flatten()[0]
    embeddings = load("embeddings.npz")["emb"]

    # Revert the word_indices list to a proper usable format
    word_index_dictionary = {}
    for i in range(len(word_indices)):
        word_index_dictionary[word_indices[i]] = i

    print("Done constructing the index dictionary")

    #find_similar(embeddings, word_index_dictionary, 'story', 10)
    #find_similar(embeddings, word_index_dictionary, 'good', 10)

    find_similar(embeddings, word_index_dictionary, 'problem', 10)
    find_similar(embeddings, word_index_dictionary, 'scene', 10)