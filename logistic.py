from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import time
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
from scipy.ndimage import filters
import urllib
from numpy import random
import os
import shutil
import string
import operator
import tensorflow as tf
import naivebayes.py

# ======================================================================================================================
# ========================================= Running the code ===========================================================
# ======================================================================================================================


run_part4  = False  #logistic regression via. tensor flow
tuning_lam = False


# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================


def get_train(neg_dict, pos_dict):
    '''param dict {pos, neg} which is a dictionary containing (word, probability)
       returns:
            X is a MxN vector, where M is the number of reviews
                                     N is the number of words
            Y is a Mx1 vector, where M is the number of reviews
    '''
    combined_list = []

    num_words, combined_list = get_numwords(neg_dict, pos_dict)

    x = zeros((1600, len(combined_list)))
    y = zeros((1600, 2))  # one hot encoding
    cur_row = 0  # keep track of which column we're on

    path = 'training_set'

    for review in os.listdir(path + "/neg"):  # [1,0] corresponds to a negative review
        with open(os.getcwd() + "/" + path + "/neg/" + review) as f:
            unique_words = set(f.read().split())
            i = 0
            for word in combined_list:
                if word in unique_words:
                    x[cur_row, i] = 1
                    y[cur_row] = [1, 0]
                i += 1
        cur_row += 1
        # print(cur_row)

    for review in os.listdir(path + "/pos"):  # [0,1] corresponds to a positive review
        with open(os.getcwd() + "/" + path + "/pos/" + review) as f:
            unique_words = set(f.read().split())
            i = 0
            for word in combined_list:
                if word in unique_words:
                    x[cur_row, i] = 1
                    y[cur_row] = [0, 1]
                i += 1
        cur_row += 1
        # print(cur_row)
    return x, y


def get_test(neg_dict, pos_dict):
    combined_list = []

    num_words, combined_list = get_numwords(neg_dict, pos_dict)

    x = zeros((200, len(combined_list)))
    y = zeros((200, 2))
    cur_row = 0  # keep track of which column we're on

    path = 'test_set'

    for review in os.listdir(path + "/neg"):
        with open(os.getcwd() + "/" + path + "/neg/" + review) as f:
            unique_words = set(f.read().split())
            i = 0
            for word in combined_list:
                if word in unique_words:
                    x[cur_row, i] = 1
                    y[cur_row] = [1, 0]
                i += 1
        cur_row += 1
        # print(cur_row)

    for review in os.listdir(path + "/pos"):
        with open(os.getcwd() + "/" + path + "/pos/" + review) as f:
            unique_words = set(f.read().split())
            i = 0
            for word in combined_list:
                if word in unique_words:
                    x[cur_row, i] = 1
                    y[cur_row] = [0, 1]
                i += 1
        cur_row += 1
        # print(cur_row)
    return x, y


def get_validation(neg_dict, pos_dict):
    combined_list = []

    num_words, combined_list = get_numwords(neg_dict, pos_dict)

    x = zeros((200, len(combined_list)))
    y = zeros((200, 2))
    cur_row = 0  # keep track of which column we're on

    path = 'validation_set'

    for review in os.listdir(path + "/neg"):
        with open(os.getcwd() + "/" + path + "/neg/" + review) as f:
            unique_words = set(f.read().split())
            i = 0
            for word in combined_list:
                if word in unique_words:
                    x[cur_row, i] = 1
                    y[cur_row] = [1, 0]
                i += 1
        cur_row += 1
        # print(cur_row)

    for review in os.listdir(path + "/pos"):
        with open(os.getcwd() + "/" + path + "/pos/" + review) as f:
            unique_words = set(f.read().split())
            i = 0
            for word in combined_list:
                if word in unique_words:
                    x[cur_row, i] = 1
                    y[cur_row] = [0, 1]
                i += 1
        cur_row += 1
        # print(cur_row)
    return x, y


def logistic_regression(neg_dict, pos_dict, num_words, lam):
    # Initialize Tensor Flow variables
    x = tf.placeholder(tf.float32, [None, num_words])
    W0 = tf.Variable(tf.random_normal([num_words, 2], stddev=0.01))
    b0 = tf.Variable(tf.random_normal([2], stddev=0.01))

    y = tf.nn.softmax(tf.matmul(x, W0) + b0)
    y_ = tf.placeholder(tf.float32, [None, 2])

    # lam = 0.01
    decay_penalty = lam * tf.reduce_sum(tf.square(W0))
    reg_NLL = -tf.reduce_sum(y_ * tf.log(y)) + decay_penalty

    train_step = tf.train.AdamOptimizer(0.0005).minimize(reg_NLL)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    test_x, test_y = get_test(neg_dict, pos_dict)
    val_x, val_y = get_validation(neg_dict, pos_dict)
    batch_xs, batch_ys = get_train(neg_dict, pos_dict)

    # Run the TF, collect accuracies in a separate array
    test_results = []
    validation_results = []
    for i in range(500):
        # print i
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

        if i % 5 == 0:
            test_accuracy = sess.run(accuracy, feed_dict={x: test_x, y_: test_y})
            validation_accuracy = sess.run(accuracy, feed_dict={x: val_x, y_: val_y})
            train_accuracy = sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys})

            print("i=", i)
            print("Test:", test_accuracy)
            print("Validation:", validation_accuracy)
            print("Train:", train_accuracy)
            print("Penalty:", sess.run(decay_penalty))
            print(sess.run(reg_NLL, feed_dict={x: batch_xs, y_: batch_ys}))
            test_results.append(test_accuracy)
            validation_results.append(validation_accuracy)
    return max(test_results), max(validation_results)

# ======================================================================================================================
# ========================================== Definitions end ===========================================================
# ======================================================================================================================


if run_part4 == True:
    neg_dict, pos_dict = generate_dict('training_set')
    num_words = get_numwords(neg_dict, pos_dict)[0]

    lam = 1
    logistic_regression(neg_dict, pos_dict, num_words, lam)

if tuning_lam:

    neg_dict, pos_dict = generate_dict('training_set')
    num_words = get_numwords(neg_dict, pos_dict)[0]

    # store the highest test and validation accuracy
    highest_test = []
    highest_validation = []

    for lam in [0, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1]:
        results = logistic_regression(neg_dict, pos_dict, num_words, lam)
        print(results)
        highest_test.append(results[0])
        highest_validation.append(results[1])

    print(highest_test)
    print(highest_validation)

    '''Output: [0.85500002, 0.83999997, 0.85500002, 0.85000002, 0.86500001, 0.86000001, 0.85000002, 0.86000001, 0.86500001, 0.875]
               [0.83999997, 0.83999997, 0.82499999, 0.83999997, 0.81999999, 0.81, 0.85000002, 0.82499999, 0.83499998, 0.85000002]'''