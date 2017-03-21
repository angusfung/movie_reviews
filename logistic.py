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
from naivebayes import *

# ======================================================================================================================
# ========================================= Running the code ===========================================================
# ======================================================================================================================


run_part4  = False  # Logistic regression via. Tensor Flow
tuning_lam = False # Finding the optimal regularization parameter - Takes time!


# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================


def get_train(neg_dict, pos_dict):
    '''
    Generates matricies for the training set according to dictionaries specified
    :param neg_dict: dictionary of negative reviews
    :param pos_dict: dictionary of positive reviews
    :return: x and y matricies prepared for classificaiton
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
    '''
    Generates matricies for the test set according to dictionaries specified
    :param neg_dict: dictionary of negative reviews
    :param pos_dict: dictionary of positive reviews
    :return: x and y matricies prepared for classificaiton
            X is a MxN vector, where M is the number of reviews
                                     N is the number of words
            Y is a Mx1 vector, where M is the number of reviews
    '''
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
    '''
    Generates matricies for the validation set according to dictionaries specified
    :param neg_dict: dictionary of negative reviews
    :param pos_dict: dictionary of positive reviews
    :return: x and y matricies prepared for classificaiton
            X is a MxN vector, where M is the number of reviews
                                     N is the number of words
            Y is a Mx1 vector, where M is the number of reviews
    '''
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


def logistic_regression(neg_dict, pos_dict, num_words, lam, total_iterations=500, print_iterations = True):
    '''
    Performs logistic regression with Tensor Flow
    :param neg_dict: dictionary of negative reviews
    :param pos_dict: dictionary of positive reviews
    :param num_words: total number of words
    :param lam: regularizing parameter lambda
    :param total_iterations: total number of training iterations
    :param print_iterations: boolean variable indicating if printing results while training is required
    :return: performance
    '''
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

# ======================================================================================================================
# ========================================== Definitions end ===========================================================
# ======================================================================================================================


if run_part4:
    neg_dict, pos_dict = generate_count_dict('training_set')
    num_words = get_numwords(neg_dict, pos_dict)[0]

    lam = 0.01
    tot_iters= 100

    results = logistic_regression(neg_dict, pos_dict, num_words, lam, tot_iters, True)
    train_results = results[0]
    validation_results = results[1]
    test_results = results[2]

    x_axis = linspace(0, tot_iters, len(train_results))
    plt_training = plt.plot(x_axis, train_results, label='Training')
    plt_test = plt.plot(x_axis, test_results, label='Test')
    plt_validation = plt.plot(x_axis, validation_results, label='Validation')
    plt.ylim([0.2, 1.05])
    plt.xlabel('# of Iterations')
    plt.ylabel('Performance')
    plt.title('Performance vs. # of iterations')
    plt.legend(["Training", "Test", "Validation"], loc=7)
    plt.savefig("logistic_regression_curve.png")

if tuning_lam:
    neg_dict, pos_dict = generate_count_dict('training_set')
    num_words = get_numwords(neg_dict, pos_dict)[0]
    tot_iters = 100

    # Store the highest (optimal) accuracies
    highest_train = 0
    highest_validation = 0
    highest_test = 0
    optimal_lambda = 0

    for lam in [0, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1]:
        results = logistic_regression(neg_dict, pos_dict, num_words, lam, tot_iters, False)
        train_results = results[0]
        validation_results = results[1]
        test_results = results[2]

        validation = validation_results[-1]
        print("======\nLambda: " + str(lam))
        print("Validation: " + str(validation))

        if validation > highest_validation:
            highest_train = train_results[-1]
            highest_validation = validation
            highest_test = test_results[-1]
            optimal_lambda = lam

    print("Final train: " + str(highest_train))
    print("Final validation: " + str(highest_validation))
    print("Final test: " + str(highest_test))
    print("Optimal lambda: " + str(optimal_lambda))
