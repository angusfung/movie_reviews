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
from logistic import *

# ======================================================================================================================
# ========================================= Running the code ===========================================================
# ======================================================================================================================


print_nb  = True # Prints top 100 coefficients obtained from NB algorithm
print_lr = True # Prints top 100 coefficients obtained from Logistic Regression algorithm


# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================

def print_nb_coefficients():
    '''
    Prints top 100 positive coefficients with words of Naive Bayes classifier built
    According to Piazza responses, taking only positives works
    '''
    neg_dict, pos_dict = generate_count_dict('training_set')
    m = 0.5
    k = 16.
    total_neg_sum = sum(neg_dict.values())
    for key in neg_dict.keys():
        neg_dict[key] = log((neg_dict[key]+m*k)/(total_neg_sum+k))

    total_pos_sum = sum(pos_dict.values())
    for key in pos_dict.keys():
        pos_dict[key] = log((pos_dict[key] + m * k) / (total_pos_sum + k))

    vocabulary = {}
    for key in neg_dict.keys():
        if key in pos_dict:
            vocabulary[key] = pos_dict[key] - neg_dict[key]
        else:
            vocabulary[key] = log(m*k/(total_pos_sum + k)) - neg_dict[key]
    for key in pos_dict.keys():
        if not key in neg_dict:
            vocabulary[key] = pos_dict[key] - log(m*k/(total_neg_sum + k))

    # Vocabulary now has difference of log(probabilities)
    top100 = sorted(vocabulary.items(), key=lambda x: x[1], reverse=True)[:100]
    for theta in top100:
        print(theta)
    return True

def print_lr_coefficients():
    '''
    Prints top 100 positive coefficients with words of Logistic Regression classifier built
    According to Piazza responses, taking only positives works
    '''
    neg_dict, pos_dict = generate_count_dict('training_set')
    num_words = get_numwords(neg_dict, pos_dict)[0]

    # Optimal parameters
    lam = 0.01
    total_iterations = 100

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
    for i in range(total_iterations):
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    coefficients = sess.run(W0)

    # Take difference of the weights
    coefficients = coefficients[:,0] - coefficients[:,1]

    num_words, combined_list = get_numwords(neg_dict, pos_dict)
    sorting_indices = np.argsort(coefficients)
    for i in range(1, 101):
        index = sorting_indices[-i]
        print(combined_list[index], coefficients[index])

    return True

# ======================================================================================================================
# ========================================== Definitions end ===========================================================
# ======================================================================================================================



if print_nb:
    print('Naive Bayes coefficients')
    print_nb_coefficients()

if print_lr:
    print('Logistic Regression coefficients')
    print_lr_coefficients()
