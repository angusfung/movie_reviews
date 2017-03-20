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
    Prints top 100 coefficients of Naive Bayes classifier built
    '''
    dicts = generate_count_dict('training_set')
    vocabulary = merge(dicts[0], dicts[1], lambda x,y: x+y)
    m = 0.5
    k = 16.
    total_sum = sum(vocabulary.values())
    for key in vocabulary.keys():
        vocabulary[key] = (vocabulary[key]+m*k)/(total_sum+k)

    top100 = sorted(vocabulary.values(), key=lambda x: x, reverse=True)[:100]
    for theta in top100:
        print(theta)
    return True

def print_lr_coefficients():
    '''
    Prints top 100 coefficients of Logistic Regression classifier built
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
    coefficients = np.reshape(coefficients,(coefficients.shape[0]*2))
    coefficients.sort()
    for i in range(1,101):
        print(coefficients[-i])
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

# Naive Bayes coefficients
# 0.00296314598703
# 0.00296314598703
# 0.00296314598703
# 0.00296130208785
# 0.00295945818866
# 0.00295761428948
# 0.0029557703903
# 0.00290598511236
# 0.00288385832216
# 0.00286726322952
# 0.00284882423769
# 0.0028469803385
# 0.00281379015321
# 0.00281194625403
# 0.00274556588344
# 0.00265337092429
# 0.00264415142838
# 0.00262755633573
# 0.00262386853736
# 0.00262386853736
# 0.00258330275534
# 0.00255564426759
# 0.00254642477168
# 0.00254089307413
# 0.00252061018312
# 0.00247082490518
# 0.00246713710681
# 0.00243025912315
# 0.00242288352642
# 0.00240813233296
# 0.00235097145828
# 0.00225693259995
# 0.00225324480159
# 0.0022514009024
# 0.00222005461629
# 0.00220161562446
# 0.00219977172528
# 0.00218502053182
# 0.00215736204407
# 0.00215551814489
# 0.00215367424571
# 0.00211864016123
# 0.0021002011694
# 0.00205594758901
# 0.00202828910127
# 0.00200616231107
# 0.00199141111761
# 0.00198772331924
# 0.00196559652904
# 0.00196375262986
# 0.00195084533558
# 0.0019379380413
# 0.00192871854538
# 0.00192871854538
# 0.00191949904947
# 0.00184389918297
# 0.00181624069522
# 0.00176645541728
# 0.00175907982055
# 0.00171667013934
# 0.00168163605487
# 0.00167241655895
# 0.00165766536549
# 0.00165213366794
# 0.00163922637366
# 0.00163922637366
# 0.00162078738183
# 0.00161894348265
# 0.00160972398673
# 0.00160419228918
# 0.00159866059163
# 0.00158575329735
# 0.00156178260797
# 0.00154703141451
# 0.0015378119186
# 0.00151937292677
# 0.00151384122922
# 0.00151199733003
# 0.00150646563249
# 0.00149724613657
# 0.00149540223739
# 0.00147696324556
# 0.00145299255618
# 0.00143639746353
# 0.00143270966517
# 0.00142349016925
# 0.0014179584717
# 0.00139398778232
# 0.00139029998396
# 0.00136079759703
# 0.0013534220003
# 0.00134604640357
# 0.00133498300847
# 0.00132576351255
# 0.00132391961337
# 0.00132023181501
# 0.00130548062154
# 0.00129626112563
# 0.00129441722644
# 0.00129257332726
# Logistic Regression coefficients
# 0.0765352
# 0.0738327
# 0.0717637
# 0.0708311
# 0.0700011
# 0.0694907
# 0.0692778
# 0.0689915
# 0.0688441
# 0.0682681
# 0.0679314
# 0.0664813
# 0.0664744
# 0.0658342
# 0.0658183
# 0.065587
# 0.065554
# 0.0654179
# 0.0652592
# 0.0649749
# 0.0648926
# 0.0647923
# 0.0647718
# 0.0646058
# 0.0642095
# 0.0641996
# 0.0638675
# 0.0637341
# 0.0637309
# 0.0637108
# 0.0636003
# 0.0634619
# 0.0632691
# 0.0632318
# 0.0631412
# 0.0631382
# 0.0631284
# 0.0629109
# 0.0628384
# 0.062678
# 0.0626355
# 0.0625265
# 0.0624754
# 0.0622977
# 0.0622803
# 0.0619633
# 0.0618165
# 0.0617895
# 0.061713
# 0.0616907
# 0.0615553
# 0.061436
# 0.0613275
# 0.0612571
# 0.0611944
# 0.061089
# 0.0610779
# 0.0610112
# 0.0609904
# 0.060868
# 0.0607908
# 0.0607561
# 0.0606542
# 0.0606137
# 0.0605032
# 0.0604989
# 0.0604659
# 0.0603349
# 0.0602425
# 0.0602032
# 0.0600332
# 0.0600249
# 0.0599061
# 0.0598568
# 0.0598315
# 0.0598103
# 0.0598095
# 0.0597996
# 0.0596995
# 0.0596169
# 0.0595299
# 0.0594574
# 0.059394
# 0.059376
# 0.0592836
# 0.0592715
# 0.0592546
# 0.0592366
# 0.059232
# 0.059228
# 0.0592156
# 0.0591965
# 0.0591364
# 0.0591048
# 0.0590974
# 0.0590885
# 0.0590684
# 0.0590564
# 0.0590228
# 0.0589663