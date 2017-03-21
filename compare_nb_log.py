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

    top100 = sorted(vocabulary.items(), key=lambda x: x[1], reverse=True)[:100]
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

    num_words, combined_list = get_numwords(neg_dict, pos_dict)

    sorting_indices = np.argsort(coefficients)
    for i in range(1, 101):
        index = sorting_indices[-i]
        print(combined_list[index % num_words], coefficients[index])

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
# ('the', 0.0029631459870300132)
# ('of', 0.0029631459870300132)
# ('and', 0.0029631459870300132)
# ('to', 0.002961302087847045)
# ('a', 0.0029594581886640764)
# ('is', 0.0029576142894811083)
# ('in', 0.0029557703902981401)
# ('that', 0.0029059851123579969)
# ('with', 0.0028838583221623775)
# ('it', 0.0028672632295156631)
# ('for', 0.0028488242376859801)
# ('as', 0.0028469803385030119)
# ('but', 0.0028137901532095831)
# ('this', 0.0028119462540266149)
# ('on', 0.0027455658834397568)
# ('are', 0.0026533709242913435)
# ('an', 0.0026441514283765022)
# ('by', 0.0026275563357297878)
# ('be', 0.002623868537363851)
# ('his', 0.002623868537363851)
# ('one', 0.0025833027553385491)
# ('who', 0.0025556442675940252)
# ('at', 0.0025464247716791835)
# ('film', 0.002540893074130279)
# ('from', 0.0025206101831176278)
# ('its', 0.0024708249051774846)
# ('not', 0.0024671371068115478)
# ('have', 0.0024302591231521826)
# ('he', 0.0024228835264203095)
# ('has', 0.0024081323329565633)
# ('all', 0.0023509714582845469)
# ('movie', 0.002256932599953165)
# ('i', 0.0022532448015872286)
# ('out', 0.00225140090240426)
# ('was', 0.0022200546162937994)
# ('so', 0.0022016156244641168)
# ('more', 0.0021997717252811486)
# ('like', 0.0021850205318174024)
# ('they', 0.0021573620440728781)
# ('when', 0.0021555181448899099)
# ('about', 0.0021536742457069417)
# ('you', 0.0021186401612305447)
# ('up', 0.0021002011694008617)
# ('or', 0.0020559475890096234)
# ('some', 0.0020282891012650991)
# ('if', 0.0020061623110694798)
# ('which', 0.001991411117605734)
# ('what', 0.0019877233192397972)
# ('into', 0.0019655965290441778)
# ('just', 0.0019637526298612097)
# ('their', 0.0019508453355804318)
# ('even', 0.001937938041299654)
# ('only', 0.0019287185453848125)
# ('than', 0.0019287185453848125)
# ('there', 0.0019194990494699712)
# ('time', 0.001843899182968272)
# ('can', 0.0018162406952237479)
# ('no', 0.0017664554172836047)
# ('most', 0.0017590798205517316)
# ('good', 0.0017166701393434613)
# ('him', 0.001681636054867064)
# ('much', 0.0016724165589522228)
# ('her', 0.0016576653654884765)
# ('would', 0.0016521336679395718)
# ('been', 0.0016392263736587939)
# ('other', 0.0016392263736587939)
# ('get', 0.0016207873818291111)
# ('also', 0.0016189434826461427)
# ('after', 0.0016097239867313015)
# ('will', 0.0016041922891823967)
# ('do', 0.0015986605916334918)
# ('story', 0.0015857532973527139)
# ('them', 0.0015617826079741264)
# ('films', 0.0015470314145103802)
# ('two', 0.0015378119185955389)
# ('first', 0.0015193729267658561)
# ('make', 0.0015138412292169513)
# ('we', 0.0015119973300339832)
# ('way', 0.0015064656324850782)
# ('character', 0.0014972461365702369)
# ('characters', 0.0014954022373872685)
# ('well', 0.001476963245557586)
# ('very', 0.0014529925561789984)
# ('see', 0.0014363974635322838)
# ('any', 0.0014327096651663475)
# ('while', 0.001423490169251506)
# ('does', 0.0014179584717026012)
# ('because', 0.0013939877823240137)
# ('too', 0.0013902999839580771)
# ('where', 0.0013607975970305847)
# ('little', 0.0013534220002987118)
# ('had', 0.0013460464035668386)
# ('how', 0.001334983008469029)
# ('off', 0.0013257635125541877)
# ('she', 0.0013239196133712193)
# ('plot', 0.0013202318150052827)
# ('over', 0.0013054806215415367)
# ('could', 0.0012962611256266952)
# ('then', 0.001294417226443727)
# ('really', 0.0012925733272607586)
# Logistic Regression coefficients
# ('yglesias', 0.073168658)
# ('repetitions', 0.072413407)
# ('trueman', 0.070467986)
# ('spake', 0.069864698)
# ('chao', 0.069448344)
# ('currently', 0.068477906)
# ('ruber', 0.068297476)
# ('scott', 0.06734249)
# ('chummingup', 0.067067392)
# ('restauranteur', 0.066924736)
# ('header', 0.066338941)
# ('swinging', 0.066182174)
# ('sentinels', 0.065907203)
# ('transparent', 0.065444306)
# ('refuses', 0.065425225)
# ('emphasis', 0.065421931)
# ('pitiful', 0.064866327)
# ('yellowish', 0.064687736)
# ('lloyd', 0.064619899)
# ('stylistically', 0.064347073)
# ('feel', 0.064313345)
# ('roaches', 0.064275943)
# ('yodas', 0.064154081)
# ('designed', 0.064147726)
# ('indepth', 0.063962147)
# ('considers', 0.063948676)
# ('julias', 0.063908011)
# ('delete', 0.063781314)
# ('contained', 0.063720003)
# ('burrito', 0.063519895)
# ('inflation', 0.063073911)
# ('reversals', 0.063066356)
# ('assuring', 0.062919408)
# ('loyality', 0.062895611)
# ('toswallow', 0.06287922)
# ('recap', 0.062774405)
# ('infant', 0.062733494)
# ('melee', 0.062683977)
# ('mindcharlie', 0.062619247)
# ('ulee', 0.062614553)
# ('arming', 0.062465582)
# ('pseudoepic', 0.062314399)
# ('jumanji', 0.062177036)
# ('mena', 0.062045816)
# ('sec', 0.061926343)
# ('subsequent', 0.061891031)
# ('investigating', 0.061847921)
# ('senator', 0.061817877)
# ('smolder', 0.061594538)
# ('villagers', 0.061581247)
# ('informationsomething', 0.061552167)
# ('morphed', 0.061505664)
# ('subordination', 0.061502427)
# ('professionalleon', 0.061369006)
# ('edie', 0.061365418)
# ('beek', 0.061357059)
# ('verbose', 0.061282575)
# ('travellers', 0.061228082)
# ('climb', 0.061162092)
# ('goldie', 0.061139662)
# ('convert', 0.061112732)
# ('crosscountry', 0.061012734)
# ('mart', 0.060817488)
# ('occasions', 0.060801797)
# ('postcannibal', 0.060663909)
# ('clockwatchers', 0.060598381)
# ('that', 0.060590714)
# ('fruition', 0.060547307)
# ('santostefano', 0.060465876)
# ('strident', 0.060459312)
# ('gassner', 0.060459312)
# ('quotation', 0.060411099)
# ('harsher', 0.060341623)
# ('incohesive', 0.060237996)
# ('infestation', 0.060231242)
# ('weaknesses', 0.060214013)
# ('fearful', 0.060170993)
# ('subpar', 0.060092177)
# ('spadowskis', 0.060077965)
# ('night', 0.060073424)
# ('kio', 0.060033757)
# ('tendencies', 0.059976161)
# ('reexamine', 0.059943315)
# ('decisively', 0.059851155)
# ('jersey', 0.059811413)
# ('divesting', 0.059782039)
# ('selfdestructs', 0.059746247)
# ('reconciliation', 0.059737939)
# ('ziembicki', 0.059710994)
# ('blame', 0.059678063)
# ('makeup', 0.059629817)
# ('greatness', 0.059566919)
# ('must', 0.059523866)
# ('onepiece', 0.05951095)
# ('would', 0.059457909)
# ('tornatore', 0.059449837)
# ('foresee', 0.059404302)
# ('squeeze', 0.059303995)
# ('keenan', 0.059256054)
# ('similar', 0.059251834)