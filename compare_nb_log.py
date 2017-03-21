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

# Naive Bayes coefficients
# ('outstanding', 1.5072543968951333)
# ('satisfying', 1.3588343917768615)
# ('era', 1.1954100751642738)
# ('wonderfully', 1.1481133604612079)
# ('allows', 1.1356908404626509)
# ('perfectly', 1.1227036449358385)
# ('superb', 1.094868845942397)
# ('breathtaking', 1.0843975460751007)
# ('offbeat', 1.0631201476278154)
# ('delightful', 1.0303303248048241)
# ('finest', 1.0303303248048241)
# ('magnificent', 1.0303303248048241)
# ('german', 1.0043548384015644)
# ('anger', 1.0021594478381282)
# ('terrific', 1.0009164395985302)
# ('memorable', 0.9964287731291428)
# ('beautifully', 0.95622235265110156)
# ('religion', 0.93502014500050024)
# ('everyday', 0.91254728914844208)
# ('marvelous', 0.90111859332481714)
# ('ordered', 0.89679893218030138)
# ('lovingly', 0.89679893218030138)
# ('astounding', 0.89679893218030138)
# ('fantastic', 0.88722948116415168)
# ('flaws', 0.88437641218174434)
# ('tony', 0.88269432599875941)
# ('refreshing', 0.87617964497756518)
# ('excellent', 0.86884285491371216)
# ('courage', 0.86602727351354858)
# ('nevertheless', 0.85970480777405989)
# ('captures', 0.84800876801087099)
# ('debate', 0.84800876801087099)
# ('melancholy', 0.84800876801086922)
# ('seamless', 0.84800876801086922)
# ('fascinating', 0.84800876801086922)
# ('annual', 0.84800876801086922)
# ('friendship', 0.83617431036386591)
# ('portrays', 0.83250458147490569)
# ('topnotch', 0.82965962934267345)
# ('hearts', 0.82553591215881106)
# ('subtle', 0.82269096002657882)
# ('fits', 0.82269096002657882)
# ('portrayed', 0.8190212311376186)
# ('politics', 0.81902123113761682)
# ('detailed', 0.81902123113761682)
# ('perfect', 0.81000764983225615)
# ('dazzling', 0.80718677349061529)
# ('spielbergs', 0.80718677349061529)
# ('depiction', 0.80718677349061529)
# ('anna', 0.80718677349061529)
# ('ordinary', 0.80718677349061352)
# ('slip', 0.79671547362331907)
# ('gattaca', 0.79671547362331907)
# ('ideals', 0.79671547362331907)
# ('coens', 0.79671547362331907)
# ('portrayal', 0.79191930135982602)
# ('maintains', 0.79191930135982602)
# ('feelgood', 0.79191930135982602)
# ('spoil', 0.79191930135982602)
# ('pays', 0.77901589652391934)
# ('upper', 0.77901589652391934)
# ('themes', 0.77901589652391934)
# ('commanding', 0.77901589652391934)
# ('effortlessly', 0.77901589652391756)
# ('tucker', 0.77901589652391756)
# ('innocence', 0.77901589652391756)
# ('manipulation', 0.77901589652391756)
# ('elliot', 0.77901589652391756)
# ('faced', 0.77901589652391756)
# ('thematic', 0.77901589652391756)
# ('avoids', 0.77901589652391756)
# ('addresses', 0.77901589652391756)
# ('questioning', 0.77901589652391756)
# ('natural', 0.7732852218149322)
# ('hilarious', 0.7700472265411582)
# ('colors', 0.77004722654115731)
# ('poignant', 0.76796606033733461)
# ('animators', 0.76462715907181966)
# ('performed', 0.76462715907181966)
# ('warns', 0.76462715907181966)
# ('period', 0.75839660932118313)
# ('destined', 0.75839660932118313)
# ('exceptional', 0.75839660932118313)
# ('masterpiece', 0.75589347910306337)
# ('mesmerizing', 0.75269858820654534)
# ('engaged', 0.75002835965066517)
# ('joy', 0.74847917266383668)
# ('contrast', 0.74847917266383668)
# ('damon', 0.74264825235304421)
# ('weakest', 0.74264825235304244)
# ('oscar', 0.73734320012334997)
# ('decades', 0.73609085180688361)
# ('brilliant', 0.73486611191098916)
# ('flawless', 0.73406450866165152)
# ('assigned', 0.73406450866165152)
# ('obiwan', 0.7302257323544854)
# ('conveys', 0.7302257323544854)
# ('tool', 0.7302257323544854)
# ('embodies', 0.7302257323544854)
# ('palpable', 0.7302257323544854)
# Logistic Regression coefficients
# ('fled', 0.1248212)
# ('priced', 0.12417657)
# ('craving', 0.12296744)
# ('bested', 0.12208948)
# ('plump', 0.11905482)
# ('31st', 0.11650236)
# ('wartorn', 0.11641815)
# ('dierdre', 0.11551502)
# ('wrapping', 0.11469546)
# ('obtuse', 0.11460398)
# ('excite', 0.11301324)
# ('spying', 0.11275588)
# ('outperform', 0.11237419)
# ('gutbusting', 0.11119756)
# ('wring', 0.11107261)
# ('concede', 0.11079349)
# ('superbowl', 0.11046429)
# ('nothingness', 0.11019677)
# ('contaminated', 0.10996445)
# ('burke', 0.10965322)
# ('unkempt', 0.10936677)
# ('mens', 0.10901233)
# ('outshining', 0.10889457)
# ('jerky', 0.1088909)
# ('cheng', 0.10866033)
# ('dragged', 0.10843267)
# ('dinos', 0.10842248)
# ('intermittently', 0.10768691)
# ('toaster', 0.10762677)
# ('sprightly', 0.10746579)
# ('mull', 0.1074148)
# ('spare', 0.10740757)
# ('stamps', 0.10735935)
# ('teresas', 0.10727585)
# ('satisfactorily', 0.10694055)
# ('gypsy', 0.10677495)
# ('contortions', 0.1067455)
# ('stu', 0.10669623)
# ('thereafter', 0.10605255)
# ('alluded', 0.10589411)
# ('kersey', 0.10562639)
# ('barter', 0.10555466)
# ('policeman', 0.10541075)
# ('gnosis', 0.10521144)
# ('realists', 0.10491891)
# ('formidable', 0.1048215)
# ('humbled', 0.10471277)
# ('austins', 0.10464242)
# ('buffet', 0.1043836)
# ('simpleton', 0.10355058)
# ('nibble', 0.10325704)
# ('bombastically', 0.10310829)
# ('tediouslyand', 0.10278364)
# ('moviesa', 0.10277981)
# ('wager', 0.10276391)
# ('property', 0.10272495)
# ('corresponded', 0.1026255)
# ('curly', 0.10245524)
# ('siouxsie', 0.10240315)
# ('rickys', 0.10234327)
# ('bostonians', 0.10220879)
# ('personification', 0.10220585)
# ('spader', 0.10220437)
# ('coaster', 0.10204513)
# ('eighteen', 0.10202323)
# ('greedier', 0.10183892)
# ('avarice', 0.10181448)
# ('fellas', 0.10178645)
# ('matted', 0.10169203)
# ('pledges', 0.10166472)
# ('actionmovie', 0.10166286)
# ('outgrown', 0.10165636)
# ('interpretations', 0.10164339)
# ('greenblatt', 0.10163821)
# ('jokesotherwise', 0.10142158)
# ('exhausted', 0.10134804)
# ('sandler', 0.10124214)
# ('reworks', 0.10122174)
# ('malloy', 0.10118885)
# ('marry', 0.10112122)
# ('loretta', 0.10112093)
# ('lunkheads', 0.10109934)
# ('afloat', 0.10092185)
# ('bayou', 0.10090986)
# ('beginners', 0.10086909)
# ('mishandled', 0.10084329)
# ('hotandbothered', 0.10083415)
# ('ninetyfive', 0.10073271)
# ('disenfranchised', 0.10066444)
# ('hamburg', 0.1006294)
# ('historians', 0.10046709)
# ('operatives', 0.10040833)
# ('composing', 0.10037219)
# ('lemmons', 0.10028771)
# ('57', 0.10025149)
# ('deceiver', 0.099880472)
# ('shenanigans', 0.099584609)
# ('forming', 0.09956786)
# ('distilled', 0.099551305)
# ('switzerland', 0.099493146)