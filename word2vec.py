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
#import tensorflow as tf

embeddings = load("embeddings.npz")["emb"]

#generate ([context], target) pairs for each word 

def word2vec(path):
    '''
    Generates word occurence dictionaries for positive and negative reviews
    :param path: the location of the training set
    :return:
    '''
    word_context = {}
    # Go through the negative reviews
    for review in os.listdir(path + "/neg"):
        i = 0 
        with open(os.getcwd()+"/"+path+"/neg/"+review) as f:
            text = f.read().split()
            for word in text:
                if i == 0: #skip the first word of each review
                    i += 1
                    continue
                elif i == len(text) - 1: #skip the last word of each review
                    continue
                else: #obtain ([context], target)
                    word_context[(text[i-1], text[i+1])] = word
                i += 1
                
    # Go through the positive reviews
    for review in os.listdir(path + "/pos"):
        i = 0 
        with  open(os.getcwd()+"/"+path+"/pos/"+review)as f:
            text = f.read().split()
            for word in text:
                if i == 0:
                    i += 1
                    continue
                elif i == len(text) - 1:
                    continue
                else:
                    word_context[(text[i-1], text[i+1])] = word
                i += 1
    return word_context            
a = word2vec('training_set')