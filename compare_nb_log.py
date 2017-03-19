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


download   = False #download dataset (just pull from github, don't need to run this)
run_part1  = False #prints each word and its frequency
run_part2  = False #prints naive bayes classification performance
tuning_mk  = False #tuning m and k for naive bayes
run_part3  = False #printing 10 most important words for negative and positive reviews
run_part4  = False  #logistic regression via. tensor flow
tuning_lam = False


# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================




# ======================================================================================================================
# ========================================== Definitions end ===========================================================
# ======================================================================================================================
