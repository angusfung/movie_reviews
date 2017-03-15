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

#download the polarity dataset and seperate it into training, testing (200), validation (200)

#1000 positive and negative reviews in polarity dataset
download = True

def download_dataset(neg_index, pos_index, set_type,size): 
    '''param index: list of random, unique numbers
       param set_type: string that specifies training/test/validation set
       param size: the size of the set
    '''
    #969 in neg, 948 in pos
    
    filename = "\review_polarity\txt_sentoken"
    
    curr_directory = os.getcwd() #get current directory
    
    if not os.path.exists(os.getcwd()+"\\training_set"): #make the folders if they don't exist
        os.makedirs(os.getcwd()+"\\training_set\\neg")
        os.makedirs(os.getcwd()+"\\training_set\\pos")
    if not os.path.exists(os.getcwd()+"\\test_set"): 
        os.makedirs(os.getcwd()+"\\test_set\\neg")
        os.makedirs(os.getcwd()+"\\test_set\\pos")
    if not os.path.exists(os.getcwd()+"\\validation_set"): 
        os.makedirs(os.getcwd()+"\\validation_set\\neg")
        os.makedirs(os.getcwd()+"\\validation_set\\pos")
    
    #generate the training set, saving 200 images for testing set, 200 for training set
    for j in range(2):
        for i in range(int(size/2)):
            if j==0:  #if j==0, neg reviews; else, pos reviews
                rand_num = neg_index[i]
                #print(i, rand_num)
                file_source = os.getcwd()+"\\review_polarity\\txt_sentoken\\neg\\"
                image_name = os.listdir(file_source)[rand_num]
                file_source = file_source + image_name
                file_destination = os.getcwd() + set_type + "\\neg"
            else:
                rand_num = pos_index[i]
                #print(i, rand_num)
                file_source = os.getcwd()+"\\review_polarity\\txt_sentoken\\pos\\"
                image_name = os.listdir(file_source)[rand_num]
                file_source = file_source + image_name
                file_destination = os.getcwd() + set_type + "\\pos"
            shutil.copy2(file_source, file_destination)
            
            #process the text file
            f = open(file_destination + "\\"+ image_name, "r+")
            data = f.read()
            exclude = set(string.punctuation)
            file_text = ''.join(ch for ch in data if ch not in exclude).lower()
            f.seek(0)
            f.write(file_text)
            f.truncate()
            f.close()
                
        
    
    return
    
if download == True:
    #generate 2000 unique numbers
    random.seed(0) 
    neg_index = random.choice(range(1000), 1000, replace=False) #i.e 0-size
    random.seed(1) 
    pos_index = random.choice(range(1000), 1000, replace=False) #i.e 0-size
    
    #generate training set (size = 1600)
    download_dataset(neg_index, pos_index, "\\training_set",1600)
    print("Please wait while the training set is being processed...")
    #generate the testing set (size = 200)
    download_dataset(neg_index[800:],pos_index[800:], "\\test_set", 200)
    print("Please wait while the testing set is being processed...")
    #generate the validation set (size = 200)
    download_dataset(neg_index[900:], pos_index[900:], "\\validation_set", 200) 
    print("Please wait while the validation set set is being processed...") 