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

# ======================================================================================================================
# ========================================= Running the code ===========================================================
# ======================================================================================================================


prepare_data   = False #download dataset (just pull from github, don't need to run this)
run_part1  = False #prints each word and its frequency
run_part2  = False #prints naive bayes classification performance
tuning_mk  = False #tuning m and k for naive bayes
run_part3  = True #printing 10 most important words for negative and positive reviews


# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================


def split_dataset(neg_index, pos_index, set_type, size):
    '''
    Extracts a sample of size specified for the purpose of testing/validation or training
    :param neg_index: list of random numbers for positive reviews
    :param pos_index: list of random numbers for negative reviews
    :param set_type: extracting validation/test or training
    :param size: size of the set
    '''
    #969 in neg, 948 in pos
    
    filename = "/review_polarity/txt_sentoken"

    # Get current directory
    curr_directory = os.getcwd()

    # Make the folders if they don't exist
    if not os.path.exists(os.getcwd()+"/training_set"):
        os.makedirs(os.getcwd()+"/training_set/neg")
        os.makedirs(os.getcwd()+"/training_set/pos")
    if not os.path.exists(os.getcwd()+"/test_set"): 
        os.makedirs(os.getcwd()+"/test_set/neg")
        os.makedirs(os.getcwd()+"/test_set/pos")
    if not os.path.exists(os.getcwd()+"/validation_set"): 
        os.makedirs(os.getcwd()+"/validation_set/neg")
        os.makedirs(os.getcwd()+"/validation_set/pos")
    
    # Generate the training set, saving 200 images for testing set, 200 for training set
    for j in range(2):
        for i in range(int(size/2)):
            if j==0:  # j=0 => neg reviews; else, pos reviews
                rand_num = neg_index[i]
                file_source = os.getcwd()+"/review_polarity/txt_sentoken/neg/"
                review_filename = os.listdir(file_source)[rand_num]
                file_source = file_source + review_filename
                file_destination = os.getcwd() + set_type + "/neg"
            else:
                rand_num = pos_index[i]
                file_source = os.getcwd()+"/review_polarity/txt_sentoken/pos/"
                review_filename = os.listdir(file_source)[rand_num]
                file_source = file_source + review_filename
                file_destination = os.getcwd() + set_type + "/pos"
            shutil.copy2(file_source, file_destination)
            
            # Process the text file
            f = open(file_destination + "/"+ review_filename, "r+")
            data = f.read()
            exclude = set(string.punctuation)
            file_text = ''.join(ch for ch in data if ch not in exclude).lower()
            f.seek(0)
            f.write(file_text)
            f.truncate()
            f.close()
    return
    
def merge(d1, d2, merge_fn=lambda x,y:y):  
    '''
    Merges 2 dictionaries together accodring to merge_fn
    :param d1: first dictionary
    :param d2: second dictionary
    :param merge_fn: function according to which merge will be performed
    :return: merged dictionary
    '''

    result = dict(d1)
    for k,v in d2.items():
        if k in result:
            result[k] = merge_fn(result[k], v)
        else:
            result[k] = v
    return result
     
def generate_count_dict(path):
    '''
    Generates word occurence dictionaries for positive and negative reviews
    :param path: the location of the training set
    :return: 2 dictionaries for positive and negative reviews
    '''

    pos_dict = {} #format = (key, value) = (word, frequency)
    neg_dict = {}
    
    # Go through the negative reviews
    for review in os.listdir(path + "/neg"):
        with open(os.getcwd()+"/"+path+"/neg/"+review) as f:
            unique_words = set(f.read().split())
            for word in unique_words:
                if not neg_dict.get(word): #check if the word is in dictionary
                    neg_dict[word] = 1
                else: #the word is in dictionary
                    neg_dict[word] += 1
                    
    # Go through the positive reviews
    for review in os.listdir(path + "/pos"):
        with  open(os.getcwd()+"/"+path+"/pos/"+review)as f:
            unique_words = set(f.read().split())
            for word in unique_words:
                if not pos_dict.get(word): #check if the word is in dictionary
                    pos_dict[word] = 1
                else: #the word is in dictionary
                    pos_dict[word] += 1
    return neg_dict, pos_dict

def generate_frequency_dict(path): #same as above, but normalizing by length (Piazza)
    '''
    Generates word frequency dictionaries for positive and negative reviews
    (normalizing by length)
    :param path: the location of the training set
    :return: 2 dictionaries for positive and negative reviews
    '''
    pos_dict = {} #format = (key, value) = (word, frequency)
    neg_dict = {}

    # Go through the negative reviews
    for review in os.listdir(path + "/neg"):
        with open(os.getcwd()+"/"+path+"/neg/"+review) as f:
            unique_words = set(f.read().split())
            for word in unique_words:
                if not neg_dict.get(word): #check if the word is in dictionary
                    neg_dict[word] = 1. / len(unique_words)
                else: #the word is in dictionary
                    neg_dict[word] += 1. / len(unique_words)

    # Go through the positive reviews
    for review in os.listdir(path + "/pos"):
        with  open(os.getcwd()+"/"+path+"/pos/"+review)as f:
            unique_words = set(f.read().split())
            for word in unique_words:
                if not pos_dict.get(word): #check if the word is in dictionary
                    pos_dict[word] = 1. / len(unique_words)
                else: #the word is in dictionary
                    pos_dict[word] += 1. / len(unique_words)
    return neg_dict, pos_dict
    
def smoothen_probability(dict, m, k, size = 800):
    '''
    Applies delta smoothing to probabilities
    :param dict: word frequency dictionary
    :param m: parameter m required for delta smoothing
    :param k: parameter k required for delta smoothing
    :param size: size can be specified (default 800)
    :return: smoothened word frequency dictionary
    '''
    denominator_pos = k + sum(dict.values()) * len(dict.keys())
    for word in dict:
        dict[word] = (dict[word]* len(dict.keys()) + m*k) / denominator_pos
    return dict

def classify(path, m, k,  neg_dict, pos_dict, size):
    '''
    Performe the Naive Bayes classification of reviews specified in neg_dict/pos_dict
    :param path: location of the dataset
    :param m: parameter m required for delta smoothing
    :param k: parameter k required for delta smoothing
    :param neg_dict: dictionary of negative reviews
    :param pos_dict: dictionary of positive reviews
    :param size: size of the training set
    :return: scores obtained on negative and positive reviews
    '''
    neg_score = 0.
    pos_score = 0.


    # Go through the negative reviews
    for review in os.listdir(path + "/neg"):
        with open(os.getcwd()+"/"+path+"/neg/"+review) as f:
            unique_words = set(f.read().split())
            neg_sum = 0. #compute P(ai|negative)
            pos_sum = 0. #compute P(ai|positive) [then take the largest]
            denominator_neg = k + sum(neg_dict.values()) * len(unique_words)
            denominator_pos = k + sum(pos_dict.values()) * len(unique_words)

            for word in unique_words:
                if neg_dict.get(word): #if word is in dictionary
                    neg_sum += log((neg_dict[word] * len(unique_words) + m * k) / denominator_neg)
                else:
                    neg_sum += log((m * k) / denominator_neg)
                if pos_dict.get(word):
                    pos_sum += log((pos_dict[word] * len(unique_words) + m * k) / denominator_pos)
                else:
                    pos_sum += log((m * k) / denominator_pos)

            if neg_sum > pos_sum:
                # Guessed correcctly
                neg_score += 1


    # Go through the positive reviews
    for review in os.listdir(path + "/pos"):
        with open(os.getcwd()+"/"+path+"/pos/"+review) as f:
            unique_words = set(f.read().split())
            neg_sum = 0. #compute P(ai|negative)
            pos_sum = 0. #compute P(ai|positive) [then take the largest]
            denominator_neg = k + sum(neg_dict.values()) * len(unique_words)
            denominator_pos = k + sum(pos_dict.values()) * len(unique_words)

            for word in unique_words:
                if neg_dict.get(word):  # if word is in dictionary
                    neg_sum += log((neg_dict[word] * len(unique_words) + m * k) / denominator_neg)
                else:
                    neg_sum += log((m * k) / denominator_neg)
                if pos_dict.get(word):
                    pos_sum += log((pos_dict[word] * len(unique_words) + m * k) / denominator_pos)
                else:
                    pos_sum += log((m * k) / denominator_pos)

            if pos_sum > neg_sum:
                # Guessed correcctly
                pos_score += 1
    
    return neg_score / (size/2.), pos_score / (size/2.)

def max_validation():
    '''
    Performs classification on different values of m and k for finding the optimal parameters
    '''

    combined_score = 0.
    mk = 0

    for m in np.arange(0.1,0.9,0.1):
        for k in np.arange(5.,20.,1.):
            dicts = generate_frequency_dict('training_set')
            score = classify('validation_set', m, k, dicts[0], dicts[1], 200)
            if ((score[0] + score[1])/2.) > combined_score:
                combined_score = (score[0] + score[1])/2
                mk = (m,k) 
            print("validation set: " + "m = " + str(m) + " k = " + str(k), score)
            print("=====================================================")
    print (combined_score,mk)

def get_numwords(neg_dict, pos_dict):
    '''
    Combines dictionary keys into one list (vocabulary)
    :param neg_dict: dictionary of negative reviews
    :param pos_dict: dictionary of positive reviews
    :return: list containing all words
    '''
    
    combined_list = []
    
    # Convert the keys of the dicts into a list
    for word in neg_dict:
        combined_list.append(word)
    for word in pos_dict:
        combined_list.append(word)

    # Remove duplicates in the list
    combined_list = list(set(combined_list))
    num_words = len(combined_list) 
    
    return num_words, combined_list
        
def most_predictive(neg_dict, pos_dict):
    '''
    Obtains and prints the top 10 words that distinguish a review polarity
    :param neg_dict: dictionary of negative reviews
    :param pos_dict: dictionary of positive reviews
    '''
    neg_words = {}
    pos_words = {}
    
    #take the difference of the probabilities
    for word in neg_dict:
        if word not in pos_dict:
            neg_words[word] = neg_dict[word]
        else:
            neg_words[word] = neg_dict[word] - pos_dict[word]
    
    for word in pos_dict:
        if word not in neg_dict:
            pos_words[word] = pos_dict[word]
        else:
            pos_words[word] = pos_dict[word] - neg_dict[word]

    neg_words = sorted(neg_words.items(), key=lambda x: x[1], reverse=True)[:10]
    pos_words = sorted(pos_words.items(), key=lambda x: x[1], reverse=True)[:10]

    print("Top 10 negative words:")
    for i in neg_words:
        print(i[0])
    print("Top 10 positive words:")
    for i in pos_words:
        print(i[0])
    return neg_words, pos_words

# ======================================================================================================================
# ========================================== Definitions end ===========================================================
# ======================================================================================================================


if prepare_data:
    #generate 2000 unique numbers
    random.seed(0) 
    neg_index = random.choice(range(1000), 1000, replace=False) #i.e 0-size
    random.seed(1) 
    pos_index = random.choice(range(1000), 1000, replace=False) #i.e 0-size
    
    #generate training set (size = 1600)
    split_dataset(neg_index, pos_index, "/training_set", 1600)
    print("Please wait while the training set is being processed...")
    #generate the testing set (size = 200)
    split_dataset(neg_index[800:], pos_index[800:], "/test_set", 200)
    print("Please wait while the testing set is being processed...")
    #generate the validation set (size = 200)
    split_dataset(neg_index[900:], pos_index[900:], "/validation_set", 200)
    print("Please wait while the validation set set is being processed...")

if run_part1:
    
    #combine all the dictionaries together, computing word frequency
    dicts1 = generate_count_dict('training_set')
    dicts2 = generate_count_dict('test_set')
    dicts3 = generate_count_dict('validation_set')
    
    neg_dict = merge(dicts1[0], dicts2[0], lambda x,y: x+y)
    neg_dict = merge(neg_dict , dicts3[0], lambda x,y: x+y)
    
    pos_dict = merge(dicts1[1], dicts2[1], lambda x,y: x+y)
    pos_dict = merge(pos_dict , dicts3[1], lambda x,y: x+y)
    
    print("=======================================================")
    words = ['annoying', 'terrible', 'stupid', 'wasted']
    for word in words:
        print("Occurrence of " + word + " in the negative dataset is " + str(neg_dict[word]))
    for word in words:
        print("Occurrence of " + word + " in the positive dataset is " + str(pos_dict[word]))
    print("=======================================================")
    words = ['like', 'liked', 'enjoy', 'enjoyed', 'love', 'loved', 'good']
    for word in words:
        print("Occurrence of " + word + " in the negative dataset is " + str(neg_dict[word]))
    for word in words:
        print("Occurrence of " + word + " in the positive dataset is " + str(pos_dict[word]))
    print("=======================================================")
    words = ['hilarious', 'terrific']
    for word in words:
        print("Occurrence of " + word + " in the negative dataset is " + str(neg_dict[word]))
    for word in words:
        print("Occurrence of " + word + " in the positive dataset is " + str(pos_dict[word]))

if run_part2:
    dicts = generate_frequency_dict('training_set')
    m = 0.5
    k = 16.

    neg_dict = dicts[0]
    pos_dict = dicts[1]
    score1 = classify('validation_set', m, k, neg_dict, pos_dict, 200)
    score2 = classify('test_set', m, k, neg_dict, pos_dict, 200)
    score3 = classify('training_set', m, k, neg_dict, pos_dict, 1600)
    print("Performances (negative, positive)")
    print("Training performance: " + str(score3))
    print("Validation performance: " + str(score1))
    print("Test performance: "+ str(score2))
    print("Overall:")
    print("Training performance: " + str((score3[0]+score3[1])/2.))
    print("Validation performance: " + str((score1[0]+score1[1])/2.))
    print("Test performance: " + str((score2[0]+score2[1])/2.))

if tuning_mk:
    max_validation()
    
if run_part3:
    dicts = generate_frequency_dict('training_set')
    m = 0.5
    k = 16.
    neg_dict = smoothen_probability(dicts[0], m, k)
    pos_dict = smoothen_probability(dicts[1], m, k)
    most_predictive(neg_dict, pos_dict)
