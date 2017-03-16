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


download   = False
run_part2  = False
run_tuning = False #I've ran this before & saved the results in a text file called mkscore.
run_part3  = False
run_part4  = True

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
    
def generate_dict(path):
    '''param path is the location of the training set
            e.g 'training_set'
       return two dicts, with each word occurance and its frequency
    '''
    pos_dict = {} #format = (key, value) = (word, frequency)
    neg_dict = {}
    
    #go through the negative review first
    for review in os.listdir(path + "\\neg"):
        with open(os.getcwd()+"\\"+path+"\\neg\\"+review) as f:
            unique_words = set(f.read().split())
            for word in unique_words:
                if not neg_dict.get(word): #check if the word is in dictionary
                    neg_dict[word] = 1
                else: #the word is in dictionary
                    neg_dict[word] += 1
                    
    #go through the positive reviews
    for review in os.listdir(path + "\\pos"):
        with  open(os.getcwd()+"\\"+path+"\\pos\\"+review)as f:
            unique_words = set(f.read().split())
            for word in unique_words:
                if not pos_dict.get(word): #check if the word is in dictionary
                    pos_dict[word] = 1
                else: #the word is in dictionary
                    pos_dict[word] += 1
    return neg_dict, pos_dict
    
def calculate_prob(dict, m, k):
    
    size = 800
    # m = 15
    # k = 30
    
    for word in dict:
        dict[word] = (dict[word] + m*k) / (800 + k)
    
    return dict

def classify(path, neg_dict, pos_dict):
    '''param path is the location of the training set
            e.g 'testing set'
       param dict {neg, pos} have key = word, value = probability
    '''
    prior = log(0.5)
    neg_score = 0
    pos_score = 0

    #go through the negative review first
    for review in os.listdir(path + "\\neg"):
        with open(os.getcwd()+"\\"+path+"\\neg\\"+review) as f:
            unique_words = set(f.read().split())
            neg_sum = 0 #compute P(ai|negative)
            pos_sum = 0 #compute P(ai|positive) [then take the largest]
            neg_prob = 0
            pos_prob = 0
            
            for word in unique_words:
                if neg_dict.get(word): #if word is in dictionary
                    neg_sum += log(neg_dict[word])
                if pos_dict.get(word):
                    pos_sum += log(pos_dict[word])
            neg_prob = exp(neg_sum + prior)
            pos_prob = exp(pos_sum + prior)
            if neg_prob > pos_prob: 
                neg_score += 1 #since we know the review must be negative
                
    #go through the positive review 
    for review in os.listdir(path + "\\pos"):
        with open(os.getcwd()+"\\"+path+"\\pos\\"+review) as f:
            unique_words = set(f.read().split())
            neg_sum = 0 #compute P(ai|negative)
            pos_sum = 0 #compute P(ai|positive) [then take the largest]
            neg_prob = 0
            pos_prob = 0
            
            for word in unique_words:
                if neg_dict.get(word): #if word is in dictionary
                    neg_sum += log(neg_dict[word])
                if pos_dict.get(word):
                    pos_sum += log(pos_dict[word])
            neg_prob = exp(neg_sum + prior)
            pos_prob = exp(pos_sum + prior)
            if pos_prob > neg_prob:
                pos_score += 1 #since we know the review must be positive
    
    return neg_score / 100, pos_score / 100

def max_validation():
    '''loops through all possible values of m, k and prints out the scores
    '''
    dicts = generate_dict('training_set')
    for m in range(25, 50, 2):
        for k in range(25, 50,2):
            neg_dict = calculate_prob(dicts[0],m,k)
            pos_dict = calculate_prob(dicts[1],m,k)
            score = classify('validation_set', neg_dict, pos_dict)
            score1 = classify('test_set', neg_dict, pos_dict)
            print("validation set: " + "m = " + str(m) + " k = " + str(k), score, "test set: ", "m = " + str(m) + " k = " + str(k), score1)

def most_predictive(neg_dict, pos_dict):
    
    neg_words = dict(sorted(neg_dict.items(), key=operator.itemgetter(1), reverse=True)[:10])
    pos_words = dict(sorted(pos_dict.items(), key=operator.itemgetter(1), reverse=True)[:10])
    print(neg_words)
    print(pos_words)
    return neg_words, pos_words
            
def get_train(neg_dict, pos_dict):
    global num_words
    
    combined_list = []
    
    #convert the keys of the dicts into a list
    for word in neg_dict:
        combined_list.append(word)
    for word in pos_dict:
        combined_list.append(word)
    #remove duplicates in the list
    combined_list = list(set(combined_list))
    num_words = len(combined_list) 
    
    #X is a MxN vector, where M is the number of reviews
    #                         N is the number of words 
    #Y is a Mx1 vector, where M is the number of reviews
    x = zeros((1600,len(combined_list)))
    y = zeros((1600,2)) #one hot encoding
    cur_row = 0 #keep track of which column we're on
    
    path = 'training_set'
    
    for review in os.listdir(path + "\\neg"): #[1,0] corresponds to a negative review
        with open(os.getcwd()+"\\"+path+"\\neg\\"+review) as f:
            unique_words = set(f.read().split())
            i=0
            for word in combined_list:
                if word in unique_words:
                    x[cur_row, i] = 1
                    y[cur_row] = [1, 0]
                i+=1
        cur_row += 1
        #print(cur_row)
        
    for review in os.listdir(path + "\\pos"): #[0,1] corresponds to a positive review
        with open(os.getcwd()+"\\"+path+"\\pos\\"+review) as f:
            unique_words = set(f.read().split())
            i=0
            for word in combined_list:
                if word in unique_words:
                    x[cur_row, i] = 1
                    y[cur_row] = [0, 1]
                i+=1
        cur_row += 1
        #print(cur_row)
    return x, y


def get_test(neg_dict, pos_dict):

    combined_list = []
    
    #convert the keys of the dicts into a list
    for word in neg_dict:
        combined_list.append(word)
    for word in pos_dict:
        combined_list.append(word)
    #remove duplicates in the list
    combined_list = list(set(combined_list))
    
    x = zeros((200,len(combined_list)))
    y = zeros((200,2))
    cur_row = 0 #keep track of which column we're on
    
    path = 'test_set'
    
    for review in os.listdir(path + "\\neg"): 
        with open(os.getcwd()+"\\"+path+"\\neg\\"+review) as f:
            unique_words = set(f.read().split())
            i=0
            for word in combined_list:
                if word in unique_words:
                    x[cur_row, i] = 1
                    y[cur_row] = [1, 0]
                i+=1
        cur_row += 1
        #print(cur_row)
        
    for review in os.listdir(path + "\\pos"): 
        with open(os.getcwd()+"\\"+path+"\\pos\\"+review) as f:
            unique_words = set(f.read().split())
            i=0
            for word in combined_list:
                if word in unique_words:
                    x[cur_row, i] = 1
                    y[cur_row] = [0, 1]
                i+=1
        cur_row += 1
        #print(cur_row)
    return x, y
    
def logistic_regression(neg_dict, pos_dict, num_words):
    
    '''This part may make no sense at all...noob at tensorflow
    '''
    # Initialize Tensor Flow variables
    x = tf.placeholder(tf.float32, [None, num_words])
    W0 = tf.Variable(tf.random_normal([num_words, 1], stddev=0.01))
    b0 = tf.Variable(tf.random_normal([1], stddev=0.01))
    
    y = tf.nn.sigmoid(tf.matmul(x, W0)+b0)
    y_ = tf.placeholder(tf.float32, [None, 2])
    
    lam = 0.00000
    decay_penalty =lam*tf.reduce_sum(tf.square(W0))
    reg_NLL = -tf.reduce_sum(y_*tf.log(y))+decay_penalty
    
    train_step = tf.train.AdamOptimizer(0.0005).minimize(reg_NLL)
    
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    test_x, test_y = get_test(neg_dict, pos_dict)
    
    
    for i in range(5000):
    #print i  
        batch_xs, batch_ys = get_train(neg_dict, pos_dict)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
        
        
        if i % 1 == 0:
            print("i=",i)
            print("Test:", sess.run(accuracy, feed_dict={x: test_x, y_: test_y}))
            batch_xs, batch_ys = get_train(neg_dict, pos_dict)
        
            print("Train:", sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys}))
            print("Penalty:", sess.run(decay_penalty))

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

if run_part2 == True:
    dicts = generate_dict('training_set')
    m = 50
    k = 50
    neg_dict = calculate_prob(dicts[0], m, k)
    pos_dict = calculate_prob(dicts[1], m, k)
    score = classify('validation_set', neg_dict, pos_dict)
    score = classify('test_set', neg_dict, pos_dict)
    print(score)

if run_tuning == True:
    max_validation()
    '''I've ran this before and saved the results in a text file called mkscore.
       Highest Performance Tuning:
        validation set: m = 50 k = 50 (0.57, 0.82) test set:  m = 50 k = 50 (0.58, 0.78)
    '''
    
if run_part3 == True:
    dicts = generate_dict('training_set')
    m = 50
    k = 50
    neg_dict = calculate_prob(dicts[0], m, k)
    pos_dict = calculate_prob(dicts[1], m, k)
    most_predictive(neg_dict, pos_dict)

if run_part4 == True:
    dicts = generate_dict('training_set')
    m = 50
    k = 50
    neg_dict = calculate_prob(dicts[0], m, k)
    pos_dict = calculate_prob(dicts[1], m, k)
    x,y = get_train(neg_dict, pos_dict)
    logistic_regression(neg_dict, pos_dict, num_words)
    