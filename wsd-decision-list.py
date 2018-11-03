# Team name: Evil Geniuses
# Authors:
# Vijayasaradhi Muthavarapu (Vijay)
# Nikhil Reddy Pathuri (Nikhil)
# Dilip Molugu (Dilip)
# Date: 10/31/2018

# 1. This program helps in determing the sense of an ambigous word in a text. It takes a XML file for training which has the senseid, cantext and ambigous word and tries to
#   predict the sesnse id in a test data.

# 2. Inorder to make this program work first we need to install the required libraries. Then in the command line arguments you need to pass a minimum of 4 arguments.
#   -> The first argument should be the training data with senseid.
#   -> The second argument should be the testing data with no sesnseid.
#   -> The third argument should be an empty file on which we print our vectors and probabilities for each feature.
#   -> The third argument is the standard output file to which our program prints the results to.

# An example to run this program:
# >python decision-list.py line-train.xml line-test.xml my-decision-list.txt > my-line-answers.txt
# Sample Output:
# my-decision-list:
# line =1:
# 	 vector senseid = phone:[0.19101123595505617]
# 	 vector senseid = product:[0.0]
# 	 Probability senseid = phone:0.19101123595505617
# 	 Probability senseid = product:0.0
# line =2:
# 	 vector senseid = phone:[0.0039081582804103565, 0.016853932584269662]
# 	 vector senseid = product:[0.0800798580301686, 0.0]
# 	 Probability senseid = phone:6.58678361866914e-05
# 	 Probability senseid = product:0.0
# my-line-answers:
# <answer instance="line-n.w8_059:8174:" senseid="phone"/>
# <answer instance="line-n.w7_098:12684:" senseid="phone"/>
# <answer instance="line-n.w8_106:13309:" senseid="phone"/>
# <answer instance="line-n.w9_40:10187:" senseid="phone"/>
# <answer instance="line-n.w9_16:217:" senseid="phone"/>

# 3. About our code and algorithm:
# Training data: We used the XML file provided to us with senseid of the ambigous word for training.
# Program Logic:
# Step 1: Our program reads the training XMl file and extracts required tags like senseid and head values using regular expressions.
# Step 2: Next we clean the texts by removing unnecessary tags, lemmatize the data to bring the words to their base form and tokenize the data.
# Step 3: Next we extract words before and after head value.
# Step 4: Now we create our Bag of Words seperately for each senseid by selecting n prev words and n future words. Here we got highest accuracy of 90.47% for 1 prev and 1 future.
#         we have tried with different number of words like (1,0)-89%,(0,1)-88.8%,(2,1)-86%,(1,2)-85%,(2,2)-83%....
# Step 5: We calculate frequency distributions for the bag of words we created in previous step. We also calculate combined frequencies of all senseid.
# Step 6: Now we read the test XML file and again perform the same cleaning and Bag of words steps mentioned earlier in steps 2,3,and 4
# Step 7: Predict the senseid by calculating the conditional probabilities of the bag of words for the context.
# Step 8: Print the vectors and probabilities to my-decision-list for the use of debugging
# Step 9: Print the output to STDOUT

# Key-Feature: p(s)*P(f/s) Used this formula to calculate likelihood.

# 4. Results:
# Accuracy = 0.9047619047619048

#  Confusion Matrix:

#          phone  product
# phone       63        9
# product      3       51

import sys
import re
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from random import random
import numpy as np
import pandas as pd
import csv
from xml.dom import minidom
import xml.etree.ElementTree as et
import os
from nltk.stem import WordNetLemmatizer
import nltk

#reading data
train = open(sys.argv[1], "r")
test = open(sys.argv[2], "r")
my_list = open(sys.argv[3], "r+")
my_list.truncate(0)

#reading each line
sense = []
for line in train:
    sense.append(line)

#extract sense id
str1 = ''.join(sense)
sense1 = []
pattern = re.compile(r'senseid="(.*)"')
matches = pattern.finditer(str1)
for match in matches:
    sense1.append(match.group(1))

#Compute Frequency Distribution
fdist_sense1 = FreqDist(sense1)

#extract ambiguous word
sense2 = []
pattern1 = re.compile(r'<head>(.*)</head>')
matches1 = pattern1.finditer(str1)
for match1 in matches1:
    sense2.append(match1.group(1))

# words after ambigous word
feat = []
pattern2 = re.compile(r'<head>(.*)</head>(.*)')
matches2 = pattern2.finditer(str1)
for match2 in matches2:
    feat.append(match2.group(2))

#Removing unnecessary tags
x = []
for line in feat:
    x.append(re.sub(r'(<s>|<\/s>)','',line))

#Lemmatizing
x4 = []
lemmatizer = WordNetLemmatizer()
for line in x:
        y3 = lemmatizer.lemmatize(line)
        x4.append(y3)

#tokenizing
x5 = []
tokenizer = RegexpTokenizer(r'\w+')
for line in x4:
        y4 = tokenizer.tokenize(line)
        x5.append(y4)

#extracting words before the ambigous word
feat1 = []
pattern3 = re.compile(r'(.*)<head>(.*)</head>')
matches3 = pattern3.finditer(str1)
for match3 in matches3:
    feat1.append(match3.group(1))

#removing unnecessary tags
x1 = []
for line in feat1:
    x1.append(re.sub(r'(<s>|<\/s>)','',line))

#Lemmatizing
x3 = []
lemmatizer = WordNetLemmatizer()
for line in x1:
        y2 = lemmatizer.lemmatize(line)
        x3.append(y2)

#Tokenizing
x2 = []
tokenizer = RegexpTokenizer(r'\w+')
for line in x1:
        y1 = tokenizer.tokenize(line)
        x2.append(y1)

#Selecting 1 prev words
past2 = []
for line in x2:
    past2.append(line[-1:])

#Selecting 1 future words
fut2 = []
for line in x5:
    fut2.append(line[:1])

#combining
combined = [x+y for x,y in zip(past2,fut2)]

#creating bag of words for each sense
bof_phone = []
bof_product = []

for i in range(len(sense1)):
    if(sense1[i]=="phone"):
        bof_phone.append(combined[i])
    else:
        bof_product.append(combined[i])

#converting 2d list to 1d list
temp =[]
for item in bof_phone:
    temp += item
bof_phone = temp

temp = []
for item in bof_product:
    temp += item
bof_product = temp

#Freq Distributions
fdist_phone = FreqDist(bof_phone)
fdist_product = FreqDist(bof_product)

total = bof_phone + bof_product
fdist_total = FreqDist(total)

#################

#testing

#################

#reading each line
t_sense = []
for line in test:
    t_sense.append(line)

#extracting instance id
str2 = ''.join(t_sense).lower()
t_sense1 = []
pattern = re.compile(r'instance id="(.*)"')
matches = pattern.finditer(str2)
for match in matches:
    t_sense1.append(match.group(1))

#extracting ambigous word
t_sense2 = []
pattern = re.compile(r'<head>(.*)</head>')
matches = pattern.finditer(str2)
for match in matches:
    t_sense2.append(match.group(1))

#extracting future words
t_feat = []
pattern = re.compile(r'<head>(.*)</head>(.*)')
matches = pattern.finditer(str2)
for match in matches:
    t_feat.append(match.group(2))

#removing unncessary tags
x = []
for line in t_feat:
    x.append(re.sub(r'(<s>|<\/s>)','',line))

#lemmatizing
te1 = []
for line in x:
    te0 = lemmatizer.lemmatize(line)
    te1.append(te0)

#tokenizing
te2 = []
for line in te1:
        te_1 = tokenizer.tokenize(line)
        te2.append(te_1)

#extracting prev words
t_feat1 = []
pattern = re.compile(r'(.*)<head>(.*)</head>')
matches = pattern.finditer(str2)
for match in matches:
    t_feat1.append(match.group(1))

#removing unnecessary tags
x1 = []
for line in t_feat1:
    x1.append(re.sub(r'(<s>|<\/s>)','',line))

#lemmatizing
te1_1 = []
for line in x1:
    te0_0 = lemmatizer.lemmatize(line)
    te1_1.append(te0_0)

#tokenizing
te2_2 = []
for line in te1_1:
        te_1_1 = tokenizer.tokenize(line)
        te2_2.append(te_1_1)

#selecting prev 1 word
t_past2 = []
for line in te2_2:
    t_past2.append(line[-1:])

#selecting 1 future words
t_fut2 = []
for line in te2:
    t_fut2.append(line[:1])

#combining bag of test words
t_combined = [x+y for x,y in zip(t_past2,t_fut2)]

# Calculating Probabilities
output = []
temp = 0
for j in range(len(t_combined)):
    my_list.write("line =" + str(j+1) + ":\n")
    prob_phone = []
    prob_product = []
    for i in range(len(t_combined[j])):
        #sense phone
        if t_combined[j][i] in total:
            prob_word = fdist_phone[t_combined[j][i]]/fdist_total[t_combined[j][i]]
            prob_feature = fdist_phone[t_combined[j][i]]/fdist_sense1["phone"]
            prob_phone.append(prob_word * prob_feature)
        else:
            prob_phone.append(fdist_sense1["phone"]/len(sense1))
            temp+=1

        #sense product
        if t_combined[j][i] in total:
            prob_word = fdist_product[t_combined[j][i]]/fdist_total[t_combined[j][i]]
            prob_feature = fdist_product[t_combined[j][i]]/fdist_sense1["product"]
            prob_product.append(prob_word * prob_feature)
        else:
            prob_product.append(fdist_sense1["product"]/len(sense1))
            temp+=1

    my_list.write("\t vector senseid = phone:"+str(prob_phone)+"\n")
    my_list.write("\t vector senseid = product:"+str(prob_product)+"\n")
    final_phone = 1
    final_product = 1

    for item in prob_phone:
        final_phone = final_phone * item
    for item in prob_product:
        final_product = final_product * item

    my_list.write("\t Probability senseid = phone:"+str(final_phone)+"\n")
    my_list.write("\t Probability senseid = product:"+str(final_product)+"\n")

    if(final_phone>final_product):
        output.append("phone")
    else:
        output.append("product")

#output
for i in range(len(output)):
    print("<answer instance=\""+str(t_sense1[i])+"\" senseid=\""+str(output[i])+"\"/>")
