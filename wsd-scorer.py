# Team name: Evil Geniuses
# Authors:
# Vijayasaradhi Muthavarapu (Vijay)
# Nikhil Reddy Pathuri (Nikhil)
# Dilip Molugu (Dilip)
# Date: 10/31/2018

# 1. This program helps in calculating the Accuracy and Confusion Matrix for the predictions we made using the tagger.py by comparing its results to the Gold Standard file.

# 2. Inorder to make this program work first we need to install the required libraries. Then in the command line arguments you need to pass a minimum of 2 arguments. 
#   -> The first argument should be the test results data obtained from executing tagger.py.
#   -> The second argument should be the Gold Standard data with correct sense-id's.

# An example to run this program:
# >python scorer.pl my-line-answers.txt line-answers.txt
# Sample Output:
# Accuracy = 0.9047619047619048

#  Confusion Matrix:

#          phone  product
# phone       63        9
# product      3       51

# 3. About our code and algorithm:
# Step 1: It reads two text files from the arguments and extracts sense-id from them using regular expressions.
# Step 2: Calculate the Accuracy of the tagger.py program.
# Step 3: Compute the Confusion Matrix for gold standard values and predicted values using the confusion_matrix() function from sklearn.metrics package and print the 
#         results to the standard output.

import sys
import re
from sklearn.metrics import confusion_matrix
import pandas as pd

#reading data
answers = open(sys.argv[1], "r")
gold_std = open(sys.argv[2], "r")

#extracting sense-id from answers
answers2 = []
for line in answers:
    answers2.append(line)

str1 = ''.join(answers2)
sense_ans = []
pattern = re.compile(r'senseid="(.*)"')
matches = pattern.finditer(str1)
for match in matches:
    sense_ans.append(match.group(1))

#extracting sense-id from gold standard

gold_std2 = []
for line in gold_std:
    gold_std2.append(line)

str1 = ''.join(gold_std2)
sense_gold = []
pattern = re.compile(r'senseid="(.*)"')
matches = pattern.finditer(str1)
for match in matches:
    sense_gold.append(match.group(1))

#compute Accuracy
sum = 0
for i in range(len(sense_gold)):
    if(sense_ans[i]==sense_gold[i]):
        sum+=1
accuracy = sum/len(sense_gold)
print("Accuracy = "+str(accuracy))

#create Confusion Matrix

c = confusion_matrix(sense_gold,sense_ans)
c = pd.DataFrame(c,index=["phone","product"],columns=["phone","product"])

print("\n Confusion Matrix:\n")
print(c)