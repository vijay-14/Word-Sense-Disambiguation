1. This program helps in determining the sense of an ambiguous word in a text. It takes an XML file for training which has the senseid, context and ambiguous word and tries to predict the sense id in a test data.

2. Inorder to make this program work first we need to install the required libraries. Then in the command line arguments you need to pass a minimum of 4 arguments.
  -> The first argument should be the training data with senseid.
  -> The second argument should be the testing data with no sesnseid.
  -> The third argument should be an empty file on which we print our vectors and probabilities for each feature.
  -> The third argument is the standard output file to which our program prints the results to.

An example to run this program:
>python decision-list.py line-train.xml line-test.xml my-decision-list.txt > my-line-answers.txt
Sample Output:
my-decision-list:
line =1:
	 vector senseid = phone:[0.19101123595505617]
	 vector senseid = product:[0.0]
	 Probability senseid = phone:0.19101123595505617
	 Probability senseid = product:0.0
line =2:
	 vector senseid = phone:[0.0039081582804103565, 0.016853932584269662]
	 vector senseid = product:[0.0800798580301686, 0.0]
	 Probability senseid = phone:6.58678361866914e-05
	 Probability senseid = product:0.0
my-line-answers:
<answer instance="line-n.w8_059:8174:" senseid="phone"/>
<answer instance="line-n.w7_098:12684:" senseid="phone"/>
<answer instance="line-n.w8_106:13309:" senseid="phone"/>
<answer instance="line-n.w9_40:10187:" senseid="phone"/>
<answer instance="line-n.w9_16:217:" senseid="phone"/>

3. About our code and algorithm:
Training data: We used the XML file provided to us with senseid of the ambigous word for training.
Program Logic:
Step 1: Our program reads the training XMl file and extracts required tags like senseid and head values using regular expressions.
Step 2: Next we clean the texts by removing unnecessary tags, lemmatize the data to bring the words to their base form and tokenize the data.
Step 3: Next we extract words before and after head value.
Step 4: Now we create our Bag of Words seperately for each senseid by selecting n prev words and n future words. Here we got highest accuracy of 90.47% for 1 prev and 1 future.
        we have tried with different number of words like (1,0)-89%,(0,1)-88.8%,(2,1)-86%,(1,2)-85%,(2,2)-83%....
Step 5: We calculate frequency distributions for the bag of words we created in previous step. We also calculate combined frequencies of all senseid.
Step 6: Now we read the test XML file and again perform the same cleaning and Bag of words steps mentioned earlier in steps 2,3,and 4
Step 7: Predict the senseid by calculating the conditional probabilities of the bag of words for the context.
Step 8: Print the vectors and probabilities to my-decision-list for the use of debugging
Step 9: Print the output to STDOUT

Key-Feature: p(s)*P(f/s) Used this formula to calculate likelihood.

4. Results:
Accuracy = 0.9047619047619048

 Confusion Matrix:

         phone  product
phone       63        9
product      3       51

(c) 2018 Vijayasaradhi Muthavarapu, Nikhil Reddy Pathuri, Dilip Molugu
