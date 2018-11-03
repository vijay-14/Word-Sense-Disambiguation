1. This program helps in calculating the Accuracy and Confusion Matrix for the predictions we made using the tagger.py by comparing its results to the Gold Standard file.

2. Inorder to make this program work first we need to install the required libraries. Then in the command line arguments you need to pass a minimum of 2 arguments. 
  -> The first argument should be the test results data obtained from executing tagger.py.
  -> The second argument should be the Gold Standard data with correct sense-id's.

An example to run this program:
>python scorer.pl my-line-answers.txt line-answers.txt
Sample Output:
Accuracy = 0.9047619047619048

 Confusion Matrix:

         phone  product
phone       63        9
product      3       51

3. About our code and algorithm:
Step 1: It reads two text files from the arguments and extracts sense-id from them using regular expressions.
Step 2: Calculate the Accuracy of the tagger.py program.
Step 3: Compute the Confusion Matrix for gold standard values and predicted values using the confusion_matrix() function from sklearn.metrics package and print the 
        results to the standard output.
