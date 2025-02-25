#-------------------------------------------------------------------------
# AUTHOR: your name
# FILENAME: title of the source file
# SPECIFICATION: description of the program
# FOR: CS 4210- Assignment #2
# TIME SPENT: how long it took you to complete the assignment
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard vectors and arrays

#Importing some Python libraries
from sklearn.neighbors import KNeighborsClassifier
import csv

db = []

#Reading the data in a csv file
with open('binary_points.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         db.append (row)

#Loop your data to allow each instance to be your test set
for i in db:

    #Add the training features to the 2D array X removing the instance that will be used for testing in this iteration.
    #For instance, X = [[1, 3], [2, 1,], ...]].
    #Convert each feature value to float to avoid warning messages
    #--> add your Python code here

    #Transform the original training classes to numbers and add them to the vector Y.
    #Do not forget to remove the instance that will be used for testing in this iteration.
    #For instance, Y = [1, 2, ,...].
    #Convert each feature value to float to avoid warning messages
    #--> add your Python code here

    #Store the test sample of this iteration in the vector testSample
    #--> add your Python code here

    #Fitting the knn to the data
    clf = KNeighborsClassifier(n_neighbors=1, p=2)
    clf = clf.fit(X, Y)

    #Use your test sample in this iteration to make the class prediction. For instance:
    #class_predicted = clf.predict([[1, 2]])[0]
    #--> add your Python code here

    #Compare the prediction with the true label of the test instance to start calculating the error rate.
    #--> add your Python code here

#Print the error rate
#--> add your Python code here






