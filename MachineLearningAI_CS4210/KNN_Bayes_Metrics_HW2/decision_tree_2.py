#-------------------------------------------------------------------------
# AUTHOR: Maayan Israel
# FILENAME: decision_tree_2.py
# SPECIFICATION: 
#   This program reads in a dataset and trains a decision tree on it. 
#   It then tests the decision tree on a test dataset and calculates the accuracy of the decision tree. 
#   This process is repeated 10 times and the average accuracy is calculated.
# FOR: CS 4210- Assignment #2
#-----------------------------------------------------------*/

from sklearn import tree
import csv

dataSets = ['contact_lens_training_1.csv', 'contact_lens_training_2.csv', 'contact_lens_training_3.csv']

dbTest = []
with open('contact_lens_test.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i > 0:
            dbTest.append(row)

XTest = []
YTest = []

for i in range(len(dbTest)):
    row = [
        1 if dbTest[i][0] == 'Young' else 2 if dbTest[i][0] == 'Prepresbyopic' else 3,
        1 if dbTest[i][1] == 'Myope' else 2,
        1 if dbTest[i][2] == 'No' else 2,
        1 if dbTest[i][3] == 'Normal' else 2
    ]
    XTest.append(row)
    YTest.append(1 if dbTest[i][4] == 'Yes' else 2)

for ds in dataSets:

    dbTraining = []
    X = []
    Y = []

    with open(ds, 'r') as csvfile:
         reader = csv.reader(csvfile)
         for i, row in enumerate(reader):
             if i > 0: 
                dbTraining.append (row)

    temp = []
    tempy = []

    for i in range(len(dbTraining)):
        row = [
            1 if dbTraining[i][0] == 'Young' else 2 if dbTraining[i][0] == 'Prepresbyopic' else 3,
            1 if dbTraining[i][1] == 'Myope' else 2,
            1 if dbTraining[i][2] == 'No' else 2,
            1 if dbTraining[i][3] == 'Normal' else 2
        ]
        temp.append(row)
        tempy.append(1 if dbTraining[i][4] == 'Yes' else 2)

    X = temp
    Y = tempy

    averageAccuracy = 0

    for i in range (10):

        clf = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth=5)
        clf = clf.fit(X, Y)

        class_predicted = []

        for data in XTest:
            class_predicted.append(clf.predict([data])[0])

        correct_predictions = sum([1 for j in range(len(class_predicted)) if class_predicted[j] == YTest[j]])
        accuracy = correct_predictions / len(YTest)
        averageAccuracy += accuracy
    
    averageAccuracy /= 10
    print(f'Average accuracy for {ds}: {averageAccuracy}')