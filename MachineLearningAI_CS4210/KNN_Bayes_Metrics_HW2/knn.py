#-------------------------------------------------------------------------
# AUTHOR: Maayan Israel
# FILENAME: knn.py
# SPECIFICATION: This program reads a csv file and uses the KNN algorithm to predict the class of each instance in the file and calculates the error rate.
# FOR: CS 4210- Assignment #2
#-----------------------------------------------------------*/

from sklearn.neighbors import KNeighborsClassifier
import csv

db = []

with open('email_classification.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: 
         db.append(row)

numErrors = 0

for i in db:
    X = []
    Y = []
    XTest = []
    YTest = []
    for row in db:
        if row != i:
            X.append([float(val) for val in row[:-1]])
            if row[-1] == 'ham':
                Y.append(1)
            elif row[-1] == 'spam':
                Y.append(2)
        elif row == i:
            XTest = [float(val) for val in row[:-1]]
            if row[-1] == 'ham':
                YTest = 1
            elif row[-1] == 'spam':
                YTest = 2

    clf = KNeighborsClassifier(n_neighbors=1, p=2)
    clf = clf.fit(X, Y)

    class_predicted = clf.predict([XTest])[0]
    if class_predicted != YTest:
        numErrors += 1

errorRate = numErrors / len(db)
print("Error rate: ", errorRate)