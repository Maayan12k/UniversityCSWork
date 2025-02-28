#-------------------------------------------------------------------------
# AUTHOR: Maayan Israel
# FILENAME: naive_bayes.py
# SPECIFICATION: This program reads a csv file and uses the NAIVE BAYES algorithm to predict the class of each instance in the file and ouputs the probabilities of each class for each instance with confidence >= 0.75.
# FOR: CS 4210- Assignment #2
#-----------------------------------------------------------*/

import csv
from sklearn.naive_bayes import GaussianNB

db = []
with open('weather_training.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: 
         db.append(row)

X = []
Y = []
for i in range(len(db)):
    row = [
        1 if db[i][1] == 'Overcast' else 2 if db[i][1] == 'Rain' else 3,
        1 if db[i][2] == 'Cool' else 2 if db[i][2] == 'Mild' else 3,
        1 if db[i][3] == 'Normal' else 2,
        1 if db[i][4] == 'Strong' else 2
    ]
    X.append(row)
    Y.append(1 if db[i][5] == 'Yes' else 2)

clf = GaussianNB(var_smoothing=1e-9)
clf.fit(X, Y)


test = []
with open('weather_test.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: 
         test.append(row)

XTest = []
for i in range(len(test)):
    row = [
        1 if test[i][1] == 'Overcast' else 2 if test[i][1] == 'Rain' else 3,
        1 if test[i][2] == 'Cool' else 2 if test[i][2] == 'Mild' else 3,
        1 if test[i][3] == 'Normal' else 2,
        1 if test[i][4] == 'Strong' else 2
    ]
    XTest.append(row)

print(f"{'Day':<6} {'Outlook':<10} {'Temperature':<12} {'Humidity':<8} {'Wind':<6} {'PlayTennis':<12} {'Confidence':<10}")
for i in range(len(XTest)):
    result = clf.predict_proba([XTest[i]])[0]
    maxVal = max(result)
    if maxVal >= 0.75:
        playTennis = "Yes" if maxVal == result[0] else "No"
        print(f"{test[i][0]:<6} {test[i][1]:<10} {test[i][2]:<12} {test[i][3]:<8} {test[i][4]:<6} {playTennis:<12} {maxVal:<10.3f}")