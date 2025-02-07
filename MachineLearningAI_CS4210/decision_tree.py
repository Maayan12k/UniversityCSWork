#-------------------------------------------------------------------------
# AUTHOR: Maayan Israel
# FILENAME: decision_tree.py
# SPECIFICATION: simple decision tree classifier
# FOR: CS 4210- Assignment #1
# TIME SPENT: 20 minutes
#-----------------------------------------------------------*/

from sklearn import tree
import matplotlib.pyplot as plt
import csv
db = []
X = []
Y = []

with open('contact_lens.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         db.append (row)

temp = []
tempy = []

for i in range(len(db)):
    row = [
        1 if db[i][0] == 'Young' else 2 if db[i][0] == 'Prepresbyopic' else 3,
        1 if db[i][1] == 'Myope' else 2,
        1 if db[i][2] == 'No' else 2,
        1 if db[i][3] == 'Normal' else 2
    ]
    temp.append(row)
    tempy.append(1 if db[i][4] == 'Yes' else 2)

X = temp
print(X)

Y = tempy
print(Y)

#fitting the decision tree to the data
clf = tree.DecisionTreeClassifier(criterion = 'entropy')
clf = clf.fit(X, Y)

#plotting the decision tree
tree.plot_tree(clf, feature_names=['Age', 'Spectacle', 'Astigmatism', 'Tear'], class_names=['Yes','No'], filled=True, rounded=True)
plt.show()