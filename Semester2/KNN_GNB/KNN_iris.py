import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors
from sklearn.metrics import accuracy_score
from sklearn import datasets
import random

#Load the dataset in
iris = datasets.load_iris()
x = iris.data
y = iris.target

#some prints
#print(iris)
#print(x)
#print(y)

#splitting dataset in two halfs randomly
index = np.ones(150, int)
for i in range(0, 150):
    index[i] = i
index_train = random.sample(index, 75)
index_train.sort()
index_test = []
for i in range(0, 150):
    if i not in index_train:
        index_test.append(i)
print("Indexes used for training data: " + str(index_train) + "\nIndexes used for testing: " + str(index_test))
train_x = []
train_y = []
test_x = []
test_y = []
for i in range(0, 150):
    if i in index_train:
        train_x.append(x[i])
        train_y.append(y[i])
    if i in index_test:
        test_x.append(x[i])
        test_y.append(y[i])
print("Test iris types:" + str(train_y) + "\nTypes of irises in test subset: " + str(test_y))
#setting up a classifier
classifier = neighbors.KNeighborsClassifier(7)
classifier.fit(train_x, train_y)
predictions = classifier.predict(test_x)

print("Predistions:" + str(predictions))
print("Accuracy(%)=",accuracy_score(test_y, predictions)*100)


