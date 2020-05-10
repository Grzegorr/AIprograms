import numpy as np
import matplotlib.pyplot as plt
from sklearn import naive_bayes
from sklearn.metrics import accuracy_score

X_train = np.random.randn(500, 2)
Y_train = np.logical_xor(X_train[:, 0] > 0, X_train[:, 1] > 0)

X_test = np.random.randn(500, 2)
Y_test = np.logical_xor(X_test[:, 0] > 0, X_test[:, 1] > 0)

classifier = naive_bayes.GaussianNB()
classifier.fit(X_train, Y_train)
predictions = classifier.predict(X_test)
print("Accuracy(%)=",accuracy_score(Y_test, predictions)*100)

plt.figure(1)
plt.subplot(131)
plt.scatter(X_train[:,0], X_train[:,1], s=50, c=Y_train, cmap=plt.cm.Paired)
plt.title("Training Data")
plt.subplot(132)
plt.scatter(X_test[:,0], X_test[:,1], s=50, c=predictions, cmap=plt.cm.Paired)
plt.title("Test Data")
plt.subplot(133)
plt.scatter(X_test[:,0], X_test[:,1], s=50, c=Y_test, cmap=plt.cm.Paired)
plt.title("Ground-Truth Data")
plt.show()
