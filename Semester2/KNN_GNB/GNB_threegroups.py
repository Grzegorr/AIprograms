import numpy as np
import matplotlib.pyplot as plt
from sklearn import naive_bayes
from sklearn.metrics import accuracy_score

def getData(num_instances_per_class):
	A = np.random.rand(num_instances_per_class,2)+1
	B = np.random.rand(num_instances_per_class,2)+1*0.01
	C = np.random.rand(num_instances_per_class,2)-1
	A = np.append(A,np.ones([len(A),1])*1,1)
	B = np.append(B,np.ones([len(B),1])*2,1)
	C = np.append(C,np.ones([len(C),1])*3,1)
	data = np.concatenate([A,B])
	data = np.concatenate([data,C])
	return data

train = getData(100)
test = getData(100)

classifier = naive_bayes.GaussianNB()
classifier.fit(train[:, :2], train[:, 2])
predictions = classifier.predict(test[:, :2])
print("Accuracy(%)=",accuracy_score(test[:, 2], predictions)*100)

plt.figure(1)
plt.subplot(131)
plt.scatter(train[:,0], train[:,1], s=50, c=train[:,2], cmap=plt.cm.Paired)
plt.title("Training Data")
plt.subplot(132)
plt.scatter(test[:,0], test[:,1], s=50, c=predictions, cmap=plt.cm.Paired)
plt.title("Test Data")
plt.subplot(133)
plt.scatter(test[:,0], test[:,1], s=50, c=test[:,2], cmap=plt.cm.Paired)
plt.title("Ground-Truth Data")
plt.show()
