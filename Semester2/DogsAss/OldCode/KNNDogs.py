import numpy as np
from sklearn import neighbors
from sklearn.metrics import accuracy_score
import random
import cv2
import psutil
from sklearn import naive_bayes
import time


class KNN_GNBDogs:
    
    downScale = 10 #how much a picture was resized. values: 1,5,10
    width = 0
    height = 0
    file = ""
    train_no = 10000
    test_no = 2500
    train_indexes = []
    test_indexes = []
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    
    #initializes 3 values, needed to fully specify a test
    #These are resizing scale, no of samples to train on, no. of sample to validate
    def __init__(self, downScale, no_train,  no_test):
        self.downScale = downScale
        self.train_no = no_train
        self.test_no = no_test
     
    #puts togethet bunch of other functions for convenience
    def runBothClassifications(self):
        self.getIndexes()
        self.generateVariables()
        self.readInTrain()
        self.readInTest()
        self.runClassifierGNB()
        self.runClassifierKNN()
    
    #Check for available RAM, since KNN and GNB are memory intensive
    def memAvailable(self):
            mem = psutil.virtual_memory()
            avl = mem.available
            tot = mem.total
            perc = float(avl)/float(tot)
            print(perc*100)
    
    #This function sets hight, width and a file to read in data
    #It allows to try out the algorithm wit various rescalings
    def generateVariables(self):
        resize = self.downScale
        if resize == 1:
            self.file = "resized_train"
            self.file2 = "resized_test"
            self.width = 500
            self.height = 400
        if resize == 5:
            self.file = "resized_train1_5"
            self.file2 = "resized_test1_5"
            self.width = 100
            self.height = 80
        if resize == 10:
            self.file = "resized_train1_10"
            self.file2 = "resized_test1_10"
            self.width = 50
            self.height = 40
        
    #gets indexes and randomly assigns them for
    #training and validation datasets
    def getIndexes(self):
        #splitting dataset for training and dataset
        index = np.ones(3001, int)
        for i in range(0, 3001):
            index[i] = i
        self.train_indexes = random.sample(index, self.train_no)
        index = np.ones(500, int)
        for i in range(3001, 3501):
            index[i-3001] = i
        self.test_indexes = random.sample(index, self.test_no)
        #index_train = random.sample(index, self.train_no)
        #index_train.sort()
        #self.train_indexes = index_train
        #index = np.delete(index, index_train)
        #index_test = []
        #index_test = random.sample(index, self.test_no)
        #index_test.sort()
        #self.test_indexes = index_test
        #print(index_train)
        #print(index_test)
    
    #Reads in and converst to a row matrix pictures which
    #are to be used for tranig
    def readInTrain(self):
        o=0
        for i in self.train_indexes:
            path = "DogsData/"+ self.file + "/dog." + str(i) + ".jpg"
            img = cv2.imread(path,1)
            img_serial = []
            for g in range(0, self.height):
                for j in range(0, self.width):
                    for k in range(0, 3):
                        img_serial.append(img[g][j][k])
            self.x_train.append(img_serial)
            self.y_train.append(0)
            path = "DogsData/"+ self.file + "/cat." + str(i) + ".jpg"
            img = cv2.imread(path,1)
            img_serial = []
            for g in range(0, self.height):
                for j in range(0, self.width):
                    for k in range(0, 3):
                        img_serial.append(img[g][j][k])
            self.x_train.append(img_serial)
            self.y_train.append(1)
            #self.memAvailable()
            o = o+1
            #print("Train input no. " + str(o) + " was read in.")

            
    #Reads in and converst to a row matrix pictures which
    #are to be used for validation
    def readInTest(self):
        o=0
        for i in self.test_indexes:
            path = "DogsData/"+ self.file2 + "/dog." + str(i) + ".jpg"
            img = cv2.imread(path,1)
            img_serial = []
            for g in range(0, self.height):
                for j in range(0, self.width):
                    for k in range(0, 3):
                        img_serial.append(img[g][j][k])
            self.x_test.append(img_serial)
            self.y_test.append(0)
            path = "DogsData/"+ self.file2 + "/cat." + str(i) + ".jpg"
            img = cv2.imread(path,1)
            img_serial = []
            for g in range(0, self.height):
                for j in range(0, self.width):
                    for k in range(0, 3):
                        img_serial.append(img[g][j][k])
            self.x_test.append(img_serial)
            self.y_test.append(1)
            #self.memAvailable()
            o = o+1
            #print("Test input no. " + str(o) + " was read in.")
    
    
    #trains and runs KNN classifier
    #then prints predictions accuracy
    def runClassifierKNN(self):
        #setting up a classifier
        self.start = time.time()
        classifier = neighbors.KNeighborsClassifier(3)
        classifier.fit(self.x_train, self.y_train)
        predictions = classifier.predict(self.x_test)
        self.stop = time.time()
        print("KNN classifier output:")
        print("---Parameters: downscale:" + str(self.downScale)+ " learning samples: " + str(self.train_no)+ " test samples: " + str(self.test_no))
        print("Accuracy(%)=" +  str(accuracy_score(self.y_test, predictions)*100))
        print("Time needed: " + str(self.stop-self.start))
        
    #trains and runs GNN classifier
    #then prints predictions accuracy
    def runClassifierGNB(self):
        self.start = time.time()
        #setting up a classifier
        classifier = naive_bayes.GaussianNB()
        classifier.fit(self.x_train, self.y_train)
        predictions = classifier.predict(self.x_test)
        self.stop = time.time()
        print("GNB classifier output:")
        print("---Parameters: downscale:" + str(self.downScale)+ " learning samples: " + str(self.train_no)+ " test samples: " + str(self.test_no))
        print("Accuracy(%)=" +  str(accuracy_score(self.y_test, predictions)*100))
        print("Time needed: " + str(self.stop-self.start))


KNN = KNN_GNBDogs(10, 500, 500)
KNN.runBothClassifications()
KNN = KNN_GNBDogs(10, 1500, 500)
KNN.runBothClassifications()
KNN = KNN_GNBDogs(10, 3000, 500)
KNN.runBothClassifications()
























