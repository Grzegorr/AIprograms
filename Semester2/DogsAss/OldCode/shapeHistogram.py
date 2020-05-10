import cv2
import numpy as np
from matplotlib import pyplot as plt


class Histogramer:
    #Arrays to store hights and widths of analyzed pictures
    shape1 = []
    shape2 = []
    
    #scanning thorough training data and extracting hights and widths of images
    def appendPictures(self):
        for i in range(0, 3001):
            path = "DogsData/train/dog." + str(i) + ".jpg"
            img = cv2.imread(path,1)
            #cv2.imshow('original',  img)
            #cv2.waitKey(0)
            #print(img.shape)
            self.shape1.append(img.shape[0])
            self.shape2.append(img.shape[1])
            path = "DogsData/train/cat." + str(i) + ".jpg"
            img = cv2.imread(path,1)
            #cv2.imshow('original',  img)
            #cv2.waitKey(0)
            #print(img.shape)
            self.shape1.append(img.shape[0])
            self.shape2.append(img.shape[1])
	for i in range(3001, 500):
            path = "DogsData/train/dog." + str(i) + ".jpg"
            img = cv2.imread(path,1)
            #cv2.imshow('original',  img)
            #cv2.waitKey(0)
            #print(img.shape)
            self.shape1.append(img.shape[0])
            self.shape2.append(img.shape[1])
            path = "DogsData/train/cat." + str(i) + ".jpg"
            img = cv2.imread(path,1)
            #cv2.imshow('original',  img)
            #cv2.waitKey(0)
            #print(img.shape)
            self.shape1.append(img.shape[0])
            self.shape2.append(img.shape[1])
    
    #Drawing histograms of widths and hights of processed pictures 
    #as well as printing its maximum and minimum values
    def drawHistogram(self):
        print(np.amin(self.shape1))
        print(np.amax(self.shape1))
        plt.hist(self.shape1, bins = np.amax(self.shape1) - np.amin(self.shape1)/100)
        plt.title("Histogram of hights of the pictures.")
        plt.xlabel("No. of pixels")
        plt.ylabel("No. of images")
        plt.show()
        print(np.amin(self.shape2))
        print(np.amax(self.shape2))
        plt.hist(self.shape2, bins = np.amax(self.shape2) - np.amin(self.shape2)/100)
        plt.title("Histogram of widths of the pictures.")
        plt.xlabel("No. of pixels")
        plt.ylabel("No. of images")
        plt.show()
            
            

H = Histogramer()
H.appendPictures()
H.drawHistogram()
