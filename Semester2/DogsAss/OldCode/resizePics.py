import cv2

class Resizer:
    
    #Resize train data to 500x400
    def resizeTrain(self):
        for i in range(0, 3001):
            path = "DogsData/train/dog." + str(i) + ".jpg"
            img = cv2.imread(path,1)
            resized = cv2.resize(img,  (500, 400))
            path = "DogsData/resized_train/dog." + str(i) + ".jpg"
            cv2.imwrite(path,resized)
            path = "DogsData/train/cat." + str(i) + ".jpg"
            img = cv2.imread(path,1)
            resized = cv2.resize(img,  (500, 400))
            path = "DogsData/resized_train/cat." + str(i) + ".jpg"
            cv2.imwrite(path,resized)

    #Resizing test data to 500x400
    def resizeTest(self):
        for i in range(3001, 3501):
            path = "DogsData/test1/dog." + str(i) + ".jpg"
            img = cv2.imread(path,1)
            resized = cv2.resize(img,  (500, 400))
            path = "DogsData/resized_test1/dog." + str(i) + ".jpg"
            cv2.imwrite(path,resized)
            path = "DogsData/test1/cat." + str(i) + ".jpg"
            img = cv2.imread(path,1)
            resized = cv2.resize(img,  (500, 400))
            path = "DogsData/resized_test1/cat." + str(i) + ".jpg"
            cv2.imwrite(path,resized)
            
    #Resize train data to 100x80
    def resizeTrain1_5(self):
        for i in range(0, 3001):
            path = "DogsData/train/dog." + str(i) + ".jpg"
            img = cv2.imread(path,1)
            resized = cv2.resize(img,  (100, 80))
            path = "DogsData/resized_train1_5/dog." + str(i) + ".jpg"
            cv2.imwrite(path,resized)
            path = "DogsData/train/cat." + str(i) + ".jpg"
            img = cv2.imread(path,1)
            resized = cv2.resize(img,  (100, 80))
            path = "DogsData/resized_train1_5/cat." + str(i) + ".jpg"
            cv2.imwrite(path,resized)

    #Resizing test data to 100x80
    def resizeTest1_5(self):
        for i in range(3001, 3501):
            path = "DogsData/test1/dog." + str(i) + ".jpg"
            img = cv2.imread(path,1)
            resized = cv2.resize(img,  (100, 80))
            path = "DogsData/resized_test1_5/dog." + str(i) + ".jpg"
            cv2.imwrite(path,resized)
            path = "DogsData/test1/cat." + str(i) + ".jpg"
            img = cv2.imread(path,1)
            resized = cv2.resize(img,  (100, 80))
            path = "DogsData/resized_test1_5/cat." + str(i) + ".jpg"
            cv2.imwrite(path,resized)
            
    #Resize train data to 50x40
    def resizeTrain1_10(self):
        for i in range(0, 3001):
            path = "DogsData/train/dog." + str(i) + ".jpg"
            img = cv2.imread(path,1)
            resized = cv2.resize(img,  (50, 40))
            path = "DogsData/resized_train1_10/dog." + str(i) + ".jpg"
            cv2.imwrite(path,resized)
            path = "DogsData/train/cat." + str(i) + ".jpg"
            img = cv2.imread(path,1)
            resized = cv2.resize(img,  (50, 40))
            path = "DogsData/resized_train1_10/cat." + str(i) + ".jpg"
            cv2.imwrite(path,resized)

    #Resizing test data to 50x40
    def resizeTest1_10(self):
        for i in range(3001, 3501):
            print(i)
            path = "DogsData/test1/dog." + str(i) + ".jpg"
            img = cv2.imread(path,1)
            resized = cv2.resize(img,  (50, 40))
            path = "DogsData/resized_test1_10/dog." + str(i) + ".jpg"
            cv2.imwrite(path,resized)
            path = "DogsData/test1/cat." + str(i) + ".jpg"
            img = cv2.imread(path,1)
            resized = cv2.resize(img,  (50, 40))
            path = "DogsData/resized_test1_10/cat." + str(i) + ".jpg"
            cv2.imwrite(path,resized)


R = Resizer()
R.resizeTest()
R.resizeTest1_5()
R.resizeTest1_10()
R.resizeTrain()
R.resizeTrain1_5()
R.resizeTrain1_10()













