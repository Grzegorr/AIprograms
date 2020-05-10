from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import time
import cv2
from keras import backend as K
import numpy as np
from matplotlib import pyplot as plt
from email_sender import send_email
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16
from keras.models import Model


class NN:
    
    #Loading in all parameters
    #Also data read in and preparation
    def __init__(self):
        
        self.img_rows = 80
        self.img_cols = 100
        
        self.batch_size = 128
        self.num_classes = 2
        self.epochs = 40
        
        
    
    #pipeline of training
    def train(self, model_name,if_aug,if_gray, fold):
        self.x_train = []
        self.y_train = []
        self.x_test = []
        self.y_test = []
        self.readInTrain(if_gray)
        self.readInTest(if_gray)
        #self.readIn7Fold(if_gray,1)
        self.data_prep()
        self.start_time = time.time()
        input_shape = self.if_chanel_first(if_gray)
        self.model_pipeline(model_name, input_shape,if_aug,if_gray)
        message = "Accuracy plot and model files in the attachment. "
        if if_gray == "True":
            message = message + "Test was in grayscale. "
        else:
            message = message + "Test was in colour. "
        if if_aug == "True":
            message = message + "Image augumentation was used. "
        else:
            message = message + "Image augumentation was NOT used. " 
        send_email("Model \"" + str(model_name) + "\" was trained.", message,
                                        "Trained_Models/" + str(model_name) + ".png")
        
    def model_pipeline(self, model_name, input_shape,if_aug,if_gray):
        model = self.compile_model(input_shape, model_name, if_gray)
        if model != None:
            self.train_model(model, self.start_time, model_name,if_aug)
            self.plot_acc(self.model_for_history, model_name)
        
    
    def readInTrain(self,if_gray):
        for i in range(0, 3001):
            path = "DogsData/resized_train1_5/dog." + str(i) + ".jpg"
            #print(path)
            img = cv2.imread(path,1)
            if if_gray == "True":
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            self.x_train.append(img)
            self.y_train.append(0)
            path = "DogsData/resized_train1_5/cat." + str(i) + ".jpg"
            img = cv2.imread(path,1)
            if if_gray == "True":
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            self.x_train.append(img)
            self.y_train.append(1)
            
    def readInTest(self,if_gray):
        for i in range(3001, 3501):
            path = "DogsData/resized_test1_5/dog." + str(i) + ".jpg"
            img = cv2.imread(path,1)
            if if_gray == "True":
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            self.x_test.append(np.array(img))
            self.y_test.append(0)
            path = "DogsData/resized_test1_5/cat." + str(i) + ".jpg"
            img = cv2.imread(path,1)
            if if_gray == "True":
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            self.x_test.append(np.array(img))
            self.y_test.append(1)
    
    #num is 0-6 to set which of the ranges is to be used for test    
    def readIn7Fold(self,  num, if_gray):
        for i in range(0, 3001):
            path = "DogsData/resized_train1_5/dog." + str(i) + ".jpg"
            #print(path)
            img = cv2.imread(path,1)
            if if_gray == "True":
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            self.x.append(img)
            self.y.append(0)
            path = "DogsData/resized_train1_5/cat." + str(i) + ".jpg"
            img = cv2.imread(path,1)
            if if_gray == "True":
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            self.x.append(img)
            self.y.append(1)
        for i in range(3001, 3501):
            path = "DogsData/resized_test1_5/dog." + str(i) + ".jpg"
            img = cv2.imread(path,1)
            if if_gray == "True":
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            self.x.append(np.array(img))
            self.y.append(0)
            path = "DogsData/resized_test1_5/cat." + str(i) + ".jpg"
            img = cv2.imread(path,1)
            if if_gray == "True":
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            self.x.append(np.array(img))
            self.y.append(1)
        
        #select one of 7 "500" as test, rest is training
        full = range(0, 3500)
        range1 = range(num * 500,  (num+1)*500)
        range2 = [x for x in full if x not in range1]
        for i in range1:
            self.x_test.append(self.x(i))
            self.y_test.append(self.y(i))
        for i in range2:
            self.x_train.append(self.x(i))
            self.y_train.append(self.y(i))
        
    
    
    def data_prep(self):
        self.x_train = np.array(self.x_train,  dtype=np.float32)
        self.x_test = np.array(self.x_test,  dtype=np.float32)
        self.x_train /= 255
        self.x_test /= 255
        print('x_train shape:', self.x_train.shape)
        print(self.x_train.shape[0], 'train samples')
        print(self.x_test.shape[0], 'test samples')
        # convert class vectors to binary class matrices
        self.y_train = keras.utils.to_categorical(self.y_train, self.num_classes)
        self.y_test = keras.utils.to_categorical(self.y_test, self.num_classes)
    
    def if_chanel_first(self,if_gray):
        if if_gray != "True":
            if K.image_data_format() == 'channels_first':
                self.x_train = self.x_train.reshape(self.x_train.shape[0], 3, self.img_rows, self.img_cols)
                self.x_test = self.x_test.reshape(self.x_test.shape[0], 3, self.img_rows, self.img_cols)
                return (3, self.img_rows, self.img_cols)
            else:
                self.x_train = self.x_train.reshape(self.x_train.shape[0], self.img_rows, self.img_cols, 3)
                self.x_test = self.x_test.reshape(self.x_test.shape[0], self.img_rows, self.img_cols, 3)
                return (self.img_rows, self.img_cols, 3)
        else:
            if K.image_data_format() == 'channels_first':
                self.x_train = self.x_train.reshape(self.x_train.shape[0], 1, self.img_rows, self.img_cols)
                self.x_test = self.x_test.reshape(self.x_test.shape[0], 1, self.img_rows, self.img_cols)
                return (1, self.img_rows, self.img_cols)
            else:
                self.x_train = self.x_train.reshape(self.x_train.shape[0], self.img_rows, self.img_cols, 1)
                self.x_test = self.x_test.reshape(self.x_test.shape[0], self.img_rows, self.img_cols, 1)
                return (self.img_rows, self.img_cols, 1)
    
    def compile_model(self, input_shape, model_name,if_gray):
        #full list of models = 2conv1ff, 3conv1ff, 4conv1ff, 5conv1ff, 5conv2ff
        if model_name == "2conv1ff":
            model = Sequential()
            model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
            model.add(Conv2D(64, (3, 3), activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.2))
            model.add(Flatten())
            model.add(Dense(128, activation='relu'))
            model.add(Dropout(0.2))
            model.add(Dense(self.num_classes, activation='softmax'))
            model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])
            model.summary()
            
        if model_name == "3conv1ff":
            model = Sequential()
            model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
            model.add(Conv2D(64, (3, 3), activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Conv2D(64, (3, 3), activation='relu'))
            model.add(Dropout(0.2))
            model.add(Flatten())
            model.add(Dense(128, activation='relu'))
            model.add(Dropout(0.2))
            model.add(Dense(self.num_classes, activation='softmax'))
            model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])
            model.summary()            
            
        if model_name == "4conv1ff":
            model = Sequential()
            model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
            model.add(Conv2D(64, (3, 3), activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Conv2D(64, (3, 3), activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Conv2D(64, (3, 3), activation='relu'))
            model.add(Dropout(0.2))
            model.add(Flatten())
            model.add(Dense(128, activation='relu'))
            model.add(Dropout(0.2))
            model.add(Dense(self.num_classes, activation='softmax'))
            model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])
            
        if model_name == "5conv1ff":
            model = Sequential()
            model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
            model.add(Conv2D(64, (3, 3), activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Conv2D(64, (3, 3), activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Conv2D(64, (3, 3), activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Conv2D(64, (3, 3), activation='relu'))
            model.add(Dropout(0.2))
            model.add(Flatten())
            model.add(Dense(128, activation='relu'))
            model.add(Dropout(0.2))
            model.add(Dense(self.num_classes, activation='softmax'))
            model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])
            
        if model_name == "5conv2ff":
            model = Sequential()
            model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
            model.add(Conv2D(64, (3, 3), activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Conv2D(64, (3, 3), activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Conv2D(64, (3, 3), activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Conv2D(64, (3, 3), activation='relu'))
            model.add(Dropout(0.2))
            model.add(Flatten())
            model.add(Dense(128, activation='relu'))
            model.add(Dropout(0.2))
            model.add(Dense(128, activation='relu'))
            model.add(Dropout(0.2))
            model.add(Dense(self.num_classes, activation='softmax'))
            model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])
            
        if model_name == "5conv2ffdrop50":
            model = Sequential()
            model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
            model.add(Conv2D(64, (3, 3), activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Conv2D(64, (3, 3), activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Conv2D(64, (3, 3), activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Conv2D(64, (3, 3), activation='relu'))
            model.add(Dropout(0.5))
            model.add(Flatten())
            model.add(Dense(128, activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(128, activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(self.num_classes, activation='softmax'))
            model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])
            
        if model_name == "5conv2ffdrop50_bignumbers":
            model = Sequential()
            model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
            model.add(Conv2D(64, (3, 3), activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Conv2D(64, (3, 3), activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Conv2D(128, (3, 3), activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Conv2D(128, (3, 3), activation='relu'))
            model.add(Dropout(0.5))
            model.add(Flatten())
            model.add(Dense(600, activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(300, activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(self.num_classes, activation='softmax'))
            model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])
            
        if model_name == "VGG16":  
            if if_gray == "True":
                print("VGG16 cannot work for grayscale. Skipping this model.")
                return None
            model = VGG16(include_top=False, weights='imagenet', input_shape=input_shape, pooling="max")
            for layer in model.layers:
                layer.trainable = False
            dropout1 = Dropout(0.5)(model.layers[-1].output)
            class1 = Dense(600, activation='relu')(dropout1)
            dropout2 = Dropout(0.5)(class1)
            class2 = Dense(300, activation='relu')(dropout2)
            output = Dense(self.num_classes, activation='softmax')(class2)
            model = Model(inputs=model.inputs, outputs=output)
            model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])
            model.summary()
            for layer in model.layers:
                print(str(layer.trainable))

        #remove dropouts
        #Smaller dense
        if model_name == "VGG16v2":  
            if if_gray == "True":
                print("VGG16 cannot work for grayscale. Skipping this model.")
                return None
            model = VGG16(include_top=False, weights='imagenet', input_shape=input_shape, pooling="max")
            for layer in model.layers:
                layer.trainable = False
            FF = Dense(256, activation='relu')(model.layers[-1].output)
            FF2 = Dense(128, activation='relu')(FF)
            output = Dense(self.num_classes, activation='softmax')(FF2)
            model = Model(inputs=model.inputs, outputs=output)
            model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])
            model.summary()
            for layer in model.layers:
                print(str(layer.trainable))
                
        if model_name == "VGG16v3":  
            if if_gray == "True":
                print("VGG16 cannot work for grayscale. Skipping this model.")
                return None
            model = VGG16(include_top=False, weights='imagenet', input_shape=input_shape, pooling="max")
            #for layer in model.layers:
             #   layer.trainable = False
            FF = Dense(256, activation='relu')(model.layers[-1].output)
            FF2 = Dense(128, activation='relu')(FF)
            output = Dense(self.num_classes, activation='softmax')(FF2)
            model = Model(inputs=model.inputs, outputs=output)
            model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])
            model.summary()
            for layer in model.layers:
                print(str(layer.trainable))
            
        return model
        
    
    def train_model(self, model, start_time, model_name,if_aug):
        if model == None:
            print("No model presented for train_model")
            return
        if if_aug == "True":
            datagen = ImageDataGenerator(rotation_range=20,width_shift_range=0.2,height_shift_range=0.2,horizontal_flip=True,   featurewise_center = True)      
            self.model_for_history = model.fit(datagen.flow(self.x_train, self.y_train, batch_size=self.batch_size), epochs=self.epochs, verbose=1, validation_data=(self.x_test, self.y_test))
        else:
            self.model_for_history = model.fit(self.x_train, self.y_train, batch_size=self.batch_size, epochs=self.epochs, verbose=1, validation_data=(self.x_test, self.y_test))
        score = model.evaluate(self.x_test, self.y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
        print("--- %s seconds ---" % (time.time() - start_time))
        model.save("Trained_Models/" + str(model_name) + ".h5")
        
    def plot_acc(self, history, model_name):
        # Plot training & validation accuracy values
        plt.cla()
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.ylim((0.45,1))
        plt.legend(['Train-set', 'Test-set'], loc='upper left')
        plt.savefig("Trained_Models/" + str(model_name) + ".png")
        
        
if __name__ == '__main__':
    Net = NN()
    #for model in ["2conv1ff", "3conv1ff", "4conv1ff", "5conv1ff", "5conv2ff", "5conv2ffdrop50", "5conv2ffdrop50_bignumbers"]:
    for model in ["VGG16v3"]:
    #for model in ["5conv2ffdrop50_bignumbers","VGG16", "2conv1ff", "5conv1ff", "5conv2ff", "5conv2ffdrop50"]:
        #Net.train(model,"True","True", 999)
        #Net.train(model,"False","True", 999)
        #Net.train(model,"True","False", 999)
        #Net.train(model,"False","False", 999)
        Net.train(model,"True","False", 0)
        Net.train(model,"True","False", 1)
        Net.train(model,"True","False", 2)
        Net.train(model,"True","False", 3)
        Net.train(model,"True","False", 4)
        Net.train(model,"True","False", 5)
        Net.train(model,"True","False", 6)
        










