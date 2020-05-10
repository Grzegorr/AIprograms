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
        self.x_train = []
        self.y_train = []
        self.x_test = []
        self.y_test = []
        
        self.img_rows = 40
        self.img_cols = 50
        
        self.batch_size = 128
        self.num_classes = 2
        self.epochs = 50
        
        self.readInTrain()
        self.readInTest()
        self.data_prep()
    
    #pipeline of training
    def train(self, model_name,if_aug):
        self.start_time = time.time()
        input_shape = self.if_chanel_first()
        self.model_pipeline(model_name, input_shape,if_aug)
        send_email("Model \"" + str(model_name) + "\" was trained.", "Accuracy plot and model files in the attachment.",
                                        "Trained_Models/" + str(model_name) + ".png")
        
    def model_pipeline(self, model_name, input_shape,if_aug):
        model = self.compile_model(input_shape, model_name)
        self.train_model(model, self.start_time, model_name,if_aug)
        self.plot_acc(self.model_for_history, model_name)
        
    
    def readInTrain(self):
        for i in range(0, 3001):
            path = "DogsData/resized_train1_10/dog." + str(i) + ".jpg"
            img = cv2.imread(path,1)
            #img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            self.x_train.append(img)
            self.y_train.append(0)
            path = "DogsData/resized_train1_10/cat." + str(i) + ".jpg"
            img = cv2.imread(path,1)
            #img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            self.x_train.append(img)
            self.y_train.append(1)
            
    def readInTest(self):
        for i in range(3001, 3501):
            path = "DogsData/resized_test1_10/dog." + str(i) + ".jpg"
            img = cv2.imread(path,1)
            #img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            self.x_test.append(np.array(img))
            self.y_test.append(0)
            path = "DogsData/resized_test1_10/cat." + str(i) + ".jpg"
            img = cv2.imread(path,1)
            #img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            self.x_test.append(np.array(img))
            self.y_test.append(1)
    
    
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
    
    def if_chanel_first(self):
        if K.image_data_format() == 'channels_first':
            self.x_train = self.x_train.reshape(self.x_train.shape[0], 3, self.img_rows, self.img_cols)
            self.x_test = self.x_test.reshape(self.x_test.shape[0], 3, self.img_rows, self.img_cols)
            return (3, self.img_rows, self.img_cols)
        else:
            self.x_train = self.x_train.reshape(self.x_train.shape[0], self.img_rows, self.img_cols, 3)
            self.x_test = self.x_test.reshape(self.x_test.shape[0], self.img_rows, self.img_cols, 3)
            return (self.img_rows, self.img_cols, 3)
    
    def compile_model(self, input_shape, model_name):
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
            model = VGG16(include_top=False, weights='imagenet', input_shape=input_shape, pooling="max")
            for layer in model.layers:
                layer.trainable = False
            model.summary()
            class1 = Dense(128, activation='relu', kernel_initializer='he_uniform')(model.layers[-1].output)
            output = Dense(self.num_classes, activation='softmax')(class1)
            model = Model(inputs=model.inputs, outputs=output)
            model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])
            model.summary()
            
        return model
    
    def train_model(self, model, start_time, model_name,if_aug):
        if if_aug == "True":
            datagen = ImageDataGenerator(rotation_range=20,width_shift_range=0.2,height_shift_range=0.2,horizontal_flip=True)      
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
    for model in ["VGG16", "5conv2ffdrop50_bignumbers"]:
        Net.train(model,"True")
        Net.train(model,"False")










