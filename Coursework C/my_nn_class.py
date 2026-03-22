##############################################################################################################
#   Project:        Computational Intelligence Coursework C
#   Description:    Library containing the custom machine learning and neural network class used in 
#                   this project.
##############################################################################################################

# --- Import packages ----------------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from keras.callbacks import EarlyStopping

# --- ML/Neural Network Class ----------------------------------------------------------------------------

class NeuralNetwork:
    """Neural Network"""

    def __init__(self, inDataShape, num_classes):
        """Neural Network

        Initialise the Neural Network

        :param: inDataShape - tuple of data shape to be modelled by neural network
        :param: num_classes - defines the number of output neurons of the model (classification classes)
        :return: model - initialised neural network model
        """
        self.inputShape = inDataShape
        self.trainLoss = 0

        self.model = keras.Sequential(
                [
                    keras.Input(shape=self.inputShape),
                    keras.layers.Dense(32, input_shape=self.inputShape, activation="relu", kernel_regularizer='l1_l2'), #change from 'l1' for sub7
                    keras.layers.Dense(16, activation="relu"), 
                    keras.layers.Dropout(0.1), 
                    keras.layers.Dense(num_classes, activation="softmax"), # regression means final dense layer has to be 1
                ]
            )
   
        # Display model summary.
        self.model.summary()
                
    def TrainNN(self, input, target, batch_size = 64, epochs = 500):  #change epochs to 50, batch size = 32
        """ Train Neural Network

        Train the input Neural Network model. Config the model with losses and metrics.

        :param: input - numpy array of input data to be trained
        :param: target - numpy array of target data to be trained
        :param: batch_size - integer number of training examples utilised in one iteration
        :param: epochs - integer number of cycles through the full training dataset
        """
        # Configure the model and start training
        # Configure the model and start training
        optimizer = Adam(learning_rate=0.0005)
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        es_callback = EarlyStopping(monitor='val_loss', patience=5)
        history = self.model.fit(input, target, epochs=epochs, batch_size=batch_size, verbose=1, validation_split=0.2, callbacks=[es_callback])
        
        # summarize history for accuracy
        plt.figure(7)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        
        # summarize history for loss
        plt.figure(8)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

    def ValNN(self, testInput, testTarget):
        """ Test Neural Network on known data

        Test the input Neural Network model and predict the labels of the data values on the basis of the trained model

        :param: self - trained neural network model
        :param: testInput - numpy array of input data to be tested
        :param: testTarget - numpy array of target data to be tested

        :return: score - accuracy of neural network model
        :return: my_predictions - output array of prediction labels for the data values
        """

        # Test the model after training
        self.score = self.model.evaluate(testInput, testTarget, verbose=1)
        self.my_predictions = self.model.predict(testInput)

        return self.score, self.my_predictions

    def TestNN(self, testInput):
        """ Test Neural Network on unknown data

        Test the input Neural Network model and predict the labels of the data values on the basis of the trained model

        :param: self - trained neural network model
        :param: testInput - numpy array of input data to be tested

        :return: class_pred - output array of prediction labels for the data values
        """

        # Test the model after training
        class_out = self.model.predict(testInput)
        self.class_pred = np.zeros(len(class_out), dtype=np.int64)
        i = 0
        n1 = 0
        n2 = 0
        n3 = 0
        n4 = 0
        n5 = 0
        
        for l in class_out:
            self.class_pred[i] = l.argmax()

            # Set counters
            if self.class_pred[i] == 1:
                n1 = n1 + 1
            elif self.class_pred[i] == 2:
                n2 = n2 + 1
            elif self.class_pred[i] == 3:
                n3 = n3 + 1
            elif self.class_pred[i] == 4:
                n4 = n4 + 1 
            else:
                n5 = n5 + 1
            
            i = i + 1

        print("Number of 1s predicted: ", n1)   
        print("Number of 2s predicted: ", n2)    
        print("Number of 3s predicted: ", n3)   
        print("Number of 4s predicted: ", n4)  
        print("Number of 5s predicted: ", n5)       

        return self.class_pred    

