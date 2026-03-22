##############################################################################################################
#   Project:        Bladder Pressure Prediction
#   Authors:        Maria N. Petrou
#   Description:    Library containing the custom machine learning and neural network class used in 
#                   this project.
##############################################################################################################

# --- Import packages ----------------------------------------------------------------------------------------

from unicodedata import bidirectional
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interp
from tensorflow import keras
import pandas
from keras.callbacks import History 

# --- Setup Callbacks ------------------------------------------------------------------------------------------

# This callback is used to track the loss at each epoch of training and validation data
history = History()

# This callback will stop the training when there is no improvement in the loss for three consecutive epochs
earlyStop = keras.callbacks.EarlyStopping(monitor='loss', patience=3)


# --- ML/Neural Network Class ----------------------------------------------------------------------------

class NeuralNetwork:
    """Neural Network"""

    def __init__(self, inDataShape, ntype):
        """Neural Network

        Initialise the Neural Network

        :param: inDataShape - tuple of data shape to be modelled by neural network
        :param: type - defines the name of neural network to be initialised: MLP = Multi Layer Perceptron,
                       LSTM = Long Short Term Memory, CNN = Convolutional Neural Network

        :return: model - initialised neural network model
        """
        self.inputShape = inDataShape[1:]
        self.trainLoss = 0

        if ntype == "MLP":
            # Initialises model for a Multi Layer Perceptron Neural Network model
            # Input layer with shape of input Data
            # 2 hidden layers with 75 and 10 neurons respectively
            # Output layer with 1 neuron
            self.model = keras.Sequential(
                [
                    keras.Input(shape=self.inputShape),
                    keras.layers.Flatten(),
                    keras.layers.Dense(15, input_shape=self.inputShape, activation="relu"), #75
                    keras.layers.Dense(15, activation="relu"), # 10
                    keras.layers.Dense(1, activation="linear"), # regression means final dense layer has to be 1
                ]
            )
        elif ntype == "LSTM":
            # Initialises model for a Long Short Term Memory Neural Network model
            # Input layer with shape of input Data
            # 1 hidden layers with 128 neurons
            # Output layer with 1 neuron
            self.model = keras.Sequential(
                [
                    keras.Input(shape=(self.inputShape)),
                    # layers should be as few as possible
                    keras.layers.Dense(16), #16
                    keras.layers.LSTM(32, kernel_regularizer=keras.regularizers.l2(0.001)), #32 #0.001
                    keras.layers.BatchNormalization(),
                    # keras.layers.LSTM(128, input_shape=(self.inputShape), return_sequences=True, kernel_regularizer=keras.regularizers.l2(0.000001)),
                    # keras.layers.LSTM(128, return_sequences=True, kernel_regularizer=keras.regularizers.l2(0.000001)),
                    # keras.layers.LSTM(128, kernel_regularizer=keras.regularizers.l2(0.000001)),
                    keras.layers.Dropout(0.2), # 0.2 Dropout layer randomly sets input units to 0 to prevent overfitting
                    keras.layers.Dense(1, activation="relu"),  
                ]
            )
        elif ntype == "LSTM_PCA":
            # Initialises model for a Long Short Term Memory Neural Network model
            self.model = keras.Sequential(
                [
                    keras.Input(shape=(self.inputShape)),
                    keras.layers.Dense(512),
                    keras.layers.Bidirectional(keras.layers.LSTM(256, kernel_regularizer=keras.regularizers.l1(0.001))), #32
                    keras.layers.BatchNormalization(),
                    keras.layers.Dense(128),
                    keras.layers.Dense(64),
                    keras.layers.Dense(32),
                    keras.layers.Dropout(0.5), # 0.5 Dropout layer randomly sets input units to 0 to prevent overfitting
                    keras.layers.Dense(1, activation="softplus"),  # tested and deosn't work: sigmoid, softmax, softsign, tanh
                                                                  # tested and works: relu, softplus, selu, elu
                ]
            )
        elif ntype == "multiLSTM":
            # Initialise model for a multitask learning version of LSTM model
            self.model = keras.Sequential(
                [
                    keras.Input(shape=(self.inputShape)),
                    keras.layers.Dense(16),
                    keras.layers.LSTM(32, kernel_regularizer=keras.regularizers.l2(0.001)), #32
                    keras.layers.BatchNormalization(),
                    keras.layers.Dropout(0.5), # 0.5 Dropout layer randomly sets input units to 0 to prevent overfitting
                    keras.layers.Dense(1, activation="relu"),  
                ]
            )
        elif ntype == "CuDNNLSTM":
            #initialise model for a CuDNN LSTM model
            self.model = keras.Sequential(
                [
                    keras.Input(shape=(self.inputShape)),
                    keras.layers.Dense(16),
                    keras.layers.LSTM(16, kernel_regularizer=keras.regularizers.l2(0.000001)),
                    keras.layers.BatchNormalization(),
                    keras.layers.Dropout(0,5),
                    keras.layers.Dense(1, activation='softmax'),
                ]
            )
        elif ntype == "CNN":
            # Initialises model for a Convolutional Neural Network model
            # Input layer with shape of input Data
            # 1 Convolutional layer and 1 Max pooling layer
            # Output layer with 1 neuron
            self.model = keras.Sequential(
                [
                    keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape = (12, 12, 1)),
                    keras.layers.MaxPooling2D(pool_size=(2, 2)),
                    keras.layers.Flatten(input_shape=(1, 12)),
                    keras.layers.Dropout(0.1), # increase to 0.5 if it starts overfitting
                    keras.layers.Dense(1, activation="softplus"), # selu, relu, softplus work, tanh, softmax and softsign don't
                ]
            )
        elif ntype == "TDNN":
            # TDNN = 1D CNN with no pooling and dilation
            self.model = keras.Sequential(
                [
                    keras.layers.Conv1D(32, kernel_size=5, activation="relu", input_shape = (12, 12, 1), dilation_rate=2),
                    keras.layers.Flatten(input_shape=(1, 12)),
                    keras.layers.Dropout(0.1),
                    keras.layers.Dense(1, activation="softplus"), # selu, relu, softplus work, tanh, softmax and softsign don't
                ]
            )
                    
        # Display model summary.
        self.model.summary()
                
    def TrainNN(self, input, target, x_val, y_val, batch_size = 32, epochs = 50):  #change epochs to 50, batch size = 32
        """ Train Neural Network

        Train the input Neural Network model. Config the model with losses and metrics.

        :param: input - numpy array of input data to be trained
        :param: target - numpy array of target data to be trained
        :param: batch_size - integer number of training examples utilised in one iteration
        :param: epochs - integer number of cycles through the full training dataset
        """
        # Configure the model and start training
        # advised optimizer: 'adam', 'RMSprop'
        self.model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_squared_error'])
       
        # Overfitting limits
        early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10)
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=5)

        # Fit the model to the training data.
        # hist = self.model.fit(input, target, batch_size=batch_size, epochs=epochs, validation_split=0.5, callbacks=[History(), early_stopping, reduce_lr]) #ERROR HERE early_stopping
        
        # Use this model to fit training data otherwise the LSTM won't work - consider adding the callbacks into this one
        # hist = self.model.fit(input, target, batch_size=batch_size, epochs=epochs, validation_data=(x_val, y_val), callbacks=[reduce_lr])
        hist = self.model.fit(input, target, batch_size=batch_size, epochs=epochs, validation_data=(x_val, y_val), callbacks=[reduce_lr, early_stopping])
        
        # use history to visualise losses in train and validation to check for overfitting
        trainLoss = hist.history['loss']
        valLoss = hist.history['val_loss']
        # noEpochs = range(epochs)

        # plot figure of model train and validation accuracy against number of epochs
        # validation loss should be similar to train loss. 
        # validation loss >> train loss = overfitting
        # validation loss << train loss = underfitting
        plt.figure()
        # plt.plot(noEpochs, trainLoss,'k-o', label='train')
        # plt.plot(noEpochs, valLoss,'r-o', label='validation')
        plt.plot(trainLoss,'k-o', label='train')
        plt.plot(valLoss,'r-o', label='validation')
        plt.legend()
        plt.pause(0.001)

    def TestNN(self, testInput, testTarget):
        """ Test Neural Network

        Test the input Neural Network model and predict the labels of the data values on the basis of the trained model

        :param: model - trained neural network model
        :param: input - numpy array of input data to be tested
        :param: target - numpy array of target data to be tested

        :return: score - accuracy of neural network model
        :return: my_predictions - output array of prediction of labels of the data values
        """

        self.score = self.model.evaluate(testInput, testTarget, verbose=1)
        self.my_predictions = self.model.predict(testInput)

        return self.score, self.my_predictions

