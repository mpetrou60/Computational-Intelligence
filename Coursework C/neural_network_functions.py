##############################################################################################################
#   Project:        Bladder Pressure Prediction
#   Authors:        Maria N. Petrou
#   Description:    Library containing all custom machine learning and neural network functions used in 
#                   this project.
##############################################################################################################

# --- Import packages ----------------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interp
from tensorflow import keras
import pandas

# --- ML/Neural Network Functions ----------------------------------------------------------------------------

# build the model
def InitMLP_NN(inDataShape):
    """ Initialise Neural Network

    Initialise the Multi-Layer Perceptron Neural Network

    :param: inDataShape - tuple of data shape to be modelled by neural network

    :return: model - initialised neural network model
    """
    inputShape = inDataShape[1:]

    model = keras.Sequential(
        [
            keras.Input(shape=inputShape),
            keras.layers.Flatten(),
            keras.layers.Dense(75, input_shape=inputShape, activation="relu"), 
            keras.layers.Dense(10, activation="relu"),
            keras.layers.Dense(1, activation="linear"), # regression means final dense layer has to be 1
        ]
    )

    # Display model summary.
    model.summary()

    # Simply return the model object so that we can use it in later functions by passing it as an argument.
    return model

# build the model
def InitLSTM_NN(inDataShape):
    """ Initialise Neural Network

    Initialise the Long Short Term Memory Neural Network

    :param: inDataShape - tuple of data shape to be modelled by neural network

    :return: model - initialised neural network model
    """
    inputShape = inDataShape[1:]

    model = keras.Sequential(
        [
            keras.Input(shape=(inputShape)),
            keras.layers.LSTM(128, input_shape=(inputShape)),
            keras.layers.Dense(1, activation="relu"), # regression means final dense layer has to be 1
        ]
    )

    # Display model summary.
    model.summary()

    # Simply return the model object so that we can use it in later functions by passing it as an argument.
    return model

# train the model
def TrainNN(model, input, target, batch_size = 32, epochs = 50):
    """ Traing Neural Network

    Train the input Neural Network model. Config the model with losses and metrics.

    :param: model - initialised neural network model
    :param: input - numpy array of input data to be trained
    :param: target - numpy array of target data to be trained
    :param: batch_size - integer number of training examples utilised in one iteration
    :param: epochs - integer number of cycles through the full training dataset
    """

    # Configure the model and start training: mean_squared_error, mean_absolute_error
    model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_squared_error'])
    model.fit(input, target, epochs=epochs, batch_size=batch_size, verbose=1, validation_split=0.2)
    # No need to return model as it was passed as a reference not a copy 
    # (i.e. the changes made in this function will have applied to the original model in memory)
    pass

# evaluate the model
def TestNN(model, input, target):
    """ Test Neural Network

    Test the input Neural Network model and predict the labels of the data values on the basis of the trained model

    :param: model - trained neural network model
    :param: input - numpy array of input data to be tested
    :param: target - numpy array of target data to be tested

    :return: score - accuracy of neural network model
    :return: my_predictions - output array of prediction of labels of the data values
    """
    score = model.evaluate(input, target, verbose=1)
    my_predictions = model.predict(input)
    
    return score, my_predictions

# infer network function will do something in the future
# def InferNN():
#    return 0

# function to sort data after being shuffled by train_test_split()
def SortData(OutData, TargetData, Predictions):
    """ Sort shuffled data

    Sort shuffled data against a sorted array. Used if the data isn't ascending or descending
    but does need to be ordered

    :param: OutData - Unshuffled data array that the data will be ordered against
    :param: TargetData - numpy array of shuffled data that should match OutData
    :param: Predictions - numpy array of shuffled data that should be ordered in the same way as TargetData

    :return: TargetDataSorted - numpy array of sorted data
    :return: PredictionsSorted - numpy array of sorted data
    """
    position = np.zeros(TargetData.shape)
    TargetDataSorted = np.zeros(TargetData.shape)
    PredictionsSorted = np.zeros(TargetData.shape)

    for m in range(len(TargetData)):
        result = np.where(OutData == TargetData[m])
        position[m] = int(min(result[0]))

    t = pandas.DataFrame({
        'predictions': Predictions, 
        'position': position.flatten(), 
        'targetData': TargetData.flatten()
    })
    t.sort_values(by=['position'], inplace=True)

    TargetDataSorted = t['targetData']
    PredictionsSorted = t['predictions']
    return TargetDataSorted, PredictionsSorted

# function to plot a graph of output predictions against actual output
def PlotRegression(predictions, targetData, tData, sort=True):
    """ plot output of neural network

    Plots the predictions of the neural network against the expected output of the neural network

    :param: predictions - numpy array of data points predicted by neural network
    :param: TargetData - numpy array of expected data points from precorded data
    :param: tData - numpy array of time (x-axis) data points
    :param: sort - Boolean control to be set to True if data should be sorted in an ascending pattern
    """
    t = pandas.DataFrame({'predictions': predictions, 'targetData': targetData})
    if sort:
        t.sort_values(by=['targetData'], inplace=True)

    if (tData.shape[0] != t['targetData'].shape[0]):
        tDataInterp = interp.interp1d(np.arange(tData.size), tData[:, 0])
        tData = tDataInterp(np.linspace(0, tData.size-1, t['targetData'].size))
        tData = np.expand_dims(tData, -1)
    plt.figure()
    plt.plot(tData.tolist(), t['targetData'].tolist(), label='Measured')
    plt.plot(tData.tolist(), t['predictions'].tolist(), label='Predicted')
    plt.ylabel('Normalised Pressure')
    plt.xlabel('Time')
    plt.legend()
    plt.show()

    # function to plot a graph of output predictions against actual output
def PlotRegression2(predictions, targetData, tData, predictions2, targetData2):
    """ plot output of neural network

    Plots the predictions of the neural network against the expected output of the neural network

    :param: predictions - numpy array of data points predicted by neural network
    :param: TargetData - numpy array of expected data points from precorded data
    :param: tData - numpy array of time (x-axis) data points
    :param: sort - Boolean control to be set to True if data should be sorted in an ascending pattern
    """
    print(len(predictions), len(targetData), len(predictions2), len(targetData2))
    t = pandas.DataFrame({'predictions': predictions, 'targetData': targetData, 'predictions2': predictions2, 'targetData2': targetData2})

    if (tData.shape[0] != t['targetData'].shape[0]):
        tDataInterp = interp.interp1d(np.arange(tData.size), tData[:, 0])
        tData = tDataInterp(np.linspace(0, tData.size-1, t['targetData'].size))
        tData = np.expand_dims(tData, -1)

    plt.figure()
    plt.plot(tData.tolist(), t['targetData'].tolist(), label='Measured 1')
    plt.plot(tData.tolist(), t['predictions'].tolist(), label='Predicted 1')
    plt.plot(tData.tolist(), t['targetData2'].tolist(), label='Measured 2')
    plt.plot(tData.tolist(), t['predictions2'].tolist(), label='Predicted 2')
    plt.ylabel('Normalised Pressure')
    plt.xlabel('Time')
    plt.legend()
    plt.show()