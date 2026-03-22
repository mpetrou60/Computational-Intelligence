##############################################################################################################
#   Project:        Bladder Pressure Prediction
#   Authors:        Maria N. Petrou
#   Description:    Library containing all custom data functions used in this project.
##############################################################################################################

# --- Import packages ----------------------------------------------------------------------------------------

import numpy as np
import scipy.interpolate as interp
import pandas
from sklearn.model_selection import train_test_split
from scipy.ndimage import zoom

# --- Data Functions -----------------------------------------------------------------------------------------

# function to load a single file and return data and size
def LoadData(FileName, Transpose=False):
    """ Load Data from file

    Load prerecorded data array from a file

    :param: FileName - string filename
    :param: Transpose - Boolean statement defining whether the output array should be transposed

    :return: data - numpy array of data loaded from file
    :return: dataSize - tuple shape of data array
    """
    if (Transpose == True):
        data = pandas.read_csv(FileName, header=None).transpose()
    else:
        data = pandas.read_csv(FileName, header=None)

    data = data.to_numpy()
    dataSize = data.shape

    return data, dataSize

def ZoomData(arr, C):
    """ Cubic Spline Data

    Interpolate data to fit a specified length

    :param: arr - data array to be interpolated
    :param: C - array of specified interpolation length

    :return: arr - interpolated data array
    """
    zoom_rate = C.shape[0] / arr.shape[0]
    arr = zoom(arr, zoom_rate)
    return arr

# function to normalise data
def NormaliseData(inData, outData):
    """ Normalise array

    Normalise input and target data

    :param: inData - numpy array of input data
    :param: outData - numpy array of target data

    :return: inData - numpy array of normalised input data
    :return: outData - numpy array of normalised target data
    """
    inData = inData.astype("float32") / 255
    #outData = keras.utils.to_categorical(outData)

    if (inData.shape[0] != outData.shape[0]):
        outDataInterp = interp.interp1d(np.arange(outData.size), outData[:, 0])
        outData = outDataInterp(np.linspace(0, outData.size-1, inData[:, 0].size))
        outData = np.expand_dims(outData, -1)

    return inData, outData

def InterpData(inData, outData, tempData):
    """ Interpolate array

    Use interpolation to expand new data so that it is the same length as the rest of the data

    :param: inData - numpy array of input data
    :param: outData - numpy array of target data
    :param: tempData - numpy array of original data

    :return: inData - numpy array of interpolated input data
    :return: outData - numpy array of interpolated target data
    """

    inDatafinal = tempData
    for i in range(inData[0,:].size):
        inDataInterp = interp.interp1d(np.arange(inData[:, i].size), inData[:, i])
        inDatatest = inDataInterp(np.linspace(0, inData[:, i].size-1, tempData[:, 0].size))
        inDatafinal[:,i] = inDatatest

    inData = inDatafinal

    return inData, outData

# basic function to split test and train data 
def TestTrainGen_Basic(inData, outData):
    """ Test train split function

    Split input and target data into test and train arrays

    :param: inData - numpy array of input data
    :param: outData - numpy array of target data

    :return: inputTrain - numpy array of input data to be trained
    :return: targetTrain - numpy array of target data to be trained
    :return: inputTest - numpy array of input data to be tested
    :return: targetTest - numpy array of target data to be tested
    """
    inputTrain, inputTest, targetTrain, targetTest = train_test_split(
        inData, outData, random_state=42, test_size=0.2
    )
    
    trainData = np.append(inputTrain, targetTrain, axis=1)
    testData = np.append(inputTest, targetTest, axis=1)

    return inputTrain, targetTrain, inputTest, targetTest

# function to split test and train data by taking a snapshot at a random point
def TestTrainGen_Window(inData, outData, windowSize):
    """ Test train split with window function

    Split input and target data into test and train arrays using a rolling window

    :param: inData - numpy array of input data
    :param: outData - numpy array of target data

    :return: inputTrain - numpy array of input data to be trained
    :return: targetTrain - numpy array of target data to be trained
    :return: inputWindowed - numpy array of input data to be tested
    :return: outputWindowed - numpy array of target data to be tested
    """
    inputWindowed = np.array([inData[i:i+windowSize] for i in range(0,len(inData)-windowSize)]);
    outputWindowed = np.array([outData[i+windowSize] for i in range(0,len(outData)-windowSize)]);

    inputTrain, inputTest, targetTrain, targetTest = train_test_split(
       inputWindowed, outputWindowed, random_state=12, test_size=0.2
    )

    return inputTrain, targetTrain, inputWindowed, outputWindowed

