##############################################################################################################
#   Project:        Bladder Pressure Prediction
#   Authors:        Maria N. Petrou
#   Description:    Library containing all custom filter functions used in this project.
##############################################################################################################

# --- Import packages ----------------------------------------------------------------------------------------

import numpy as np
from scipy.signal import butter, filtfilt, lfilter
from scipy.fft import fft, ifft
import pandas

# --- Filter Functions from FYP report ------------------------------------------------------------------------

# function that implements the standard anomaly detection and removal method
def AnomalyDetection(data):
    """ Detect any outliers

    Detect any outliers in input data and replace with mean value

    :param: data - array of input points to be checked for anomalies

    :return: data - array of output points where anomalous points have been replaced with averaged values
    
    >>> AnomalyDetection([5,6,7,50,9])
    [5,6,7,15.4,9]

    """
    columns = data[:, 0];
    rows = data[0, :];

    for i in range(columns):
        sd = np.std(data[:,i])
        mean = np.mean(data[:,i])
        uT = mean + 3.5 * sd
        lT = mean - 3.5 * sd
        for j in range(rows):
            if data[j][i] > uT or data [j][i] < lT:
                data[j][i] = mean

    return data

# efficient function for getting windows of an array
def RollingWindow(data, window):
    """ a rolling window function

    Perform a Rolling Window on input data points

    :param: data - array input points to have windowed
    :param: window - integer window width

    :return: np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides - Create a view into the array with the given shape and strides
    """
    shape = data.shape[:-1] + (data.shape[-1] - window + 1, window)
    strides = data.strides + (data.strides[-1],)
    return np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)

# function to align signals
def AlignSignals(arr1, arr2):
    """ Function to align two input arrays

    Aligns two input arrays such that the last element of arr2 should align with the element in arr1[maxIndex]

    :param: arr1 - first array of two to be aligned against arr2
    :param: arr2 - second array of two to be aligned against arr1

    :return: arr1 - aligned array relative to arr2, any misalingment will have been fixed through concatenation
    :return: arr2 - aligned array relative to arr1, any misalingment will have been fixed through concatenation
    """
    assert(isinstance(arr1, (np.ndarray,list)))
    assert(isinstance(arr2, (np.ndarray,list)))
    len1 = len(arr1)
    len2 = len(arr2)
    corr = np.correlate(arr1,arr2,"full")
    # The last element of arr2 should align with the element in arr1[maxIndex]
    maxIndex = np.argmax(corr)
    # concatenate with zeros to ensure both the returned arrays are of the same length
    arr1 = np.concatenate((np.zeros(max(len2-maxIndex-1,0)),arr1))
    arr2 = np.concatenate((np.zeros(max(maxIndex+1-len2,0)),arr2))
    arr1 = np.concatenate((arr1, (np.zeros(max(maxIndex-len1+1,0)))))
    arr2 = np.concatenate((arr2, (np.zeros(max(maxIndex+1-len2,0)))))
    return arr1,arr2

# --- Dr Metcalfe's Filter Functions ------------------------------------------------------------------------

# function to downsample data
def DownsampleData(data, downsamplingFactor):
    """ Function to downsample input data

    Downsamples input data by a predefined downsampling factor

    :param: data - array of input data points to be decimated
    :param: downsamplingFactor - integer that defines how much data should be downsampled by

    :return: downsampledData - output array that has been downsampled

    >>> DownsampleData([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 2)
    [1, 3, 5, 6, 8, 10]
    """
    downsampledData = scipy.signal.decimate(data, downsamplingFactor)
    return downsampledData

# function that applies a bandpass filter in the frequency domain
def BandpassFilter(data):
    """ Bandpass Filter Function

    passes frequencies within a certain range and rejects (attenuates) frequencies outside that range

    :param: data - array to be filtered

    :return: data - bandpass filtered array of limited frequency range
    """
    nt = data.shape(1)
    fmin = 10
    fmax = 10e3
    f = (np.arange(-nt/2, (nt/2-1)+1, 1))/nt*(1/dt)

    dft = np.fft.fftfreq(fft(data), 1)
    dfiltft = np.zeros(dft.shape)
    I = np.argwhere(f>fmin & f<fmax)
    dfiltft[I, :] = dft[I, :] * np.matlib.repmat(scipy.signal.tukey(len(I),0.1),1,nu);
    I = np.argwhere( f<-fmin & f>-fmax);
    dfiltft[I, :] = dft[I, :] * np.matlib.repmat(scipy.signal.tukey(len(I),0.1),1,nu);
    dft = dfiltft;
    data = ifft(np.fft.ifftfreq(dft,1))
    data = real(data)
    return data

# function to find the moving variance of input data
def MovingVariance(data):
    """ Moving Variance Function

    Moving Variance computes the moving variance of the input signal along each channel independently over time. 
    In the sliding window method, a window of specified length moves over the data sample by sample, and the block 
    computes the variance over the data in the window.

    :param: data - array to be windowed

    :return: dataVar - array of variance over the data in the window
    """
    win = 200e-3
    dt = 1/500000
    nData = pandas.Series(data)
    dataVar = nData.rolling(win/dt, center=True, min_periods=1).var(ddof=0)
    return dataVar

# function for a butterworth lowpass filter
def LowButterFilter(data, Wn, order=2):
    """ Lowpass Butterworth Filter Function

    The lowpass filter is a filter that allows the signal with the frequency is lower than the cutoff frequency 
    and attenuates the signals with the frequency is more than cutoff frequency.
    The Butterworth filter is a type of signal processing filter designed to have a frequency response that is 
    as flat as possible in the passband.

    :param: data - array to be filtered
    :param: Wn - The critical frequency. For a Butterworth filter, this is the point at which the gain drops 
               to 1/sqrt(2) that of the passband (the “-3 dB point”)
    :param: order - integer, filter order

    :return: filteredData - array of lowpass filtered data

    >>> LowButterFilter([0, 1, 2, 3, 4, 5, 6, 7, 6, 5, 4, 3, 2, 1, 0], 0.1)
    [0.21436272 1.17059835 2.07872186 2.89924046 3.59231001 4.11974146 4.44828138 4.55453443 4.43057967 
    4.08671466 3.5488775  2.85317196 2.04103539 1.15596834 0.24142538]
    """

    b, a = butter(order,Wn, btype='low')
    filteredData = filtfilt(b, a, data, padlen=0)

    return filteredData

# ---Filter Functions -------------------------------------------------------------------------------------

def RollingAnomalyDetection(dataNew, dataOld):
    """ Detect any outliers

    Detect any outliers in input data and replace with mean value

    :param: dataNew - new integer data point introduced to system
    :param: dataOld - the previous 2 data points stored in memory

    :return: dataArray - new integer data point, or mean value if defined as outlier

    >>> RollingAnomalyDetection(5, np.array([3,4])
    np.array([5,3,4])

    >>> RollingAnomalyDetection(90, np.array([3,4])
    np.array([17,3,4])
    """
    dataArray = np.insert(dataOld, 0, dataNew, axis=0)

    sd = np.std(dataArray)
    mean = np.mean(dataArray)
    # 3.5 doesn't give great sensitivity, but I don't know if that's just because the dataset is so small
    upperLimit = mean + 3.5 * sd 
    lowerLimit = mean - 3.5 * sd

    # Only change the newest data input
    if dataArray[0] > upperLimit or dataArray[0] < lowerLimit:
       dataArray[0] = mean

    # DELETE LATER
    print("Upper Limit: ", upperLimit)
    print("Lower Limit: ", lowerLimit)
    print("Mean: ", mean)
    print("Standard Deviation: ", sd)

    return dataArray

def RollingLowButterFilter(newdata, Wn, order=2):
    """ Lowpass Butterworth Filter Function

    The lowpass filter is a filter that allows the signal with the frequency is lower than the cutoff frequency 
    and attenuates the signals with the frequency is more than cutoff frequency.
    The Butterworth filter is a type of signal processing filter designed to have a frequency response that is 
    as flat as possible in the passband.

    :param: data - array to be filtered
    :param: Wn - The critical frequency. For a Butterworth filter, this is the point at which the gain drops 
               to 1/sqrt(2) that of the passband (the “-3 dB point”)
    :param: order - integer, filter order

    :return: filteredData - array of lowpass filtered data

    >>> RollingLowButterFilter([0, 1, 2], 0.1)
    array([0.10288995, 0.10984431, 0.11168396])
    """

    #b, a = butter(order,Wn, btype='low')
    #filteredData = filtfilt(b, a, data, padlen=0)



    filteredData = LowButterFilter(data, Wn)

    return filteredData

if __name__ == '__main__':
    import doctest
    doctest.testmod()

