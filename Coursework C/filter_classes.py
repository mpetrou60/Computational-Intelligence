##############################################################################################################
#   Project:        Bladder Pressure Prediction
#   Authors:        Maria N. Petrou
#   Description:    Library containing all custom filter classes used in this project.
##############################################################################################################

# --- Import packages ----------------------------------------------------------------------------------------

import numpy as np
from scipy.signal import butter, filtfilt
import pandas
import math

# --- Basic Anomaly Detection --------------------------------------------------------------------------------

class AnomalyDetectionDelay:
    """Basic Anomaly Detection class with internal buffer"""

    def __init__(self, buffer_length):
        """Basic Anomaly Detection class with internal buffer

        A basic Anomaly Detection method with internal buffer to act as delay for data to settle.

        :param: buffer_length - the length of the buffer

        :return: AnomalyDetectionDelay class instance
        """
        # Define class parameters
        self.buffer_length = buffer_length
        self.currentNum = 0;

        # Make buffer
        self.buffer = np.zeros(buffer_length)
    
    def clear_buffer(self):
        """Clears interal buffer

        Clears an internal buffer.
        """
        self.buffer = np.zeros(self.buffer_length)

    def check_data(self, data):
        """Input data point to buffer of Anomaly Detection Class

        A data point is added into the buffer of the rolling average example class, using a shift register design.

        :param: data - the point to be added to the buffer

        """
        #1 - CHECK
        #2 - ADD TO BUFFER
        #3 - RETURN CHECK RESULT FOR USE

        self.currentNum = min(self.currentNum + 1, self.buffer_length)

        # Check data
        if self.currentNum >= 4: #should this be (buffer_length - 1)
            data = self.detector(data)
        
        # Shift buffer along
        self.buffer[1:] = self.buffer[0:-1]

        # Store the new input data
        self.buffer[0] = data

    def detector(self, data):
        """ Detect any outliers

        Detect any outliers in input data and replace with mean value
        """
        # detect anomalies
        sd = np.std(self.buffer)
        mean = np.mean(self.buffer)
        uT = mean + 3.5 * sd
        lT = mean - 3.5 * sd
        if data > uT or data < lT:
                data = mean

        return data

# --- Efficient Anomaly Detection ----------------------------------------------------------------------------

class AnomalyDetectionEfficient:
    """Efficient Anomaly Detection class without internal buffer"""

    def __init__(self, numData):
        """Efficient Anomaly Detection class without internal buffer

        An efficient Anomaly Detection class that uses approximations of mean and variance to calculate anomalies

        :param: numData - the number of data points the detector is averaging

        :return: AnomalyDetectionEfficient class instance
        """
        # Define class parameters
        self.numData = numData
        self.currentNum = 0
        self.Total = 0
        self.Mean = 0
        self.Var = 0

    def check_data(self, data):
        """Input data point to buffer of Anomaly Detection Class

        A data point is added into the buffer of the rolling average example class, using a shift register design.

        :param: data - the point to be added to the buffer

        """
        # Set data
        
        self.currentNum = min(self.currentNum + 1, self.numData)

        if self.currentNum >= 4:
            data = self.detector(data)

        self.Total = self.Total + data - self.Mean
        
        if self.currentNum >= 3:
            self.Var = (self.currentNum - 2)/(self.currentNum - 1)*self.Var + ((data - self.Mean)**2)/(self.currentNum-1)
        
        if self.currentNum >= 2:
            self.Mean = self.Total / self.currentNum

        return data

    def detector(self, data):
        """ Detect any outliers

        Detect any outliers in input data and replace with mean value
        """
        # detect anomalies
        sd = math.sqrt(self.Var)

        uT = self.Mean + 3.5 * sd
        lT = self.Mean - 3.5 * sd
        print('Data:',data,'uT:',uT,'lT:',lT,'Mean:',self.Mean,'Var:',self.Var)
        if data > uT or data < lT:
            return self.Mean
        return data

# --- Low Pass Filter ----------------------------------------------------------------------------------------

class LowPassFilter:
    """Butterworth Lowpass Filter Class with internal buffer"""

    def __init__(self, buffer_length, order, Wn):
        """Butterworth Lowpass Filter Class with interal buffer

        A butterworth lowpass filter class, with an internal buffer.

        :param: buffer_length - the length of the buffer
        :param: Wn - The critical frequency. For a Butterworth filter, this is the point at which the gain drops 
               to 1/sqrt(2) that of the passband (the “-3 dB point”)
        :param: order - integer, filter order

        :return: LowPassFilter class instance
        """
        # Define class parameters
        self.buffer_length = buffer_length
        self.order = order
        self.Wn = Wn

        # Make buffer
        self.buffer = np.zeros(buffer_length)
    
    def clear_buffer(self):
        """Clears interal buffer

        Clears an internal buffer.
        """
        self.buffer = np.zeros(self.buffer_length)

    def input_data(self, data):
        """Input data point to buffer of Lowpass Filter Class

        A data point is added into the buffer of the rolling average example class, using a shift register design.

        :param: data - the point to be added to the buffer

        """
        # Shift buffer along
        self.buffer[1:] = self.buffer[0:-1]

        # Store the new input data
        self.buffer[0] = data

    def butterFilter(self):
        """apply the lowpass filter

        Perform the butterworth lowpass filter calculation to the class.

        :return: filteredData - Rolling average class instance
        """
        # apply low pass filter to data
        b, a = butter(self.order,self.Wn, btype='low')
        filteredData = filtfilt(b, a, self.buffer, padlen=0)

        return filteredData

