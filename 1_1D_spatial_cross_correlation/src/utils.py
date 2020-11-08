import numpy as np


def LoadSensorData(path=''):
    """Load specific sensor data for this project

    Args:
        path (str, optional): Path to the sensor data. Defaults to ''.

    Returns:
        array (numpy array): sensor data
    """
    return np.loadtxt(path, skiprows=1, delimiter="\n", unpack=False)


def TakeTimeSnapshot(data_1D, start=0, n=100):
    """ sample n points from signal data by slicing numpy array

    Args:
        data_1D (numpy array): input a 1D numpy array
        start (int, optional): start position of slice. Defaults to 0.
        n (int, optional): size of slice. Defaults to 100.
    """
    try:
        snapshot = data_1D[start:start + n]
    except:
        print('start value or length out of bounds of input signal')
    else:
        return snapshot


def Make1DSignalSameLength(pat, temp):
    """makes two 1D signals pat & temp the same size

    Args:
        pat (numpy array): 1D numpy array
        temp (numpy array): 1D numpy array

    Returns:
        pat, temp (tuple of two arrays): returns two signals the same size
    """
    lp, lt = [vec.shape[0] for vec in [pat, temp]]  # get length of inputs
    if lp > lt:
        temp = np.pad(temp, (lp - lt, 0), mode='constant', constant_values=(0, 0))
    elif lp < lt:
        pat = np.pad(pat, (0, lt - lp), mode='constant', constant_values=(0, 0))
    else:
        pass

    return pat, temp


def CalcNormedCrossCorr(pat, temp):
    """calculates cross correlation to two 1D signals

    Args:
        pat (numpy array): 1D numpy array
        temp (numpy array): 1D numpy array

    Returns:
        normalised cross correlation [float]: cross correlation of pat and temp.
    """
    pat, temp = Make1DSignalSameLength(pat, temp)

    # calulate normalised cross correlation
    norm = np.sqrt(((pat**2).sum()) * ((temp**2).sum()))
    cc = (pat * temp).sum()
    return cc / norm


