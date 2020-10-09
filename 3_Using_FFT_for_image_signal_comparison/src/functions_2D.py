# libraries
import logging
import os
from pyprojroot import here
from PIL import Image
import numpy as np

class ImageOperators:
    def __init__(self):
        self




# 2D functions

# read in images
def OpenImage(image_path):
    im=Image.open(image_path)
    return(im)

def ImageToArray(im):
    '''converts an image to a NumPy array'''
    image_array=np.asarray(im)
    return(image_array)

def ConvertRGBAtoGreyscale(im_array):
    '''converts an RGBA array into greyscale array'''
    gs_array=np.dot(im_array[...,:3],[0.299, 0.587, 0.114])
    return(gs_array)

def Make2DArraySameSize(array1,array2):
    '''given a small and big array, the small array is resized 
    to be the same size as the big array'''
    # if not 2D make it break
    diff_x=array1.shape[0]-array2.shape[0]
    diff_y=array1.shape[1]-array2.shape[1]

    # which is the big shape
    if (diff_x<0 and diff_y<0):
        # array2 is the big shape
        reshaped_array = np.zeros(array2.shape)
        small_array=array1
        big_array=array2
    elif (diff_x>0 and diff_y>0):  
        # array2 is the big shape
        reshaped_array = np.zeros(array1.shape) 
        small_array=array2
        big_array=array1
    else:
        # something should break
        pass

    reshaped_array[:small_array.shape[0],:small_array.shape[1]]=small_array
    
    return(reshaped_array,big_array)

def CheckImage(np_array):
    '''input an numpy array, and this returns an image, this function
    is used for manual checks throughout the code'''
    import matplotlib.pyplot as plt
    to_plot=Image.fromarray(np_array) # convert numpy array to image
    plt.imshow(to_plot) 
    plt.show()  

def Get2DLag(array1, array2):
    '''input two 2D arrays and return lag'''
    # step 1: get FFT
    a1, a2=list(map(np.fft.fft2,[array1, array2]))

    # step 2: conjugate one of the signals
    a1_conj=a1.conjugate()
    
    # step 3: get lag by taking inverse FFT of s1_conj*s2
    norm=abs(a2*a1_conj)
    a1a2_inv=np.fft.ifft2(a2*a1_conj/norm)
    lag=np.unravel_index(a1a2_inv.argmax(), a1a2_inv.shape)
    
    # step 4: lag is expressed from the top/left or bottom/right of big image
    if (array1[55,50]==array2[lag[0]+55,lag[1]+50]):
        return(lag)
    elif (array2[55,50]==array1[-lag[0]+array1.shape[0]+55,-lag[1]+array1.shape[1]+50]):
        adjusted_lag=(array1.shape[0]-lag[0],array1.shape[1]-lag[1])
        return(adjusted_lag)
    else:
        # no match found
        return(np.nan)
     