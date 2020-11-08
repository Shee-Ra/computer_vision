import os
from PIL import Image
import numpy as np 

def Get2DLag(array1, array2): # template, pattern
    '''input two 2D arrays and return lag'''
    # step 1: get FFT
    a1, a2 = list(map(np.fft.fft2, [array1, array2]))

    # step 2: conjugate one of the signals
    a1_conj = a1.conjugate()
    
    # step 3: get lag by taking inverse FFT of s1_conj*s2
    norm = abs(a1_conj * a2)
    a1a2_inv = np.fft.ifft2(a2 * a1_conj / norm)
    lag = np.unravel_index(a1a2_inv.argmax(), a1a2_inv.shape)
    
    return lag
     