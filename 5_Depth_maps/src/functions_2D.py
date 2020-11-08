import os
from pyprojroot import here
from PIL import Image
import numpy as np
import numpy.ma as ma
from scipy import signal
from scipy import ndimage, misc
import math
import logging
import time
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors


# sort out logging
logging.basicConfig(level=logging.INFO,     # set level
                    format='%(levelname)s '  # logging level
                    'on line %(lineno)d: '   # line ID
                    '%(message)s '           # message
                    )
log = logging.getLogger()

def ConvertRGBAtoGreyscale(im_array):
    """ Convert an array representing an RGB impage into greyscale  

    Args:
        im_array (numpy array): n x m x 4 array representing a 2D RGB image

    Returns:
        im_array (numpy array): converted to n x m array    
    """
    gs_array=np.dot(im_array[...,:3],[0.299, 0.587, 0.114])
    return(gs_array)

def load_image_pairs(pairs):
    """ Loads image pairs suffixed by _left and _right and returns greyscale

    Args:
        pairs (jpg): [description]

    Returns:
        left ,right (tuple): returns greyscale left / right image pairs read in by PIL
    """
    left = Image.open(pairs[0])
    right = Image.open(pairs[1])

    left = np.asarray(left)
    right = np.asarray(right)

    left = ConvertRGBAtoGreyscale(left)
    right = ConvertRGBAtoGreyscale(right)
    return left, right    

def SplitAxis(im, num_pixels=50, overlap=0, axis=0):
    """ Given an 2D array this function will split one of the axes into intervals of
        length num_pixels, and includes any overlaps.

    Args:
        im (numpy 2D array): numpy 2D array generated from an image
        num_pixels (int, optional): size axis is split into, e.g. 50 pixels. Defaults to 50.
        overlap (int, optional): window overlap for search stragegy. Defaults to 0.
        axis (int, optional): rows / y-axis (0) or columns / x-axis (1). Defaults to 0.

    Returns:
        list of tupples: list of tuples defining (start, end) for an interval of num_pixels long with 
                         overlap. 
    """

    try:
        assert overlap > 0 and overlap < 1 
    except AssertionError:
        log.info("overlap is zero, or must be a fraction between (0,1). Assume overlap = 0")
        a_overlap_shifts = np.array([0])
    else:
        a_overlap = math.floor(num_pixels * (1 - overlap))      # convert overlap in
        a_overlap_shifts = np.arange(0, num_pixels, a_overlap)  # convert into list for use in for loop 
        log.debug(f'x-overlaps: {a_overlap_shifts}')
    
    try:
        assert axis == 0 or axis == 1
    except AssertionError:
        log.info("axis value needs to be 0 or 1")
    else:
        a_max = im.shape[axis]
        log.debug(f'length of axis: {(a_max)}')

        a_splits = np.arange(0, num_pixels + a_max, num_pixels) # cols
        a_splits[-1] = a_max
        log.debug(f'x breaks: {a_splits}')
        
        a_breaks = []
        for i, a_pixel in enumerate(a_splits):
            for l, a_overlap in enumerate(a_overlap_shifts):
                try:
                    a_start = a_pixel + a_overlap
                    a_end = a_splits[i+1] + a_overlap
                    assert a_end <= a_max        
                except:
                    pass
                else:
                    a_breaks.append((a_start, a_end))
        log.debug(f'x_breaks: {a_breaks}')
        return(a_breaks) 

def GetWindows(im, window_size=50, overlap=0):
    """ Given an 2D array this function will split the array into
        windows of size 50x50, and includes any overlaps.

    Args:
        im (numpy 2D array): numpy 2D array generated from an image
        window_size (int, optional): window size image is split into, e.g. 50x50 pixels. Defaults to 50.
        overlap (int, optional): window overlap for search stragegy. Defaults to 0.

    Returns:
        row_breaks, col_breaks (tuple): list of row-col values which slice an array into search areas 
    """
    row_breaks = SplitAxis(im=im, num_pixels=window_size, overlap=overlap, axis=0)
    col_breaks = SplitAxis(im=im, num_pixels=window_size, overlap=overlap, axis=1)

    return row_breaks, col_breaks 


def Make2DArraySameSize(array1, array2):
    """ Makes two arrays the same size by left/bottom pading arrays with zeros

    Args:
        array1 (2D numpy array): two arrays to be resized
        array2 (2D numpy array): two arrays to be resized

    Returns:
        array1 (2D numpy array): resized
        array2 (2D numpy array): resized
    """
    '''given a small and big array, the small array is resized 
    to be the same size as the big array'''
    
    array2_pad_col = max(0,array1.shape[1] - array2.shape[1])
    array2_pad_row = max(0,array1.shape[0] - array2.shape[0])
    array2 = np.pad(array2, 
                   ((0,array2_pad_row), (array2_pad_col,0)), 
                   'constant', constant_values=((0,0),(0,0)))
    
    array1_pad_col = max(0,array2.shape[1] - array1.shape[1])
    array1_pad_row = max(0,array2.shape[0] - array1.shape[0])
    array1 = np.pad(array1, 
                   ((0, array1_pad_row), (array1_pad_col, 0)), 
                   'constant', constant_values=((0,0),(0,0)))

    return array1, array2


def Get2DLag(array1, array2):
    """ Inputs two 2D arrays and return lag'

    Args:
        array1 (2D numpy array): two arrays representing images
        array2 (2D numpy array): two arrays representing images

    Returns:
        tuple: row, column location of array2 in a array1
    """
    
    a1, a2=list(map(np.fft.fft2,[array1, array2]))
    a1_conj=a1.conjugate()
    norm=abs(a2*a1_conj)

    try:
        a1a2_inv = np.fft.ifft2((a2*a1_conj + 1) / (norm + 1)) # add one to denom/num to deal with zeros
        lag = np.unravel_index(a1a2_inv.argmax(), a1a2_inv.shape)
    except:
        lag = 0
             
    return(lag)   

def FindTemplate(pattern, template, row_offset=0, col_offset=0, use_built_in_method=True):
    """ Calculates a template in a pattern

    Args:
        pattern (2D numpy array): 2D array representing an image
        template (2D numpy array): 2D array representing an image
        row_offset (int, optional): [description]. Defaults to 0.
        col_offset (int, optional): [description]. Defaults to 0.
        use_built_in_method (bool, optional): [description]. Defaults to True.

    Returns:
        rr, cc: row, column location of template in pattern
    """
    # pattern, big
    # template, small

    if(use_built_in_method):
        # my method
        template, pattern = Make2DArraySameSize(pattern, template)
        rr, rc = Get2DLag(pattern, template)
        rr = rr + row_offset#row_search_coords[0]
        rc = rc + col_offset#col_search_coords[0]
    else:
        # use signal.correlate    
        conv = signal.correlate(pattern, template, method='fft', mode='same')
        rr, rc = np.where(conv == conv.max())            
        rr = rr[0] + row_offset#row_search_coords[0]
        rc = rc[0] + col_offset#col_search_coords[0]
        
    return rr, rc


def GetWindowSearchArea(im, rs, re, cs, ce, padding=0):
        """ Defines a row search area in an image 

        Args:
            im (2D numpy array): an array that represent image
            rs (int): defines the window, rs "row start" is the top row
            re (int): defines the window, re "row end" is the bottom row
            cs (int): defines the window, cs "col start" is the top col
            ce (int): defines the window, ce, "col end" is the bottom col
            grid_size (int, optional): builds n windows around the central window. Defaults to 3.

        Retuns: 
            row_search (tuple of ints): coords of row search area (start, end)
            col_search (tuple of ints): coords of col search area (start, end) 
        """
        # Test re > rs
        row_size = re - rs
        col_size = ce - cs

        col_padding = col_size * padding
        row_padding = row_size * padding

        row_search = (max(0, rs - row_padding), min(im.shape[0], re + row_padding))
        col_search = (max(0, cs - col_padding), min(im.shape[1], ce + col_padding))
        log.debug(f'search area cols: {col_search} search area rows: {row_search}')
            
        # search_area = im[row_search[0]:row_search[1], col_search[0]:col_search[1]]
        return row_search, col_search  

def GetFlatSearchArea(im, rs, re, cs, ce, padding=0):
        """defines a row search area in an image 

        Args:
            im (2D numpy array): an array that represent image
            rs (int): defines the window, rs "row start" is the top row
            re (int): defines the window, re "row end" is the bottom row
            cs (int): defines the window, cs "col start" is the top col
            ce (int): defines the window, ce, "col end" is the bottom col
            grid_size (int, optional): builds n windows around the central window. Defaults to 3.

        Retuns:
            row_search (tuple of ints): coords of row search area (start, end)
            col_search (tuple of ints): coords of col search area (start, end)  
        """
        # Test re > rs
        row_size = re - rs
        col_size = ce - cs

        padding_size = col_size * padding

        row_search = (rs, re)
        col_search = (max(0, cs - padding_size), min(im.shape[1], ce + padding_size))
        log.debug(f'right search area cols: {col_search} and search area rows: {row_search}')
            
        # search_area = im[row_search[0]:row_search[1], col_search[0]:col_search[1]]
        return row_search, col_search          

def GetDepthMap(left, right, windowsize, overlap, padding, windowsearch=True, use_my_lag_function=True):
    """ Returns a depth map

    Args:
        left (2D numpy array): template (smaller) image represented by a numpy array
        right (2D numpy array): pattern (bigger) image represented by a numpy array
        windowsize (int): grid search area in pixels to divide up left image
        overlap (float): between (0,9). will search grids overlap
        padding (int): size of window search, in units of windowsize
        windowsearch (bool, optional): search window square (True) or flat (False) . Defaults to True.
        use_my_lag_function (bool, optional): use homemade function (True) or scipy function (False). Defaults to True.

    Returns:
        lags: lag values mapped to a 2D region the same shape as left (template / small image)
    """
    left = np.pad(left, ((0,windowsize), (0,windowsize)), 'constant', constant_values=((0,0),(0,0))) # pad with zeros
    
    # break up template (left), return row/col breaks
    left_row_breaks, left_col_breaks = GetWindows(left, window_size=windowsize, overlap=overlap)
    
    lags_mapped_to_2D = np.ones(left.shape)
    lags_raw = np.zeros([len(left_row_breaks),len(left_col_breaks)])
    for i, lr in enumerate(left_row_breaks):     
        for j, lc in enumerate(left_col_breaks): 
            left_square = left[lr[0]:lr[1], lc[0]:lc[1]] # get window in template

            # execute search strategy in pattern (right / large image)
            if(windowsearch):
                row_search_coords, col_search_coords = GetWindowSearchArea(right, rs=lr[0], re=lr[1], cs=lc[0], ce=lc[1], padding=padding)
            else:
                row_search_coords, col_search_coords = GetFlatSearchArea(right, rs=lr[0], re=lr[1], cs=lc[0], ce=lc[1], padding=padding)    
            right_search_area = right[row_search_coords[0]:row_search_coords[1], col_search_coords[0]:col_search_coords[1]]

            # locate template in pattern
            rr, rc = FindTemplate(pattern=right_search_area, 
                                  template=left_square, 
                                  row_offset=row_search_coords[0], 
                                  col_offset=col_search_coords[0],
                                  use_built_in_method=use_my_lag_function)

            # calculate lag
            lag_col = ((lc[1] + lc[0]) / 2) - rc
            lag_row = ((lr[1] + lr[0]) / 2) - rr
            lag = np.sqrt(lag_col**2 + lag_row**2)
            lags_raw[i][j] = lag # collect lag values

            # map lag to 2D area the same size as pattern image
            try:
                lags_mapped_to_2D[left_row_breaks[i][0]:left_row_breaks[i+1][0],
                                  left_col_breaks[j][0]:left_col_breaks[j+1][0]] = lag
                
                # average any overlaps
                if i > 0 and j > 0:
                    lags_mapped_to_2D[left_row_breaks[i][0]:left_row_breaks[i-1][1],
                                      left_col_breaks[j][0]:left_col_breaks[j-1][1]] = (lags_raw[i][j] + lags_raw[i-1][j-1] + lags_raw[i][j-1] + lags_raw[i-1])/4                     
            except:
                lags_mapped_to_2D[left_row_breaks[i][0]:left_row_breaks[i][1],
                                  left_col_breaks[j][0]:left_col_breaks[j][1]] = lag 

    # cut off padding and return result
    return lags_mapped_to_2D[0:(left.shape[0]-windowsize),0:left.shape[1]-windowsize] 

         