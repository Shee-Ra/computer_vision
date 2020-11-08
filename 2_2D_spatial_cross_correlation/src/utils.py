from PIL import Image
import numpy as np
import logging

logging.basicConfig(level=logging.INFO,     # set level
                    format='%(levelname)s '  # logging level
                    'on line %(lineno)d: '   # line ID
                    '%(message)s '           # message
                    )
log = logging.getLogger()


def CalcNormedCrossCorr2D(pat, temp):
    """calclates cross correlation of pat and temp
    This function assumes pat is a fixed size and temp is a sample <= pat as
    is the usecase we have here

    Args:
        pat (2D numpy array): array representation of an image
        temp (2D numpy array): array representation of an image

    Returns:
        normalised cross correlation (float)    
    """
    # handle egde case when lengths of temp does not equal pat
    dim_diff=abs(np.subtract(pat.shape, temp.shape)) # get difference in dimensions
    log.debug(f'dimension difference between pat and temp: {dim_diff}')

    if dim_diff.sum()>0:
        log.debug(f'before padding temp has dimensions: {temp.shape}')
        npad=((dim_diff[0],0), (dim_diff[1],0))                              # new dimension
        temp=np.pad(temp, pad_width=npad, mode='constant',constant_values=0) # padding
        log.debug(f'temp now has dimensions: {temp.shape}')
    else:
        pass

    # calulate normalised cross correlation
    norm=np.sqrt(((pat**2).sum()) * ((temp**2).sum()))
    log.debug(f'temp and pat dimensions: {temp.shape, pat.shape}')
    cc=(pat*temp).sum()
    return cc=cc/norm


def ImageToArray(image_path):
    """converts an image to a NumPy array

    Args:
        image_path (str): path to image files

    Returns:
        (2D numpy array): numpy array of image
    """
    image_=Image.open(image_path)
    return np.asarray(image_)


def CheckImage(np_array):
    """shows an image, given a 2D array

    Args:
        np_array (numpy array)
    """
    to_plot=Image.fromarray(np_array) # convert numpy array to image
    pyplot.imshow(to_plot) 
    pyplot.show() 


def ConvertRGBAtoGreyscale(im_array):
    """converts an RGBA array into greyscale array

    Args:
        im_array (3D numpy array): numpy array representing an image in RGB

    Returns:  
        gs_array (2D numpy array): greyscale'd version of the image.
    """

    '''converts an RGBA array into greyscale array'''
    return np.dot(im_array[...,:3],[0.299, 0.587, 0.114])

def GetSearchAreaInPatternArray(pattern_array):
    """returns features from pattern_array. These are searched for in the template

    Args:
        pattern_array (2D numpy array): pattern array of template (small image)

    Returns:
        selected_row (numpy array): row from template, 
        solid_row (numpy array): the solid rows in selected_row, 
        count_of_transparent_pixels_to_left (numpy array): used to find pattern, 
        selected_column (numpy array): selects column in solid_row, used to get selected_pixel, 
        selected_pixel (numpy array): used to search for in the maze  
    """

    # 1. Rocketman has transparency, so choose the longest row of solid pixels
    max_pixels = np.apply_along_axis(sum, 0, pattern_array[:, :, 3]).max()

    # 2. Select a solid row of pixels in rocketman
    selected_row = np.where(np.apply_along_axis(sum, 0, pattern_array[:,:,3]) == max_pixels)[0][0]
    solid_pixels = pattern_array[selected_row:selected_row+1,:,:][:,:,3][0] == 255
    solid_row = pattern_array[selected_row:selected_row+1, solid_pixels, :]

    # 3. This counts the number of transparent pixels to the left of "solid_row", it comes in useful later
    count_of_transparent_pixels_to_left = np.argmax(solid_pixels)+1

    # 4. select random pixel in solid_row
    selected_column = np.random.random_integers(low=0, high=solid_row.shape[1]) # choose column
    selected_pixel = solid_row[:,selected_column:selected_column+1,:][0][0]     #

    return selected_row, solid_row, count_of_transparent_pixels_to_left, selected_column, selected_pixel    