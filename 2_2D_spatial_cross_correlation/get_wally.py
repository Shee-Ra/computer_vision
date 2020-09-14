import numpy as np
from pyprojroot import here
import time
import os
import logging
from PIL import Image
from matplotlib import image, pyplot

# ------------------------
# sort out logging
# ------------------------
logging.basicConfig(level=logging.DEBUG,     # set level
                    format='%(levelname)s '  # logging level
                    'on line %(lineno)d: '   # line ID
                    '%(message)s '           # message
                    )
log = logging.getLogger()

# ------------------------
# sort out working directory
# ------------------------
p=here()
os.chdir(p)
log.debug(f'current working directory is: {os.getcwd()}')

# ------------------------
# my functions
# ------------------------
def CalcNormedCrossCorr2D(pat, temp):
    '''calclates cross correlation of pat and temp
    This function assumes pat is a fixed size and temp is a sample <= pat as
    is the usecase we have here'''
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
    normd_cc=cc/norm
    return(normd_cc)

def ImageToArray(image_path):
    '''converts an image to a NumPy array'''
    image_=Image.open(image_path)
    image_array=np.asarray(image_)
    return(image_array)

def CheckImage(np_array):
    '''input an numpy array, and this returns an image, this function
    is used for manual checks throughout the code'''
    to_plot=Image.fromarray(np_array) # convert numpy array to image
    pyplot.imshow(to_plot) 
    pyplot.show()   

def GetLocation(small_image='2_2D_spatial_cross_correlation/data/wallypuzzle_rocket_man.png',
                big_image='2_2D_spatial_cross_correlation/data/wallypuzzle.png'):  
    rocketman_path=small_image
    map_path=big_image

    # ------------------------
    # load data --------------
    # ------------------------
    pattern_array, template_array=list(map(ImageToArray,[rocketman_path,map_path]))

    # -----------------------------------------------------------------
    # select a row / pixel in rocketman to anchor search --------------
    # -----------------------------------------------------------------
    
    # 1. Rocketman has transparency, so choose the longest row of solid pixels
    max_pixels=np.apply_along_axis(sum,0,pattern_array[:,:,3]).max()
    # 2. Select a solid row of pixels in rocketman
    selected_row=np.where(np.apply_along_axis(sum,0,pattern_array[:,:,3])==max_pixels)[0][0]
    solid_pixels=pattern_array[selected_row:selected_row+1,:,:][:,:,3][0]==255
    solid_row=pattern_array[selected_row:selected_row+1,solid_pixels,:]
    # 3. This counts the number of transparent pixels to the left of "solid_row", it comes in useful later
    count_of_transparent_pixels_to_left=np.argmax(solid_pixels)+1
    # 4. select random pixel in solid_row
    selected_column=np.random.random_integers(low=0, high=solid_row.shape[1]) # choose column
    selected_pixel=solid_row[:,selected_column:selected_column+1,:][0][0]     # 

    # -----------------------------------------------------------------
    # now begin search in maze ----------------------------------------
    # -----------------------------------------------------------------
    # find random pixel in the maze
    indices_of_selected_pixel_in_template = np.where(np.all(template_array == selected_pixel, axis=2))

    # Now for all (x,y's) in the above, get 2D arrays (rows,cols) in the maze and compare with the strip solid_row 
    for i in range(indices_of_selected_pixel_in_template[0].shape[0]):
        r=indices_of_selected_pixel_in_template[0][i] # row in maze same colour as selected_pixel
        c=indices_of_selected_pixel_in_template[1][i] # columns in maze same colour as selected_pixel
        
        # check if edges match
        left_edge=c-selected_column
        right_edge=left_edge+solid_row.shape[1]
        if (left_edge>0 and right_edge>0):
            pattern=template_array[r:r+1,left_edge:right_edge,:] # take slice from map
            
            # CheckImage(pattern)   # check image - this is manual for now
            # CheckImage(solid_row) # check image - this is manual for now
                
            # compare edges in slice from map to edges from rocketman
            right_match=np.all(solid_row[:,-1,:]==pattern[:,-1,:])
            left_match=np.all(solid_row[:,0,:]==pattern[:,0,:])

            if (right_match & left_match):
                log.debug('edge match found')
                # extract 2D region from template
                r_bottom_maze=r-selected_row
                r_top_maze=r_bottom_maze+pattern_array.shape[0]
                c_left_maze=c-(count_of_transparent_pixels_to_left+selected_column)
                c_right_maze=c_left_maze+pattern_array.shape[1]
                # check image
                log.info(f'Rocketman found at lag {(r_top_maze,c_left_maze)}.')
                # CheckImage(template_array[r_bottom_maze:r_top_maze,c_left_maze:c_right_maze,:])

start=time.time()
GetLocation(small_image='2_2D_spatial_cross_correlation/data/wallypuzzle_rocket_man.png',
                big_image='2_2D_spatial_cross_correlation/data/wallypuzzle.png')
end=time.time()
log.info(f'Time taken: {end-start}s')