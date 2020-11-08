from src.functions_2D import *
from PIL import Image
import numpy as np
import time
import logging
from scipy import signal
from scipy import ndimage, misc
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
from scipy.ndimage import gaussian_filter
import numpy.ma as ma

def main(win=30, 
         o=0.50, 
         pad=0, 
         wss=False, 
         my_lag=True):
    filenames=os.listdir('5_Depth_maps/data/')
    jpgs = [x for x in filenames if '.jpg' in x]
    file_pairs = []

    for x in jpgs:
        name = x.split('_')[0]
        if x.split('_')[1] == 'left.jpg':
            file_pairs.append(('5_Depth_maps/data/'+x, '5_Depth_maps/data/'+name+'_right.jpg'))            

    ims = ['giraffe', 'block', 'rubiks']
    for k, p in enumerate(file_pairs):
        left, right = load_image_pairs(p)
        log.debug('image pair loaded')
        
        try:
        # if True:
            lags_mapped_to_2D = GetDepthMap(left, 
                                            right, 
                                            windowsize=win, 
                                            overlap=o, 
                                            padding=pad, 
                                            windowsearch=wss, 
                                            use_my_lag_function=my_lag)

            # save file
            filename=f'5_Depth_maps/data/results/{ims[k]}_mylagfnc_{my_lag}_window_{int(win)}px_overlap_{int(o*100)}pc_padding_{int(pad)}sq.pdf'
            cmap = plt.cm.jet
            image = cmap(lags_mapped_to_2D)
            plt.imshow(lags_mapped_to_2D)
            plt.colorbar()
            plt.savefig(filename)
            plt.clf()
            print(f'output saved to {filename}')

        except:
            print(f'error, likely with overlap, try a smaller overlap')
            pass   

if __name__ == "__main__":
    start = time.time()
    main(win=45, 
         o=0.90, 
         pad=0, 
         wss=False, 
         my_lag=True)
    end = time.time()
    log.info(f'Time taken: {end-start}s')