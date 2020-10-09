# import functions
from src.functions_2D import *
import numpy as np
import time

# sort out logging
logging.basicConfig(level=logging.INFO,     # set level
                    format='%(levelname)s '  # logging level
                    'on line %(lineno)d: '   # line ID
                    '%(message)s '           # message
                    )
log = logging.getLogger()

# sort out working directory
p=here()
os.chdir(p)
log.debug(f'current working directory is: {os.getcwd()}')

start=time.time()
rocketman_path='3_Using_FFT_for_image_signal_comparison/data/wallypuzzle_rocket_man.png'
map_path='3_Using_FFT_for_image_signal_comparison/data/wallypuzzle.png'

# load images
pattern_image, template_image=list(map(OpenImage,[rocketman_path,map_path]))

# convert image to an array
pattern_array, template_array=list(map(ImageToArray,[pattern_image, template_image]))

# convert to greyscale
gs_pattern_array, gs_template_array=list(map(ConvertRGBAtoGreyscale,[pattern_array, template_array]))

# resize smaller image
pattern_array_resized, template_array_resized=Make2DArraySameSize(gs_pattern_array, gs_template_array)

# Get lag
lag=Get2DLag(pattern_array_resized, template_array_resized)
 
print(f'the lag is: {lag}')
end=time.time()
print(f'run time is {end-start}')