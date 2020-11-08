from src.functions_2D import Get2DLag
from PIL import Image
import numpy as np
import time
import os

# load useful libraries from part 2
from shutil import copyfile
func_2D_src = os.path.join(os.getcwd(), '2_2D_spatial_cross_correlation','src','utils.py')
func_2D_dst = os.path.join(os.getcwd(), '3_Using_FFT_for_image_signal_comparison','src','utils_2D.py')
copyfile(src=func_2D_src, dst=func_2D_dst)
from src.utils_2D import ConvertRGBAtoGreyscale, Make2DArraySameSize

start = time.time()

# paths
data_2D = os.path.join(os.getcwd(), '2_2D_spatial_cross_correlation', 'data')
rocketman_path = os.path.join(data_2D, 'wallypuzzle_rocket_man.png')
map_path = os.path.join(data_2D, 'wallypuzzle.png')

# load images
pattern_image = Image.open(map_path)
template_image = Image.open(rocketman_path)

# convert image to an array
pattern_array = np.asarray(pattern_image)
template_array = np.asarray(template_image)

# convert to greyscale
gs_pattern_array = ConvertRGBAtoGreyscale(pattern_array)
gs_template_array = ConvertRGBAtoGreyscale(template_array)

# resize smaller image
pattern_array_resized, template_array_resized = Make2DArraySameSize(gs_pattern_array, gs_template_array)

# Get lag
lag = Get2DLag(template_array_resized, pattern_array_resized)
 
print(f'the lag is: {lag}')
end = time.time()
print(f'run time is {end-start}')