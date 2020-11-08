import numpy as np
import time
import logging
from PIL import Image
from matplotlib import image, pyplot
import os
from itertools import permutations
from src.utils import CalcNormedCrossCorr2D, ImageToArray, ConvertRGBAtoGreyscale, CheckImage, GetSearchAreaInPatternArray


logging.basicConfig(level=logging.INFO,     # set level
                    format='%(levelname)s '  # logging level
                    'on line %(lineno)d: '   # line ID
                    '%(message)s '           # message
                    )
log = logging.getLogger()


def main(small_image='2_2D_spatial_cross_correlation/data/wallypuzzle_rocket_man.png',
                big_image='2_2D_spatial_cross_correlation/data/wallypuzzle.png'):
    
    pattern_array = ImageToArray(small_image)  # Rocketman
    template_array = ImageToArray(big_image)   # Map

    # sample a row and pixel from rocketman
    while True:
        try: 
            selected_row, solid_row, count_of_transparent_pixels_to_left, selected_column, selected_pixel = GetSearchAreaInPatternArray(pattern_array)
        except:
            continue
        break    

    # begin searching maze for rocketman
    indices_of_selected_pixel_in_template = np.where(np.all(template_array == selected_pixel, axis=2))

    # Now for all (x,y's) in the above, get 2D arrays (rows,cols) in the maze and compare with the strip solid_row
    ccmax = 0
    for i in range(indices_of_selected_pixel_in_template[0].shape[0]):
        r = indices_of_selected_pixel_in_template[0][i]  # row in maze same colour as selected_pixel
        c = indices_of_selected_pixel_in_template[1][i]  # columns in maze same colour as selected_pixel

        # check if edges match
        left_edge = c-selected_column
        right_edge = left_edge+solid_row.shape[1]
        if (left_edge > 0 and right_edge > 0):
            pattern = template_array[r:r+1, left_edge:right_edge,:]  # take slice from map

            # compare edges in slice from map to edges from rocketman
            right_match = np.all(solid_row[:, -1, :] == pattern[:, -1, :])
            left_match = np.all(solid_row[:, 0, :] == pattern[:, 0, :])

            if (right_match & left_match):
                log.debug('edge match found')
                # extract 2D region from template
                r_bottom_maze = r - selected_row
                r_top_maze = r_bottom_maze + pattern_array.shape[0]
                c_left_maze = c-(count_of_transparent_pixels_to_left + selected_column)
                c_right_maze = c_left_maze + pattern_array.shape[1]

                # check image
                search_area = [i for i in permutations([-2, -1, 0, 1, -2], 2)]
                for indices in search_area:
                    gs_temp = ConvertRGBAtoGreyscale(template_array[r_bottom_maze + indices[0]:r_top_maze + indices[0], c_left_maze + indices[1]:c_right_maze + indices[1], :])
                    gs_pat = ConvertRGBAtoGreyscale(pattern_array)
                    cc = CalcNormedCrossCorr2D(gs_temp, gs_pat)
                    if ccmax < cc:
                        ccmax = cc
                        coords = (r_bottom_maze + indices[0], r_top_maze + indices[0], c_left_maze + indices[1], c_right_maze + indices[1])
                log.info(f'Rocketman found at lag {(coords[1], coords[2])}.\n bottom row: {coords[0]}, top row: {coords[1]}, col left: {coords[2]}, col right: {coords[3]}')
                return((coords[1], coords[3]))


if __name__ == "__main__":
    start = time.time()
    main(small_image='2_2D_spatial_cross_correlation/data/wallypuzzle_rocket_man.png',
                big_image='2_2D_spatial_cross_correlation/data/wallypuzzle.png')
    end = time.time()
    log.info(f'Time taken: {end-start}s')