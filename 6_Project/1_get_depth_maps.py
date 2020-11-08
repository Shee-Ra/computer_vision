import cv2
from PIL import Image
import numpy as np
import os
from matplotlib import pyplot as plt
from src.utils import *
from tqdm import tqdm

processed_frame_path = os.path.join(os.getcwd(), '6_Project/data/processed/')
allfiles = sorted(os.listdir(processed_frame_path))
results_path = os.path.join(os.getcwd(), '6_Project/data/results/')

file_pairs = []
for jpg in allfiles[1:]:
    name = jpg.split('_')[0] + '_' + jpg.split('_')[1]
    if jpg.split('_')[2] == 'left.jpg':
        file_pairs.append((processed_frame_path + jpg, processed_frame_path + name + '_right.jpg'))

for k, p in enumerate(tqdm(file_pairs)):
    left, right = load_image_pairs(p)
    filename = p[0].split('/')[-1][0:-9] + '.jpg'
    full_path = os.path.join(results_path, filename)

    results_so_far = os.listdir(results_path)
    if filename in results_so_far:
      continue
    else:
      while_loop_go = True
      try_overlap = 0.95
      while while_loop_go:
        try:
      # if True:
          lags_mapped_to_2D = GetDepthMap(left, 
                                          right, 
                                          windowsize=20,
                                          overlap=try_overlap,
                                          padding=0,
                                          windowsearch=True,
                                          use_my_lag_function=True)

          # save file
          cmap = plt.cm.jet
          image = cmap(lags_mapped_to_2D)
          plt.imshow(lags_mapped_to_2D)
          plt.savefig(full_path)
          plt.clf()
          print(f'{filename} saved')

          while_loop_go = False

        except:
          f = open(os.path.join(results_path, "fails.txt"),"a+")
          f.write(f'{filename} fails for overlap {try_overlap}\n')
          f.close() 

          try_overlap = try_overlap - 0.05
          if try_overlap < 0.5:
            while_loop_go= False
          print(f'hyperparameter causes error (likely overlap is too large)')
          continue
