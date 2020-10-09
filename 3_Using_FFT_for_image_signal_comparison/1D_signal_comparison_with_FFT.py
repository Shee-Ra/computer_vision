# import functions
from src.functions_1D import LoadSensorData, Get1DFFT, Get1DLag, MakeSameSize
import numpy as np
import time

start=time.time()
# load files
signal_1='3_Using_FFT_for_image_signal_comparison/data/sensor1Data.txt'
signal_2='3_Using_FFT_for_image_signal_comparison/data/sensor2Data.txt'
s1, s2 = map(LoadSensorData, [signal_1,signal_2],[None,None])

# check size of s1, s2 is the same
s1, s2 = MakeSameSize(s1, s2)

# get lag
point_lag, time_lag, distance=Get1DLag(s1,s2)
print(f'distance between points is: {point_lag}, difference in time is: {time_lag}s, and the distance apart is: {distance}m')
end=time.time()
print(f'run time is {end-start}')