import numpy as np
import os
import logging
from math import floor
import time
import matplotlib.pyplot as plt
from src.utils import LoadSensorData, TakeTimeSnapshot, CalcNormedCrossCorr


def main(main_folder, print_results=True):
    startTime = time.time()
    logging.basicConfig(level=logging.INFO,
                        format='%(levelname)s '
                        'on line %(lineno)d: '
                        '%(message)s ')
    log = logging.getLogger()

    p1 = os.path.join(main_folder, '1_1D_spatial_cross_correlation/data/sensor1Data.txt')
    p2 = os.path.join(main_folder, '1_1D_spatial_cross_correlation/data/sensor2Data.txt')
    s1 = LoadSensorData(p1)
    s2 = LoadSensorData(p2)

    # Take a random sample of points in s1 (5% of the sample), and find matching points in s2
    s1_sample = np.random.choice(range(s1.shape[0]), floor(s1.shape[0]*0.005), replace=False)  # take a sample from s1
    s1_matches = s1_sample[np.array([s1[s] in s2 for s in s1_sample])]  # find where s2 matches sample in s1
    log.debug(f'matches found: {s1_matches.shape[0]}')

    # Calculate cross correlation for sets of matching points in s1 & s2
    index_of_s1_matches = np.empty(0)
    index_of_s2_matches = np.empty(0)
    cc_s1_s2 = np.empty(0)

    for match in s1_matches:
        s2_matches = np.where(s2 == s1[match]) # index where s2 matches samples from s1 (this can be many points)

        for s2_match in s2_matches[0]:       # just select one point

            # get cross correlation
            samp1 = TakeTimeSnapshot(s1, match, 100)
            samp2 = TakeTimeSnapshot(s2, s2_match, 100)
            cc = CalcNormedCrossCorr(samp1, samp2)

            # save results
            index_of_s1_matches = np.append(index_of_s1_matches, match)
            index_of_s2_matches = np.append(index_of_s2_matches, s2_match)
            cc_s1_s2 = np.append(cc_s1_s2, cc)

    ccmax = cc_s1_s2.max()
    lags = index_of_s2_matches[cc_s1_s2 == ccmax] - index_of_s1_matches[cc_s1_s2 == ccmax]
    lag_vals = np.unique(lags, return_counts=True)
    max_freq_val = lag_vals[1].max()
    lag_max_index = np.where(lag_vals[1] == max_freq_val)[0][0]
    lag = lag_vals[0][lag_max_index]

    endTime = time.time()

    # report results
    if(print_results):
        log.info(f'The lag between signals is: {abs(lag)} points, which corresponds to {abs((lag*333)/44100)} m')
        log.info(f'sample of points used to confirm where s1=s2: {"%.2f" % (max_freq_val/s1.shape[0]*100)} % ({max_freq_val} points)')
        log.info(f'time taken: {"%.2f" % (endTime-startTime)} s')

    return endTime-startTime, round(abs((lag*333)/44100), 2)


if __name__ == "__main__":
    main_folder = './'
    main(main_folder)
