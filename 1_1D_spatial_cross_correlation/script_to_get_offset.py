import numpy as np
from pyprojroot import here
import os
import logging
from math import floor
import time

startTime=time.time()
# ------------------------
# functions
# ------------------------
def LoadSensorData(path=''):
    '''Load sensor data into numpy array'''
    sensor_data=np.loadtxt(path, skiprows=1, delimiter="\n", unpack=False)
    return(sensor_data)  # assumes files take same form every time

def TakeTimeSnapshot(data_1D, start=0, n=100):
    '''sample n points from signal data'''

    # test: make sure start and n are in bounds
    snapshot=data_1D[start:start+n]
    return(snapshot)

def CalcNormedCrossCorr(pat, temp):
    '''calclates cross correlation of pat and temp'''
    # handle egde case when lengths of pat and temp are unequal
    lp, lt=[vec.shape[0] for vec in [pat,temp]] # get length of inputs
    if lp>lt:
        temp=np.pad(temp,(lp-lt,0),mode='constant',constant_values=(0,0))
    elif lp<lt:
        pat=np.pad(pat,(0,lt-lp),mode='constant',constant_values=(0,0))
    else:
        pass

    # calulate normalised cross correlation
    norm=np.sqrt(((pat**2).sum()) * ((temp**2).sum()))
    cc=(pat*temp).sum()
    normd_cc=cc/norm
    return(normd_cc) 

# ------------------------
# sort out logging
# ------------------------
logging.basicConfig(level=logging.INFO,     # set level
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
# load data
# ------------------------
p1=here('./1_1D_spatial_cross_correlation/data/sensor1Data.txt') # paths to sensor data
p2=here('./1_1D_spatial_cross_correlation/data/sensor2Data.txt')
s1, s2 =list(map(LoadSensorData,[p1,p2]))                        # load data
log.debug(f'current working directory is: {os.getcwd()}')

# ------------------------
# Take a random sample of points in s1 (5% of the sample), and find matching points in s2
# ------------------------
s1_sample=np.random.choice(range(s1.shape[0]), floor(s1.shape[0]*0.005), replace=False) # take a sample from s1
s1_matches=s1_sample[np.array([s1[s] in s2 for s in s1_sample])]   # find where s2 matches sample in s1
log.debug(f'matches found: {s1_matches.shape[0]}')

# ------------------------
# Calculate cross correlation for sets of matching points in s1 & s2
# ------------------------
index_of_s1_matches=np.empty(0)
index_of_s2_matches=np.empty(0)
cc_s1_s2=np.empty(0)

for match in s1_matches:
    s2_matches=np.where(s2==s1[match]) # index where s2 matches samples from s1 (this can be many points)
    for s2_match in s2_matches[0]:

        # get cross correlation
        samp1=TakeTimeSnapshot(s1, match, 100)    # take a sample of points from s1 and scale between 0-2
        samp2=TakeTimeSnapshot(s2, s2_match, 100) # take a sample of points from s1 and scale between 0-2
        cc=CalcNormedCrossCorr(samp1,samp2)       # take cross correlation

        # save results
        index_of_s1_matches=np.append(index_of_s1_matches,match)
        index_of_s2_matches=np.append(index_of_s2_matches,s2_match)
        cc_s1_s2=np.append(cc_s1_s2,cc)
        
        ccmax=cc_s1_s2.max()
        lag=index_of_s2_matches[cc_s1_s2==ccmax]-index_of_s1_matches[cc_s1_s2==ccmax] # calculate lag by id-ing where cross correlation is 1
endTime=time.time()

# report results
log.info(f'The lag between signals is: {abs(lag[0])}s, which corresponds to {abs(lag[0]*333)} m')
log.info(f'sample of points used to confirm where s1=s2: {"%.2f" % (lag.shape[0]/s1.shape[0]*100)} % ({lag.shape[0]} points)')
log.info(f'time taken: {"%.2f" % (endTime-startTime)} s')

#44.1khz


