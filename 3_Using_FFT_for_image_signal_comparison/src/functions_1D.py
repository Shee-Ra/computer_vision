# library
import numpy as np
import logging

# ------------------------
# sort out logging
# ------------------------
logging.basicConfig(level=logging.DEBUG,     # set level
                    format='%(levelname)s '  # logging level
                    'on line %(lineno)d: '   # line ID
                    '%(message)s '           # message
                    )
log = logging.getLogger()


# 1D functions

def LoadSensorData(path='', mrows=10):
    '''Load sensor data into numpy array'''
    sensor_data=np.loadtxt(path, skiprows=1, delimiter="\n", max_rows=mrows, unpack=False)
    return(sensor_data)  # assumes files take same form every time

def MakeSameSize(signal_1, signal_2):
    len1=signal_1.shape[0]
    len2=signal_2.shape[0]
    diff=abs(len2-len1)

    if (len1<len2):
        signal_1=np.pad(len1,(diff,0),'constant',(0,0))
    elif (len2<len1):
        signal_2=np.pad(len2,(diff,0),'constant',(0,0))
    else:
        pass

    return(signal_1,signal_2)

def Get1DFFT(s,sampling_rate=44100):
    # test: make sure sig is 1D
    # create time samples
    n=s.shape[0]
    dt=1/(sampling_rate) ## 44kHz sampling rate
    t=np.linspace(0,n-1,n)*dt
    
    # Fourier transformed
    f=np.fft.fftfreq(n,dt)
    y=np.fft.fft(s)
    
    return(f,y)

def Get1DLag(signal_1, signal_2, sampling_rate=44100,speed_sound=333):
    # step 1: get FFT
    signal_f, s1=Get1DFFT(signal_1)
    kernel_f, s2=Get1DFFT(signal_2)
    # step 2: conjugate one of the signals
    s1_conj= -s1.conjugate()
    
    # step 3: get lag by taking inverse FFT of s1_conj*s2
    s2_point_lag=np.argmax(np.abs(np.fft.ifft(s1_conj*s2))) # this is how much s2 lags in-front/behind s1
    
    # step 4: lag can be expressed as s1/s2 being ahead/behind this sorts that out
    if (signal_1[0]==signal_2[s2_point_lag]): # s2 is ahead of s1, do nothing (e.g. if i=0 for s1, this is the value i, for which s1=s2)
        log.info(f'when first signal is at i=0 value is: {signal_1[0]}, second signal is ahead by {s2_point_lag} with value {signal_2[s2_point_lag]}')
    elif (signal_1[-1:]==signal_2[s2_point_lag-1:s2_point_lag]):
        log.info(f'when second signal is at i=end value is: {signal_1[-1:]}, first signal lags behind by {s2_point_lag} with value {signal_2[s2_point_lag-1:s2_point_lag]}')
        s2_point_lag=s1.shape[0]-s2_point_lag # how much s1 is 'behind' s2 (e.g. if i=last point for s1, this is the value i, for which s1=s2), so recalculate to express as above
    else:
        log.info(f'no match between input signals, NaN returned')
        s2_point_lag=np.nan

    # step 5: calculate time lag and thus distance
    s2_time_lag=s2_point_lag/sampling_rate
    distance = speed_sound*s2_time_lag
    return(s2_point_lag, s2_time_lag, distance)