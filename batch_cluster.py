import hyperSLIC # Note dont have centroids respawn after being orphaned
import hyperspy.api as hs
import numpy as np
import time
import multiprocessing as mp
import numba as nb
from numba import jit
from tqdm import tqdm
import os
import copy
import glob
import gc
from scipy.stats.stats import pearsonr


@jit(nopython=True)
def normalise(channel):
    channel = (channel - np.amin(channel))+1 # breaks if value of 0 is given
    channel = channel/np.amax(channel)
    return channel

@jit(nopython = True)
def quick_wheat(ranges, reshaped, percentile):
    wheat = reshaped.T[ranges >= percentile]
    return wheat.T

def wheat_from_chaff(data, percentile,mode):
    reshaped = np.reshape(data, (np.shape(data)[0]*np.shape(data)[1], np.shape(data)[2]))
    print(np.shape(reshaped))
    print('Reshaped')
    if mode == 'range':
        ranges = np.ptp(reshaped, axis=0)
    if mode == 'var':
        ranges = np.var(reshaped,axis=0)
    ranges = np.ptp(reshaped, axis=0)
    print(np.shape(ranges))
    print('Ranges aquried')
    percentile = np.percentile(ranges,percentile)
    print(percentile)
    wheat = quick_wheat(ranges, reshaped, percentile)
    print(np.shape(wheat))
    print('Wheat seperated')
    maxes = np.amax(wheat, axis=0)
    minis = np.amin(wheat,axis=0) 
    
    wheat = wheat - minis

    wheat = wheat/maxes
    
    wheat = np.reshape(wheat,(np.shape(data)[0],np.shape(data)[1],-1))
    return wheat

## Script start
errors = []
dirc_list = glob.glob(r'Z:\smf57\dg606\20230624_centered\*.zspy')
constant = 3
for dirc in dirc_list:
    try:
        print(dirc)
        s = hs.load(dirc,lazy=True)
        s = s.inav[:,:]
        s_copy = s.deepcopy()
        s = s.rebin(scale=[1,1,2,2])
        s.compute()
        s_copy.compute()

        s.isig[(s.axes_manager[2].size//2)-constant:(s.axes_manager[2].size//2)+constant,(s.axes_manager[3].size//2)-constant:(s.axes_manager[3].size//2)+constant] = 0#np.zeros((constant*2,constant*2))
        raveled = np.reshape(s.data, (s.axes_manager[1].size, s.axes_manager[0].size,s.axes_manager[2].size*s.axes_manager[3].size))
        wheat = wheat_from_chaff(raveled,98,'range')
        print('Complete', np.shape(wheat))
        wheat_hs = hs.signals.Signal1D(wheat)
        test = hyperSLIC.SLIC(wheat_hs,'regular',700,0.9,1) # seed_num, m, searchspace
        summed = s.T.sum().data
        t0 = time.time()
        for i in tqdm(range(10)):

            test.find_closest_centeroid()
            test.update_centeroids()
        t1 = time.time()
        print(f'Loop time: {np.round((t1-t0),3)}s')


        summed_channels = np.zeros(test.k,dtype='object')
        number_of_occurances = np.zeros(test.k)

        for row in tqdm(range(np.shape(test.closest_centeroid)[0])):
            for col in range(np.shape(test.closest_centeroid)[1]):

                arg = int(test.closest_centeroid[row,col])
                number_of_occurances[arg] += 1 
                channel = test.dot_data[row,col].astype(np.float32)# Use the copy not the data with the masked bright feild
                summed_channels[arg]+=channel

        for i in range(len(number_of_occurances)):
            occurances = number_of_occurances[i]
            if occurances == 0:
                number_of_occurances[i] = np.nan


        mean_channels = summed_channels/number_of_occurances

        updated_centroids = copy.deepcopy(test.closest_centeroid)
        for i in tqdm(range(len(mean_channels))):
            channel_oi = mean_channels[i]
            for counter,comparison_channel in enumerate(mean_channels[i:]):

                try:
                    r = pearsonr(channel_oi,comparison_channel)
                    if r[0] > 0.8:
                        args = np.where(updated_centroids == i)
                        for j in range(len(args[0])):
                                row = args[0][j]
                                col = args[1][j]
                                updated_centroids[row,col] = i+counter
                except:
                    pass
        print('Clusters combined')
        summed_patterns = np.zeros((test.k,np.shape(s_copy.inav[0,0].data)[0],np.shape(s_copy.inav[0,0].data)[1]),dtype = 'float32')

        number_of_occurances = np.zeros(test.k)


        for row in tqdm(range(np.shape(test.closest_centeroid)[0])):
            for col in range(np.shape(test.closest_centeroid)[1]):

                arg = int(updated_centroids[row,col])
                number_of_occurances[arg] += 1 
                pattern = s_copy.data[row,col].astype(np.float32)# Use the copy not the data with the masked bright feild
                summed_patterns[arg] += pattern

        print('Patterns summed')
        mean_patterns = np.zeros_like(summed_patterns)
        for i in tqdm(range(len(mean_patterns))):
            occurances = number_of_occurances[i]
            if occurances == 0:
                mean_patterns[i] = np.zeros((256,256))
            else:
                mean_patterns[i] = summed_patterns[i]/number_of_occurances[i]

        np.save(f'{dirc[:-5]}_mean_cluster_patterns',mean_patterns)
        np.save(f'{dirc[:-5]}_clusters',updated_centroids)
        print(f'{dirc}_mean_cluster_patterns and {dirc}_clusters saved')
        gc.collect()
        del s_copy
        del s
    except Exception as e: 
        print('##################################################')
        print(e)
        print('##################################################')
        errors.append([dirc])
print('Complete :)')
print('Errors')
print(errors)