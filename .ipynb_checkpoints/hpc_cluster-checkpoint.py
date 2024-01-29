import hyperSLIC
import hyperspy.api as hs
import numpy as np
import matplotlib.pyplot as plt
import numba as nb
from numba import jit
from tqdm import tqdm
import os


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
    print('Wheat separated')
    maxes = np.amax(wheat, axis=0)
    minis = np.amin(wheat,axis=0) 
    
    wheat = wheat - minis

    wheat = wheat/maxes
    
    wheat = np.reshape(wheat,(np.shape(data)[0],np.shape(data)[1],-1))
    return wheat



s = hs.load(r"C:\Users\tas72\Downloads\SimulatedData\SimulatedData\data\SEND.hspy",lazy=True)
edx = hs.load(r"C:\Users\tas72\Downloads\SimulatedData\SimulatedData\data\EDS.hspy")
s.data = s.data.astype('uint16')
s.compute()
raveled = np.reshape(s.data, (s.axes_manager[1].size, s.axes_manager[0].size,s.axes_manager[2].size*s.axes_manager[3].size))
wheat_edx = wheat_from_chaff(edx.data,95,'range')
wheat_edx = hs.signals.Signal1D(wheat_edx)
wheat = wheat_from_chaff(raveled,98,'range')

sed_weight_factor = 0.2
edx_weight_factor = 3

wheat_hs = hs.signals.Signal1D(np.append(sed_weight_factor*wheat,edx_weight_factor*wheat_edx.data,axis=2))

epochs = 2 #(5 works okay with these params)
iterations = 5
cluster_number = 100
m_value = 1
searchspace = 1
method = 'semi'
test = hyperSLIC.SLIC(wheat_hs,method,cluster_number,m_value,searchspace)

accumulator = np.zeros((test.width,test.height,test.k))
for k in range(epochs):
    test = hyperSLIC.SLIC(wheat_hs,method,cluster_number,m_value,searchspace) # centroid initalisation, seed_num, m, searchspace
    for i in tqdm(range(iterations)):
        test.find_closest_centeroid()
        test.update_centeroids()
    for i in range(np.shape(accumulator)[0]):
        for j in range(np.shape(accumulator)[1]):
            cluster = test.closest_centeroid[i,j]
            accumulator[i,j,int(cluster)] += 1
np.save('accumulator.npy',accumulator)
print('Complete')