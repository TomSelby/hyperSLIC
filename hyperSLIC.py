import random
import numpy as np
import hyperspy.api as hs
from numba import jit,prange,njit
import numba as nb

@jit(nopython=True)
def pythag_d(x1,y1,x2,y2):
    d = np.sqrt(((x1-x2)**2) + ((y1-y2)**2))
    return d
@jit(nopython=True)
def calc_channel_dist(current_channels,centeroid_channels):
    distance_channels = current_channels - centeroid_channels
    distance_channels = distance_channels**2
    distance_channels = np.sum(distance_channels) # don't need to sqrt as just using it squared next line
    return distance_channels
@jit(nopython=True)
def calc_total_dist(distance_channels,distance_x_y,dom_size,m):
    total_dist = np.sqrt(distance_channels + ((distance_x_y/dom_size)**2)*(m**2))
    #total_dist = np.sqrt((distance_channels/m)**2+(distance_x_y/dom_size)**2)
    return total_dist

@njit(parallel=True)
def distance_to_cent_oi(vasinity_data,centeroid_xy,centeroid_channels,row_lb,col_lb, dom_size, m):
    distance_to_centeroid_oi = np.zeros_like(vasinity_data[:,:,0])
    
    for i in prange(np.shape(distance_to_centeroid_oi)[0]): # cover entire vasinity data
        for j in prange(np.shape(distance_to_centeroid_oi)[1]):
            
            distance_x_y = pythag_d(centeroid_xy[0],centeroid_xy[1],i+row_lb,j+col_lb)
            current_channels = vasinity_data[i,j]                   
            distance_channels = calc_channel_dist(current_channels,centeroid_channels)
            distance_to_centeroid_oi[i,j] = calc_total_dist(distance_channels,distance_x_y,dom_size,m)
            
    return distance_to_centeroid_oi
                            
@njit(parallel=True)
def find_new_centeroid(channel_len, coords,dot_data,shape_coords):
    sum_channels = np.zeros((channel_len))
    
    for i in range(shape_coords):
        
        channels = dot_data[coords[0][i],coords[1][i]]
        sum_channels += channels
   
    mean_x = np.mean(coords[0])
    mean_y = np.mean(coords[1])
    mean_channels = sum_channels/shape_coords
    return (mean_x, mean_y, mean_channels)
                            
class SLIC():
    def __init__(self,data,mode,k,m,search_space):
        '''Initialise the data defines the height, width and number of channels in the data then defines the starting grid based off the number of clusters (k). Mode is random default regular, search_space is how much to find distances beyond S, classically this is 2'''
        self.data = data
        self.dot_data = data.data
        self.image = self.data.T.sum()
        self.width = data.axes_manager[1].size
        self.height = data.axes_manager[0].size
        self.channels = data.axes_manager[2].size
        self.mode = mode
        self.k = k
        self.search_space = search_space
        self.m = m
        self.num_each_side = int(np.sqrt(self.k))
        self.est_domain_size = self.width/(self.num_each_side+1) # inbetween distance
        
       

        
        # Initialise the centeroids
        self.initial_xy_centeroids, self.initial_channel_centeroids = self.find_initial_centeroids()
        self.xy_centeroids = self.initial_xy_centeroids
        self.channel_centeroids = self.initial_channel_centeroids
    
    def find_initial_centeroids(self):
        if self.mode == 'random':
            seed_positions = [((random.randint(0,self.width-1)),(random.randint(0,self.height-1))) for x in range(self.k)]
        elif self.mode == 'regular':
            
            seeds = [(self.est_domain_size+(x*self.est_domain_size)) for x in range(self.num_each_side)]
            seed_positions = []
            for seed_x in seeds:
                for seed_y in seeds:
                    seed_positions.append((seed_x,seed_y))
        elif self.mode == 'semi':
            seeds = [(self.est_domain_size+(x*self.est_domain_size)) for x in range(self.num_each_side)]
            seed_positions = []
            for seed_x in seeds:
                for seed_y in seeds:
                    seed_positions.append((seed_x+np.random.randint(-2,2),seed_y+np.random.randint(-2,2)))
        channel_positions = []
        
        for seed in seed_positions: # for initial channel position just take the value at the x/y initialised x/y value
            
            channel_value_at_seed = self.data.inav[seed]
            channel_positions.append(channel_value_at_seed.data)
        return seed_positions, channel_positions
        
    
    def find_closest_centeroid(self):    
        closest_centeroid = np.zeros((self.width,self.height),dtype=int) #initalise array of current closest centeroids
        distances_arr = np.zeros((self.width,self.height))+np.inf     
        
        
        for counter, centeroid in enumerate(self.xy_centeroids): #find the upper and lower bounds to check for x/y
                      
            centeroid_row_lb = round(centeroid[0]-(self.est_domain_size*self.search_space))
            if centeroid_row_lb <= 0:
                centeroid_row_lb = 0
            centeroid_col_lb = round(centeroid[1]-(self.est_domain_size*self.search_space))
            if centeroid_col_lb <= 0:
                centeroid_col_lb = 0
            centeroid_row_ub = round(centeroid[0]+(self.est_domain_size*self.search_space))
            if centeroid_row_ub >= self.width:
                centeroid_row_ub = self.width
            centeroid_col_ub = round(centeroid[1]+(self.est_domain_size*self.search_space))
            if centeroid_col_ub >= self.height:
                centeroid_col_ub = self.height
            centeroid_channels = self.channel_centeroids[counter] # get current centeroids
            vasinity_data = self.dot_data[centeroid_row_lb:centeroid_row_ub,centeroid_col_lb:centeroid_col_ub]
            
            
            distance_to_centeroid_oi = distance_to_cent_oi(vasinity_data,centeroid, centeroid_channels,centeroid_row_lb,centeroid_col_lb, self.est_domain_size, self.m)
            
            
            distances_to_check = distances_arr[centeroid_row_lb:centeroid_row_ub,centeroid_col_lb:centeroid_col_ub] 
            
            args_to_change = np.where(distance_to_centeroid_oi <= distances_to_check)
            for i in range(len(args_to_change[0])):
                arg = (args_to_change[0][i], args_to_change[1][i])
                closest_centeroid[arg[0]+centeroid_row_lb,arg[1]+centeroid_col_lb] = counter
                distances_arr[arg[0]+centeroid_row_lb,arg[1]+centeroid_col_lb] = distance_to_centeroid_oi[arg[0],arg[1]]
                
        self.closest_centeroid = closest_centeroid
        
            
                   
    def update_centeroids(self):             
        for counter in range(len(self.xy_centeroids)):            
            coords = np.where(self.closest_centeroid == counter)
            if np.shape(coords)[1] == 0:
                self.xy_centeroids[counter] = self.initial_xy_centeroids[counter]
                self.channel_centeroids[counter] = self.initial_channel_centeroids[counter]
            else:
                (mean_x,mean_y, mean_channels) = find_new_centeroid(self.channels, coords,self.dot_data,np.shape(coords)[1])
                ## Update
                self.xy_centeroids[counter] = (mean_x,mean_y) 
                self.channel_centeroids[counter] = mean_channels
           
            
    
        
        
    
    
            
        
    

  
        

    
    
 
