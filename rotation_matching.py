import numpy as np
from shapely import geometry
from shapely.ops import transform
from metpy.units import atleast_1d, check_units, concatenate, units
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from pyproj import Geod

def rot_storm_matcher(shear_maxes1,shear_lats1, shear_lons1,ax,f,time_start,month,d_beg,h_beg,min_beg,sec_beg,d_end,h_end,min_end,sec_end,rlons,rlats,max_lons_c,max_lats_c,proj):
    
    rot_mag1=[] 
    rot_lat1=[]
    rot_lon1=[] 
    rot_storm_lon1=[] 
    rot_storm_lat1=[]
    
    for i in range(len(shear_maxes1)):
        g = Geod(ellps='sphere')
        dist_rot = np.zeros((np.asarray(max_lons_c).shape[0]))
        for j in range(dist_rot.shape[0]):
            distance_rot = g.inv(shear_lons1[i], shear_lats1[i],
                               max_lons_c[j], max_lats_c[j])
            dist_rot[j] = distance_rot[2]/1000.
        if np.min(np.asarray(dist_rot)) < 15.0:
            rot_mag1.append((shear_maxes1[i]))
            rot_lon1.append((shear_lons1[i]))
            rot_lat1.append((shear_lats1[i]))
            rot_storm_lon1.append((max_lons_c[np.where(dist_rot == np.min(dist_rot))[0][0]]))
            rot_storm_lat1.append((max_lats_c[np.where(dist_rot == np.min(dist_rot))[0][0]]))
            
    return rot_mag1, rot_lat1, rot_lon1, rot_storm_lon1, rot_storm_lat1
        
                

        