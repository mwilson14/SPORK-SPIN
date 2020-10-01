import numpy as np
from shapely import geometry
from shapely.ops import transform
from metpy.units import atleast_1d, check_units, concatenate, units
from matplotlib.path import Path
from matplotlib.patches import PathPatch
import matplotlib.pyplot as plt
from pyproj import Geod
import csv
from metpy.calc import wind_direction, wind_speed, wind_components

def rot_storm_matcher_qc(shear_maxes1,shear_lats1, shear_lons1,ax,f,time_start,month,d_beg,h_beg,min_beg,sec_beg,d_end,h_end,min_end,sec_end,rlons,rlats,max_lons_c,max_lats_c,proj, tracking_ind, rlons_2d, rlats_2d, REFall, KDPall, CCall, grad_ffd, grad_mag, ZDR_sum_stuff, az_masked, storm_relative_dir,station, dt, forest_loaded_meso, rot_lev):
    
    rot_mag1=[] 
    rot_lat1=[]
    rot_lon1=[] 
    rot_storm_lon1=[] 
    rot_storm_lat1=[]
    
    rot_mag2=[] 
    rot_lat2=[]
    rot_lon2=[] 
    rot_storm_lon2=[] 
    rot_storm_lat2=[]
    az_area2 = []
    print(shear_maxes1, 'shear maxes')
    
    for i in range(len(shear_maxes1)):
        g = Geod(ellps='sphere')
        dist_rot = np.zeros((np.asarray(max_lons_c).shape[0]))
        forw_rot = np.zeros((np.asarray(max_lons_c).shape[0]))
        back_rot = np.zeros((np.asarray(max_lons_c).shape[0]))
        rawangle_rot = np.zeros((np.asarray(max_lons_c).shape[0]))
        for j in range(dist_rot.shape[0]):
            distance_rot = g.inv(shear_lons1[i], shear_lats1[i],
                               max_lons_c[j], max_lats_c[j])
            dist_rot[j] = distance_rot[2]/1000.
            back_rot[j] = distance_rot[1]
            if distance_rot[1] < 0:
                back_rot[j] = distance_rot[1] + 360
            forw_rot[j] = np.abs(back_rot[j] - storm_relative_dir)
            rawangle_rot[j] = back_rot[j] - storm_relative_dir
            #Account for weird angles
            if forw_rot[j] > 180:
                forw_rot[j] = 360 - forw_rot[j]
                rawangle_rot[j] = (360-forw_rot[j])*(-1)
            rawangle_rot[j] = rawangle_rot[j]*(-1)
        if np.min(np.asarray(dist_rot)) < 30.0:
            print('got a close one')
            rot_mag1.append((shear_maxes1[i]))
            rot_lon1.append((shear_lons1[i]))
            rot_lat1.append((shear_lats1[i]))
            rot_storm_lon1.append((max_lons_c[np.where(dist_rot == np.min(dist_rot))[0][0]]))
            rot_storm_lat1.append((max_lats_c[np.where(dist_rot == np.min(dist_rot))[0][0]]))



            postsize = 15

            lond = np.abs(rlons_2d-shear_lons1[i])
            latd = np.abs(rlats_2d-shear_lats1[i])

            xminlon = np.where(lond==np.min(lond))[0][0]
            yminlon = np.where(lond==np.min(lond))[1][0]

            xminlat = np.where(latd==np.min(latd))[0][0]
            yminlat = np.where(latd==np.min(latd))[1][0]

            REFpost = REFall[rot_lev,xminlat-postsize:xminlat+postsize, yminlon-postsize:yminlon+postsize]
            KDPpost = KDPall[rot_lev,xminlat-postsize:xminlat+postsize, yminlon-postsize:yminlon+postsize]
            CCpost = CCall[rot_lev,xminlat-postsize:xminlat+postsize, yminlon-postsize:yminlon+postsize]
            gradffdpost = grad_ffd[xminlat-postsize:xminlat+postsize, yminlon-postsize:yminlon+postsize]
            gradmagpost = grad_mag[xminlat-postsize:xminlat+postsize, yminlon-postsize:yminlon+postsize]
            ZDRdpost = ZDR_sum_stuff[xminlat-postsize:xminlat+postsize, yminlon-postsize:yminlon+postsize]
            azpost = az_masked[rot_lev,xminlat-postsize:xminlat+postsize, yminlon-postsize:yminlon+postsize]

            print('shape of az', np.shape(azpost[azpost>(.002*493)].flatten()))
            print('points in square', np.shape(azpost.flatten()))

            #print('az percent area', np.shape(azpost[azpost>(.002*493)].flatten())[0]/np.shape(azpost.flatten())[0])
            try:
                
                meanZ = np.mean(REFpost)
                maxZ = np.max(REFpost)
                meanKDP = np.mean(KDPpost)
                meanCC = np.mean(CCpost)
                meangradffd = np.mean(gradffdpost)
                meangradmag = np.mean(gradmagpost)
                meanZDRd = np.mean(ZDRdpost)
                maxZDRd = np.mean(ZDRdpost)
                maxrot = shear_maxes1[i]/493
                az_area = np.shape(azpost[azpost>(.002*493)].flatten())[0]/np.shape(azpost.flatten())[0]
                rot_50 = np.percentile(azpost, 50)
                rot_90 = np.percentile(azpost, 90)
            
            except:
                meanZ = 0
                maxZ = 0
                meanKDP = 0
                meanCC = 0
                meangradffd = 0
                meangradmag = 0
                meanZDRd = 0
                maxZDRd = 0
                maxrot = 0
                az_area = 0
                rot_50 = 0
                rot_90 = 0
                
                
            lonpost = rlons_2d[xminlat-postsize:xminlat+postsize, yminlon-postsize:yminlon+postsize]
            latpost = rlats_2d[xminlat-postsize:xminlat+postsize, yminlon-postsize:yminlon+postsize]
            
            if (rawangle_rot[np.where(dist_rot == np.min(dist_rot))[0][0]] > 0):
                directions_raw = 360 - rawangle_rot[np.where(dist_rot == np.min(dist_rot))[0][0]]
            else:
                directions_raw = (-1) * rawangle_rot[np.where(dist_rot == np.min(dist_rot))[0][0]]
                
            xc, yc = wind_components(np.min(dist_rot)*units('m/s'), directions_raw * units('degree'))
            MESO_X = np.zeros((1, 14))

            MESO_X[:,0] = az_area
            MESO_X[:,1] = np.min(dist_rot)
            MESO_X[:,2] = rot_50
            MESO_X[:,3] = maxrot
            MESO_X[:,4] = meanCC
            MESO_X[:,5] = meanKDP
            MESO_X[:,6] = meanZ
            MESO_X[:,7] = meangradffd
            MESO_X[:,8] = meangradmag
            MESO_X[:,9] = maxZDRd
            MESO_X[:,10] = meanZDRd
            MESO_X[:,11] = rot_90
            MESO_X[:,12] = xc
            MESO_X[:,13] = yc
            
            pred_meso = forest_loaded_meso.predict(MESO_X)
            if pred_meso[0]==1:
                rot_mag2.append((shear_maxes1[i]))
                az_area2.append((az_area))
                rot_lon2.append((shear_lons1[i]))
                rot_lat2.append((shear_lats1[i]))
                rot_storm_lon2.append((max_lons_c[np.where(dist_rot == np.min(dist_rot))[0][0]]))
                rot_storm_lat2.append((max_lats_c[np.where(dist_rot == np.min(dist_rot))[0][0]]))

                
    return rot_mag2, rot_lat2, rot_lon2, rot_storm_lon2, rot_storm_lat2, az_area2