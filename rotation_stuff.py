from numba import jit
import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from metpy.calc import wind_direction
from metpy.units import units
import numpy.ma as ma

@jit(nopython=True)
def calc_regr(A_x, A_y, VEL, xregr_array, yregr_array):
    #Create a definition to use with numba
    for level in [4,12,20,28]:
        for i in range(3, 496, 1):
            for j in range(3, 496, 1):

                vel_regr = VEL[level, i-3:i+4, j-3:j+4]
                #print(x_r.shape, vel_regr.shape)
                xregr_array[level, i, j] = np.linalg.lstsq(A_x, vel_regr.flatten())[0][0]
                yregr_array[level, i, j] = np.linalg.lstsq(A_y, vel_regr.flatten())[0][0]

    return xregr_array, yregr_array

def get_rotation(VEL, REF, rlons_2d, rlats_2d, bin_size=7):
    x_rng = np.arange(0,bin_size,1)
    y_r, x_r = np.meshgrid(x_rng, x_rng)
    A_x = np.vstack([x_r.flatten(), np.ones(len(x_r.flatten()))]).T
    A_y = np.vstack([y_r.flatten(), np.ones(len(y_r.flatten()))]).T
    A_x = A_x.astype(np.float32)
    A_y = A_y.astype(np.float32)
    xregr_array = np.zeros((41,500,500))
    yregr_array = np.zeros((41,500,500))
    
    xregr_array, yregr_array = calc_regr(A_x, A_y, VEL, xregr_array, yregr_array)
    
    x_shape = np.arange(0,500,1)

    y_shp, x_shp = np.meshgrid(x_shape, x_shape) 
    print(y_shp)
    azim = wind_direction(250*units('m/s')-x_shp*units('m/s'), y_shp*units('m/s')-250*units('m/s'))+ 90*units('degree')
    azim = azim + 270*units('degree')
    azim[azim > 360*units('degree')] = azim[azim > 360*units('degree')]- 360 * units('degree')
    print(azim.shape)
    xcomp = np.sin(azim.to('radian'))
    ycomp = np.cos(azim.to('radian'))

    #Get projection by taking the dot product
    az_shear = (ycomp * xregr_array + xcomp * yregr_array)
    az_masked = np.copy(az_shear)
    az_masked = ma.masked_where(REF < 20, az_masked)
    az_masked = ma.filled(az_masked, fill_value=-2)
    
    dist = 5
    thresh = 0.002 * 493
    #image_max = ndi.maximum_filter(az_masked[lev, :, :], size=dist, mode='constant')
    coordinates4 = peak_local_max(az_masked[4, :, :], min_distance=dist, threshold_abs=thresh)
    coordinates6 = peak_local_max(az_masked[12, :, :], min_distance=dist, threshold_abs=thresh)
    coordinates8 = peak_local_max(az_masked[20, :, :], min_distance=dist, threshold_abs=thresh)
    coordinates10 = peak_local_max(az_masked[28, :, :], min_distance=dist, threshold_abs=thresh)
    
    shear_maxes1 = az_masked[4,:,:][coordinates4[:,0], coordinates4[:,1]]
    shear_maxes15 = az_masked[12,:,:][coordinates6[:,0], coordinates6[:,1]]
    shear_maxes2 = az_masked[20,:,:][coordinates8[:,0], coordinates8[:,1]]
    shear_maxes25 = az_masked[28,:,:][coordinates10[:,0], coordinates10[:,1]]

    shear_lats1 = rlats_2d[:,:][coordinates4[:,0], coordinates4[:,1]]
    shear_lats15 = rlats_2d[:,:][coordinates6[:,0], coordinates6[:,1]]
    shear_lats2 = rlats_2d[:,:][coordinates8[:,0], coordinates8[:,1]]
    shear_lats25 = rlats_2d[:,:][coordinates10[:,0], coordinates10[:,1]]

    shear_lons1 = rlons_2d[:,:][coordinates4[:,0], coordinates4[:,1]]
    shear_lons15 = rlons_2d[:,:][coordinates6[:,0], coordinates6[:,1]]
    shear_lons2 = rlons_2d[:,:][coordinates8[:,0], coordinates8[:,1]]
    shear_lons25 = rlons_2d[:,:][coordinates10[:,0], coordinates10[:,1]]

    return az_masked, shear_maxes1, shear_maxes15, shear_maxes2, shear_maxes25, shear_lats1, shear_lats15, shear_lats2, shear_lats25, shear_lons1, shear_lons15, shear_lons2, shear_lons25

