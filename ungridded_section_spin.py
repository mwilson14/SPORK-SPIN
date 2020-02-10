import numpy as np
import numpy.ma as ma
import pyart

def quality_control_spin(radar1,n,calibration):
    print('Pre-grid Organization Section')
    ni = 0
    tilts = []
    for i in range(radar1.nsweeps):
        try:
            radar2 = radar1.extract_sweeps([i])
        except:
            continue
        #Checking to make sure the tilt in question has all needed data and is the right elevation
        if ((np.max(np.asarray(radar2.fields['differential_reflectivity']['data'])) != np.min(np.asarray(radar2.fields['differential_reflectivity']['data'])))):
            if ((np.mean(radar2.elevation['data']) < .65)):
                if (ni==0):
                    tilts.append(i)
                    n = ni+1

            else:
                tilts.append(i)

    radar = radar1.extract_sweeps(tilts)
    n = n+1

    radar3 = radar.extract_sweeps([0])
    range_i = radar3.range['data']
    # mask out last 10 gates of each ray, this removes the "ring" around the radar.
    radar.fields['differential_reflectivity']['data'][:, -10:] = np.ma.masked
    ref_ungridded_base = radar3.fields['reflectivity']['data']

    #Get 2d ranges
    range_2d = np.zeros((ref_ungridded_base.shape[0], ref_ungridded_base.shape[1]))
    for i in range(ref_ungridded_base.shape[0]):
        range_2d[i,:]=range_i
    
    #Use the calibration factor on the ZDR field
    radar.fields['differential_reflectivity']['data'] = radar.fields['differential_reflectivity']['data']-calibration
    cc_m = radar.fields['cross_correlation_ratio']['data'][:]
    refl_m = radar.fields['reflectivity']['data'][:]
    radar.fields['differential_reflectivity']['data'][:] = ma.masked_where(cc_m < 0.80, radar.fields['differential_reflectivity']['data'][:])
    radar.fields['differential_reflectivity']['data'][:] = ma.masked_where(refl_m < 20.0, radar.fields['differential_reflectivity']['data'][:])

    #Get stuff for QC control rings
    radar_t = radar.extract_sweeps([radar.nsweeps-1])
    last_height = radar_t.gate_altitude['data'][:]
    rlons_h = radar_t.gate_longitude['data']
    rlats_h = radar_t.gate_latitude['data']
    ungrid_lons = radar3.gate_longitude['data']
    ungrid_lats = radar3.gate_latitude['data']
    gate_altitude = radar3.gate_altitude['data'][:]

    try:
        #Add in another radar object for getting velocity
        dealiased_vel_u1 = pyart.correct.dealias_unwrap_phase(radar1)
        velocity_final_u1 = dealiased_vel_u1['data']

        #Create dictionary
        spin_nwsdict = {}
        spin_nwsdict['units'] = 'm/s'
        spin_nwsdict['standard_name'] = 'dealiased_velocity'
        spin_nwsdict['long_name'] = 'dealiased_radial_velocity'
        spin_nwsdict['coordinates'] = 'elevation azimuth range'
        spin_nwsdict['data'] = velocity_final_u1
        spin_nwsdict['valid_min'] = -999
        spin_nwsdict['valid_max'] = 999
        spin_nwsdict['Clipf'] = 3906250000.0
        #Add field to radar
        radar1.add_field('Good_VEL', spin_nwsdict)
        # mask out first 20 gates of each ray, this removes the "ring" around the radar.
        radar1.fields['Good_VEL']['data'][:, 0:20] = np.ma.masked

        ni = 0
        tilts_v = []
        for i in range(radar1.nsweeps):
            radar2 = radar1.extract_sweeps([i])
            print(np.mean(radar2.elevation['data']))
            #Checking to make sure the tilt in question has all needed data and is the right elevation
            if ((np.max(np.asarray(radar2.fields['Good_VEL']['data'])) != np.min(np.asarray(radar2.fields['Good_VEL']['data'])))):
                if ((np.mean(radar2.elevation['data']) < .65)):
                    if (ni==0):
                        tilts_v.append(i)
                        n = ni+1
                    else:
                        print('extra low-level tilt')
                else:
                    tilts_v.append(i)
            else:
                print('bad split cut')

        radar_v = radar1.extract_sweeps(tilts_v)
        
    except:
        radar_v = np.asarray([0,0,0])
        

    return radar,radar_v,n,range_2d,last_height,rlons_h,rlats_h,ungrid_lons,ungrid_lats