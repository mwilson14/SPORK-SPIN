import numpy as np
import numpy.ma as ma
from shapely import geometry
from shapely.ops import transform
from metpy.units import check_units, concatenate, units
from matplotlib.path import Path
from pyproj import Geod
from radar_storm_motion import radar_motion
from datetime import datetime, timedelta

def storm_objects_new(refc,proj,REFlev,REFlev1,big_storm,smoothed_ref,ax,rlons,rlats,storm_index,tracking_index,scan_index,tracks_dataframe,track_dis, time_start):
    g = Geod(ellps='sphere')
    #Inputs,
    #refc: Contours of smoothed_ref at first reflectivity threshold (REFlev)
    #proj: Projection of Earth's surface to be used for accurate area and distance calculations
    #REFlev: First reflectivity threshold
    #REFlev1: Second reflectivity threshold
    #big_storm: Threshold for "big storm" classification in km^2
    #smoothed_ref: Gaussian filtered reflectivity, masked below 20dBz
    #ax: Subplot object to be built on with each contour
    #rlons,rlats: Full volume geographic coordinates, longitude and latitude respectively
    #storm_index: Index to count and label storm objects as they are realized
    #tracking_index: Index to track which iteration of tracking algorithm is present
    #scan_index: Index used to track radar scans
    #tracks_dataframe: Final Pandas dataframe to be used to collect all output data
    #track_dis: Maximum distance for dislocation of storm obects between iterations to be considered same object
    

    #Empty arrays made for storm characteristics
    ref_areas = []
    max_lons_c = []
    max_lats_c = []
    storm_ids = []
    if np.max(smoothed_ref) > REFlev[0]:
        for level in refc.collections:
            #Loops through each closed reflectivity polygon in the contour list
            for contour_poly in level.get_paths(): 
                for n_contour,contour in enumerate(contour_poly.to_polygons()):
                    contour_a = np.asarray(contour[:])
                    xa = contour_a[:,0]
                    ya = contour_a[:,1]
                    polygon_new = geometry.Polygon([(i[0], i[1]) for i in zip(xa,ya)])
                    #Eliminates 'holes' in the polygons
                    if n_contour == 0:
                        polygon = polygon_new
                    else:
                        polygon = polygon.difference(polygon_new)

                #Transform the polygon's coordinates to the proper projection and calculate area
                #Testing a try statement here to eliminate some errors
                try:
                    pr_area = (transform(proj, polygon).area * units('m^2')).to('km^2')
                except:
                    continue
                #Use the polygon boundary to select all points within the polygon via a mask
                boundary = np.asarray(polygon.boundary.xy)
                polypath = Path(boundary.transpose())
                coord_map = np.vstack((rlons[0,:,:].flatten(), rlats[0,:,:].flatten())).T 
                maskr = polypath.contains_points(coord_map).reshape(rlons[0,:,:].shape)
                meanr = np.mean(smoothed_ref[maskr])
                
                if pr_area > 10 * units('km^2') and meanr > REFlev[0]:
                    #For large reflectivity contours with embedded supercells, find the embedded storm cores
                    #Normal/"big storm" cutoff 300 km^2
                    if pr_area > big_storm * units('km^2'):
                        rlon_2 = rlons[0,:,:]
                        rlat_2 = rlats[0,:,:]
                        smoothed_ref_m = ma.masked_where(maskr==False, smoothed_ref)
                        smoothed_ref_m = ma.filled(smoothed_ref_m, fill_value = -2)
                        rlon2m = ma.MaskedArray(rlon_2, mask=maskr)
                        rlat2m = ma.MaskedArray(rlat_2, mask=maskr)
                        
                        #This section uses the 2nd reflectivity threshold to subdivide big storms, in a similar manner to the 
                        #previous section's method for finding storms
                        refc1 = ax.contour(rlon2m,rlat2m,smoothed_ref_m,REFlev1, linewidths = 3, linestyle = '--', alpha=.01)
                        #Look for reflectivity centroids
                        for level1 in refc1.collections:
                            for contour_poly1 in level1.get_paths(): 
                                for n_contour1,contour1 in enumerate(contour_poly1.to_polygons()):
                                    contour_a1 = np.asarray(contour1[:])
                                    xa1 = contour_a1[:,0]
                                    ya1 = contour_a1[:,1]
                                    polygon_new1 = geometry.Polygon([(i[0], i[1]) for i in zip(xa1,ya1)])
                                    if n_contour1 == 0:
                                        polygon1 = polygon_new1
                                    else:
                                        polygon1 = polygon1.difference(polygon_new1)

                                pr_area1 = (transform(proj, polygon1).area * units('m^2')).to('km^2')
                                boundary1 = np.asarray(polygon1.boundary.xy)
                                polypath1 = Path(boundary1.transpose())
                                maskr1 = polypath1.contains_points(coord_map).reshape(rlons[0,:,:].shape)
                                meanr1 = np.mean(smoothed_ref[maskr1])
                                #Add objects that fit requirements to the list of storm objects
                                if pr_area1 > 10 * units('km^2') and meanr1 > REFlev1[0]:
                                    ref_areas.append((pr_area.magnitude))
                                    max_lons_c.append((polygon1.centroid.x))
                                    max_lats_c.append((polygon1.centroid.y))
                                    #For tracking, assign ID numbers and match current storms to any previous storms that are close enough to be the same
                                    if tracking_index == 0:
                                        storm_ids.append((storm_index))
                                        storm_index = storm_index + 1
                                    else:
                                        max_lons_p = np.asarray(tracks_dataframe['storm_lon'].loc[scan_index-1].iloc[:])
                                        max_lats_p = np.asarray(tracks_dataframe['storm_lat'].loc[scan_index-1].iloc[:])
                                        storm_ids_p = np.asarray(tracks_dataframe['storm_id1'].loc[scan_index-1].iloc[:])
                                        bearings = np.asarray(tracks_dataframe['storm direction'].loc[scan_index-1].iloc[:])
                                        speeds_p = np.asarray(tracks_dataframe['storm speed'].loc[scan_index-1].iloc[:])
                                        bearings[np.isnan(bearings)] = 0
                                        bearings = bearings + 180
                                        bearings[bearings > 360] = bearings[bearings > 360] - 360
                                        speeds_p[np.isnan(speeds_p)] = 0  
                                        #Next, get the number of elapsed seconds
                                        end_time = time_start
                                        try:
                                            start_time = tracks_dataframe['times'].loc[scan_index-1].iloc[:][0]
                                        except:
                                            start_time = tracks_dataframe['times'].loc[scan_index-1].iloc[:]
                                        print(start_time, 'st')
                                        print(end_time, 'et')
                                        try:
                                            stormdte = (np.datetime64(end_time) - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
                                            stormdtce = datetime.utcfromtimestamp(stormdte)
                                            stormdti = (np.datetime64(start_time) - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
                                            stormdtci = datetime.utcfromtimestamp(stormdti)
                                            storm_dur = stormdtce-stormdtci
                                            storm_sec = storm_dur.seconds
                                        except:
                                            storm_sec=0

                                        prj_lon, prj_lat, prj_bear = g.fwd(max_lons_p, max_lats_p, bearings, speeds_p*storm_sec)
                                        ax.scatter(prj_lon, prj_lat, s=200, color='r')
                                        dist_track = np.zeros((np.asarray(prj_lon).shape[0]))
                                        for i in range(prj_lon.shape[0]):
                                            distance_track = g.inv(polygon1.centroid.x, polygon1.centroid.y,
                                                                   prj_lon[i], prj_lat[i])
                                            dist_track[i] = distance_track[2]/1000.
                                        if np.min(dist_track) < track_dis:
                                            storm_ids.append((storm_ids_p[np.where(dist_track == np.min(dist_track))[0][0]]))
                                        else:
                                            storm_ids.append((storm_index))
                                            storm_index = storm_index + 1

                    #Do the same thing for objects from the 1st reflectivity threshold
                    else:
                        ref_areas.append((pr_area.magnitude))
                        max_lons_c.append((polygon.centroid.x))
                        max_lats_c.append((polygon.centroid.y))
                        if tracking_index == 0:
                            storm_ids.append((storm_index))
                            storm_index = storm_index + 1
                        else:
                            #To fix the problem of the algorithm crashing when it loses track of a storm, putting a try statement here
                            try:
                                max_lons_p = np.asarray(tracks_dataframe['storm_lon'].loc[scan_index-1].iloc[:])
                                max_lats_p = np.asarray(tracks_dataframe['storm_lat'].loc[scan_index-1].iloc[:])
                                storm_ids_p = np.asarray(tracks_dataframe['storm_id1'].loc[scan_index-1].iloc[:])
                                bearings = np.asarray(tracks_dataframe['storm direction'].loc[scan_index-1].iloc[:])
                                speeds_p = np.asarray(tracks_dataframe['storm speed'].loc[scan_index-1].iloc[:])
                                bearings[np.isnan(bearings)] = 0
                                bearings = bearings + 180
                                bearings[bearings > 360] = bearings[bearings > 360] - 360
                                speeds_p[np.isnan(speeds_p)] = 0   
                                #Next, get the number of elapsed seconds
                                end_time = time_start
                                try:
                                    start_time = np.asarray(tracks_dataframe['times'].loc[scan_index-1].iloc[:])[0]
                                except:
                                    start_time = np.asarray(tracks_dataframe['times'].loc[scan_index-1].iloc[:])[0]
                                print(start_time, 'st')
                                print(end_time, 'et')
                                try:
                                    stormdte = (np.datetime64(end_time) - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
                                    stormdtce = datetime.utcfromtimestamp(stormdte)
                                    stormdti = (np.datetime64(start_time) - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
                                    stormdtci = datetime.utcfromtimestamp(stormdti)
                                    storm_dur = stormdtce-stormdtci
                                    storm_sec = storm_dur.seconds
                                except:
                                    storm_sec=0

                                prj_lon, prj_lat, prj_bear = g.fwd(max_lons_p, max_lats_p, bearings, speeds_p*storm_sec)
                                ax.scatter(prj_lon, prj_lat, s=200, color='r')
                                dist_track = np.zeros((np.asarray(prj_lon).shape[0]))
                                for i in range(prj_lon.shape[0]):
                                    distance_track = g.inv(polygon.centroid.x, polygon.centroid.y,
                                                           prj_lon[i], prj_lat[i])
                                    dist_track[i] = distance_track[2]/1000.

                                if np.min(dist_track) < track_dis:
                                    storm_ids.append((storm_ids_p[np.where(dist_track == np.min(dist_track))[0][0]]))
                                else:
                                    storm_ids.append((storm_index))
                                    storm_index = storm_index + 1
                            #If previous storms cant be accessed due to a long gap with no storms, add new indicies
                            except:
                                storm_ids.append((storm_index))
                                storm_index = storm_index + 1

    storm_ids = np.asarray(storm_ids)
    max_lons_c = np.asarray(max_lons_c)
    max_lats_c = np.asarray(max_lats_c)
    #Removing duplicate storm ids and giving them new ids
    unique_ids = np.unique(storm_ids)
    for i_d in unique_ids:
        ids_i = storm_ids[np.where(storm_ids==i_d)]
        index_i = np.where(storm_ids==i_d)[0]
        lats_id = max_lats_c[np.where(storm_ids==i_d)]
        lons_id = max_lons_c[np.where(storm_ids==i_d)]
        if (ids_i.shape[0]) > 1:
            print('duplicate')
            dist_duplicates = np.zeros((np.asarray(lats_id).shape[0]))
            for k_d in range(ids_i.shape[0]):
                print(max_lons_p[np.where(storm_ids_p==i_d)])
                print(max_lats_p[np.where(storm_ids_p==i_d)])
                print(lons_id)
                print(lats_id)
                print(lons_id[k_d])
                print(lats_id[k_d])
                print(lons_id[k_d].shape, lats_id[k_d].shape)
                print(max_lons_p[np.where(storm_ids_p==i_d)].shape, max_lats_p[np.where(storm_ids_p==i_d)].shape)
                #Add a try statement for the extremely rare circulsgtance that a duplicate slips through for one scan.
                #Reconsider this section later
                try:
                    distance_duplicates = g.inv(lons_id[k_d], lats_id[k_d],
                                           max_lons_p[np.where(storm_ids_p==i_d)], max_lats_p[np.where(storm_ids_p==i_d)])
                except:
                    distance_duplicates = g.inv(lons_id[k_d], lats_id[k_d],
                                           max_lons_p[np.where(storm_ids_p==i_d)][0], max_lats_p[np.where(storm_ids_p==i_d)][0])
                dist_duplicates[k_d] = distance_duplicates[2]/1000.
            dist_duplicates=np.asarray(dist_duplicates)
            index_ns = np.where(dist_duplicates==np.max(dist_duplicates))[0]
            print(index_ns, "index_ns")
            print(int(index_ns))
            print(index_ns[0])
            print(index_i, 'index_i')
            index_rep = index_i[int(index_ns)]
            storm_ids[index_rep] = storm_index
            storm_index = storm_index + 1
    #Calculate storm motion vectors for all cells.
    alg_speeds = []
    alg_directions = []
    for c in range(len(storm_ids)):
        print(storm_ids[c], 'storm id')
        print(max_lons_c[c], 'storm lon')
        print(max_lats_c[c], 'storm lat')
        stormid_t = storm_ids[c]
        max_lon_te = max_lons_c[c]
        max_lat_te = max_lats_c[c]
            #Get previous locations of this storm
        track_token = 0
        try:
            storm_t = tracks_dataframe.xs(stormid_t, level = 1)
            max_lon_str = np.asarray(storm_t['storm_lon'].iloc[:])[:]
            max_lat_str = np.asarray(storm_t['storm_lat'].iloc[:])[:]
            times_str = np.asarray(storm_t['times'].iloc[:])[:]
            print('length of obs', np.shape(max_lon_str)[0])
            len_str = np.shape(max_lon_str)[0]
            ind_tr_start = len_str - 10
            if ind_tr_start < 0:
                ind_tr_start = 0
            track_token = 0
            
        except:
            print('looks like a new cell')
            track_token = 1
            alg_speeds.append(np.nan)
            alg_directions.append(np.nan)
        if track_token == 0:
            print(radar_motion(times_str[ind_tr_start], time_start, max_lon_str[ind_tr_start], max_lat_str[ind_tr_start], max_lon_te, max_lat_te))
            st_speed, st_direction = radar_motion(times_str[ind_tr_start], time_start, max_lon_str[ind_tr_start], max_lat_str[ind_tr_start], max_lon_te, max_lat_te)
            alg_speeds.append(st_speed)
            alg_directions.append(st_direction)
        
    #Returning Variables,
    #storm_ids: List of Storm IDs to track from iteration to iteration, incremented using storm_index
    #max_lons_c,max_lats_c: Centroid coordinates of storm objects
    #ref_areas: Area of storm objects at reflectivity contours of REFlev or REFlev1 for standard and "big storms" respectively
    #storm_index: Index to count and label storm objects as they are realized; now including possible new objects from this iteration
    return storm_ids,max_lons_c,max_lats_c,ref_areas,storm_index, alg_speeds, alg_directions