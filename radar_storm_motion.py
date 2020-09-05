from pyproj import Geod
import numpy as np
from datetime import datetime, timedelta

#Create geod object for later distance and area calculations
g = Geod(ellps='sphere')
def radar_motion(start_time, end_time, sloni, slati, slone, slate):

    stormdte = (np.datetime64(end_time) - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
    stormdtce = datetime.utcfromtimestamp(stormdte)
    stormdti = (np.datetime64(start_time) - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
    stormdtci = datetime.utcfromtimestamp(stormdti)
    distance_track = g.inv(sloni, slati,
                           slone, slate)
    dist_track = distance_track[2]                                        
    if distance_track[1] < 0:
        back = distance_track[1] + 360
    else:
        back = distance_track[1]
    storm_dur = stormdtce-stormdtci
    storm_sec = storm_dur.seconds
    speed = dist_track/storm_sec
    direc = back
    return speed, direc