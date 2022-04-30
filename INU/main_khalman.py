import threading
import numpy as np
import imuhat.Qwiic.IMU_controller as IMU_cont
from SIM7600X.python.GPS import GPS
from imuhat.Qwiic.khalman import khalman_gps_imu
dt_imu = 0.3
dt_gps = 1.5
position = np.array(np.zeros(3))
velocity = np.array(np.zeros(3))
GPS_thread = threading.Thread(target=GPS.get_gps_position,args=())
states = np.array([*position,*velocity])
khalman_filt = threading.Thread(target=khalman_gps_imu, args=(states,dt_imu,dt_gps))
IMU_thread = threading.Thread(target=IMU_cont.start,args=())