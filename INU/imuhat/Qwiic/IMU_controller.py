

from __future__ import print_function
import qwiic_icm20948
import numpy as np
import time
import sys
from khalman import khalman_gps_imu
def start(filter_khal,dt):
	print("\nSparkFun 9DoF ICM-20948 Sensor  Example 1\n")
	IMU = qwiic_icm20948.QwiicIcm20948()

	if IMU.connected == False:
		print("The Qwiic ICM20948 device isn't connected to the system. Please check your connection", \
			file=sys.stderr)
		return

	IMU.begin()

	while True:
		if IMU.dataReady():
			IMU.getAgmt() # read all axis and temp from sensor, note this also updates all instance variables
			imu_data = np.hstack([IMU.axRaw,IMU.ayRaw,IMU.azRaw,
								  IMU.gxRaw,IMU.gyRaw,IMU.gzRaw,
								  IMU.mxRaw,IMU.myRaw,IMU.mzRaw])
			
			acel = filter_khal.get_accel_imu(imu_data)
			filter_khal.update_imu(acel)
			time.sleep(dt)
		else:
			time.sleep(dt)

if __name__ == '__main__':
	try:
		dt = 0.3
		filter_khal = khalman_gps_imu(np.vstack([0,0,0,0,0,0]),dt,1.5)
		start(filter_khal,dt)
	except (KeyboardInterrupt, SystemExit) as exErr:
		print("\nEnding Example 1")
		sys.exit(0)


