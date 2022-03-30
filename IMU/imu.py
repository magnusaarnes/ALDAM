import qwiic_icm20948
import time
import sys
import numpy as np

def IMUGetData():
    print("Starting ICM20948")
    IMU = qwiic_icm20948.QwiicIcm20948(address=0x69)
    if not IMU.connected:
        print("Could not initialize IMU. Check hardware connections")
        return


    # this does not work, don't know why
    # IMU.begin()    
    print("Data collection beginning")
    
    # IMU.swReset()
    #disable low power
    # IMU.lowPower(0)
    #if not IMU.magWhoIAm():
    #    print("Magnetometer is having an existential crisis")
    
    while True:
        if IMU.dataReady():
            IMU.getAgmt()
            a_x, a_y, a_z = IMU_convert_accelerometer_data(IMU.axRaw, IMU.ayRaw, IMU.azRaw)
            print("----------------------")
            print("ACC: {:.3f}, {:.3f}, {:.3f}".format(a_x, a_y, a_z))
            print("GYR: {: 06d}, {: 06d}, {:06d}".format(IMU.gxRaw, IMU.gyRaw, IMU.gzRaw))
            print("MAG: {: 06d}, {: 06d}, {:06d}".format(IMU.mxRaw, IMU.myRaw, IMU.mzRaw))

            time.sleep(0.1)
        else:
            print("Waitin for data")
            time.sleep(0.1)


# takes in raw data from accelerometer and returns in 'g'
# default resolution is +/- 2g 
def IMU_convert_accelerometer_data(x, y, z, res=2):
    # np. interp maps value from one range to another
    x = np.interp(x, [-2**15, 2**15], [-res, res])
    y = np.interp(y, [-2**15, 2**15], [-res, res])
    z = np.interp(z, [-2**15, 2**15], [-res, res])
    return x, y, z


if __name__ == "__main__":
    try:
        IMUGetData()
    except(KeyboardInterrupt, SystemExit) as exErr:
        print("\n Ending data collection")
        sys.exit(0)
