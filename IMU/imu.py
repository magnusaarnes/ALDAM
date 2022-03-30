import qwiic_icm20948
import time
import sys

def IMUGetData():
    print("Starting ICM20948")
    IMU = qwiic_icm20948.QwiicIcm20948(address=0x69)
    if not IMU.connected:
        print("Could not initialize IMU. Check hardware connections")
        return


    # this does not work, don't know why
    IMU.begin()    
    print("Data collection beginning")
    
    # IMU.swReset()
    #disable low power
    # IMU.lowPower(0)
    #if not IMU.magWhoIAm():
    #    print("Magnetometer is having an existential crisis")
    
    while True:
        if IMU.dataReady():
            IMU.getAgmt()
            print("----------------------")
            print("ACC: {: 06d}, {: 06d}, {:06d}".format(IMU.axRaw, IMU.ayRaw, IMU.azRaw))
            print("GYR: {: 06d}, {: 06d}, {:06d}".format(IMU.gxRaw, IMU.gyRaw, IMU.gzRaw))
            print("MAG: {: 06d}, {: 06d}, {:06d}".format(IMU.mxRaw, IMU.myRaw, IMU.mzRaw))

            time.sleep(0.1)
        else:
            print("Waitin for data")
            time.sleep(0.1)

if __name__ == "__main__":
    try:
        IMUGetData()
    except(KeyboardInterrupt, SystemExit) as exErr:
        print("\n Ending data collection")
        sys.exit(0)
