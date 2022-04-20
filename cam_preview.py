from picamera import PiCamera
from time import sleep

cam = PiCamera()
cam.start_preview()
sleep(10)
cam.stop_preview()
