import cv2
import numpy as np


def take_webcam_photo():
    # define a video capture object
    vid = cv2.VideoCapture(0)
    # Capture the video frame by frame
    ret, frame = vid.read()
    vid.release()
    return frame

