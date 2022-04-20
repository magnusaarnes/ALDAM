import cv2
#import PIL
import numpy as np
import json
import picamera
from picamera.array import PiRGBArray
#from matplotlib import pyplot as plt
import time
from Image_encoding.base64 import encode_image
import requests
import threading
from frame import Frame

url = "https://aldam-saliency.herokuapp.com/upload_img/"
cam = picamera.PiCamera()
cam.resolution = (1920, 1080)
cam.framerate = 10
raw_capture = PiRGBArray(cam, size=(1920, 1080))

#allow camera to wake up
time.sleep(0.1)

def main():
    frames = []
    # Iterator cycling through 0-9 in order to have the threads save
    # up to 10 different "temporary" images
    i = 0
    for image_frame in cam.capture_continuous(raw_capture, format="bgr", use_video_port=True):
        image = image_frame.array
        
        ##################
        # --- Fetch INS data ---
        # Not implemented in our project, but should be
        # fetched either from drone which has some INS,
        # or an INS should be implemented on the RPi as
        # well, in order to get pose of camera.
        ##################
        
        # Create a frame object with the current image-frame and some metadata
        frames.append(Frame(
            image=image,
            center_coord=[0.0, 0.0],
            height_above_sea=25,
            timestamp=time.time(),
            orientation=[0, 0, 0]))
        
        centroids = frames[i].find_centroids()
        Xc = frames[i].find_camera_coords()
        Xw = frames[i].find_world_coords()
        metadata = frames[i].get_metadata()
        
        send_marked_image = True
        if send_marked_image:
            cv2.imwrite(f'Temporary_images/temp{i}.png', frames[i].marked_image)
        else:
            cv2.imwrite(f'Temporary_images/temp{i}.png', frames[i].image)

        # Check that there are any detections
        num_detections = centroids.shape[1]
        if num_detections > 0:
            print("Detections - Sending things :(")
            t = threading.Thread(target=thread_upload_image, args=(f'Temporary_images/temp{i}.png', json.dumps(metadata)))
            t.start()
        else:
            print("No detections - no sending :)")

        if i % 10 == 0: i = 0
        i += 1
        
        # Clear frame buffer
        raw_capture.truncate(0)


def thread_upload_image(filename, metadata):
    img_str = encode_image(filename)
    data = {'image' : img_str, 'metadata': str(metadata)}
    try:
        r = requests.post(url=url, data=data)
    except:
        print("An error occured while trying to upload an image")


if __name__ == "__main__":
    main()