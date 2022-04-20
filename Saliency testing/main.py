import cv2
import PIL
import numpy as np
import json
import picamera
#from matplotlib import pyplot as plt
import time
import base64
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
        start = time.time()
        
        image = image_frame.array
        
        print(f"It took {round(time.time() - start, 3)}s to capture image")
        
        ##################
        # Fetch INS data.
        # Not implemented in our project, but shoul be
        # fetched either from drone which has some INS,
        # or an INS should be implemented on the RPi as
        # well, in order to get pose of camera.
        ##################
        
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
        
        #fig = plt.figure()
        #ax = fig.add_subplot(projection='3d')
        #ax.scatter(Xw[0,:], Xw[1,:], Xw[2,:], marker='x')
        #ax.set_xlabel('X Label')
        #ax.set_ylabel('Y Label')
        #ax.set_zlabel('Z Label')
        #plt.figure()
        #plt.imshow(cv2.cvtColor(frames[i].image, cv2.COLOR_BGR2RGB))
        #plt.scatter(centroids[0,:], centroids[1,:], c="r", marker="x")
        
        # Create PIL image obj in order to add metadata
        pil_image = PIL.Image.fromarray(cv2.cvtColor(frames[i].image, cv2.COLOR_BGR2RGB))
        
        # Temporarily save img so a thread can pick it up and upload it
        pil_image.save(f'temp{i}.png')
        
        # Check that there are any detections
        num_detections = centroids.shape[1]
        if num_detections > 0:
            t = threading.Thread(target=thread_upload_image, args=(f'temp{i}.png', json.dumps(metadata)))
            t.start()

        if i % 10 == 0: i = 0
        i += 1
        
        # Clear frame buffer
        raw_capture.truncate(0)
        
        print(f"It took {round(time.time() - start, 3)}s to process image")


def thread_upload_image(filename, metadata):
    img_str = base64.b64encode(filename)
    data = {'image' : img_str, 'metadata': metadata}
    try:
        r = requests.post(url=url, data=data)
    except:
        print("An error occured while trying to upload an image")
        


if __name__ == "__main__":
    main()