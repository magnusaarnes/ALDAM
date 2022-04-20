import picamera
from picamera.array import PiRGBArray
import time
from Image_encoding.base64 import encode_image
import requests
import cv2
import threading

filename = "test.jpg"
url = "https://aldam-saliency.herokuapp.com/upload_img/"
cam = picamera.PiCamera()
cam.resolution = (1920, 1080)
cam.framerate = 10
raw_capture = PiRGBArray(cam, size=(1920, 1080))

#allow camera to wake up
time.sleep(0.1)

def thread_upload_image(filename, metadata):
    img_str = encode_image(filename)
    data = {'image' : img_str, 'metadata', metadata}
    r = requests.post(url=url, data=data)
    #print("Upload done")

def capture_video_stream():
    for frame in cam.capture_continuous(raw_capture, format="bgr", use_video_port=True):
        start = time.time()
        img = frame.array
        cv2.imwrite(filename, img)
        metadata = ""
        t = threading.Thread(target=thread_upload_image, args=(filename, metadata,))
        t.start()
        raw_capture.truncate(0)
        print(f"It took {round(time.time() - start, 3)}s to capture image")
        
def capture_image_and_upload():
    start = time.time()
    cam.capture(filename)
    img_str = encode_image(filename)
    data = {'image' : img_str}
    r = requests.post(url=url, data=data)
    print(f"It took {round(time.time() - start, 3)}s to capture and upload the image")

if __name__ == "__main__":
    capture_video_stream()
    #    for i in range(3):
#        capture_image_and_upload()
