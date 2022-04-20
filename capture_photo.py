import picamera
import time
from Image_encoding.base64 import encode_image
import requests

filename = "test.jpg"
url = "https://aldam-saliency.herokuapp.com/upload_img/"
cam = picamera.PiCamera()


def capture_image_and_upload():
    #start = time.time()
    cam.capture(filename)
    img_str = encode_image(filename)
    data = {'image' : img_str}
    r = requests.post(url=url, data=data)
    #print(f"It took {round(time.time() - start, 3)}s to capture and upload the image")
