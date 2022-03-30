import base64
import requests
import time
img_path = "/home/magnus/Dropbox/Bilder/viktigikkeslett.jpg"


with open(img_path, "rb") as img:
    #start = time.time()
    a = base64.b64encode(img.read())
    data = { 'image' : a }
    #x = requests.post("http://localhost:8000/upload_img/", data=data)
    x = requests.post("https://aldam-saliency.herokuapp.com/upload_img/", data=data)
    #print(x.json())
    # print(f"Server returned: {x} after {time.time()-start}")
