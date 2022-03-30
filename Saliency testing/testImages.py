import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import base64
import requests
from frame import Frame
    

if __name__ == "__main__":
    in_dataset  = os.path.join(os.getcwd(), 'Images')
    image_files = [ f for f in os.listdir(in_dataset) if os.path.isfile(os.path.join(in_dataset,f)) ]
    images = []
    for i in range(len(image_files)):
        images.append(cv2.imread( os.path.join(in_dataset,image_files[i])))
        
    with open('Images/image1.png', "rb") as img:
        image_base64 = base64.b64encode(img.read())
        data = { 'image' : image_base64 }
        print(image_base64)
        x = requests.post("https://aldam-saliency.herokuapp.com/upload_img/", data=data)
        print(x)
    
    exit()
    
    frames = []
    for i in range(len(image_files)):
        frames.append(Frame(images[i], [0.0, 0.0], 25, 0.0, [0, 0, 0]))
        frames[i].process()
        centroids = frames[i].find_centroids()
        Xc = frames[i].find_camera_coords()
        Xw = frames[i].find_world_coords()
        
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(Xw[0,:], Xw[1,:], Xw[2,:], marker='x')
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        plt.figure()
        plt.imshow(cv2.cvtColor(frames[i].image, cv2.COLOR_BGR2RGB))
        plt.scatter(centroids[0,:], centroids[1,:], c="r", marker="x")
        
        num_detections = centroids.shape[1]
        if num_detections > 0:
            image_base64 = base64.b64encode(frames[i].image)
            data = { 'image' : image_base64 }
            print(image_base64)
            #x = requests.post("http://localhost:8000/upload_img/", data=data)
            x = requests.post("https://aldam-saliency.herokuapp.com/upload_img/", data=data)
    
    plt.show()
