from turtle import color
import cv2
import PIL
import numpy as np
from matplotlib import pyplot as plt
import time
import os
import base64
import requests
from frame import Frame
    

if __name__ == "__main__":
    targetImage = PIL.Image.open("Test_images/test.png")
    for key, value in targetImage.text.items():
        print(key, ": ", value, sep="")
    
    in_dataset  = os.path.join(os.getcwd(), 'Test_images')
    image_files = [ f for f in os.listdir(in_dataset) if os.path.isfile(os.path.join(in_dataset,f)) ]
    images = []
    for i in range(len(image_files)):
        images.append(cv2.imread( os.path.join(in_dataset,image_files[i])))
    
    #image_files = ['Test_images/test.png']
    frames = []
    for i in range(len(image_files)):
        ##################
        # Fetch INS data.
        # Not implemented in our project, but shoul be
        # fetched either from drone which has some INS,
        # or an INS should be implemented on the RPi as
        # well, in order to get pose of camera.
        ##################
        
        frames.append(Frame(
            image=images[i],
            center_coord=[1.0, 3.0],
            height_above_sea=25,
            timestamp=time.time(),
            orientation=[0, 0, 45]))
        
        centroids = frames[i].find_centroids()
        Xc = frames[i].find_camera_coords()
        Xw = frames[i].find_world_coords()
        
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(Xw[0,:-1], Xw[1,:-1], Xw[2,:-1], marker='x', color='r')
        ax.scatter(Xw[0,-1], Xw[1,-1], Xw[2,-1], marker='o', color='b')
        ax.set_xlabel('North')
        ax.set_ylabel('East')
        ax.set_zlabel('Down')
        ax.set_xlim( -12, 10)
        ax.set_ylim( -8, 12)
        ax.set_zlim(-26,  2)
        ax.invert_zaxis()
        ax.invert_yaxis()
        plt.figure()
        plt.imshow(cv2.cvtColor(frames[i].image, cv2.COLOR_BGR2RGB))
        plt.scatter(centroids[0,:], centroids[1,:], c="r", marker="x")
        
        # Create PIL image obj in order to add metadata
        pil_image = PIL.Image.fromarray(cv2.cvtColor(frames[i].image, cv2.COLOR_BGR2RGB))
        
        # Add metadata
        metadata = PIL.PngImagePlugin.PngInfo()
        metadata.add_text("Time", str(frames[i].timestamp))
        metadata.add_text("Pos", str(frames[i].center_coord))
        metadata.add_text("Height", str(frames[i].height_above_sea))
        metadata.add_text("Ori", str(frames[i].orientation))
        metadata.add_text("Centroids:", np.array2string(frames[i].centroids))
        metadata.add_text("WorldCoords:", np.array2string(frames[i].Xw))
        
        # Temporarily save img with metadata
        pil_image.save('temp.png', pnginfo=metadata)
        targetImage = PIL.Image.open("temp.png")
        
        # Check that there are any detections
        num_detections = centroids.shape[1]
        print("Num det:", num_detections)
        if num_detections > 0 or True:
            with open('temp.png', "rb") as img:
                image_base64 = base64.b64encode(img.read())
                metadata = "Test"
                data = {'image' : image_base64, 'metadata': str(metadata)}
                try:
                    x = requests.post("https://aldam-saliency.herokuapp.com/upload_img/", data=data)
                    print("Uploaded shit")
                except:
                    print(f"An error occured while trying to upload image {i+1}")
    
    plt.show()
