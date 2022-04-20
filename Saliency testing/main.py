import cv2
import PIL
import numpy as np
import picamera
from matplotlib import pyplot as plt
import time
import base64
import requests
from frame import Frame


def main():
    frames = []
    # For capture blablabla
    while capture:
        image = i
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
        
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(Xw[0,:], Xw[1,:], Xw[2,:], marker='x')
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
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
        if num_detections > 0:
            with open('temp.png', "rb") as img:
                image_base64 = base64.b64encode(img.read())
                data = { 'image' : image_base64 }
                try:
                    x = requests.post("https://aldam-saliency.herokuapp.com/upload_img/", data=data)
                except:
                    print(f"An error occured while trying to upload image {i+1}")


if __name__ == "__main__":
    main()