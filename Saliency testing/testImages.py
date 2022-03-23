import cv2
import numpy as np
from matplotlib import pyplot as plt
from frame import Frame
    

if __name__ == "__main__":
    image = cv2.imread('images/image1.png')
    
    img = Frame(image, [0.0, 0.0], 25, 0.0, [0, 0, 0])
    
    img.process()
    
    centroids = img.find_centroids()
    Xc = img.find_camera_coords()
    Xw = img.find_world_coords()
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    #ax.scatter(Xc[0,:], Xc[1,:], Xc[2,:], marker='x')
    ax.scatter(Xw[0,:], Xw[1,:], Xw[2,:], marker='x')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    
    plt.figure()
    #cv2.drawContours(img.image, img.contours, -1, (0,255,0), 2)
    #cv2.imshow('contours', img.image)
    plt.imshow(cv2.cvtColor(img.image, cv2.COLOR_BGR2RGB))
    plt.scatter(centroids[0,:], centroids[1,:], c="r", marker="x")
    plt.show()
