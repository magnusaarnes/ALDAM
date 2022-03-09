import cv2
import numpy as np
from matplotlib import pyplot as plt
from frame import Frame
    

if __name__ == "__main__":
    image = cv2.imread('images/image6.png')
    
    img = Frame(image, [0.0, 0.0], 25, 0.0, [0, 0, 0])
    
    img.process()
    
    centroids = img.get_centroids()
    
    cv2.drawContours(img.image, img.contours, -1, (0,255,0), 2)
    #cv2.imshow('contours', img.image)
    plt.imshow(cv2.cvtColor(img.image, cv2.COLOR_BGR2RGB))
    plt.scatter(centroids[0,:], centroids[1,:], c="r", marker="x")
    plt.show()
