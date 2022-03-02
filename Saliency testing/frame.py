import cv2
from matplotlib import image
import numpy as np

def process_image(image, display_steps=False):
    saliency = cv2.saliency.StaticSaliencyFineGrained_create()

    # Do the saliency
    success, image_saliency_map = saliency.computeSaliency(image)
    if not success:
        print("[ERROR] Saliency failed")
        return
    
    saliency_map_image = (image_saliency_map * 255).astype("uint8")

    saliency_map_image_bin = saliency_map_image > 60
    saliency_map_image_bin = saliency_map_image_bin.astype("uint8") * 255

    saliency_map_image_bin_blur = cv2.blur(saliency_map_image_bin, (23,23), 0)

    saliency_map_image_bin_blur_bin = saliency_map_image_bin_blur > 10
    saliency_map_image_bin_blur_bin = saliency_map_image_bin_blur_bin.astype("uint8") * 255
    
    if display_steps:
        # Display original image
        cv2.imshow('image', image)
        cv2.imshow('saliency', saliency_map_image)
        cv2.imshow('saliency_bin', saliency_map_image_bin)
        cv2.imshow('saliency_bin_blur', saliency_map_image_bin_blur)
        cv2.imshow('saliency_bin_blur_bin', saliency_map_image_bin_blur_bin)
    
    return saliency_map_image_bin_blur_bin

class Frame:
    def __init__(self,
                 image: np.ndarray,
                 center_coord: np.ndarray,
                 height: float):
        """
        Initialize a cameraframe with the image and the coordinate
        representing the point where the image was taken.
        """
        self.image = image
        self.center_coord = center_coord
        self.height = height
    
    def process(self):
        self.image_processed = process_image(self.image)
        print(self.image_processed.shape)
        return self.image_processed
    
    def get_centroids(self):
        contours, _ = cv2.findContours(self.image_processed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        self.centroids = np.zeros((2, len(contours)))
        self.contours = list(contours)
        for i in range(len(self.contours)):
            self.contours[i] = np.squeeze(self.contours[i], axis=1)
            self.centroids[:,i] = np.mean(self.contours[i], axis=0)
        
        return self.centroids