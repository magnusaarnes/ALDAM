import cv2
from matplotlib import image
import numpy as np


def remove_specularity(img):
    # convert to gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    #GLARE_MIN = np.array([180//2, 0, 0],np.uint8)
    #GLARE_MAX = np.array([280//2, 0, 200],np.uint8)
    
    #frame_threshold = cv2.inRange(hsv_img, GLARE_MIN, GLARE_MAX)
    #print("SE HER")
    #print(np.min(frame_threshold), np.max(frame_threshold))
    #mask = 255 - frame_threshold
    
    # Reduce noise in the saturation channel
    hsv_img[:,:,1] = cv2.blur(hsv_img[:,:,1], (35,35), 0)
    
    # threshold grayscale image to extract glare
    mask = cv2.threshold(hsv_img[:,:,2], 220, 255, cv2.THRESH_BINARY)[1]
    kernel = np.ones((5,5),np.uint8)
    mask = cv2.dilate(mask,kernel,iterations = 1)

    # Optionally add some morphology close and open, if desired
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # use mask with input to do inpainting
    result = cv2.inpaint(img, mask, 21, cv2.INPAINT_TELEA) 

    # display it
    cv2.imshow("IMAGE", img)
    cv2.imshow("GRAY", gray)
    cv2.imshow("MASK", mask)
    cv2.imshow("RESULT", result)
    
    return result


def process_image(image, display_steps=False):
    # Remove specular reflections
    image = remove_specularity(image).copy()
    
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
    instrinsic_params = np.loadtxt('config/intrinsic_params.txt')
    dist_coeffs = np.loadtxt('config/dist_coeffs.txt')
    width, height = np.loadtxt('config/image_dims.txt', dtype='int64')
    new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(
        instrinsic_params,
        dist_coeffs,
        (width, height),
        1,
        (width, height))
    
    def __init__(self,
                 image: np.ndarray,
                 center_coord: list[float],
                 height_above_sea: float,
                 timestamp: float,
                 orientation: list[float]):
        """
        Initialize a cameraframe and undistort image.
        
        Parameters:
        - image (np.ndarray):         The image.
        - center_coord (list[float]): Longitude and latidute of where image is taken.
        - height_above_sea: (float):  Height above sea level of where image is taken.
        - timestamp (float):          Unix timestamp of when image is taken.
        - orientation (float):        Orientation of camera in terms of [roll, pitch, yaw].
        """
        self.image = image #cv2.undistort(image, self.instrinsic_params, self.dist_coeffs, None, self.new_camera_mtx)
        self.center_coord = center_coord
        self.height_above_sea = height_above_sea
        self.timestamp = timestamp
        self.orientation = orientation
    
    def process(self):
        self.image_processed = process_image(self.image, display_steps=True)
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