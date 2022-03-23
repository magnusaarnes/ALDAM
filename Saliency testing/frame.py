import cv2
import numpy as np
from common import *


def process_image(image, w, h):
    # Downsize to new dimensions
    w_new = 480
    h_new = 270
    # Tunable parameters
    blur_str                    = 0.015
    color_object_lower_thresh   = 0.3
    booster_threshold           = 0.01
    # Bilateral filtering parameters // Attempts are made to make these image-size invariant
    neighborhood_gain   = 0.027
    sigmaColor_gain     = 0.14375
    sigmaSpace_gain     = 0.14375
    filter_neighborhood = int(neighborhood_gain*w_new)
    sigmaColor          = int(sigmaColor_gain*w_new)
    sigmaSpace          = int(sigmaSpace_gain*w_new)
    
    processed_img = cv2.resize(image.copy(), (w_new, h_new))
    # Bilateral filtering on input image
    processed_img = cv2.bilateralFilter(processed_img, filter_neighborhood, sigmaColor=sigmaColor,sigmaSpace=sigmaSpace)
    
    # Color normalization
    processed_img = processed_img / 255.0
    for c in range(3):
        processed_img[:,:,c] = processed_img[:,:,c] / (1.0/255.0 + processed_img[:,:,0] + processed_img[:,:,1] + processed_img[:,:,2]) # Normalize colors
    
    # Blurred version for autocomparison
    blur_str = 0.02
    blur_rad = int(w_new*blur_str)
    blurred_processed_img = cv2.blur(processed_img,(blur_rad,blur_rad))
    
    # For each channel, compute pixel-wise distance from blurred normals
    processed_img = np.absolute(processed_img - blurred_processed_img)
    
    # Merge channels into 1 grayscale image (yes this does look dumb but it actually runs faster than 'sum of channels / 3')
    processed_img = cv2.cvtColor((processed_img*255).astype(np.ubyte), cv2.COLOR_BGR2GRAY)/255.0

    # Boost pixels that stand out from their surroundings (compare with blurred self), over a certain threshold
    processed_img_max   = np.max(processed_img)
    processed_img_blur  = cv2.blur(processed_img, (blur_rad, blur_rad))
    processed_img_diff  = np.absolute(processed_img - processed_img_blur) / processed_img_max
    processed_img       = np.where(processed_img_diff < booster_threshold, processed_img, processed_img/processed_img_max)
    
    ## Thresholding and detection
    # Threshold values  [Input is slightly blurred, not sure why/if this might be a good idea]
    _disc, processed_img = cv2.threshold(cv2.blur(processed_img, (5,5)), color_object_lower_thresh, 1, cv2.THRESH_BINARY)
    
    # Perform morphology to remove isolated pixels
    processed_img = cv2.morphologyEx(processed_img, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    
    processed_img = (processed_img*255).astype(np.ubyte)
    
    # Return image in original size
    return cv2.resize(processed_img, (w, h))


class Frame:
    K = np.loadtxt('config/intrinsic_params.txt')
    dist_coeffs = np.loadtxt('config/dist_coeffs.txt')
    dist_coeffs = np.loadtxt('config/dist_coeffs.txt')
    width, height = np.loadtxt('config/image_dims.txt', dtype='int64')
    #pixel_density = np.array([[1.55e-6], [1.55e-6]]) # From specs. Probably not needed
    # Comment in next lines and remove from init later
    #new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(
    #    K,
    #    dist_coeffs,
    #    (width, height),
    #    1,
    #    (width, height))
    
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
        self.image = image
        self.width = image.shape[1]
        self.height = image.shape[0]
        new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(
            self.K,
            self.dist_coeffs,
            (self.width, self.height),
            1,
            (self.width, self.height))
        #self.image = cv2.undistort(image, self.K, self.dist_coeffs, None, self.new_camera_mtx)
        #x, y, w, h = self.roi
        #self.image = self.image[y:y+h, x:x+w]
        #self.width, self.height = w, h
        self.center_coord = center_coord
        self.height_above_sea = height_above_sea
        self.timestamp = timestamp
        self.orientation = orientation
    
    def process(self):
        self.image_processed = process_image(self.image, self.width, self.height)
        return self.image_processed
    
    def find_centroids(self):
        #contours, _ = cv2.findContours(self.image_processed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #self.centroids = np.zeros((2, len(contours)))
        #self.contours = list(contours)
        #for i in range(len(self.contours)):
        #    self.contours[i] = np.squeeze(self.contours[i], axis=1)
        #    self.centroids[:,i] = np.mean(self.contours[i], axis=0)
        
        #self.centroids = np.loadtxt('Positions/image1.txt').T
        
        assert hasattr(self, 'image_processed'), "Run `Frame.process() first`"
        
        contours, _ = cv2.findContours(self.image_processed, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
        centroids = np.zeros((2, len(contours)))
        contours = list(contours)
        for i in range(len(contours)):
            contours[i] = np.squeeze(contours[i], axis=1)
            centroids[:,i] = np.mean(contours[i], axis=0)
        self.centroids =  centroids.astype(float)
        
        return self.centroids
    
    def find_camera_coords(self):
        assert hasattr(self, 'K'), "Could not find camera matrix"
        assert hasattr(self, 'centroids'), "Run `Frame.find_centroids() first`"
        # focal lengths and principal point (~center)
        fx_pixels = self.K[0,0]
        fy_pixels = self.K[1,1]
        f = np.array([[fx_pixels], [fy_pixels]])
        print("Focal lengths", f)
        cx = self.K[0,2]
        cy = self.K[1,2]
        principal_point = np.array([[cx], [cy]])
        print("Principal point:", principal_point)
        n_points = self.centroids.shape[1]
        self.Xc = np.empty((3,n_points))
        
        # Camera coord Z is aproximately equal height aboe sea level
        Zc = self.height_above_sea
        print("Height: ", Zc)
        print("Centroids - CxCy:\n", (self.centroids - principal_point)[:,:4])
        self.Xc[:2,:] = Zc * (self.centroids - principal_point) / f
        self.Xc[2,:] = Zc
        print(self.Xc[:,:4])
        
        return self.Xc
    
    def find_world_coords(self):
        assert hasattr(self, 'Xc'), "Run `Frame.find_camera_coords() first`"
        t_c_to_w = translate(150, 25, -self.height_above_sea)
        R_c_to_w = rotate_z(np.deg2rad(90 + 90))
        
        T_c_to_w = t_c_to_w @ R_c_to_w
        
        n_points = self.Xc.shape[1]
        Xc_tilde = np.column_stack((np.vstack((self.Xc, np.ones(n_points))), np.array([0,0,0,1])))
        self.Xw = (T_c_to_w @ Xc_tilde)[:3,:]
        
        return self.Xw
        