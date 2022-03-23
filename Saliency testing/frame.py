import cv2
import numpy as np
from common import *


def process_image(image, display_steps=False):
    # Cook some code from Sindre!
    return image


class Frame:
    K = np.loadtxt('config/intrinsic_params.txt')
    dist_coeffs = np.loadtxt('config/dist_coeffs.txt')
    dist_coeffs = np.loadtxt('config/dist_coeffs.txt')
    width, height = np.loadtxt('config/image_dims.txt', dtype='int64')
    #pixel_density = np.array([[1.55e-6], [1.55e-6]]) # From specs. Probably not needed
    new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(
        K,
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
        self.image = image #cv2.undistort(image, self.K, self.dist_coeffs, None, self.new_camera_mtx)
        #x, y, w, h = self.roi
        #self.image = self.image[y:y+h, x:x+w]
        #self.width, self.height = w, h
        self.center_coord = center_coord
        self.height_above_sea = height_above_sea
        self.timestamp = timestamp
        self.orientation = orientation
    
    def process(self):
        self.image_processed = process_image(self.image, display_steps=True)
        return self.image_processed
    
    def find_centroids(self):
        #contours, _ = cv2.findContours(self.image_processed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #self.centroids = np.zeros((2, len(contours)))
        #self.contours = list(contours)
        #for i in range(len(self.contours)):
        #    self.contours[i] = np.squeeze(self.contours[i], axis=1)
        #    self.centroids[:,i] = np.mean(self.contours[i], axis=0)
        
        self.centroids = np.loadtxt('Positions/image1.txt').T
        
        return self.centroids
    
    def find_camera_coords(self):
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
        t_c_to_w = translate(150, 25, -self.height_above_sea)
        R_c_to_w = rotate_z(np.deg2rad(90 + 90))
        
        T_c_to_w = t_c_to_w @ R_c_to_w
        
        n_points = self.Xc.shape[1]
        Xc_tilde = np.column_stack((np.vstack((self.Xc, np.ones(n_points))), np.array([0,0,0,1])))
        self.Xw = (T_c_to_w @ Xc_tilde)[:3,:]
        
        return self.Xw
        