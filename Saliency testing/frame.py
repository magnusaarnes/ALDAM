import cv2
import numpy as np
from common import *
from colorDeviancyDetection import detect_color_deviancies


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
    
    def find_centroids(self):
        # Downsize to new dimensions
        w_new = 480
        h_new = 270
        processed_img = cv2.resize(self.image.copy(), (w_new, h_new))
        scale_x = self.image.shape[1] / w_new
        scale_y = self.image.shape[0] / h_new
        _, centroids, _, _ = detect_color_deviancies(processed_img)
        self.centroids = centroids
        self.centroids[0,:] = (self.centroids[0,:].astype(np.float64) * scale_x).astype(np.int32)
        self.centroids[1,:] = (self.centroids[1,:].astype(np.float64) * scale_y).astype(np.int32)
        
        return self.centroids
    
    def find_camera_coords(self):
        assert hasattr(self, 'K'), "Could not find camera matrix"
        assert hasattr(self, 'centroids'), "Run `Frame.find_centroids() first`"
        # focal lengths and principal point (~center)
        fx_pixels = self.K[0,0]
        fy_pixels = self.K[1,1]
        f = np.array([[fx_pixels], [fy_pixels]])
        
        cx = self.K[0,2]
        cy = self.K[1,2]
        principal_point = np.array([[cx], [cy]])
        
        n_points = self.centroids.shape[1]
        self.Xc = np.empty((3,n_points))
        
        # Camera coord Z is aproximately equal height aboe sea level
        Zc = self.height_above_sea
        
        self.Xc[:2,:] = Zc * (self.centroids - principal_point) / f
        self.Xc[2,:] = Zc
        
        return self.Xc
    
    def find_world_coords(self):
        assert hasattr(self, 'Xc'), "Run `Frame.find_camera_coords() first`"
        t_c_to_w = translate(150, 25, -self.height_above_sea)
        R_c_to_w = rotate_z(np.deg2rad(90 + 90))
        
        T_c_to_w = t_c_to_w @ R_c_to_w
        
        n_points = self.Xc.shape[1]
        Xc_tilde = np.vstack((self.Xc, np.ones(n_points)))
        add_camera_pos = False
        if add_camera_pos:
            Xc_tilde = np.column_stack((Xc_tilde, np.array([0,0,0,1])))
        self.Xw = (T_c_to_w @ Xc_tilde)[:3,:]
        
        return self.Xw
    
    def get_metadata(self):
        assert hasattr(self, 'Xw'), "Run `Frame.find_world_coords() first`"
        
        self.metadata = dict()
        self.metadata["Time"]        = str(self.timestamp)
        self.metadata["Pos"]         = str(self.center_coord)
        self.metadata["Height"]      = str(self.height_above_sea)
        self.metadata["Ori"]         = str(self.orientation)
        self.metadata["Centroids"]   = np.array2string(self.centroids)
        self.metadata["WorldCoords"] = np.array2string(self.Xw)
        