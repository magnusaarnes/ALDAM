from locale import normalize
from tabnanny import verbose
import cv2, sys
import numpy as np
from scipy.optimize import linear_sum_assignment, least_squares

# Last inn bilde

# Step 1: Bilateral filter
# Step 2: Image Normalization
# Step 2: Saliency

images = [cv2.imread(f'images/image{i+1}.png') for i in range(5)]
positions = [np.loadtxt(f'Positions/image{i+1}.txt', ndmin=2).T for i in range(5)]
use_images = [2, 3]

saliency = cv2.saliency.StaticSaliencyFineGrained_create()


def residuals(p):
    processed_images = [images[i].copy() for i in use_images]
    #Bilateral filter
    dia = round(p[0])
    sigma_color = p[1]
    sigma_space = p[2]
    
    # Gaussian blur filter size
    f_size = 2*round(p[3])+1 # Must be odd number
    
    # Thresholds
    thr1 = p[4]
    thr2 = p[5]
    
    points_hat = []
    res = np.zeros(len(use_images))
    
    for i, im_ind in enumerate(use_images):
        processed_images[i] = cv2.bilateralFilter(processed_images[i], dia, sigma_color, sigma_space)
        
        success, processed_images[i] = saliency.computeSaliency(processed_images[i])
        if not success:
            print("[ERROR] Saliency failed")
            return
        processed_images[i] = (processed_images[i] * 255).astype("uint8")
        
        processed_images[i] = processed_images[i] > thr1
        processed_images[i] = processed_images[i].astype("uint8") * 255

        processed_images[i] = cv2.blur(processed_images[i], (f_size,f_size), 0)

        processed_images[i] = processed_images[i] > thr2
        processed_images[i] = processed_images[i].astype("uint8") * 255
        
        contours, _ = cv2.findContours(processed_images[i], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        centroids = np.zeros((2, len(contours)))
        contours = list(contours)
        for j in range(len(contours)):
            contours[j] = np.squeeze(contours[j], axis=1)
            centroids[:,j] = np.mean(contours[j], axis=0)
        points_hat.append(centroids)

        # Calculate residuals
        dist = lambda points : np.dot(points[0]-points[1], points[0]-points[1])
        
        n_pred = centroids.shape[1]
        n_gt = positions[im_ind].shape[1]
        cost_matrix = np.zeros((n_pred, n_gt))

        for pred in range(n_pred):
            for gt in range(n_gt):
                distance = dist([centroids[:,pred], positions[im_ind][:,gt]])
                cost_matrix[pred,gt] = distance if distance > 400 else -1

        pred_ind, gt_ind = linear_sum_assignment(-cost_matrix)
        
        # Remove indices corresponding to costs of -1 (iou to low)
        ind_remove = np.array([i for i in range(len(pred_ind)) if cost_matrix[pred_ind[i],gt_ind[i]] < 0])
        if len(ind_remove > 0):
            pred_ind = np.delete(pred_ind, ind_remove)
            gt_ind = np.delete(gt_ind, ind_remove)
        
        #print("cost_matrix:", cost_matrix)
        #print("pred_ind:", pred_ind)
        #print("gt_ind:", gt_ind)
        #print("cost_matrix.shape:", cost_matrix.shape)
        #print("cost_matrix[pred_ind, gt_ind]", cost_matrix[pred_ind, gt_ind])
        
        res[i] = np.mean(cost_matrix[pred_ind, gt_ind])
    
    return res


for i in range(5):
    pass
    #print(positions[i].shape, ":\n", positions[i])
    #cv2.imshow(f"Image {i+1}", images[i])

dia_range       = range(1, 20, 4)
sigma_col_range = range(30, 130, 4)
sigma_sp_range  = range(30, 130, 4)
f_size_range    = range(3, 50, 4)
thr1_range      = range(30, 100, 10)
thr2_range      = range(4, 20, 4)

n_dia       = len(dia_range)
n_sigma_col = len(sigma_col_range)
n_sigma_sp  = len(sigma_sp_range)
n_f_size    = len(f_size_range)
n_thr1      = len(thr1_range)
n_thr2      = len(thr2_range)

print("Num iters:", n_dia*n_sigma_col*n_sigma_sp*n_f_size*n_thr1*n_thr2)

cost = np.zeros((n_dia, n_sigma_col, n_sigma_sp, n_f_size, n_thr1, n_thr2, len(use_images)))
for i_dia, dia in enumerate(dia_range):
    print("i_dia:", i_dia, "of", n_dia)
    for i_sigma_col, sigma_col in enumerate(sigma_col_range):
        print("i_sigma_col:", i_sigma_col, "of", n_sigma_col)
        for i_sigma_sp, sigma_sp in enumerate(sigma_sp_range):
            print("i_sigma_sp:", i_sigma_sp, "of", n_sigma_sp)
            for i_f_size, f_size in enumerate(f_size_range):
                print("i_f_size:", i_f_size, "of", n_f_size)
                for i_thr1, thr1 in enumerate(thr1_range):
                    print("i_thr1:", i_thr1, "of", n_thr1)
                    for i_thr2, thr2 in enumerate(thr2_range):
                        print("i_thr2:", i_thr2, "of", n_thr2)
                        p = [dia, sigma_col, sigma_sp, f_size, thr1, thr2]
                        cost[i_dia, i_sigma_col, i_sigma_sp, i_f_size, i_thr1, i_thr2, :] \
                            = residuals(p)

#p0 = [5, 40, 40, 10, 60, 10]

print(cost)

print(p)

cv2.waitKey(-1)