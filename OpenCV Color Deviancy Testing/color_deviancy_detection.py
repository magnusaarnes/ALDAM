import numpy as np
import cv2
import time
import sys

print(sys.argv)

### Set variables

### User functions
def get_centroids(image, mode=cv2.RETR_EXTERNAL, module=cv2.CHAIN_APPROX_NONE):
    contours, _ = cv2.findContours(image, mode, module)
    centroids = np.zeros((2, len(contours)))
    contours = list(contours)
    for i in range(len(contours)):
        contours[i] = np.squeeze(contours[i], axis=1)
        centroids[:,i] = np.mean(contours[i], axis=0)
    return centroids.astype(int)

def draw_centroids(centroids, image):
    marked_image = image.copy()
    if centroids.shape[0]>0 and centroids.shape[1]>0:
        for j in range(centroids.shape[1]):
            marked_image = cv2.circle(marked_image, (centroids[0,j], centroids[1,j]), radius=9, color=(0,0,255), thickness=2)
    return marked_image

def spectral_autodifference(image, blur_strength):
    # Blur normalized
    blurred = cv2.blur(image, (int(image.shape[0]*blur_strength),int(image.shape[0]*blur_strength)))
    
    # For each channel, compute pixel-wise distance from blurred normals
    color_deviancy = np.absolute(image - blurred)
    return color_deviancy

def detect_color_deviancies(frame, blur_strengths=(0.1,0.04), color_object_lower_thresh=0.3, booster_threshold=0.01,bilateral_filter_gains=(0.027,0.14375,0.14375), threshold_preblur_rad=(5,5)):
    start_time = time.time()

    res = frame.shape[0:2]
    print("Processing image of resolution",res)
    # Bilateral filtering parameters // Attempts are made to make these image-size invariant
    filter_neighborhood = int(bilateral_filter_gains[0]*res[1])
    sigmaColor          = int(bilateral_filter_gains[1]*res[1])
    sigmaSpace          = int(bilateral_filter_gains[2]*res[1])

    ## Image preprocessing
    processed_frame = frame.copy()
    
    # Pre-blurring and Bilateral Filtering on input image
    processed_frame = cv2.bilateralFilter(processed_frame, filter_neighborhood, sigmaColor=sigmaColor,sigmaSpace=sigmaSpace)

    ## Color filtering
    col_normd = processed_frame.copy()
    #Convert to floating-point / normalize values
    col_normd = col_normd/255.0
    # Normalize colors
    for c in range(3):
        col_normd[:,:,c] = col_normd[:,:,c] / (1.0/255.0 + col_normd[:,:,0] + col_normd[:,:,1] + col_normd[:,:,2]) # Normalize colors

    # For each channel, compute pixel-wise distance from blurred normals
    color_deviancy = spectral_autodifference(col_normd, blur_strengths[0])

    # Merge channels into 1 grayscale image (yes this does look dumb but it actually runs faster than 'sum of channels / 3')
    processed_deviancy = cv2.cvtColor((color_deviancy*255).astype(np.ubyte), cv2.COLOR_BGR2GRAY)/255.0

    # Boost pixels that stand out from their surroundings (compare with blurred self), over a certain threshold
    procdev_max             = np.max(processed_deviancy)
    normalized_diff         = spectral_autodifference(processed_deviancy, blur_strengths[1])/procdev_max
    processed_deviancy      = np.where(normalized_diff<booster_threshold, processed_deviancy, processed_deviancy/procdev_max)

    ## Thresholding and detection
    # Threshold values  [Input is slightly blurred, not sure why/if this might be a good idea]
    _disc, threshed_detections = cv2.threshold(cv2.blur(processed_deviancy,threshold_preblur_rad),color_object_lower_thresh,1,cv2.THRESH_BINARY)
    
    # Perform morphology to remove isolated pixels
    threshed_detections = cv2.morphologyEx(threshed_detections, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    
    # Contours and centroids
    threshed_detections = (threshed_detections*255).astype(np.ubyte)

    centroids = get_centroids(threshed_detections)
    marked_image = frame.copy()
    if centroids.shape[0]>0 and centroids.shape[1]>0:
        for j in range(centroids.shape[1]):
            marked_image = cv2.circle(marked_image, (centroids[0,j], centroids[1,j]), radius=9, color=(0,0,255), thickness=2)

    ## Image post-processing
    if len(threshed_detections.shape)<3:
        threshed_detections = cv2.cvtColor(threshed_detections, cv2.COLOR_GRAY2BGR)

    color_deviancy = (color_deviancy*255).astype(np.ubyte)      # Convert result to image-readable format
    col_normd = (col_normd*255).astype(np.ubyte)                    # Convert result to image-readable format

    # Convert from grayscale to colormap, just for the aesthetics
    processed_deviancy = cv2.applyColorMap(cv2.cvtColor((processed_deviancy*255).astype(np.ubyte),cv2.COLOR_GRAY2BGR), cv2.COLORMAP_INFERNO)

    # Image Merging
    hcon1 = cv2.hconcat([marked_image, processed_frame, col_normd])
    hcon2 = cv2.hconcat([color_deviancy, processed_deviancy, threshed_detections])
    displ = cv2.vconcat([hcon1, hcon2])

    # Downsize concat image
    displ = cv2.resize(displ, (int(480*2), int(270*2)))

    # Timings
    end_time = time.time()
    proc_time = round(end_time - start_time, 3)
    print('Processing time: ', proc_time, ' [s]')

    return marked_image, centroids, displ, threshed_detections

def detect_color_deviancies_thresholds(frame, blur_strength=0.04, color_object_lower_thresh=0.3, booster_threshold=0.01,bilateral_filter_gains=(0.027,0.14375,0.14375)):
    start_time = time.time()

    res = frame.shape[0:2]
    print(res)
    # Bilateral filtering parameters // Attempts are made to make these image-size invariant
    filter_neighborhood = int(bilateral_filter_gains[0]*res[1])
    sigmaColor          = int(bilateral_filter_gains[1]*res[1])
    sigmaSpace          = int(bilateral_filter_gains[2]*res[1])

    ## Image preprocessing
    processed_frame = frame.copy()
    
    # Pre-blurring and Bilateral Filtering on input image
    processed_frame = cv2.bilateralFilter(processed_frame, filter_neighborhood, sigmaColor=sigmaColor,sigmaSpace=sigmaSpace)

    ## Color filtering
    col_normd = processed_frame.copy()
    #Convert to floating-point / normalize values
    col_normd = col_normd/255.0
    # Normalize colors
    for c in range(3):
        col_normd[:,:,c] = col_normd[:,:,c] / (1.0/255.0 + col_normd[:,:,0] + col_normd[:,:,1] + col_normd[:,:,2]) # Normalize colors

    # For each channel, compute pixel-wise distance from blurred normals
    color_deviancy = spectral_autodifference(col_normd, blur_strength)

    # Merge channels into 1 grayscale image (yes this does look dumb but it actually runs faster than 'sum of channels / 3')
    processed_deviancy = cv2.cvtColor((color_deviancy*255).astype(np.ubyte), cv2.COLOR_BGR2GRAY)/255.0

    # Boost pixels that stand out from their surroundings (compare with blurred self), over a certain threshold
    procdev_max             = np.max(processed_deviancy)
    normalized_diff         = spectral_autodifference(processed_deviancy, blur_strength)/procdev_max
    processed_deviancy      = np.where(normalized_diff<booster_threshold, processed_deviancy, processed_deviancy/procdev_max)

    ## Thresholding and detection
    # Threshold values  [Input is slightly blurred, not sure why/if this might be a good idea]
    threshold_preblur = (5,5)
    _disc, threshed_detections = cv2.threshold(cv2.blur(processed_deviancy,threshold_preblur),color_object_lower_thresh,1,cv2.THRESH_BINARY)
    
    # Perform morphology to remove isolated pixels
    threshed_detections = cv2.morphologyEx(threshed_detections, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    
    # Contours and centroids
    threshed_detections = (threshed_detections*255).astype(np.ubyte)

    # Timings
    end_time = time.time()
    proc_time = round(end_time - start_time, 3)
    print('Processing time: ', proc_time, ' [s]\n')

    return threshed_detections