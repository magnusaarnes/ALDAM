import numpy as np
import cv2
import time
import os
import sys
import scipy.signal as signal

print(sys.argv)

### Set variables


### User functions
def strideConv(arr, arr2, s):
    return signal.convolve2d(arr, arr2[::-1, ::-1], mode='valid')[::s, ::s]

def get_centroids(image):
    contours, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    centroids = np.zeros((2, len(contours)))
    contours = list(contours)
    for i in range(len(contours)):
        contours[i] = np.squeeze(contours[i], axis=1)
        centroids[:,i] = np.mean(contours[i], axis=0)

def spectral_autodifference(image, blur_strength, color_image=True):
    # Blur normalized
    blurred = cv2.blur(image, (int(image.shape[0]*blur_strength),int(image.shape[1]*blur_strength)))
    
    # For each channel, compute pixel-wise distance from blurred normals
    if color_image:
        color_deviancy  = np.zeros(image.shape)
        for c in range(3):
            color_deviancy[:,:,c] = np.absolute(image[:,:,c] - blurred[:,:,c])
    else:
        color_deviancy = np.absolute(image - blurred)
    return color_deviancy


## Standard image resolutions
# Downsize resolution
hi = 270
wi = 480
# Concat output resolution
con_hi = 1080
con_wi = 1920

## Preprocessing variables
blur_str = 0.015
color_filter_lower_thresh = 60

## Preprocessing options
downsize                = 1
bilat_filtering         = 1

# Data directory paths
in_dataset  = os.path.join(os.getcwd(), 'Input_Dataset')
conc_output = os.path.join(os.getcwd(), 'Color_Concat_Outputs')
cont_output = os.path.join(os.getcwd(), 'Conture_Outputs')

## Prepare script
# Load images from folder into array
onlyfiles = [ f for f in os.listdir(in_dataset) if os.path.isfile(os.path.join(in_dataset,f)) ]
images = np.empty(len(onlyfiles), dtype=object)
for n in range(0, len(onlyfiles)):
  images[n] = cv2.imread( os.path.join(in_dataset,onlyfiles[n]) )

# Some user interface stuff
print("Processing images in folder ", in_dataset, "(", len(onlyfiles), " objects )\n")
if downsize:
    print("Images will be downsized to ", hi, "x",wi)
print("\n")
runtimes = np.zeros(len(onlyfiles))

## Main Loop
# For each image in list, perform saliency stuff
for i in range(0,len(images)):
    start_time = time.time()

    # Extract image
    frame = images[i]

    ## Image preprocessing
    blur_rad = int (frame.shape[0]*blur_str)

    print('Processing image "',onlyfiles[i],'", ',i+1,'/',len(images),'. Shape: ', frame.shape)
    preprocessed_frame = frame.copy()

    if downsize:        # Resize images to standard resolution
        frame = cv2.resize(frame,(wi, hi))
        preprocessed_frame = frame.copy()
    if bilat_filtering: # Bilateral Filter on input image
        preprocessed_frame = cv2.bilateralFilter(preprocessed_frame, 13, 69,69)

    ## Color filtering
    col_normd = preprocessed_frame.copy()
    
    #Convert to floating-point / normalize values
    col_normd = col_normd/255.0
    # Normalize colors
    for c in range(3):
        col_normd[:,:,c] = col_normd[:,:,c] / (1.0/255.0 + col_normd[:,:,0] + col_normd[:,:,1] + col_normd[:,:,2]) # Normalize colors

    # Blur normalized
    blurred_col_normd = cv2.blur(col_normd,(blur_rad,blur_rad))

    # For each channel, compute pixel-wise distance from blurred normals
    color_deviancy = np.zeros(col_normd.shape)
    for c in range(0,3):
        color_deviancy[:,:,c] = np.absolute(col_normd[:,:,c] - blurred_col_normd[:,:,c])
    #color_deviancy = np.absolute(col_normd - blurred_col_normd)

    """
    # Selective brightness boosting
    booster_threshold = 0.2
    boosted_deviancy = np.sum(color_deviancy, axis=2)/3
    higest_val = np.max(boosted_deviancy)
    boosted_deviancy[np.absolute(np.average(boosted_deviancy) - higest_val) > booster_threshold] = 1
    boosted_deviancy = cv2.applyColorMap(cv2.cvtColor((boosted_deviancy*255).astype(np.ubyte),cv2.COLOR_GRAY2BGR), cv2.COLORMAP_INFERNO)
    """

    processed_deviancy = color_deviancy.copy()

    booster_threshold = 0.01

    processed_deviancy = np.sum(color_deviancy, axis=2)/3
    processed_deviancy_blur = cv2.blur(processed_deviancy, (blur_rad, blur_rad))
    maxus = np.max(processed_deviancy)
    normalized_diff = (np.absolute(processed_deviancy - processed_deviancy_blur))/maxus
    processed_deviancy = np.where(normalized_diff<booster_threshold, processed_deviancy, processed_deviancy/maxus)
    #processed_deviancy = cv2.resize(processed_deviancy, (wi,hi), interpolation=cv2.INTER_AREA)
    print(booster_threshold, round(maxus,4), round(np.average(normalized_diff),4), round(np.max(normalized_diff),4))


    processed_deviancy = cv2.applyColorMap(cv2.cvtColor((processed_deviancy*255).astype(np.ubyte),cv2.COLOR_GRAY2BGR), cv2.COLORMAP_INFERNO)


    color_deviancy = (color_deviancy*255).astype(np.ubyte)      # Convert result to image-readable format
    col_normd = (col_normd*255).astype(np.ubyte)                    # Convert result to image-readable format
    blurred_col_normd = (blurred_col_normd*255).astype(np.ubyte)        # Convert result to image-readable format

    # Threshold values
    _disc, threshed_detections = cv2.threshold(cv2.cvtColor(cv2.blur(processed_deviancy,(blur_rad//2,blur_rad//2)), cv2.COLOR_BGR2GRAY),color_filter_lower_thresh,255,cv2.THRESH_BINARY)
    
    # Perform morphology to remove isolated pixels
    threshed_detections = cv2.morphologyEx(threshed_detections, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    
    ## Detection
    # Contours
    # Centroids
    #threshed_detections = cv2.cvtColor(threshed_detections, cv2.COLOR_GRAY2BGR)
    
    ## Image post-processing
    print(threshed_detections.shape)
    if len(threshed_detections.shape)<3:
        threshed_detections = cv2.cvtColor(threshed_detections, cv2.COLOR_GRAY2BGR)

    # Image Merging
    hcon1 = cv2.hconcat([frame, preprocessed_frame, col_normd])
    hcon2 = cv2.hconcat([color_deviancy, processed_deviancy,threshed_detections])
    displ = cv2.vconcat([hcon1, hcon2])

    # Downsize concat image
    displ = cv2.resize(displ, (con_wi, con_hi))

    # Save outputs
    im_name_conc = str ('concatd_' + onlyfiles[i])
    im_name_cont = str ('contured_' + onlyfiles[i])
    cv2.imwrite(os.path.join(conc_output , im_name_conc), displ)
    #cv2.imwrite(os.path.join(cont_output, im_name_cont), drawn_im)

    # Timings
    end_time = time.time()
    proc_time = round(end_time - start_time, 3)
    print('Processing time: ', proc_time, ' [s]\n')
    runtimes[i] = proc_time

print("Total processing time for",len(runtimes), "images:   ", np.sum(runtimes), "[s]\nAverage processing time:              ", round(np.average(runtimes),3),"[s]")

## Close webcam
cv2.destroyAllWindows()