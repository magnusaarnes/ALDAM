import numpy as np
import cv2
import time
import os
import sys

print(sys.argv)

### Set variables

### User functions
def get_centroids(image, mode=cv2.RETR_EXTERNAL, module=cv2.CHAIN_APPROX_NONE, radius=5, color=(0,0,255), thickness=2):
    contours, _ = cv2.findContours(image, mode, module)
    centroids = np.zeros((2, len(contours)))
    contours = list(contours)
    for i in range(len(contours)):
        contours[i] = np.squeeze(contours[i], axis=1)
        centroids[:,i] = np.mean(contours[i], axis=0)
    return centroids.astype(int)

def diagnostix(object, precursor="Object; "):
    print(precursor+"shape:", object.shape, "Type:",object.dtype)
    

def spectral_autodifference(image, blur_strength, color_image=True):
    # Blur normalized
    blurred = cv2.blur(image, (int(image.shape[0]*blur_strength),int(image.shape[0]*blur_strength)))
    
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
con_wi = int (1920*1.5)

## Tunable parameters
blur_str                    = 0.015
color_object_lower_thresh   = 0.3
booster_threshold           = 0.01

# Bilateral filtering parameters // Attempts are made to make these image-size invariant
neighborhood_gain   = 0.027
sigmaColor_gain     = 0.14375
sigmaSpace_gain     = 0.14375
filter_neighborhood = int(neighborhood_gain*wi)
sigmaColor          = int(sigmaColor_gain*wi)
sigmaSpace          = int(sigmaSpace_gain*wi)


# Data directory paths
in_dataset  = os.path.join(os.getcwd(), 'Input_Dataset')
conc_output = os.path.join(os.getcwd(), 'Concat_Outputs')
marked_output = os.path.join(os.getcwd(), 'Marked_Outputs')

## Prepare script
# Load images from folder into array
onlyfiles = [ f for f in os.listdir(in_dataset) if os.path.isfile(os.path.join(in_dataset,f)) ]
images = np.empty(len(onlyfiles), dtype=object)
for n in range(0, len(onlyfiles)):
  images[n] = cv2.imread( os.path.join(in_dataset,onlyfiles[n]) )

# Some user interface stuff
print("Processing images in folder ", in_dataset, "(", len(onlyfiles), " objects )\n")
print("Images will be downsized to ", hi, "x",wi, "\n")
runtimes = np.zeros(len(onlyfiles))

## Main Loop
# For each image in list, perform saliency stuff
for i in range(0,len(images)):
    start_time = time.time()

    # Extract image
    frame = images[i]
    print('Processing image "',onlyfiles[i],'", ',i+1,'/',len(images),'. Shape: ', frame.shape)

    ## Image preprocessing
    blur_rad = int (frame.shape[0]*blur_str)

    # Resize images to standard resolution
    frame = cv2.resize(frame,(wi, hi))
    preprocessed_frame = frame.copy()
    
    # Pre-blurring and Bilateral Filtering on input image
    #preprocessed_frame = cv2.blur(preprocessed_frame, (blur_rad, blur_rad))
    preprocessed_frame = cv2.bilateralFilter(preprocessed_frame, filter_neighborhood, sigmaColor=sigmaColor,sigmaSpace=sigmaSpace)

    ## Color filtering
    col_normd = preprocessed_frame.copy()
    
    #Convert to floating-point / normalize values
    col_normd = col_normd/255.0
    # Normalize colors
    for c in range(3):
        col_normd[:,:,c] = col_normd[:,:,c] / (1.0/255.0 + col_normd[:,:,0] + col_normd[:,:,1] + col_normd[:,:,2]) # Normalize colors
    # Blurred version for autocomparison
    blurred_col_normd = cv2.blur(col_normd,(blur_rad,blur_rad))

    # For each channel, compute pixel-wise distance from blurred normals
    color_deviancy = np.absolute(col_normd - blurred_col_normd)

    # Merge channels into 1 grayscale image (yes this does look dumb but it actually runs faster than 'sum of channels / 3')
    processed_deviancy = cv2.cvtColor((color_deviancy*255).astype(np.ubyte), cv2.COLOR_BGR2GRAY)/255.0

    # Boost pixels that stand out from their surroundings (compare with blurred self), over a certain threshold
    procdev_max             = np.max(processed_deviancy)
    processed_deviancy_blur = cv2.blur(processed_deviancy, (blur_rad, blur_rad))
    normalized_diff         = np.absolute(processed_deviancy - processed_deviancy_blur)/procdev_max
    processed_deviancy      = np.where(normalized_diff<booster_threshold, processed_deviancy, processed_deviancy/procdev_max)

    ## Thresholding and detection
    # Threshold values  [Input is slightly blurred, not sure why/if this might be a good idea]
    _disc, threshed_detections = cv2.threshold(cv2.blur(processed_deviancy,(5,5)),color_object_lower_thresh,1,cv2.THRESH_BINARY)
    
    # Perform morphology to remove isolated pixels
    threshed_detections = cv2.morphologyEx(threshed_detections, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    
    # Contours and centroids
    threshed_detections = (threshed_detections*255).astype(np.ubyte)

    centroids = get_centroids(threshed_detections)
    marked_frame = frame.copy()
    if centroids.shape[0]>0 and centroids.shape[1]>0:
        for j in range(centroids.shape[1]):
            marked_image = cv2.circle(marked_frame, (centroids[0,j], centroids[1,j]), radius=9, color=(0,0,255), thickness=2)

    ## Image post-processing
    if len(threshed_detections.shape)<3:
        threshed_detections = cv2.cvtColor(threshed_detections, cv2.COLOR_GRAY2BGR)

    color_deviancy = (color_deviancy*255).astype(np.ubyte)      # Convert result to image-readable format
    col_normd = (col_normd*255).astype(np.ubyte)                    # Convert result to image-readable format
    blurred_col_normd = (blurred_col_normd*255).astype(np.ubyte)        # Convert result to image-readable format

    # Convert from grayscale to colormap, just for the aesthetics
    processed_deviancy = cv2.applyColorMap(cv2.cvtColor((processed_deviancy*255).astype(np.ubyte),cv2.COLOR_GRAY2BGR), cv2.COLORMAP_INFERNO)


    # Image Merging
    hcon1 = cv2.hconcat([frame, preprocessed_frame, col_normd])
    hcon2 = cv2.hconcat([color_deviancy, processed_deviancy, threshed_detections])
    displ = cv2.vconcat([hcon1, hcon2])

    # Downsize concat image
    displ = cv2.resize(displ, (con_wi, con_hi))

    # Save outputs
    im_name_conc = str ('concatd_' + onlyfiles[i])
    im_name_mark = str ('marked_' + onlyfiles[i])
    cv2.imwrite(os.path.join(conc_output , im_name_conc), displ)
    cv2.imwrite(os.path.join(marked_output, im_name_mark), marked_frame)

    # Timings
    end_time = time.time()
    proc_time = round(end_time - start_time, 3)
    print('Processing time: ', proc_time, ' [s]\n')
    runtimes[i] = proc_time

print("Total processing time for",len(runtimes), "images:   ", np.sum(runtimes), "[s]\nAverage processing time:              ", round(np.average(runtimes),3),"[s]")

## Close webcam
cv2.destroyAllWindows()