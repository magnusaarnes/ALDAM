from audioop import avg
import numpy as np
import cv2
import time
import datetime as dt
import matplotlib.pyplot as plt
import os
import commons

### Set variables

## Standard image resolutions
# Downsize resolution
hi = 270
wi = 480
# Concat output resolution
con_hi = 1080
con_wi = 1920

## Preprocessing variables
blur_str = 0.01
color_filter_lower_thresh = 10

## Preprocessing options
preblurring     = 0
downsize        = 1
color_filtering = 1
denoise         = 0
bilat_filtering = 1

# Data directory paths
in_dataset  = os.path.join(os.getcwd(), 'Input_Dataset')
sali_output = os.path.join(os.getcwd(), 'Saliency_Outputs')
conc_output = os.path.join(os.getcwd(), 'Concat_Outputs')
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
if preblurring:
    print("Images will be pre-blurred with a blur radius of ", blur_str, "% of height.")
print("\n")

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
    if preblurring:     # Blur input image
        preprocessed_frame = cv2.blur(preprocessed_frame, (blur_rad, blur_rad))
    if denoise:         # Denoise input image
        preprocessed_frame = cv2.fastNlMeansDenoisingColored(preprocessed_frame,None,5,5,3,5)
    if bilat_filtering: # Bilateral Filter on input image
        preprocessed_frame = cv2.bilateralFilter(preprocessed_frame, 13, 69,69)

    # Color filtering
    col_filtd = preprocessed_frame.copy()
    if color_filtering:
        #Convert to floating-point / normalize values
        col_filtd = col_filtd/255.0

        # Normalize colors
        for c in range(0,3):
            col_filtd[:,:,c] = col_filtd[:,:,c] / (1.0/255.0 + col_filtd[:,:,0] + col_filtd[:,:,1] + col_filtd[:,:,2]) # Normalize colors
        
        # Compute channel-wise averages
        avg_color_per_row = np.average(col_filtd, axis=0)
        avg_color = (np.average(avg_color_per_row, axis=0))#*255).astype(int)
        print("Average normalized color values, BGR: ",avg_color)

        # For each channel, compute pixel-wise distance from average
        color_deviancy = np.zeros(col_filtd.shape)
        for c in range(0,3):
            color_deviancy[:,:,c] = np.absolute(col_filtd[:,:,c] - avg_color[c])
        
        # 
        color_deviancy = (color_deviancy*255).astype(np.ubyte)
        col_filtd = (col_filtd*255).astype(np.ubyte)    # Convert result to image-readable format

    # Perform saliency
    saliency = cv2.saliency.StaticSaliencyFineGrained_create()
    (success, saliencyMap) = saliency.computeSaliency(color_deviancy)
    saliencyMap = (saliencyMap*255).astype("uint8")

    # Blur saliency map
    sali_blur_rad = int (saliencyMap.shape[0]*0.01)
    processed_sali_map = saliencyMap.copy()
    processed_sali_map = cv2.blur(processed_sali_map,[sali_blur_rad, sali_blur_rad])

    # Canny Edge Detector
    low_threshold = 100
    ratio = 3
    kernel_size = 5
    detected_edges = cv2.Canny(processed_sali_map, low_threshold, low_threshold*ratio, kernel_size)
    mask = detected_edges != 0
    dst = cv2.cvtColor(processed_sali_map, cv2.COLOR_GRAY2BGR) * (mask[:,:,None].astype(frame.dtype))

    # Threshold
    _disc, threshed_detections = cv2.threshold(dst,150,255,cv2.THRESH_BINARY)
    
    ## Image post-processing
    # Image Merging
    print(frame.dtype, preprocessed_frame.dtype, color_deviancy.dtype)
    hcon1 = cv2.hconcat([frame, preprocessed_frame, color_deviancy])
    hcon2 = cv2.hconcat([cv2.cvtColor(saliencyMap, cv2.COLOR_GRAY2BGR), dst, threshed_detections])
    displ = cv2.vconcat([hcon1, hcon2])

    # Downsize concat image
    displ = cv2.resize(displ, (con_wi, con_hi))

    # Timings
    end_time = time.time()
    proc_time = round(end_time - start_time, 3)
    print('Processing time: ', proc_time, ' [s]')

    # Save outputs
    im_name_conc = str ('concatd_' + onlyfiles[i])
    im_name_cont = str ('contured_' + onlyfiles[i])
    im_name_sali = str ('saliencyd_' + onlyfiles[i])
    cv2.imwrite(os.path.join(conc_output , im_name_conc), displ)
    #cv2.imwrite(os.path.join(cont_output, im_name_cont), drawn_im)
    cv2.imwrite(os.path.join(sali_output, im_name_sali), saliencyMap)

    

## Close webcam
cv2.destroyAllWindows()