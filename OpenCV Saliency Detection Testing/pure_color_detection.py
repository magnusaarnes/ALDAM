import numpy as np
import cv2
import time
import os
import sys

print(sys.argv)

### Set variables

## Standard image resolutions
# Downsize resolution
hi = 720
wi = 1270
# Concat output resolution
con_hi = 1080
con_wi = 1920

## Preprocessing variables
blur_str = 0.015
color_filter_lower_thresh = 100

## Preprocessing options
downsize                = 1
color_filtering         = 1
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
    col_filtd = preprocessed_frame.copy()

    # Blur image
    col_filtd = cv2.blur(col_filtd, (blur_rad, blur_rad))
    
    #Convert to floating-point / normalize values
    col_filtd = col_filtd/255.0

    # Normalize colors
    for c in range(3):
        col_filtd[:,:,c] = col_filtd[:,:,c] / (1.0/255.0 + col_filtd[:,:,0] + col_filtd[:,:,1] + col_filtd[:,:,2]) # Normalize colors
    
    # Normalize brightness
    for c in range(3):
        print("Max in channel c: ", np.max(col_filtd[:,:,c]))
        col_filtd[:,:,c] = (col_filtd[:,:,c]/np.max(col_filtd[:,:,c]))
        print("Max in channel c: ", np.max(col_filtd[:,:,c]))

    # Compute channel-wise averages
    avg_color_per_row = np.average(col_filtd, axis=0)
    avg_color = (np.average(avg_color_per_row, axis=0))#*255).astype(int)
    print("Average normalized color values, BGR: ",avg_color)

    # For each channel, compute pixel-wise distance from average
    color_deviancy = np.zeros(col_filtd.shape)
    for c in range(0,3):
        color_deviancy[:,:,c] = np.absolute(col_filtd[:,:,c] - avg_color[c])
    color_deviancy = (color_deviancy*255).astype(np.ubyte)  # Convert result to image-readable format

    # Threshold values
    _disc, threshed_detections = cv2.threshold(color_deviancy,color_filter_lower_thresh,255,cv2.THRESH_BINARY)
    
    # Perform morphology to remove isolated pixels
    threshed_detections = cv2.morphologyEx(threshed_detections, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
    
    ## Detection
    """
    def get_centroids(self):
        contours, _ = cv2.findContours(self.image_processed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        centroids = np.zeros((2, len(contours)))
        contours = list(contours)
        for i in range(len(self.contours)):
            contours[i] = np.squeeze(contours[i], axis=1)
            centroids[:,i] = np.mean(contours[i], axis=0)
    """
    
    
    ## Image post-processing
    # Image Merging
    hcon1 = cv2.hconcat([frame, preprocessed_frame])
    hcon2 = cv2.hconcat([color_deviancy, threshed_detections])
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
    cv2.imwrite(os.path.join(conc_output , im_name_conc), displ)
    #cv2.imwrite(os.path.join(cont_output, im_name_cont), drawn_im)


## Close webcam
cv2.destroyAllWindows()