import numpy as np
import cv2
import time
import datetime as dt
import matplotlib.pyplot as plt
import os
import commons

## Set variables
# Image downsample scaling
scale = 1
# Standard image resolution
hi = 720
wi = 1280
downsize = False
con_hi = 1600
con_wi = 2560

blur_str = 0.03
preblur = False

threshlev1 = 80
threshlev2 = 50

# Data directory paths
in_dataset = os.path.join(os.getcwd(), 'Input_Dataset')
out_saliency = os.path.join(os.getcwd(), 'Saliency_Outputs')
conc_output = os.path.join(os.getcwd(), 'Concat_Outputs')
cont_output = os.path.join(os.getcwd(), 'Conture_Outputs')

print("Folder content at start: ", os.listdir(in_dataset))

# Load images from folder into array
onlyfiles = [ f for f in os.listdir(in_dataset) if os.path.isfile(os.path.join(in_dataset,f)) ]
images = np.empty(len(onlyfiles), dtype=object)
for n in range(0, len(onlyfiles)):
  images[n] = cv2.imread( os.path.join(in_dataset,onlyfiles[n]) )

# For each image in list, perform saliency stuff
for i in range(0,len(images)):
    print('Processing image "',onlyfiles[i],'", ',i+1,'/',len(images),'. Shape: ', images[i].shape)
    start_time = time.time()

    # Capture the video frame by frame
    frame = images[i]

    # Image preprocessing
    if downsize:
        frame = cv2.resize(frame,(frame.shape[1]//scale, frame.shape[0]//scale))
    blur_rad = int (frame.shape[0]*blur_str)
    if preblur:
        frame = cv2.blur(frame, (blur_rad, blur_rad))
    #frame = cv2.flip(frame,1)

    # Perform saliency
    saliency = cv2.saliency.StaticSaliencyFineGrained_create()
    (success, saliencyMap) = saliency.computeSaliency(frame)
    saliencyMap = (saliencyMap*255).astype("uint8")
  
    # Image post-processing
    grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _ret, thresh1 = cv2.threshold(saliencyMap,threshlev1,255,cv2.THRESH_BINARY)
    smoother = cv2.blur(thresh1, (blur_rad, blur_rad))
    _ret, thresh2 = cv2.threshold(smoother,threshlev2,255,cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    drawn_im = images[i].copy()
    cv2.drawContours(drawn_im, contours, -1, (0,255,0), 3)

    # Image Merging
    hcon1 = cv2.hconcat([images[i], cv2.cvtColor(saliencyMap, cv2.COLOR_GRAY2BGR), cv2.cvtColor(thresh1, cv2.COLOR_GRAY2BGR)])
    hcon2 = cv2.hconcat([cv2.cvtColor(smoother, cv2.COLOR_GRAY2BGR),cv2.cvtColor(thresh2, cv2.COLOR_GRAY2BGR), drawn_im])
    displ = cv2.vconcat([hcon1, hcon2])

    # Downsize concat image
    print(displ.shape)
    displ = cv2.resize(displ, (con_wi, con_hi))

    # Timings
    end_time = time.time()
    proc_time = round(end_time - start_time, 3)
    print('Processing time: ', proc_time, ' [s]')

    # Save outputs
    im_name_conc = str ('concatd_' + onlyfiles[i])
    im_name_cont = str ('contured_' + onlyfiles[i])
    cv2.imwrite(os.path.join(conc_output , im_name_conc), displ)
    cv2.imwrite(os.path.join(cont_output, im_name_cont), drawn_im)

    

## Close webcam
cv2.destroyAllWindows()