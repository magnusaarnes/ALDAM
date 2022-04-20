import os
import numpy as np
import cv2

import color_deviancy_detection as cdd

## Standard image resolutions
# Downsize resolution
hi = 270
wi = 480
# Concat output resolution
con_hi = 1080
con_wi = int (1920*1.5)

## Tunable parameters
blur_str                    = 0.02
color_object_lower_thresh   = 0.3
booster_threshold           = 0.01

# Bilateral filtering parameters // Attempts are made to make these image-size invariant
neighborhood_gain   = 0.027
sigmaColor_gain     = 0.14375
sigmaSpace_gain     = 0.14375

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

num_split = 4
quadrants_in = np.zeros((hi//2,wi//2,3,num_split)).astype(np.ubyte)
quadrants_ou = np.zeros((hi//2,wi//2,num_split)).astype(np.ubyte)

for i in range(len(images)):
    image = cv2.resize(images[i],(wi,hi))
    print("Sizes: ", wi//2, hi//2)

    up_left      = image[0:hi//2, 0:wi//2]
    up_right     = image[0:hi//2, wi//2:]
    down_left    = image[hi//2:, 0:wi//2]
    down_right   = image[hi//2:, wi//2:]

    quadrants_in[:,:,:,0] = up_left
    quadrants_in[:,:,:,1] = up_right
    quadrants_in[:,:,:,2] = down_left
    quadrants_in[:,:,:,3] = down_right

    for k in range(4):
        quadrants_ou[:,:,k] = cdd.detect_color_deviancies_thresholds(quadrants_in[:,:,:,k])

    #detections, centroids, concats = cdd.detect_color_deviancies(image)
    hcon1 = cv2.hconcat([quadrants_ou[:,:,0], quadrants_ou[:,:,1]])
    hcon2 = cv2.hconcat([quadrants_ou[:,:,2], quadrants_ou[:,:,3]])
    displ = cv2.vconcat([hcon1, hcon2])

    displ = cdd.draw_centroids(cdd.get_centroids(displ),displ)

    cv2.imshow('ass', displ)
    #cv2.imshow('ass', concats)
    cv2.waitKey(0)

cv2.destroyAllWindows()

