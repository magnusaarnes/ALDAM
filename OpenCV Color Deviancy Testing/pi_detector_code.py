import os
import numpy as np
import cv2

import color_deviancy_detection_and_saliency_test as cdd

## Standard image resolutions
# Downsize resolution
hi = 270
wi = 480
# Concat output resolution
con_hi = 1080
con_wi = int (1920*1.5)

## Tunable parameters
blur_strs                   = (0.04, 0.01)
color_object_lower_thresh   = 0.3
booster_threshold           = 0.01
threshold_preblur_rad       = (5,5)

# Bilateral filtering parameters // Attempts are made to make these image-size invariant
neighborhood_gain   = 0.027
sigmaColor_gain     = 0.14375
sigmaSpace_gain     = 0.14375

show_images             = False
delete_processed_images = False

# Data directory paths
in_dataset    = os.path.join(os.getcwd(), 'Input_Dataset')
conc_output   = os.path.join(os.getcwd(), 'Concat_Outputs')
marked_output = os.path.join(os.getcwd(), 'Marked_Outputs')
suspected_inp = os.path.join(os.getcwd(), 'Suspicious_Inputs')

if not os.path.exists('Suspicious_Inputs'):
    os.makedirs('Suspicious_Inputs')
if not os.path.exists('Marked_Outputs'):
    os.makedirs('Marked_Outputs')
if not os.path.exists('Concat_Outputs'):
    os.makedirs('Concat_Outputs')
if not os.path.exists('Input_Dataset'):
  raise Exception("Missing folder 'Input_Dataset' - no valid inputs can be found!")

## Prepare script
# Load images from folder into array
onlyfiles = [ f for f in os.listdir(in_dataset) if os.path.isfile(os.path.join(in_dataset,f)) ]
images = np.empty(len(onlyfiles), dtype=object)
for n in range(0, len(onlyfiles)):
  images[n] = cv2.imread( os.path.join(in_dataset,onlyfiles[n]) )
  

# Some user interface stuff
print("Processing images in folder ", in_dataset, "(", len(onlyfiles), " objects )\n")
print("Images will be downsized to ", hi, "x",wi, "\n")

for i in range(len(images)):
    image = cv2.resize(images[i],(wi,hi))
    print("Image:", str(onlyfiles[i]))

    detections, centroids, concats, threshed_detects = cdd.detect_color_deviancies(image, blur_strs, color_object_lower_thresh, booster_threshold,bilateral_filter_gains=(neighborhood_gain,sigmaColor_gain,sigmaSpace_gain), threshold_preblur_rad=threshold_preblur_rad)
    print("Detected objects: ", np.shape(centroids)[1], "\n")

    # Save outputs
    im_name_conc = str ('concatd_' + onlyfiles[i])
    im_name_mark = str ('marked_' + onlyfiles[i])
    im_name_susp = str ('sussy_' + onlyfiles[i])
    cv2.imwrite(os.path.join(conc_output , im_name_conc), concats)
    cv2.imwrite(os.path.join(marked_output, im_name_mark), detections)
    if np.shape(centroids)[1] > 0:
      cv2.imwrite(os.path.join(suspected_inp, im_name_susp), image)

    if show_images:
      cv2.imshow('ass', concats)
      cv2.waitKey(0)
    if delete_processed_images:
      os.remove(os.path.join(in_dataset, onlyfiles[i]))

if show_images:
  cv2.destroyAllWindows()

