import cv2
import os
import numpy as np
from sympy import true

max_detections = 3

## Set up webcam
cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while(True):
	# Load our input image
	ret, image = cam.read()

	# initialize OpenCV's objectness saliency detector and set the path to the input model files
	saliency = cv2.saliency.ObjectnessBING_create()
	saliency.setTrainingPath('models')
	# compute the bounding box predictions used to indicate saliency
	(success, saliencyMap) = saliency.computeSaliency(image)
	numDetections = saliencyMap.shape[0]

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

	output = image.copy()
	# loop over the detections
	for i in range(0, min(numDetections, max_detections)):
		# extract the bounding box coordinates
		(startX, startY, endX, endY) = saliencyMap[i].flatten()
		
		# randomly generate a color for the object and draw it on the image
		color = np.random.randint(0, 255, size=(3,))
		color = [int(c) for c in color]
		cv2.rectangle(output, (startX, startY), (endX, endY), color, 2)
		# show the output image
	cv2.imshow("Image", output)

# Close webcam
cam.release()
cv2.destroyAllWindows()