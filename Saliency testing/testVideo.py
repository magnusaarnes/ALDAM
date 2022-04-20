import cv2
import numpy as np

#url = "https://10.22.222.122:8080/video"
#vcap = cv2.VideoCapture(url)
vcap = cv2.VideoCapture(0)


saliency = cv2.saliency.StaticSaliencyFineGrained_create()
#saliency = cv2.saliency.ObjectnessBING_create()

while(True):
    ret, frame = vcap.read()
    if frame is None:
        continue
    
    success, saliencyMap = saliency.computeSaliency(frame)
    if success:
        saliencyMapImage = (saliencyMap * 255).astype("uint8")
        cv2.imshow('frame', saliencyMapImage)
	
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vcap.release()
cv2.destroyAllWindows()
