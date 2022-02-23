import cv2


vid = cv2.VideoCapture(1)

#saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
saliency = cv2.saliency.StaticSaliencyFineGrained_create()

while(True):
    ret, frame = vid.read()
    print("Test")
    success, saliencyMap = saliency.computeSaliency(frame)
    saliencyMap = (saliencyMap * 255).astype("uint8")
    
    cv2.imshow('frame', saliencyMap)
	
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()
