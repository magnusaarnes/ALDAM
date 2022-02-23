import cv2
import numpy as np


def process_image(image, display_steps=False):
    saliency = cv2.saliency.StaticSaliencyFineGrained_create()

    # Do the saliency
    success, image_saliency_map = saliency.computeSaliency(image)
    if not success:
        print("[ERROR] Saliency failed")
        return
    
    saliency_map_image = (image_saliency_map * 255).astype("uint8")

    saliency_map_image_bin = saliency_map_image > 60
    saliency_map_image_bin = saliency_map_image_bin.astype("uint8") * 255

    saliency_map_image_bin_blur = cv2.blur(saliency_map_image_bin, (23,23), 0)

    saliency_map_image_bin_blur_bin = saliency_map_image_bin_blur > 10
    saliency_map_image_bin_blur_bin = saliency_map_image_bin_blur_bin.astype("uint8") * 255
    
    if display_steps:
        # Display original image
        cv2.imshow('image', image)
        cv2.imshow('saliency', saliency_map_image)
        cv2.imshow('saliency_bin', saliency_map_image_bin)
        cv2.imshow('saliency_bin_blur', saliency_map_image_bin_blur)
        cv2.imshow('saliency_bin_blur_bin', saliency_map_image_bin_blur_bin)
    
    return saliency_map_image_bin_blur_bin
    

if __name__ == "__main__":
    image = cv2.imread('images/image1.png')
    
    image_saliency = process_image(image, display_steps=True)
    
    contours, hierarchy = cv2.findContours(image_saliency, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    img_contours = np.zeros(image.shape)
    # draw the contours on the empty image
    cv2.drawContours(image, contours, -1, (0,255,0), 2)
    cv2.imshow('contours', image)
    
    cv2.waitKey(0)