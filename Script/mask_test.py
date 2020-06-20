import pymurapi as mur
import time
import cv2 as cv
import math

auv = mur.mur_init()

hsv_mask_min_red = (0, 50, 50)
hsv_mask_max_red = (15, 255, 255)

hsv_mask_min_green = (40, 40, 20)
hsv_mask_max_green = (80, 255, 255)   

hsv_mask_min_blue = (130, 20, 0)
hsv_mask_max_blue = (150, 255, 230)

while True:
    image = auv.get_image_bottom()

    copy_img = image.copy()
    hsv_img = cv.cvtColor(copy_img, cv.COLOR_BGR2HSV)
    mask1 = cv.inRange(hsv_img, hsv_mask_min_green, hsv_mask_max_green)
    mask2 = cv.inRange(hsv_img, hsv_mask_min_red, hsv_mask_max_red)
    mask3 = cv.inRange(hsv_img, hsv_mask_min_blue, hsv_mask_max_blue)

    mask = mask1 + mask2 + mask3

    cv.imshow("img", image)
    cv.imshow('mask', mask)
    cv.waitKey(5)
    