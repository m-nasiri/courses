##---------------------------------------------------------------------------##
# @file    corner detection.py
# @author  Majid Nasiri
# @ID      95340651
# @version V1.0.0
# @date    2017-April-4
# @brief   finding corner of objects in binary images
##---------------------------------------------------------------------------##

import cv2
import numpy as np

imstr='binaryshapes0.jpg'
im=cv2.imread(imstr)
img = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
imb = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1]


DISK3=np.array([[0,0,0,1,0,0,0],
                [0,1,1,1,1,1,0],
                [0,1,1,1,1,1,0],
                [1,1,1,1,1,1,1],
                [0,1,1,1,1,1,0],
                [0,1,1,1,1,1,0],
                [0,0,0,1,0,0,0]], dtype=np.uint8)

DISK5=np.array([[0,0,0,0,0,1,0,0,0,0,0],
              [0,0,1,1,1,1,1,1,1,0,0],
              [0,1,1,1,1,1,1,1,1,1,0],
              [0,1,1,1,1,1,1,1,1,1,0],
              [0,1,1,1,1,1,1,1,1,1,0],
              [1,1,1,1,1,1,1,1,1,1,1],
              [0,1,1,1,1,1,1,1,1,1,0],
              [0,1,1,1,1,1,1,1,1,1,0],
              [0,1,1,1,1,1,1,1,1,1,0],
              [0,0,1,1,1,1,1,1,1,0,0],
              [0,0,0,0,0,1,0,0,0,0,0]], dtype=np.uint8)
     
imb_opening_disk3=cv2.morphologyEx(imb, cv2.MORPH_OPEN, DISK3)
imb_closeing_disk3=cv2.morphologyEx(imb, cv2.MORPH_CLOSE, DISK3)
closing3_opening3_diff=imb_closeing_disk3-imb_opening_disk3

imb_opening_disk5=cv2.morphologyEx(imb, cv2.MORPH_OPEN, DISK5)
imb_closeing_disk5=cv2.morphologyEx(imb, cv2.MORPH_CLOSE, DISK5)
closing5_opening5_diff=imb_closeing_disk5-imb_opening_disk5

#cv2.imshow('Input Image', imb)
#cv2.imshow('imb_opening_disk3', imb_opening_disk3)
#cv2.imshow('imb_closeing_disk3', imb_closeing_disk3)
#cv2.imshow('closing3_opening3_diff', closing3_opening3_diff)
#
#cv2.imshow('imb_opening_disk5', imb_opening_disk5)
#cv2.imshow('imb_closeing_disk5', imb_closeing_disk5)
#cv2.imshow('closing5_opening5_diff', closing5_opening5_diff)

cv2.imwrite('imb_opening_disk3.jpg', imb_opening_disk3)
cv2.imwrite('imb_closeing_disk3.jpg', imb_closeing_disk3)
cv2.imwrite('closing3_opening3_diff.jpg', closing3_opening3_diff)

cv2.imwrite('imb_opening_disk5.jpg', imb_opening_disk5)
cv2.imwrite('imb_closeing_disk5.jpg', imb_closeing_disk5)
cv2.imwrite('closing5_opening5_diff.jpg', closing5_opening5_diff)

print(cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)))

cv2.waitKey(0)
cv2.destroyAllWindows()


