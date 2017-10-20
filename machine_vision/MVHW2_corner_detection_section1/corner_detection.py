##---------------------------------------------------------------------------##
# @file    corner detection.py
# @author  Majid Nasiri
# @ID      95340651
# @version V1.0.0
# @date    2017-March-21
# @brief   finding corner of objects in binary images
##---------------------------------------------------------------------------##

import cv2
import numpy as np
from scipy import signal

# finding corner of image by an elemrnt
# finding white corner in black backgroud 
def sub_corner_detection(imb, SE):
    
    imb0=np.zeros(imb.shape, dtype=int)
    imb1=np.zeros(imb.shape, dtype=int)
    imb0[imb==0]=1
    imb1[imb>0]=1

    SE0=np.zeros(SE.shape, dtype=int)
    SE1=np.zeros(SE.shape, dtype=int)
    SE0[SE==0]=1
    SE1[SE==1]=1
    max_peak0=np.sum(SE0)
    max_peak1=np.sum(SE1)
    
    corr0=signal.correlate(imb0, SE0, 'valid')
    corr_mask0=np.zeros(corr0.shape, dtype=np.uint8)
    corr_mask0[corr0==max_peak0]=1
    corr_mask00=np.lib.pad(corr_mask0, np.int(len(SE)/2), 'constant', constant_values=0)
    
    corr1=signal.correlate(imb1, SE1, 'valid')
    corr_mask1=np.zeros(corr1.shape, dtype=np.uint8)
    corr_mask1[corr1==max_peak1]=1
    corr_mask11=np.lib.pad(corr_mask1, np.int(len(SE)/2), 'constant', constant_values=0)
    
    mask=np.multiply(corr_mask00,corr_mask11)
    return mask

# finding corner of image by an elemrnt
# finding black corner in white backgroud 
def sub_corner_detection_inverse(imb, SE):
    
    SEI=np.zeros(SE.shape, dtype=np.int8)
    SEI[SE==0]=1        
    SEI[SE <0]=-1
    SEI[SE==1]=0

    imb0=np.zeros(imb.shape, dtype=int)
    imb1=np.zeros(imb.shape, dtype=int)
    imb0[imb==0]=1
    imb1[imb>0]=1

    SE0=np.zeros(SEI.shape, dtype=int)
    SE1=np.zeros(SEI.shape, dtype=int)
    SE0[SEI==0]=1
    SE1[SEI==1]=1
    max_peak0=np.sum(SE0)
    max_peak1=np.sum(SE1)
    
    corr0=signal.correlate(imb0, SE0, 'valid')
    corr_mask0=np.zeros(corr0.shape, dtype=np.uint8)
    corr_mask0[corr0==max_peak0]=1
    corr_mask00=np.lib.pad(corr_mask0, np.int(len(SEI)/2), 'constant', constant_values=0)
    
    corr1=signal.correlate(imb1, SE1, 'valid')
    corr_mask1=np.zeros(corr1.shape, dtype=np.uint8)
    corr_mask1[corr1==max_peak1]=1
    corr_mask11=np.lib.pad(corr_mask1, np.int(len(SEI)/2), 'constant', constant_values=0)
    
    mask=np.multiply(corr_mask00,corr_mask11)
    return mask

def corner_detection(imb, SEs):
    corners=np.zeros((imb.shape[0],imb.shape[1],2*len(SEs)), dtype=np.uint8)
    
    for se_count in range(len(SEs)):
        corners[:,:,2*se_count  ]=sub_corner_detection        (imb, SEs[se_count])  #white corners
        corners[:,:,2*se_count+1]=sub_corner_detection_inverse(imb, SEs[se_count])  #black corners
            
    corners=np.sum(corners, axis=2, dtype=np.uint8)
    corners[corners>0]=255
    return corners
 

for i in range(50): print(' ')  #clear screen

imstr='binary_image_1.bmp'
im=cv2.imread(imstr)
img = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
imb = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1]

# structure element for sharp corners
#SEBL=np.array([[-1, 0, 1,-1,-1],                       
#               [-1, 0, 1,-1,-1],
#               [-1, 0, 1, 1, 1],
#               [-1, 0, 0, 0, 0],
#               [-1,-1,-1,-1,-1]], dtype=np.int8)

# structure element from the book
SEBL=np.array([[-1, 1,-1],[ 0, 1, 1],[ 0, 0,-1]])   #buttom left corner

SEBR=np.fliplr(SEBL)        #flip structure element left to right  #buttom right corner
SEUL=np.flipud(SEBL)        #flip structure element up to down     #up left corner
SEUR=np.fliplr(SEUL)        ##flip structure element left to right #up right corner

# all of element in a cell (list)
Structure_Elements = [SEBL, SEBR, SEUR, SEUL]

# finding structure elements 
detected_corners=corner_detection(imb, Structure_Elements)


cv2.imshow('Input Image', imb)
cv2.imshow('Corner Image', detected_corners)
cv2.imwrite(imstr+'_detected_corners.bmp',detected_corners)
cv2.waitKey(0)
cv2.destroyAllWindows()

##---------------------------------------------------------------------------##
# @ end
##---------------------------------------------------------------------------##
