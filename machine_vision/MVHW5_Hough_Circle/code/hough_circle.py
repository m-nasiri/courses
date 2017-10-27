#*****************************************************************************/
# @file    hough_circle.py
# @author  Majid Nasiri 95340651
# @version V1.0.0
# @date    11 May 2017
# @brief   
#*****************************************************************************/
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt

original_image = cv2.imread('..\images\original_image2.jpg')
cv2.imshow('Original Image',original_image)

output = original_image.copy()

#Gaussian Blurring of Gray Image
blur_image = cv2.GaussianBlur(original_image,(3,3),0)
cv2.imshow('Gaussian Blurred Image',blur_image)

#Using OpenCV Canny Edge detector to detect edges
edged_image = cv2.Canny(blur_image,75,150)
cv2.imshow('Edged Image', edged_image)
cv2.imwrite('..\images\Edged Image.jpg',edged_image)

height,width = edged_image.shape
acc_array = np.zeros((height, width))

start_time = time.time()

def fill_acc_array(x0,y0,radius):
    x = radius
    y=0
    decision = 1-x
    
    while(y<x):
        if(x + x0<height and y + y0<width):
            acc_array[ x + x0,y + y0]+=1;
        if(y + x0<height and x + y0<width):
            acc_array[ y + x0,x + y0]+=1;
        if(-x + x0<height and y + y0<width):
            acc_array[-x + x0,y + y0]+=1;
        if(-y + x0<height and x + y0<width):
            acc_array[-y + x0,x + y0]+=1;
        if(-x + x0<height and -y + y0<width):
            acc_array[-x + x0,-y + y0]+=1;
        if(-y + x0<height and -x + y0<width):
            acc_array[-y + x0,-x + y0]+=1;
        if(x + x0<height and -y + y0<width):
            acc_array[ x + x0,-y + y0]+=1;
        if(y + x0<height and -x + y0<width):
            acc_array[ y + x0,-x + y0]+=1; 
        y+=1
        if(decision<=0):
            decision += 2 * y + 1
        else:
            x=x-1;
            decision += 2 * (y - x) + 1
        
    
edges = np.where(edged_image==255)

R = 25
for i in range(0,len(edges[0])):
    x=edges[0][i]
    y=edges[1][i]
    fill_acc_array(x,y,R)
  
              
plt.figure(num = 1, figsize=(width/100,height/100))
plt.figimage(acc_array, cmap=plt.get_cmap('gray') )
plt.savefig('..\images\R_'+str(R)+'.jpg')

i,j = np.unravel_index(acc_array.argmax(), acc_array.shape)
cv2.circle(output,(j,i),R,(255,0,0),2)
cv2.imshow('Detected circle',output)
cv2.imwrite('..\images\R_'+str(R)+'detected_circle.jpg',output)
end_time = time.time()
time_taken = end_time - start_time
print ('Time taken for execution',time_taken)


cv2.waitKey(0)
cv2.destroyAllWindows()
plt.close('all')
