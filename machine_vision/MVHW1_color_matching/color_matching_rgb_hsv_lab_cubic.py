
import numpy as np
import cv2
from scipy.stats import norm
import matplotlib.pyplot as plt


def main():
    for i in range(50): print(' ')
    
    COLOR='c6.jpg'
    IMAGE='chalk.jpg'
    colorbgr=cv2.imread(COLOR)
    imbgr=cv2.imread(IMAGE)    
    imgray = cv2.cvtColor(imbgr, cv2.COLOR_BGR2GRAY)
    impseudobgr = cv2.cvtColor(imgray, cv2.COLOR_GRAY2BGR)  
    
    #color space selection
    colorspace='RGB'
    CS = ('RGB','HSV','LAB')
    if (colorspace==CS[0]):
        im=imbgr
        color=colorbgr
        impseudo=impseudobgr
    elif (colorspace==CS[1]):
        im = cv2.cvtColor(imbgr, cv2.COLOR_BGR2HSV)             # Convert BGR to HSV
        color = cv2.cvtColor(colorbgr, cv2.COLOR_BGR2HSV)       # Convert BGR to HSV
        impseudo = cv2.cvtColor(impseudobgr, cv2.COLOR_BGR2HSV) # Convert BGR to HSV
    elif (colorspace==CS[2]):
        im = cv2.cvtColor(imbgr, cv2.COLOR_BGR2LAB)             # Convert BGR to LAB
        color = cv2.cvtColor(colorbgr, cv2.COLOR_BGR2LAB)       # Convert BGR to LAB
        impseudo = cv2.cvtColor(impseudobgr, cv2.COLOR_BGR2LAB) # Convert BGR to LAB

        
    #print(impseudo.shape)
    
    row, col, ch=im.shape
    print('im shape=',im.shape)
    
    imMatch=np.zeros((row,col,ch), np.uint8)
    cv2.namedWindow("image")
    cv2.imshow('image',imbgr)
    #cv2.waitKey(0)
    
    #-------------------------------------------------------------------------#
    #fitting normal distribution to data 
    window=np.array([[0,0],[0,0],[0,0]])
    mu=np.array([0,0,0])
    std=np.array([0,0,0])
    maxh=np.array([0,0,0])
    clrs = ('b','g','r')
    
    for i,clr in enumerate(clrs):
        data=color[:,:,i]
        mu[i], std[i] = norm.fit(data)
        mu[i]=mu[i]+1
        std[i]=std[i]+1
       
    if (colorspace==CS[0]):
        window[0,0]=mu[0]-5*std[0]
        window[0,1]=mu[0]+5*std[0]
        window[1,0]=mu[1]-5*std[1]
        window[1,1]=mu[1]+5*std[1]    
        window[2,0]=mu[2]-20*std[2]
        window[2,1]=mu[2]+20*std[2]
    elif (colorspace==CS[1]):
        window[0,0]=mu[0]-8*std[0]
        window[0,1]=mu[0]+8*std[0]
        window[1,0]=mu[1]-8*std[1]
        window[1,1]=mu[1]+8*std[1]    
        window[2,0]=mu[2]-25*std[2]
        window[2,1]=mu[2]+25*std[2]
    elif (colorspace==CS[2]):
        window[0,0]=mu[0]-12*std[0]
        window[0,1]=mu[0]+12*std[0]
        window[1,0]=mu[1]-8*std[1]
        window[1,1]=mu[1]+8*std[1]    
        window[2,0]=mu[2]-8*std[2]
        window[2,1]=mu[2]+8*std[2]
        
    minWindow=np.amin(window)
    if (window[0,0]<0): window[0,0]=0
    if (window[1,0]<0): window[1,0]=0
    if (window[2,0]<0): window[2,0]=0
    if (minWindow>0): minWindow=0
    
    for i,clr in enumerate(clrs):
        histr = cv2.calcHist([color],[i],None,[256],[0,256])
        maxh[i]=max(histr)
        plt.figure(1)
        plt.plot(histr, color = clr)
        plt.xlim([0,256])
        
        p=np.zeros([1,np.int(255*1.1)]) #more wider
        p[0,window[i,0]:window[i,1]]=maxh[i]
        plt.plot(p[0,:], '--', linewidth=1, color = clr)
        plt.xlim([minWindow,np.int(255*1.1)])
        plt.ylim([0,np.int(np.max(maxh)*1.1)])
    
    for i,clr in enumerate(clrs):
        histr = cv2.calcHist([im],[i],None,[256],[0,256])
        maxh[i]=max(histr)
        plt.figure(2)
        plt.plot(histr, color = clr)
        plt.xlim([0,256*1.1])
    
    

    print(mu)
    print(std)
    print(window) 
    
    
    #-------------------------------------------------------------------------#
    #thresholding constraints
    imL0=im[:,:,0] 
    imL1=im[:,:,1] 
    imL2=im[:,:,2] 
            
    imL0[imL0<window[0,0]]=0    
    imL0[imL0>window[0,1]]=0
        
    imL1[imL1<window[1,0]]=0    
    imL1[imL1>window[1,1]]=0

    imL2[imL2<window[2,0]]=0    
    imL2[imL2>window[2,1]]=0

    # layer selection    
    imL=np.multiply(imL0,imL1)
    imL=np.multiply(imL ,imL2)

    #imL=imL0 + np.multiply(imL1 ,imL2)
    #imL=imL0 + imL2
    #imL=np.multiply(imL0 ,imL2)
    
    #-------------------------------------------------------------------------#
    #image Matching 
    imL[imL>0]=mu[0]
    imMatch[:,:,0]=imL
    imL[imL>0]=mu[1]
    imMatch[:,:,1]=imL
    imL[imL>0]=mu[2]    
    imMatch[:,:,2]=imL
    rawMask=imL
    mask=rawMask
    cv2.namedWindow("rawMask")
    cv2.imshow("rawMask", rawMask)
    cv2.imwrite(IMAGE[:-4]+'_'+COLOR[:-4]+'_'+colorspace+'_rawMask'+'.jpg', rawMask)
    
    
    #-------------------------------------------------------------------------#
    #morph
    morphMask=mask
    SE1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    SE2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    
    morphMask = cv2.erode(morphMask,SE2,iterations = 1)
    morphMask = cv2.dilate(morphMask,SE1,iterations = 1)
    
    mask=morphMask
    
    cv2.namedWindow("morphMask")
    cv2.imshow('morphMask',morphMask)
    cv2.imwrite(IMAGE[:-4]+'_'+COLOR[:-4]+'_'+colorspace+'_morphMask'+'.jpg', morphMask)
    
    #-------------------------------------------------------------------------#
    #extract areas
    ret,thresh = cv2.threshold(mask,1,255,cv2.THRESH_BINARY)
    im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    contourNum=len(contours)
    
    contourNum=len(contours)
    area=np.zeros([contourNum])
    for i in range(0, contourNum):
        cnt = contours[i]
        area[i] = cv2.contourArea(cnt)
        
    #-------------------------------------------------------------------------#
    #remove small area  
    areaSingleMask  = np.zeros(im.shape[:2], np.uint8)
    areaMask        = np.zeros(im.shape[:2], np.uint8)
    
    area[area<40]=0
    #print('area=',area)
    #print(np.sort(area))
    
    for i in range(contourNum-1):
        if (area[i]!=0):
            #print('i={}, area={}'.format(i,area[i]))
            cv2.drawContours(areaSingleMask, contours, i, (255), -1)
            areaMask = cv2.add(areaMask, areaSingleMask)
      

    cv2.namedWindow("areaMask")
    cv2.imshow("areaMask", areaMask)
    cv2.imwrite(IMAGE[:-4]+'_'+COLOR[:-4]+'_'+colorspace+'_areaMask'+'.jpg', areaMask)
    mask=areaMask
    
    #-------------------------------------------------------------------------#
    #pseudo coloring   
    impseudo0=impseudo[:,:,0]        
    impseudo1=impseudo[:,:,1]
    impseudo2=impseudo[:,:,2]
    impseudo0[mask>0]=mu[0]     #avarage color
    impseudo1[mask>0]=mu[1]     #avarage color
    impseudo2[mask>0]=mu[2]     #avarage color
    impseudo[:,:,0]=impseudo0
    impseudo[:,:,1]=impseudo1
    impseudo[:,:,2]=impseudo2

    if (colorspace==CS[0]): 
        impseudo=impseudo
    elif (colorspace==CS[1]):
        impseudo = cv2.cvtColor(impseudo, cv2.COLOR_HSV2BGR)
    elif (colorspace==CS[2]):
        impseudo = cv2.cvtColor(impseudo, cv2.COLOR_LAB2BGR)
    
    
    #-------------------------------------------------------------------------#
    #bounding box
#    areaArgsort=np.argsort(area)
#    for i in range(0, 2):
#       cnt = contours[areaArgsort[contourNum-i-1]]
#       area = cv2.contourArea(cnt)
#       #print(area)
#       x,y,w,h = cv2.boundingRect(cnt)
#       cv2.rectangle(impseudo,(x,y),(x+w,y+h),(0,255,0),2)
#       #cv2.imshow('Features', impseudo)
#       
#    #cv2.imwrite('boundingBoxImage.jpg', impseudo)
#    
    
    #-------------------------------------------------------------------------#
    #output
    #cv2.namedWindow("output")
    #cv2.imshow('output',imMatch)
    cv2.namedWindow("impseudo")
    cv2.imshow('impseudo',impseudo)
    cv2.imwrite(IMAGE[:-4]+'_'+COLOR[:-4]+'_'+colorspace+'_impseudo'+'.jpg', impseudo)
    plt.show()
    cv2.waitKey(0)

    
    cv2.destroyAllWindows()
    plt.close('all')
    
main()
    




