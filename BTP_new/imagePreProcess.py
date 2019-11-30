import os
import cv2
import pandas as pd
import numpy as np
import math
fPath = os.getcwd() + "/image_picam"

for i,item in enumerate(os.listdir(fPath)):
    itemPath = fPath + "/" + item
    image = cv2.imread(itemPath)
    
    height, width, channels = image.shape     
    blackImage = np.zeros((height,width))
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _,thresh = cv2.threshold(gray,75,255,cv2.THRESH_BINARY_INV)
    
    kern_dilate = np.ones((8,8),np.uint8)
    kern_erode  = np.ones((3,3),np.uint8)
    mask = cv2.erode(thresh,kern_erode,iterations = 2)
    mask = cv2.dilate(mask,kern_dilate,iterations = 2) 

    contours, hierarchy = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE) 

    centreX,  centreY = width/2, height/2
    minDis = math.inf
    Sx, Sy, Sw, Sh = 0, 0, 0, 0
    
    for contour in contours:             
        [x,y,w,h] = cv2.boundingRect(contour)
        # discard areas that are too large     
        # if h>300 and w>300:     
        #     continue            
        # discard areas that are too small     
        if h<40 or w<40:     
            continue
        
        Cx, Cy  = x + w/2, y + h/2;     
        dis = (Cx - centreX)*(Cx - centreX) + (Cy - centreY)*(Cy - centreY)
        if dis < minDis:
            Sx, Sy, Sw, Sh = x, y, w, h    
            minDis = dis

    # mask_inv = cv2.bitwise_not(mask)    
    # blackImage_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)    
    # mask_fg = cv2.bitwise_and(mask,mask,mask = mask)    
    # dst = cv2.add(blackImage_bg,mask_fg)
    # cv2.rectangle(mask,(Sx,Sy),(Sx+Sw,Sy+Sh),(255,0,255),2)    
    roi = mask[Sy: Sy + Sh, Sx: Sx + Sw]    
    blackImage[Sy: Sy + Sh, Sx: Sx + Sw] = roi                
    
    # cv2.imshow(item,blackImage)
    cv2.imwrite("processedImages/Image" + str(i) + ".jpg",blackImage)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 