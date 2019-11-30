import os
import cv2
import pandas as pd
import numpy as np
import math
mpath = os.getcwd()
fPath = mpath + "/processedImages"

distortAngles = [-7, -5, 0, 5, 7]
folders = ["UpImages", "LeftImages", "DownImages", "RightImages"]

for i,item in enumerate(os.listdir(fPath)):
    itemPath = fPath + "/" + item
    image = cv2.imread(itemPath)
    (h, w, channels) = image.shape
    temp = np.zeros((h, w, channels))
    temp = image
    center = (w/2, h/2)
    scale = 1.0
    for folder in folders:    
        for j,angle in enumerate(distortAngles):
            M = cv2.getRotationMatrix2D(center, angle, scale)        
            img = cv2.warpAffine(temp, M, (h, w))
            cv2.imwrite(folder + "/Image" + str(i) + str(j) + ".jpg",img)

        temp = np.rot90(temp)
        # cv2.imshow(item,temp)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows() 