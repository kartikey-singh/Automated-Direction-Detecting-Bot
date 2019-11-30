# import tensorflow as tf, sys
# from picamera import PiCamera
from time import sleep
import os
import cv2
import pandas as pd
import numpy as np
import math


# camera = PiCamera()
# camera.start_preview()
# sleep(5)
# camera.capture('/home/pi/Desktop/btp/image.jpg')
# camera.stop_preview()

# image_path = 'image0.jpg'

fPath = os.getcwd()
#print(fpath)
# for i,item in enumerate(os.listdir(fPath)):
item="image.jpg"
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
minDis = float('inf')
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

cv2.imshow(item,blackImage)
cv2.imwrite(fPath + "/imageaa.jpg",blackImage)
#cv2.waitKey(0)
cv2.destroyAllWindows() 

#classification
# Read in the image_data
# image_data = tf.gfile.FastGFile(image_path, 'rb').read()
# # Loads label file, strips off carriage return
# label_lines = [line.rstrip() for line
#     in tf.gfile.GFile("retrained_labels.txt")]
# # Unpersists graph from file
# with tf.gfile.FastGFile("retrained_graph.pb", 'rb') as f:
#     graph_def = tf.GraphDef()
#     graph_def.ParseFromString(f.read())
#     _ = tf.import_graph_def(graph_def, name='')
# # Feed the image_data as input to the graph and get first prediction
# score_max=0
# s=""
# with tf.Session() as sess:
# 		softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
# 		predictions = sess.run(softmax_tensor,{'DecodeJpeg/contents:0': image_data})
# 		# Sort to show labels of first prediction in order of confidence
# 		top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
# 		for node_id in top_k:
# 			human_string = label_lines[node_id]
# 			score = predictions[0][node_id]
# 			if score>score_max:
# 				s=human_string
# 				score_max=score
# 			print('%s (score = %.5f)' % (human_string, score))

# print(s)
