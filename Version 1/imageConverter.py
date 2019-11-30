import os
import pandas as pd
from PIL import Image
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPDF, renderPM

mypath = os.getcwd()
directories = []
logs = []

for f in os.listdir(mypath):
	f_path = mypath+"/"+f
	if os.path.isdir(f_path):
		if f_path[-3:] != "zed":
			directories.append(f_path)

for directory in directories:
	if not os.path.exists(directory+"Resized"):
		os.makedirs(directory+"Resized")

for directory in directories:
	for i,f in enumerate(os.listdir(directory)):
		if f.endswith(".png") or f.endswith(".jpeg") or f.endswith(".jpg") or f.endswith(".gif"):
			try:		
				image_path = os.path.join(directory,f)
				im = Image.open(image_path)
				im_rgb = im.convert('RGB')		
				# size = im_rgb.size
				# ratio = 0.8
				# reduced_size = int(size[0] * ratio), int(size[1] * ratio)     
				reduced_size = 128, 128
				im_resized = im_rgb.resize(reduced_size, Image.ANTIALIAS)
				im_resized.save(directory + "Resized/" + str(i) + ".jpg", "JPEG")
			except:
				logs.append("Cannot do it for File " + directory + f)	
		if f.endswith(".svg"):
			try:				
				image_path = os.path.join(directory,f)
				drawing = svg2rlg(image_path)				
				renderPM.drawToFile(drawing, directory + "Resized/" + str(i) + ".jpg", fmt="JPEG")
				im = Image.open(directory + "Resized/" + str(i) + ".jpg")
				im_rgb = im.convert('RGB')		
				# size = im_rgb.size
				# ratio = 0.8
				# reduced_size = int(size[0] * ratio), int(size[1] * ratio)     
				reduced_size = 128, 128
				im_resized = im_rgb.resize(reduced_size, Image.ANTIALIAS)
				im_resized.save(directory + "Resized/" + str(i) + ".jpg", "JPEG")
			except:		
				logs.append("Cannot do it for File " + directory + f)	

df = pd.DataFrame({'col1':logs})		
df.to_csv("logs.csv", sep=',',index=False)




