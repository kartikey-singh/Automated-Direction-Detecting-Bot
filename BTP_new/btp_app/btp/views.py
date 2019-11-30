from django.shortcuts import render, render_to_response, HttpResponse, redirect
from django.http import Http404,JsonResponse,HttpResponseBadRequest
from django.views.generic.edit import CreateView, DeleteView, UpdateView
from django.views import generic
from django.views.generic.list import ListView
from django.urls import reverse
from django.contrib.auth.models import User
from django.utils.decorators import method_decorator
from django.contrib.auth.decorators import login_required
from django.contrib.auth.mixins import LoginRequiredMixin
from django.contrib import messages
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth import authenticate, login, logout
from .models import *
import datetime
from django.utils import timezone
from django.template import RequestContext
import base64
import json
import os
import cv2
import pandas as pd
import numpy as np
import math
import cv2
import pandas as pd
import numpy as np
import os
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from base64 import decodestring

def processImage(item):	
	fPath = os.getcwd()		
	# itemPath = fPath + "/btp" + item
	image = cv2.imread(item)

	height, width, channels = image.shape     
	blackImage = np.zeros((height,width))	
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	_,thresh = cv2.threshold(gray,75,255,cv2.THRESH_BINARY_INV)
	kern_dilate = np.ones((8,8),np.uint8)
	kern_erode  = np.ones((3,3),np.uint8)
	mask = cv2.erode(thresh,kern_erode,iterations = 2)
	mask = cv2.dilate(mask,kern_dilate,iterations = 2) 

	_,contours, hierarchy = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE) 

	centreX,  centreY = width/2, height/2
	minDis = math.inf
	Sx, Sy, Sw, Sh = 0, 0, 0, 0
	for contour in contours:             
		[x,y,w,h] = cv2.boundingRect(contour)
		if h<40 or w<40:     
			continue
		Cx, Cy  = x + w/2, y + h/2;     
		dis = (Cx - centreX)*(Cx - centreX) + (Cy - centreY)*(Cy - centreY)
		if dis < minDis:
			Sx, Sy, Sw, Sh = x, y, w, h    
			minDis = dis
			
	roi = mask[Sy: Sy + Sh, Sx: Sx + Sw]    
	blackImage[Sy: Sy + Sh, Sx: Sx + Sw] = roi                

	processedPath = fPath + "/btp/ProcessedData/Image1.jpg"
	(h, w) = blackImage.shape
	temp = np.zeros((h, w))
	temp = blackImage
	center = (w/2, h/2)
	scale = 1.0
	M = cv2.getRotationMatrix2D(center, 0, scale)        
	img = cv2.warpAffine(temp, M, (h, w))
	
	cv2.imwrite(processedPath,img)
	return None

def predict():
	fPath = os.getcwd()
	processedPath = fPath + "/btp/ProcessedData/Image1.jpg"
	img = mpimg.imread(processedPath)
	image = color.rgb2gray(img)
	image_rescaled = rescale(image, 1.0 / 4.0, anti_aliasing=False)
	image_resized = resize(image, (image.shape[0] / 4, image.shape[1] / 4), anti_aliasing=True)
	if img.shape[0] == 480:
		image_downscaled = downscale_local_mean(image, (3, 4))
	else:
		image_downscaled = downscale_local_mean(image, (4, 3))
	# image_downscaled = downscale_local_mean(image, (4, 3))
	img1 = np.ravel(image_downscaled)

	filename = 'finalized_model.sav'	
	loaded_model = joblib.load(fPath + '/btp/' + filename)
	
	im = img1.reshape(1,-1)
	ans = loaded_model.predict(im)
	print(ans)
	return ans


# Create your views here.
def home(request):
	return render(request, 'index.html')		

@csrf_exempt
def submitCauseData(request):	
	response_data = {}  	
	if request.method == 'POST':
		# received_json_data = json.loads(request.body.decode("utf-8"))
		data = request.POST		
		myDict = dict(data)		
		imageStr = list(myDict.keys())		
		# print(type(imageStr[0]))
		image = imageStr[0].replace(' ','+')
		image = image.encode() + b'==='								
		imagedec = base64.b64decode(image)
		# print(image)
		# print(imageStr[-10:])
		# print(len(imageStr))
		# print(type(imageStr))
		# print(imagedec[:10])			
		# print(len(imagedec))
		# print(type(imagedec))
		fPath = os.getcwd()		
		image_result = open(fPath + '/btp/image.jpg', 'wb')
		image_result.write(imagedec)
		image_result.close()		
		try:            
			processImage(fPath + '/btp/image.jpg')
			ans = predict()
			response_data['status'] = str(ans[0])
			return JsonResponse(response_data)
		except:
			response_data['status'] = "-1"
			return JsonResponse(response_data)    
	return HttpResponse("HELLO")		