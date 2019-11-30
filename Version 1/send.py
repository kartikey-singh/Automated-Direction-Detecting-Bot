import requests
import os
import json
import base64
fPath = os.getcwd()
url = 'http://127.0.0.1:8000/submitcausedata/'
# url = 'https://ml-arrow-detect.herokuapp.com/submitcausedata/'
headers = {'content-type': 'application/x-www-form-urlencoded'}
# headers = {'content-type': 'application/x-www-form-urlencoded; charset=UTF-8'}
path_img = fPath + '/image129.jpg'

# a = 'eW91ciB0ZXh0'
# s = base64.b64decode(a)
# print(s)

# with open(path_img, 'rb') as img:
# 	encoded_image = base64.encodestring(img.read())
# 	name_img= os.path.basename(path_img)
# 	print(encoded_image)
# 	files = {'im':str(encoded_image)}
# 	# data_json = json.dumps(files)
# 	# print(data_json)
# 	with requests.Session() as s:
# 		r = s.post(url,files=files,headers=headers)
# 		print(r)

data = open(path_img,'rb').read()
# print(len(data))
# print(type(data))
# encoded_image = base64.encodestring(data)	
encoded_image = base64.b64encode(data)	
# print(encoded_image)
# print(encoded_image[-10:])
# print(len(encoded_image))
# print(type(encoded_image))

r = requests.post(url,data=encoded_image,headers=headers)		
s = r.json()
print(s['status'])
# print(r.json)
# data = data.encode('base64','strict')
# print(len(data))
# r = requests.post(url,data=data,headers=headers)		

# import base64
# image = open('image13.jpg', 'rb') 
# image_read = image.read() 
# # print(image_read[:10])
# image_64_encode = base64.b64encode(image_read) 

# # image_64_encode = base64.encodestring(image_read) 
# # print(image_64_encode[:10])
# image_64_decode = base64.b64decode(image_64_encode) 
# # image_64_decode = base64.decodestring(image_64_encode) 
# # print(image_64_decode[:10])
# image_result = open('image14.txt','wb') # create a writable image and write the decoding result 
# image_result.write(image_64_decode)

