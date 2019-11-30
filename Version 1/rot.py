import numpy as np
import matplotlib.image as img

image = img.imread('image3.jpg')
im = np.rot90(image)
img.imsave('image1.jpg',im)


