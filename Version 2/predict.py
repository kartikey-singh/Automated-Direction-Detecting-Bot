from keras.models import load_model
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
from keras import backend as k


def load_image(img_path, show=False):
    img = image.load_img(img_path, target_size=(224, 224))
    # (height, width, channels)
    img_tensor = image.img_to_array(img)
    # (1, height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    # imshow expects values in the range [0, 1]
    img_tensor /= 255.

    if show:
        plt.imshow(img_tensor[0])
        plt.axis('off')
        plt.show()

    return img_tensor


def predict(img_path):    
    # {'Up': 0, 'Right': 1, 'Down': 2, 'Left': 3}    
    model_loaded = load_model('models/model_2019-11-26 08:02:04.860320.h5')
    new_image = load_image(img_path)
    pred = model_loaded.predict(new_image)    
    predicted_class_indices = np.argmax(pred, axis=1)        
    return predicted_class_indices[0]


if __name__ == "__main__":
    img_path = '/home/kartikey/Desktop/Semester VII/BTPv2/Test/Right/Image42.jpg'
    print(predict(img_path))
