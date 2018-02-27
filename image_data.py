from imutils import paths
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np
import imutils
import cv2
import os
from keras import backend as K

# Funcion para devolver array de imagenes y labels de troqueles 
def get_data_label(relevant_path, img_width=None, img_height=None, img_channels=None, img_color=None):  
       data = []
       labels = []

       imagePaths = sorted(list(paths.list_images(relevant_path)))   
       # load the image, pre-process it, and store it in the data list
       for imagePath in imagePaths:
          if (img_color is not None):
            image = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE) 
          else: 
            image = cv2.imread(imagePath)
        
          if (img_width is not None and img_width is not None):
            image = cv2.resize(image, (img_width, img_height), img_channels)

          image = img_to_array(image)
          data.append(image)

          label = imagePath.split(os.path.sep)[-2]
          label = 1 if (label.endswith("1")) else 0
          labels.append(label)
 
       return (data, label) 

def extract_color_histogram(image, bins=(8, 8, 8)):
# extract a 3D color histogram from the HSV color space using
# the supplied number of `bins` per channel
     hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
     hist = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])

     # handle normalizing the histogram if we are using OpenCV 2.4.X
     if imutils.is_cv2():
        hist = cv2.normalize(hist)
        # otherwise, perform "in place" normalization in OpenCV 3 personally hate the way this is done
     else:
        cv2.normalize(hist, hist)
        
     # return the flattened histogram as the feature vector
     return hist.flatten()

# Funcion para leer imagenes como vector
def get_data_flatten(relevant_path, img_width=None, img_height=None, img_channels=None, img_color=None): 
         hist = []
         data = []
         labels = []
         imagePaths = sorted(list(paths.list_images(relevant_path)))
         # loop over the input images
         for (i, imagePath) in enumerate(imagePaths):
         # load the image and extract the class label (assuming that our
             image = cv2.imread(imagePath)
             # extract a color histogram from the image, then update the data matrix and labels list  
             hist = extract_color_histogram(image)
             data.append(hist)
         
             label = imagePath.split(os.path.sep)[-2]
             label = 1 if (label.endswith("1")) else 0
             labels.append(label)
             # show an update every 1,000 images
             if i > 0 and i % 1000 == 0:
                print("[INFO] processed {}/{}".format(i, len(imagePaths)))

         return (data, labels)

# Funcion convertir una imagen en flatten . 
def get_image_flatten(image_path, img_width=None, img_height=None, img_channels=None, img_color=None): 
    hist = []
    data = []
    if (img_color is not None):
       image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) 
    else: 
       image = cv2.imread(image_path)
        
    if (img_width is not None and img_width is not None):
       image = cv2.resize(image, (img_width, img_height), img_channels)

    # extract a color histogram from the image, then update the data matrix and labels list  
    hist = extract_color_histogram(image)
    data.append(hist)
         
    return (data)