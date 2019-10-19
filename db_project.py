"""Begin with organizing all training images for pre-processing"""

import os
import numpy as np
from pathlib import Path
from PIL import Image
os.chdir('c:/users/howar/documents/database_management/Term_project/')
os.getcwd()

# Path to ImageNet DET Data
data_dir = Path("c:/users/howar/documents/database_management/Term_project/dataset/DET/train/ILSVRC2013_train/")

"""
Function that collects all file paths within the training set folder
"""
pathlist = []

def directory_search(path):
    global pathlist
    pathlist.clear()
    for path, sub, files in os.walk(path):
        sub = [n for n in sub]
        contents = files
        for f in contents:
            #print('Print Path:', path + os.sep + os.sep + f)
            pathlist.append(path + os.sep + os.sep + f)
    print('Pathlist length: ' + str(len(pathlist)))
directory_search(data_dir)

""" 
function that filters paths for a single synset
""" 
target_files = []
def synset_filter(synset):
    synset = str(synset)
    global pathlist
    global target_files
    target_files.clear()
    target_files = [ s for s in pathlist if synset in s]
    image_count = len(target_files)
    print(target_files[0])
    return image_count
    
synset_filter('n01443537')

"""
function that uses filtered paths to create arrays for images in a 
single synset
"""
arrays = []
def gen_array():
    global target_files
    global arrays
    for f in target_files:
        x = Image.open(f)
        x_arr = np.array([np.array(x)])
        arrays.append(x_arr)
        
    print("Number of arrays entered: " + str(len(arrays)))
    print(arrays[0].shape)
gen_array()

""" 
working way to find and read an image 
This was a preliminary way of examing the contents of the images 
prior to entering the images into numpy arrays.
"""
folders = os.listdir(data_dir)
img_1 = os.path.join(data_dir, folders[0], "n00007846_6247.JPEG" )
print(img_1)
contents = Image.open(img_1)
print(contents)
x = np.array([np.array(contents)])
print(x.shape)
#######################################################################
""" testing zone """
"""
creating a folder for each method
"""
# we will modify the experiment from RealPython, which saves images to disk
# as .png files. We assume that the ImageNet DET dataset is installed.
# The ImageNet DET dataset images are in .JPEG format
lmdb_dir = Path("data/lmdb/")
hdf5_dir = Path("data/hdf5/")

"""
asking Path to create folders for the stored images
"""
lmdb_dir.mkdir(parents=True, exist_ok=True)
hdf5_dir.mkdir(parents=True, exist_ok=True)


################################# loading images into tensorflow
"""
loading tensorflow
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
tf.enable_eager_execution()

AUTOTUNE= tf.data.experimental.AUTOTUNE

import IPython.display as display
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

tf.__version__

"""
Creating list of synset names from data_dir
"""

CLASS_NAMES = np.array([item.name for item in data_dir.glob('*')])
CLASS_NAMES

"""
Load images using tf.keras.preprocessing
The .1/255 is to conver from uint8 to float32 in range [0,1]
"""
# if we run directory_search(), then we can get the total image count by using the pathlist[]
image_count = len(pathlist)
# you want a single synset, run synset_filter() and input the target_files[] list  
image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
# Define parameters for the image loader:
BATCH_SIZE = 32
IMG_HEIGHT = 224
IMG_WIDTH = 224
STEPS_PER_EPOCH = np.ceil(image_count/BATCH_SIZE)

train_data_gen = image_generator.flow_from_directory(directory=str(data_dir),
                                                     batch_size = BATCH_SIZE,
                                                     shuffle = True,
                                                     target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                     classes = list(CLASS_NAMES))
STEPS_PER_EPOCH
"""
Inspect a batch
"""
def show_batch(image_batch, label_batch):
  plt.figure(figsize=(10,10))
  for n in range(25):
      ax = plt.subplot(5,5,n+1)
      plt.imshow(image_batch[n])
      plt.title(CLASS_NAMES[label_batch[n]==1][0].title())
      plt.axis('off')
      
image_batch, label_batch = next(train_data_gen)
show_batch(image_batch, label_batch)

