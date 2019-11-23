# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 13:31:02 2019

LEARNING TFRECORDS READING AND WRITING
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

import numpy as np
import IPython.display as display
"""
Serializing in python: Using PICKLE
"""
# Example TFRecord as a Python dictionary might look like this:

my_dict = {'features' : {
    'my_ints': [5, 6],
    'my_float': [2.7],
    'my_bytes': ['data']
}}

# access values like this:
my_dict['features']['my_ints']
my_dict['features']['my_float']
my_dict['features']['my_bytes']

# in python, you can serialize the data and writ it to disk and read it again
# pickle module is used for this
os.getcwd()
os.chdir('c:/users/howar/documents')
import pickle

# pickle serializes the dictionary as a string
my_dict_str = pickle.dumps(my_dict)
my_dict_str

# outfile should be in binary mode, so use 'wb' and 'rb'
# write serialized dict to pickled file
with open('my_dict.pkl', 'wb') as f:
    f.write(my_dict_str)

# read serialized pickled file    
with open('my_dict.pkl', 'rb') as f: 
    that_dict_str = f.read()
    
# translate it back into the original form
that_dict = pickle.loads(that_dict_str)

that_dict


"""
Writing and Reading EXAMPLE records using Tensorflow
"""

# TFrecords adds detailed features to serializations

# Create a TFRecord in Example format:
#python 3, you need to precede the Byteslist value with a 'b' to indicate bytes
my_example = tf.train.Example(features=tf.train.Features(feature={
        'my_ints': tf.train.Feature(int64_list=tf.train.Int64List(value=[5,6])),
        'my_float': tf.train.Feature(float_list=tf.train.FloatList(value=[2.7])),
        'my_bytes': tf.train.Feature(bytes_list=tf.train.BytesList(value= [b'data']))
        }))

my_example.features.feature['my_ints'].int64_list.value
my_example.features.feature['my_float'].float_list.value
my_example.features.feature['my_bytes'].bytes_list.value

# writing and reading from disk are like pickle, except that the reader provides
# all records from TFRecords file.
# here, there is only one record.
my_example_str = my_example.SerializeToString()
with tf.io.TFRecordWriter('my_example.tfrecords') as writer:
    writer.write(my_example_str)

reader = tf.data.TFRecordDataset('my_example.tfrecords')
reader

for record in reader:
    example = tf.train.Example()
    example.ParseFromString(record.numpy())
    print(example)

"""
turning images into TFRecords
"""
#download two images
cat_in_snow  = tf.keras.utils.get_file('320px-Felis_catus-cat_on_snow.jpg', 'https://storage.googleapis.com/download.tensorflow.org/example_images/320px-Felis_catus-cat_on_snow.jpg')
williamsburg_bridge = tf.keras.utils.get_file('194px-New_East_River_Bridge_from_Brooklyn_det.4a09796u.jpg','https://storage.googleapis.com/download.tensorflow.org/example_images/194px-New_East_River_Bridge_from_Brooklyn_det.4a09796u.jpg')
#display cat
display.display(display.Image(filename=cat_in_snow))
display.display(display.HTML('Image cc-by: <a "href=https://commons.wikimedia.org/wiki/File:Felis_catus-cat_on_snow.jpg">Von.grzanka</a>'))
# display bridge
display.display(display.Image(filename=williamsburg_bridge))
display.display(display.HTML('<a "href=https://commons.wikimedia.org/wiki/File:New_East_River_Bridge_from_Brooklyn_det.4a09796u.jpg">From Wikimedia</a>'))


# write TFRecord file
"""
store the features, such as height, width, depth, label, and the data in bytes 
"""

# the following functions turn features into the acceptable tf.Example-compatible tf.train.Feature
# format, use these funcitons Each takes a scalar input vlaue and returns a tf.train.Feature 
# The following functions can be used to convert a value to a type compatible
# with tf.Example.

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

#create image labels and store them as a .proto
image_labels = {
    cat_in_snow : 0,
    williamsburg_bridge : 1,
}

# This is an example, just using the cat image.
image_string = open(cat_in_snow, 'rb').read()

label = image_labels[cat_in_snow]

# Create a dictionary with features that may be relevant.
def image_example(image_string, label):
  image_shape = tf.image.decode_jpeg(image_string).shape

  feature = {
      'height': _int64_feature(image_shape[0]),
      'width': _int64_feature(image_shape[1]),
      'depth': _int64_feature(image_shape[2]),
      'label': _int64_feature(label),
      'image_raw': _bytes_feature(image_string),
  }

  return tf.train.Example(features=tf.train.Features(feature=feature))

for line in str(image_example(image_string, label)).split('\n')[:15]:
  print(line)
print('...')

"""
All the features are stored in the tf.Example message. Now they can be written to a file named messages.tfrecords:
"""
# Write the raw image files to `images.tfrecords`.
# First, process the two images into `tf.Example` messages.
# Then, write to a `.tfrecords` file.
record_file = 'images.tfrecords'
with tf.io.TFRecordWriter(record_file) as writer:
  for filename, label in image_labels.items():
    image_string = open(filename, 'rb').read()
    tf_example = image_example(image_string, label)
    writer.write(tf_example.SerializeToString())

"""
Read the TFRecord file. Iterate over the records int it to read back what you wrote. Given that
in this example you will only reproduce the image, the only feaure you will need is the raw image string.
Extract it using the getters 
example.features.feature['image_raw'].bytes_list.value[0]
You can also use the labels to determine which record is the car and which one is the bridge
"""
raw_image_dataset = tf.data.TFRecordDataset('images.tfrecords')

# Create a dictionary describing the features.
image_feature_description = {
    'height': tf.io.FixedLenFeature([], tf.int64),
    'width': tf.io.FixedLenFeature([], tf.int64),
    'depth': tf.io.FixedLenFeature([], tf.int64),
    'label': tf.io.FixedLenFeature([], tf.int64),
    'image_raw': tf.io.FixedLenFeature([], tf.string),
}

def _parse_image_function(example_proto):
  # Parse the input tf.Example proto using the dictionary above.
  return tf.io.parse_single_example(example_proto, image_feature_description)

parsed_image_dataset = raw_image_dataset.map(_parse_image_function)
parsed_image_dataset

"""
Recover the images from the TFRecord file:
"""

for image_features in parsed_image_dataset:
  image_raw = image_features['image_raw'].numpy()
  label = image_features['label']
  display.display(display.Image(data=image_raw))
  print(label)
  
""" 
trying to read stored TFRecord in TFRecordsfolder
"""
os.chdir('C:\\Users\\howar\\Documents\\Database_Management\\Term_Project\\Dataset\\DET\\TF_Recordsfolder')
raw_image_dataset = tf.data.TFRecordDataset('train-00001-of-01024')

# taking imagenet_preprocessing feature description of images
image_feature_description ={
      'image/height': tf.io.FixedLenFeature([], tf.int64),
      'image/width': tf.io.FixedLenFeature([], tf.int64),
      'image/colorspace': tf.io.FixedLenFeature([], tf.string),
      'image/channels': tf.io.FixedLenFeature([], tf.int64),
      'image/class/label': tf.io.FixedLenFeature([], tf.int64),
      'image/class/synset': tf.io.FixedLenFeature([], tf.string),
      'image/class/text': tf.io.FixedLenFeature([], tf.string),
      #'image/object/bbox/xmin': tf.io.FixedLenFeature([], tf.int64),
      #'image/object/bbox/xmax': tf.io.FixedLenFeature([], tf.int64),
      #'image/object/bbox/ymin': tf.io.FixedLenFeature([], tf.int64),
      #'image/object/bbox/ymax': tf.io.FixedLenFeature([], tf.int64),
      #'image/object/bbox/label': tf.io.FixedLenFeature([], tf.int64),
      'image/format': tf.io.FixedLenFeature([], tf.string),
      'image/filename': tf.io.FixedLenFeature([], tf.string),
      'image/encoded': tf.io.FixedLenFeature([], tf.string)
}

# use parsing function above
parsed_image = raw_image_dataset.map(_parse_image_function)
parsed_image

for image_features in parsed_image:
  image_raw = image_features['image/encoded'].numpy()
  label = image_features['image/class/text']
  display.display(display.Image(data=image_raw))
  print(label)
