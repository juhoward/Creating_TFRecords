# Creating TFRecords for the ImageNet ILSVRC2017 DET Dataset

Attempting to train a neural network on the ImageNet dataset for the first time can be a daunting process riddled with problems to troubleshoot.
The steps below are a path we took toward completing that goal. 

#Generating TFRecords
##First Steps:

1. Create your TensorFlow environment. We recommend the tf-gpu version of the install.
  You can use the tf-gpu.yml environment file that we provide. It installs the dependencies for tensorflow 2.0.
  To actually use tensorflow's GPU capabilities, install CUDA and CuDNN

2. Clone Google's tensorflow/models repository in the same directory as your tf-gpu environment for convenience.
 git clone https://github.com/tensorflow/models.git

## Downloading ImageNet ILSVRC2017 dataset for object detection:

1. Download the ILSVRC2017 dataset from http://image-net.org/challenges/LSVRC/2017/downloads
  It is 55GB and can take about 10 hours if you have a fiber optic connection. If storage space or time are issues, this blog post provides a solution:
  https://blog.exxactcorp.com/deep-learning-with-tensorflow-training-resnet-50-from-scratch-using-the-imagenet-dataset/
  
2. Download the boundary box annotations from http://image-net.org/download-bboxes

3. Within the conda terminal, change directories to: C:yourpath\tensorflow\models\research\slim\datasets
  Run the bounding box parsing script from the terminal:
  python process_bounding_boxes.py <dir_where_annotations_are> [optional-synsets-textfile-to-limit-script] 1> bounding_boxes.csv
  The 1> bounding_boxes.csv portion will redirect the output of the script to a CSV file you'll search for in your current directory.

4. To create TFRecords using the Google script, you will need the following data that you can find in the TFRecords directory of this repository:
  labels_file.txt
  num_threads.txt
  synset_labels.txt
  imagenet_metadata.txt
  bounding_boxes.csv
  You can either create these files yourself, or you can use those we have supplied.

5. Run build_imagenet_data.py
  You can either edit the script to supply your data_directory or use the --dir="data_dir" flag.




  
