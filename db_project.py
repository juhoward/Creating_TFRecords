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
dirpaths = []
dirnames = []
files = []
def directory_search(path):
    global dirpaths
    global files
    dirpaths.clear()
    files.clear()
    # retrieves directory paths, and includes the given path
    for dirpath, dirnames, files in os.walk(path):
        dirpaths.append(dirpath)
    dirpaths.remove(dirpaths[0])
        #for file in files:
            #dirnames.append(dirname)
            #files.append(file)
    #num_files = len(files)
    list_len = len(dirpaths)    
    print("The number of directories is :" + str(list_len))
    #print("The number of files is: " + str(num_files))
    #print("The number of directory names within " + path + " is: " + len(dirnames))
    #print("The number of files within " + path + " is: " + len(files))
    print('The structure of the first path: \n' + dirpaths[0])
    return
directory_search(data_dir)

""" working way to find and read an image """
folders = os.listdir(data_dir)
img_1 = os.path.join(data_dir, folders[0], "n00007846_6247.JPEG" )
print(img_1)
contents = Image.open(img_1)
print(contents)
x = np.array([np.array(contents)])
print(x.shape)
""" testing zone """
