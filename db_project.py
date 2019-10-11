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
    pathlist = []
    for path, sub, files in os.walk(path):
        sub = [n for n in sub]
        contents = files
        for f in contents:
            print('Print Path:', path + os.sep + os.sep + f)
            pathlist.append(path + os.sep + os.sep + f)
    print('Pathlist length: ' + str(len(pathlist)))
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
