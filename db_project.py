import os
from pathlib import Path
os.chdir('c:/users/howar/documents/database_management/Term_project/')
os.getcwd()

# Path to ImageNet DET Data
data_dir = Path("c:/users/howar/documents/database_management/Term_project/dataset/DET/train/ILSVRC2013_train/")

# identify folders containing images in the training set
folders = []
def folder_list(data_dir):
    global folders
    folders = os.listdir(data_dir)
    return
folder_list(data_dir)
# make list of images within the folders of the training set
# identify folders containing images in the training set
folders = []
def folder_list(data_dir):
    global folders
    folders = os.listdir(data_dir)
    return
# make list of images within the folders of the training set
images = []
def gather_images(data_dir):
    folder_list(data_dir)
    global images
    # uses the list of folders from folder_list
    # makes images -> a list of images within 569 folders in data_dir
    for folder in folders:
        images.append(os.listdir(data_dir/folder))
    return
gather_images(data_dir)


print(os.listdir(data_dir/folders[0]))