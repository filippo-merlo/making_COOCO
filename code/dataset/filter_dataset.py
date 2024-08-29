#%%
from utils_powerpaint import *
from dataset import *
import pandas as pd
import pickle as pkl
import os
import json

# Import the generated images of the dataset and retrieve their code

# DECLARE PATHS
CACHE_DIR_SHARED = '/mnt/cimec-storage6/shared/hf_llms_checkpoints'
CACHE_DIR_PRIVATE = '/mnt/cimec-storage6/users/filippo.merlo'

# Path for the data folder
data_folder_path = '/mnt/cimec-storage6/users/filippo.merlo/sceneREG_data/'

# Paths for the already made dataset 
dataset_path = os.path.join(data_folder_path,'coco_search18/coco_search18_annotated.json')
path_to_images = "/mnt/cimec-storage6/users/filippo.merlo/sceneREG_data/generated_images"

### List to store the names of all image files
image_files = []

# Walk through the directory
for root, dirs, files in os.walk(path_to_images):
    for file in files:
       if file.endswith(".jpg"):
            # Append the file name to the list
            image_files.append(file)

# Display the list of image file names
print(image_files)

# load all the other data for the dataset
dataset = Dataset(dataset_path = dataset_path)
data = dataset.data

def get_square_image(image, target_bbox):
    width, height = image.size
    new_x, new_y, new_w, new_h = adjust_ratio(image, target_bbox, 0.5, 2)
    new_bbox = (new_x, new_y, new_w, new_h)
    # Create a black background image
    mask = Image.new("L", (width, height), 0)
    
    # Draw a white box on the black background
    draw = ImageDraw.Draw(mask)
    draw.rectangle([new_x, new_y, new_x + new_w, new_y + new_h], fill=255)
    
    return image, mask, new_bbox

name_bbox = {}
 # Get the masked image with target and scene category
for image_name in image_files:
    image_name = image_name.split('_')[0]
    target, scene_category, image_picture, image_picture_w_bbox, target_bbox, cropped_target_only_image, object_mask = get_coco_image_data(data, image_name)
    save_path_original = os.path.join(data_folder_path+'generated_images', f"{scene_category.replace('/', '_')}/{image_name.replace('.jpg', '')}_{scene_category.replace('/', '_')}_{target.replace('/', '_').replace(' ', '_')}_original.jpg")
    image_picture.save(save_path_original)

    # remove the object before background
    image_clean = remove_object(image_picture, object_mask)
    image, mask, new_bbox = get_square_image(image_clean, target_bbox)
    name_bbox[image_name] = new_bbox

# match the info of scene, objec, bbox 



# unify everything in a single dataset


