#%%
# IMPORTS
from config_filter import *
import os 
import torch
import math
import random as rn
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFilter
import numpy as np
from collections import Counter
import re
from tqdm import tqdm
import pandas as pd
import pickle as pkl
import json
import re
#%%
# DATASET CLASS
class Dataset:

    def __init__(self, dataset_path = None):
        if dataset_path:
            with open(dataset_path) as f:
                self.data = json.load(f)
                self.img_names = list(self.data.keys())
                '''
                for each img name there are three keys:
                Key: [fixations]
                    Key: [name], Type of Value: [str]
                    Key: [subject], Type of Value: [int]
                    Key: [condition], Type of Value: [str]
                    Key: [X], Type of Value: [list]
                    Key: [Y], Type of Value: [list]
                    Key: [T], Type of Value: [list]
                    Key: [length], Type of Value: [int]
                    Key: [split], Type of Value: [str]
                    Key: [fixOnTarget], Type of Value: [bool]
                    Key: [correct], Type of Value: [int]
                Key: [captions_val2017_annotations] or [captions_train2017_annotations]
                    Key: [image_id], Type of Value: [int]
                    Key: [id], Type of Value: [int]
                    Key: [caption], Type of Value: [str]
                Key: [instances_val2017_annotations] or [instances_train2017_annotations]
                    Key: [segmentation], Type of Value: [list]
                    Key: [area], Type of Value: [float]
                    Key: [iscrowd], Type of Value: [int]
                    Key: [image_id], Type of Value: [int]
                    Key: [bbox], Type of Value: [list]
                    Key: [category_id], Type of Value: [int]
                    Key: [id], Type of Value: [int]
                '''
# UTILS
#%%
### GENERAL FUNCTIONS
def get_files(directory):
    """
    Get all files in a directory with specified extensions.

    Args:
    - directory (str): The directory path.
    - extensions (list): A list of extensions to filter files by.

    Returns:
    - files (list): A list of file paths.
    """
    files = []
    for file in os.listdir(directory):
        if file.endswith(tuple([".json",".jpg"])):
            files.append(os.path.join(directory, file))
    return files

def reverse_dict(data):
    """
    This function reverses a dictionary by swapping keys and values.

    Args:
        data: A dictionary to be reversed.

    Returns:
        A new dictionary where keys become values and vice versa, handling duplicates appropriately.
    """
    reversed_dict = {}
    for key, value in data.items():
        for l in value:
            reversed_dict[str(l)] = key
    return reversed_dict

def subtract_in_bounds(x, y):
    """
    Subtract two numbers and ensure the result is non-negative.
    """
    if x - y > 0:
        return int(x - y) 
    else:
        return 0
    
def add_in_bounds(x, y, max):
    """
    Add two numbers and ensure the result is within a specified range.
    """
    if x + y < max:
        return int(x + y)
    else:
        return int(max)

def sum_lists(list1, list2):
    """
    Sum two lists element-wise.
    """
    if len(list1) != len(list2):
        raise ValueError("Lists must have the same length")
    
    return [x + y for x, y in zip(list1, list2)]

def cosine_similarity(vec1, vec2, epsilon=1e-10):
    """
    Compute the cosine similarity between two vectors.
    
    Args:
        vec1: A numpy array representing the first vector.
        vec2: A numpy array representing the second vector.
        epsilon: A small value to prevent division by zero (default: 1e-10).
        
    Returns:
        Cosine similarity between vec1 and vec2.
    """
    # Compute the dot product of the vectors
    dot_product = np.dot(vec1, vec2)
    
    # Compute the magnitudes of the vectors
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    
    # Add epsilon to avoid division by zero
    denominator = norm_vec1 * norm_vec2 + epsilon
    
    # Compute the cosine similarity
    cosine_sim = dot_product / denominator
    
    return cosine_sim

def select_k(alist, k, lower = True):
    """
    Find the indices and values of the k lowest/higest elements in a list.
    """
    # Step 1: Enumerate the list to pair each element with its index
    enumerated_list = list(enumerate(alist))
    
    # Step 2: Sort the enumerated list by the element values
    if lower:
        reverse = False
    else:
        reverse = True
    sorted_list = sorted(enumerated_list, key=lambda x: x[1], reverse=reverse)
    
    # Step 3: Extract the indices of the first k elements
    k_indices = [index for index, value in sorted_list[:k]]
    k_values = [value for index, value in sorted_list[:k]]
    
    return k_indices, k_values


from scipy.spatial import ConvexHull

def augment_area_within_bounds(coordinates, scale_factor, img_width, img_height):
    if not isinstance(coordinates, np.ndarray):
        coordinates = np.array(coordinates)
    
    # Flatten the array if it's a list of coordinate pairs
    if coordinates.ndim == 2 and coordinates.shape[1] == 2:
        coords = coordinates
    else:
        coords = coordinates.reshape(-1, 2)
    
    # Calculate the convex hull
    hull = ConvexHull(coords)
    hull_coords = coords[hull.vertices]
    
    # Calculate the centroid
    centroid = np.mean(hull_coords, axis=0)
    
    # Translate coordinates to the origin (centroid)
    translated_coords = hull_coords - centroid
    
    # Convert to polar coordinates
    radii = np.linalg.norm(translated_coords, axis=1)
    angles = np.arctan2(translated_coords[:, 1], translated_coords[:, 0])
    
    # Scale the radial distances
    scaled_radii = radii * scale_factor
    
    # Convert back to Cartesian coordinates
    scaled_coords = np.column_stack((scaled_radii * np.cos(angles), scaled_radii * np.sin(angles)))
    
    # Translate the coordinates back to the original centroid position
    augmented_coords = scaled_coords + centroid
    
    # Check if any point is out of bounds
    min_x, min_y = np.min(augmented_coords, axis=0)
    max_x, max_y = np.max(augmented_coords, axis=0)
    
    # Adjust scale factor if necessary
    if min_x < 0 or min_y < 0 or max_x > img_width or max_y > img_height:
        scale_x = min(img_width / (max_x - centroid[0]), centroid[0] / -min_x) if min_x < 0 or max_x > img_width else scale_factor
        scale_y = min(img_height / (max_y - centroid[1]), centroid[1] / -min_y) if min_y < 0 or max_y > img_height else scale_factor
        adjusted_scale_factor = min(scale_x, scale_y)
        scaled_radii = radii * adjusted_scale_factor
        scaled_coords = np.column_stack((scaled_radii * np.cos(angles), scaled_radii * np.sin(angles)))
        augmented_coords = scaled_coords + centroid

    return augmented_coords.astype(np.int32)

#%%
### GET COCO IMAGE DATA
def get_coco_image_data(data, img_name = None):
        # Get a random image from data
        if img_name != None:
            image = data[img_name]
            for fix in image['fixations']:
                if fix['condition'] == 'absent':
                    target = None
                    raise ValueError("Absent Target")
                if 'task' in fix.keys():
                    target = fix['task']
                    break
                else:
                    raise ValueError("Not Task")
        else:
            while img_name == None:
                img_name = rn.choice(list(data.keys()))
                image = data[img_name]
                for fix in image['fixations']:
                    if fix['condition'] == 'absent':
                        target = None
                        break
                    if 'task' in fix.keys():
                        target = fix['task']
                        break
                    else:
                        target = None
                if target == None:
                    img_name = None
                
        print('*',target)

        # Get the image picture
        images_paths = get_files(coco_images_path)
        image_picture = None
        for image_path in images_paths:
            if img_name in image_path:
                image_picture = Image.open(image_path)
                break
            
        # Get the target info 
        # get the right annotation key
        try:
            ann_key = 'instances_train2017_annotations'
            image[ann_key]
        except:
            ann_key = 'instances_val2017_annotations'

        # go through annotations (objects) for the image
        # every ann is an object
        for ann in image[ann_key]:
            id = ann['category_id'] # get the category id
            
            # get the target object info
            object_name = ''
            for cat in coco_object_cat:
                if cat['id'] == id:
                    object_name = cat['name']
            # get target object info
            if object_name == target:
                target_bbox = ann['bbox']
                target_segmentation = ann['segmentation']
                target_area = ann['area']

        if target_bbox:
            image_picture_w_bbox = Image.open(image_path)
            draw = ImageDraw.Draw(image_picture_w_bbox)
            x, y, width, height = target_bbox
            draw.rectangle([x, y, x + width, y + height], outline="red", width=3)

        # Image processing and cropping code
        # Segment the target area in the image
        # Convert the image from RGB to BGR format using OpenCV
        image_mask_cv2 = cv2.cvtColor(np.array(image_picture), cv2.COLOR_RGB2BGR)

        # Get augmented segmentation coordinates
        max_w, max_h = image_picture.size
        target_segmentation = augment_area_within_bounds(target_segmentation, 1.20, max_w, max_h)

        # Create a mask with the same height and width as the image, initialized to zeros (black)
        image_mask = np.zeros(image_mask_cv2.shape[:2], dtype=np.uint8)

        # Fill the polygon defined by target_segmentation on the mask with white (255)
        cv2.fillPoly(image_mask, [target_segmentation], 255)

        # Convert the mask to a PIL image
        image_mask_pil = Image.fromarray(image_mask)

        # Apply the mask to the image, resulting in an image where only the segmented area is visible
        target_only_image = cv2.bitwise_and(image_mask_cv2, image_mask_cv2, mask=image_mask)

        # Crop the image to a bounding box around the segmented area
        # Extract the bounding box coordinates and dimensions
        x, y, w, h = target_bbox

        # Adjust the coordinates and dimensions to include some padding
        x_c = subtract_in_bounds(x, 20)
        y_c = subtract_in_bounds(y, 20)
        w_c = add_in_bounds(x, w + 20, max_w)
        h_c = add_in_bounds(y, h + 20, max_h)

        # Crop the masked image using the adjusted coordinates and dimensions
        cropped_target_only_image = target_only_image[y_c:h_c, x_c:w_c]

        # Convert the cropped image from BGR to RGB format
        cropped_target_only_image_rgb = cv2.cvtColor(cropped_target_only_image, cv2.COLOR_BGR2RGB)

        # Convert the cropped image to a PIL image
        cropped_target_only_image_pil = Image.fromarray(cropped_target_only_image_rgb)

        # Additionally, crop the original image (without the mask) to the same bounding box

        return target, image_picture, image_picture_w_bbox, target_bbox, cropped_target_only_image_pil, image_mask_pil

def find_object_for_replacement_continuous(target_object_name, scene_name):
    # get the more similar in size with the less semantic relatedness to the scene
    size_scores = []
    semantic_relatedness_scores = []

    #filter by size
    for thing in things_words_context:
        # SEM REL
        scene_vect = scenes2vec[scene_name]
        object_vect = things2vec[thing]
        semantic_relatedness_scores.append(cosine_similarity(scene_vect, object_vect))

        # SIZE
        # target size
        things_name_target = map_coco2things[target_object_name]
        target_size_score = things_plus_size_mean_matrix[things_plus_size_mean_matrix['WordContext']==things_name_target]['Size_mean'].values[0]
        # object size
        # remove same object from the possible replacements
        if map_coco2things[target_object_name] == thing:
            size_scores.append(500)
        else:
            object_size_score = things_plus_size_mean_matrix[things_plus_size_mean_matrix['WordContext']==thing]['Size_mean'].values[0]
            size_scores.append(abs(target_size_score - object_size_score))

    idxs = list(range(len(things_words_context)))
    # filter all the objects that have more than d_max size distance
    d_max = 25
    for i, score in enumerate(size_scores):
        if score > d_max:
            idxs.remove(i)

    # remaining objects
    objects = [things_words_context[i] for i in idxs]
    semantic_relatedness_scores = [semantic_relatedness_scores[i] for i in idxs]
    
    # get 3 objects with the lowest relatedness score, near to 0
    k = 15
    r = 15
    kidxs, vals = select_k(semantic_relatedness_scores, k, lower = True)
    print(vals)
    things_names = [objects[i] for i in kidxs]
    random_3_names_lower = rn.sample(things_names, r)

    # get 3 objects with the higer relatedness score, near to 1
    kidxs, vals = select_k(semantic_relatedness_scores, k, lower = False)
    things_names = [objects[i] for i in kidxs]
    random_3_names_higer = rn.sample(things_names, r)

    # get 3 objects with relatedness score near to 0.5
    semantic_relatedness_scores_sub = [abs(score - 0.25) for score in semantic_relatedness_scores]
    kidxs, vals = select_k(semantic_relatedness_scores_sub, k, lower = True)
    things_names = [objects[i] for i in kidxs]
    random_3_names_middle = rn.sample(things_names, r)

    return random_3_names_lower, random_3_names_middle, random_3_names_higer

def get_images_names(substitutes_list):
    # get things images paths [(name, path)...]
    things_folder_names = list(set([things_words_id[things_words_context.index(n)] for n in substitutes_list]))
    images_names_list = []
    images_path_list = []
    for folder_name in things_folder_names:
        folders_path = os.path.join(things_images_path, folder_name)
        images_paths = get_all_names(folders_path)
        for i_p in images_paths:
            things_obj_name = re.sub(r"\d+", "",i_p.split('/')[-2]).replace('_',' ')
            if folder_name == things_obj_name:
                images_names_list.append(things_words_context[things_words_id.index(folder_name)].replace('/','_'))
                images_path_list.append(i_p)
    return images_path_list, images_names_list

def get_all_names(path):
    """
    This function retrieves all file and folder names within a directory and its subdirectories.

    Args:
        path: The directory path to search.

    Returns:
        A list containing all file and folder names.
    """
    names = []
    for root, dirs, files in os.walk(path):
        for name in files:
            names.append(os.path.join(root, name))
        for name in dirs:
            names.append(os.path.join(root, name))
    return names

def compare_imgs(target_patch, substitutes_list):
    # get things images paths [(name, path)...]
    images_path_list, images_names_list = get_images_names(substitutes_list)
    print(len(images_path_list))
    # embed images
    images_embeddings = []
    with torch.no_grad():
        for i_path in images_path_list:
            image = Image.open(i_path)
            image_input = vitc_image_processor(image, return_tensors="pt").to(DEVICE)
            image_outputs = vitc_model(**image_input)
            image_embeds = image_outputs.last_hidden_state[0][0].to('cpu')#.squeeze().mean(dim=1).to('cpu')
            images_embeddings.append(image_embeds)
        # embed target 
        target_input = vitc_image_processor(target_patch, return_tensors="pt").to(DEVICE)
        target_outputs = vitc_model(**target_input)
        target_embeds = target_outputs.last_hidden_state[0][0].to('cpu')#.squeeze().mean(dim=1).to('cpu')

    # compare
    similarities = []
    for i_embed in images_embeddings:
        similarities.append(cosine_similarity(target_embeds.detach().numpy(), i_embed.detach().numpy()))
    # get top k
    k = 5
    print(torch.tensor(similarities).size())
    v, indices = torch.topk(torch.tensor(similarities), k)
   
    return [images_names_list[i] for i in indices], [images_path_list[i] for i in indices]

### GEenerate image function
from gradio_client import Client

def api_upscale_image_gradio(image, scale_factor=4):
    # save temporarely image:
    path = os.path.join(data_folder_path, 'temp.jpg')
    image.save(path)
    # upscale image
    client = Client("https://bookbot-image-upscaling-playground.hf.space/")
    
    result = client.predict(
            path,	
            f"modelx{scale_factor}",	# str in 'Choose Upscaler' Radio component
            api_name="/predict"
    )
    new_image = Image.open(result)
    return new_image

def remove_object(image, object_mask):
    return simple_lama(image, object_mask.convert('L'))

def adjust_ratio(image, bbox, min_ratio, max_ratio):
    # Image size
    width, height = image.size
    
    # Calculate current aspect ratio
    x, y, w, h = bbox
    current_ratio = w / h
    
    # Adjust dimensions to fit within the desired aspect ratio range
    if current_ratio < min_ratio:
        # Adjust width to meet min_ratio
        new_w = min_ratio * h
        new_h = h
    elif current_ratio > max_ratio:
        # Adjust height to meet max_ratio
        new_w = w
        new_h = w / max_ratio
    else:
        # No adjustment needed
        return (x, y, w, h)
    
    # Center the new bounding box
    new_x = x + (w - new_w) / 2
    new_y = y + (h - new_h) / 2
    
    # Ensure the bounding box stays within the image boundaries
    if new_x < 0:
        new_x = 0
    elif new_x + new_w > width:
        new_x = width - new_w
    
    if new_y < 0:
        new_y = 0
    elif new_y + new_h > height:
        new_y = height - new_h
    
    return (int(new_x), int(new_y), int(new_w), int(new_h))

def get_image_square_patch_rescaled(image, target_bbox, padding):
    # Assuming `image`, `target_bbox`, and `adjust_ratio` are defined elsewhere in your code
    width, height = image.size
    new_x, new_y, new_w, new_h = adjust_ratio(image, target_bbox, 0.5, 2)
    
    # Ensure the bounding box dimensions are at least min_size
    side_length = max(new_w + padding * 2, new_h + padding * 2)

    # Adjust the top-left corner of the bounding box to fit within the image
    square_x = max(0, new_x + new_w // 2 - side_length // 2)
    square_y = max(0, new_y + new_h // 2 - side_length // 2)

    # Ensure the square does not go out of the right edge
    if square_x + side_length > width:
        square_x = width - side_length

    # Ensure the square does not go out of the bottom edge
    if square_y + side_length > height:
        square_y = height - side_length

    # If the side length is larger than the image dimensions, adjust it
    side_length = min(side_length, width, height)

    # Adjust side_length to be the nearest number divisible by 64
    numbers = [64, 128, 256, 512]
    side_length = min(numbers, key=lambda x: abs(x - side_length))
    #side_length =  (side_length + 64 - 1) // 64 * 64

    # Ensure the square does not go out of the right edge after adjustment
    if square_x + side_length > width:
        square_x = width - side_length

    # Ensure the square does not go out of the bottom edge after adjustment
    if square_y + side_length > height:
        square_y = height - side_length

    # Define the patch
    patch_coords = (square_x, square_y, square_x + side_length, square_y + side_length)

    # Ensure patch coordinates are valid
    patch_coords = (max(0, patch_coords[0]), max(0, patch_coords[1]), min(width, patch_coords[2]), min(height, patch_coords[3]))

    # Crop the image
    image_patch = image.crop(patch_coords)

    # upscale patch
    patch_size, _ = image_patch.size
    n_upscale = 1024/patch_size
    if n_upscale == 8:
        image_patch = api_upscale_image_gradio(image_patch, scale_factor=2)
        image_patch = api_upscale_image_gradio(image_patch, scale_factor=4)
    elif n_upscale == 4:
        image_patch = api_upscale_image_gradio(image_patch, scale_factor=4)
    elif n_upscale == 2:
        image_patch = api_upscale_image_gradio(image_patch, scale_factor=2)

    # Create the mask
    mask = Image.new('L', (1024, 1024), 0)
    draw = ImageDraw.Draw(mask)
    bbox_in_mask = (
        max(0, new_x - square_x)*n_upscale,
        max(0, new_y - square_y)*n_upscale,
        min(side_length, new_x - square_x + new_w)*n_upscale,
        min(side_length, new_y - square_y + new_h)*n_upscale
    )
    draw.rectangle(bbox_in_mask, outline=255, fill=255)

    return image_patch, mask, patch_coords, bbox_in_mask

#%%
### List to store the names of all image files
IMAGE_NAMES = []
IMAGE_SIZES = []

# Walk through the directory
for root, dirs, files in os.walk(PATH_TO_IMAGES):
    for file in files:
       if file.endswith(".jpg"):
            # Append the file name to the list
            IMAGE_NAMES.append(file)

            image_path = os.path.join(root, file)
            
            # Open the image and get its size
            with Image.open(image_path) as img:
                IMAGE_SIZES.append(img.size)

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

def resize_bbox(old_bbox, old_size, new_size):
    """
    Resizes the bounding box according to the new image size.

    :param old_bbox: Tuple of (x, y, w, h) representing the original bounding box.
    :param old_size: Tuple of (width, height) representing the original image size.
    :param new_size: Tuple of (width, height) representing the new image size.
    :return: Tuple of (x, y, w, h) representing the resized bounding box.
    """
    x, y, w, h = old_bbox
    old_width, old_height = old_size
    new_width, new_height = new_size

    # Calculate scaling factors for width and height
    scale_x = new_width / old_width
    scale_y = new_height / old_height

    # Resize the bounding box
    new_x = x * scale_x
    new_y = y * scale_y
    new_w = w * scale_x
    new_h = h * scale_y

    new_bbox = (int(new_x), int(new_y), int(new_w), int(new_h))

    return new_bbox
#%%
# Define complete dataset:
'''
{
    Key: [original_name] {
        Key: [data]
            Key: [scene]
            Key: [target]
            Key: [swapped_object]
            Key: [target_bbox]
            Key: [rel_level]
            Key: [rel_score]
            Key: [excluded]
        Key: [fixations]
            Key: [name], Type of Value: [str]
            Key: [subject], Type of Value: [int]
            Key: [condition], Type of Value: [str]
            Key: [X], Type of Value: [list]
            Key: [Y], Type of Value: [list]
            Key: [T], Type of Value: [list]
            Key: [length], Type of Value: [int]
            Key: [split], Type of Value: [str]
            Key: [fixOnTarget], Type of Value: [bool]
            Key: [correct], Type of Value: [int]
        Key: [captions_val2017_annotations] or [captions_train2017_annotations]
            Key: [image_id], Type of Value: [int]
            Key: [id], Type of Value: [int]
            Key: [caption], Type of Value: [str]
        Key: [instances_val2017_annotations] or [instances_train2017_annotations]
            Key: [segmentation], Type of Value: [list]
            Key: [area], Type of Value: [float]
            Key: [iscrowd], Type of Value: [int]
            Key: [image_id], Type of Value: [int]
            Key: [bbox], Type of Value: [list]
            Key: [category_id], Type of Value: [int]
            Key: [id], Type of Value: [int]
    }

    Key: [name] {
        Key: [data]
            Key: [scene]
            Key: [target]
            Key: [swapped_object]
            Key: [target_bbox]
            Key: [rel_level]
            Key: [rel_score]
            Key: [excluded]
    }
    
}
'''
final_dataset = {}
coco_objects_list = []
for target in coco_object_cat:
    coco_objects_list.append(target['name'])
coco_objects_list = sorted(coco_objects_list, key=len, reverse=True)
print(coco_objects_list)
# Get the masked image with target and scene category
for i, image_name in enumerate(IMAGE_NAMES[:]):
    final_dataset[image_name] = {}
    
    img_data = image_name.replace('.jpg','')
    image_number = image_name.split('_')[0]
    img_data = img_data.replace(image_number, '')
    image_number = image_number + '.jpg'

    if re.search('original', image_name):
        final_dataset[image_name].update(data[image_number])

    # get scene remove scene
    for scene in sorted(sun_scene_to_keep, key=len, reverse=True):
        if re.search(scene.replace('/','_'), img_data):
            scene_name = scene
            img_data = img_data.replace(scene.replace('/','_'), '',1)
            break
        
    # get target remove target
    for target in coco_objects_list:
        target_coco_name = target
        target_name = target.replace('/','_').replace(' ','_')
        restricted_name = '_'.join(img_data.lstrip('_').split('_')[:2])
        if target_name == restricted_name or target_name == restricted_name.split('_')[0]:
            img_data = img_data.replace(target_name, '', 1)
            break

    if not re.search('original', image_name):
        swapped_object, rel_level = img_data.lstrip('_').split('_relscore_')
    else:
        rel_level = img_data.lstrip('_').split('_relscore_')[-1]

    print(scene_name)
    print(target_name)
    print(swapped_object)
    print(rel_level)

    scene_vect = scenes2vec[scene_name]
    if re.search('original', image_name):
        object_vect = things2vec[map_coco2things[target_coco_name]]
    else:
        object_vect = things2vec[swapped_object.replace('_',' ')]

    semantic_relatedness_score = cosine_similarity(scene_vect, object_vect)
    print(semantic_relatedness_score)
    '''
    # get bbox info 
    target, image_picture, image_picture_w_bbox, target_bbox, cropped_target_only_image, object_mask = get_coco_image_data(data, image_number)
    # remove the object before background
    image_clean = remove_object(image_picture, object_mask)
    image, mask, new_bbox = get_square_image(image_clean, target_bbox)
    old_size = image.size
    final_size = IMAGE_SIZES[i] 
    final_bbox = resize_bbox(new_bbox, old_size, final_size)

    print(target_bbox)
    print(final_bbox)
    '''
    # Check With LLAVA if the object is present
    prompt_llava = f"[INST] <image>\n Is there {art} \"{object_for_replacement.replace('/',' ').replace('_',' ')}\" in the image? {full_output_clean}. Answer only with \"Yes\" or \"No\". [/INST]"
    inputs_llava = llava_processor(prompt_llava, dict_out[0], return_tensors="pt").to(LLAVA_DEVICE)
    output_llava = llava_model.generate(**inputs_llava, max_new_tokens=1)
    full_output_llava = llava_processor.decode(output_llava[0], skip_special_tokens=True)
    print(full_output_llava)

    if "Yes" in full_output_llava[-5:]:
        regenerate = False
        generated_object_counter += 1
    elif scale == 30:
        regenerate = False
    else:
        scale += 7.5





# match the info of scene, objec, bbox 


# unify everything in a single dataset


