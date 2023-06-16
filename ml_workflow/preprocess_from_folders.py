import numpy as np

import keras.utils as image
import glob

def create_labels():
    unique_flowers = [flower_dir.split("/")[2] for flower_dir in glob.glob("image_storage/flowers/*")]
    labels = []
    for flower_dir in glob.glob("image_storage/flowers/*"):
        for image_file_dir in glob.glob(f'{flower_dir}/*'):
            flower = image_file_dir.split("/")[2]
            labels.append(unique_flowers.index(flower))
    labels = np.array(labels)
    return labels


def extract_images(): 
    images = []
    for flower_dir in glob.glob("image_storage/flowers/*"):
        for image_file_dir in glob.glob(f'{flower_dir}/*'):
            img = image.load_img(image_file_dir, target_size=(60, 60))
            img_list = image.img_to_array(img)
            images.append(img_list)
    images = np.array(images)
    return images