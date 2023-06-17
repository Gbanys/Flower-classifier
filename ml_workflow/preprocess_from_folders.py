import numpy as np
from tensorflow import keras

import keras.utils as image
import tensorflow as tf
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


def augment_image_data(images, labels):

    data_augmentation = tf.keras.Sequential([
        keras.layers.RandomFlip("horizontal_and_vertical"),
        keras.layers.RandomRotation(0.2),
    ])

    new_images = []
    new_labels = []
    counter = 0
    for image, label in zip(images, labels):
        counter += 1
        print(counter)
        for i in range(0, 3):
            img = data_augmentation(image)
            new_images.append(img)
            new_labels.append(label)
    new_images = np.array(new_images)
    new_labels = np.array(new_labels)

    return new_images, new_labels