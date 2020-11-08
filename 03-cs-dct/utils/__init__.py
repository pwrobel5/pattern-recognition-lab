import os

import cv2
import numpy as np


def read_image(folder_path, file_name):
    image_path = os.path.join(folder_path, file_name)
    return cv2.imread(image_path)


def preprocess_image(image, division_factor, multiply_factor):
    shape = image.shape
    # shape has reversed order than that needed for resize
    new_shape = (int(shape[1] / division_factor * multiply_factor), int(shape[0] / division_factor * multiply_factor))
    image = cv2.resize(image, new_shape)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return np.float32(image)


def get_image_for_analysis(folder_path, file_name, division_factor=10, multiply_factor=1.5):
    image = read_image(folder_path, file_name)
    return preprocess_image(image, division_factor, multiply_factor)
