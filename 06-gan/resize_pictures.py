import os

import cv2

PICTURE_DIMENSION = (32, 32)

if __name__ == '__main__':
    image_path = "./pictures/raven/"
    image_file_names = [os.path.join(image_path, file_name) for file_name in os.listdir(image_path)]

    for image_path in image_file_names:
        image = cv2.imread(image_path)
        image = cv2.resize(image, PICTURE_DIMENSION)
        cv2.imwrite(image_path, image)
