import os
import cv2
from skimage.transform import hough_circle
from constants import *


def perform_hough_transform(picture_file):
    image = cv2.imread(picture_file, cv2.IMREAD_GRAYSCALE)
    hough_space = hough_circle(image, CIRCLE_RADIUS, normalize=False)
    hough_space.resize(IMAGE_SIZE)
    cv2.imwrite("hough/" + picture_file.split("/")[-1], hough_space)


if __name__ == '__main__':
    for folder, subs, files in os.walk("pictures/"):
        for picture in files:
            perform_hough_transform(os.path.abspath(os.path.join(folder, picture)))
