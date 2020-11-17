import collections
import math

import cv2

CANNY_LOW_THRESHOLD = 100
CANNY_HIGH_THRESHOLD = 250
BLUR_KERNEL = 21, 21
BLUR_SIGMAX = 1
DILATION_KERNEL = (11, 11)
EROSION_KERNEL = (8, 8)
MORPHOLOGY_ITERATIONS = 1
THRESHOLD = 75
THRESHOLD_MAX_VALUE = 255


def preprocess_edges(base_image_name):
    base_image = cv2.imread(base_image_name)
    blurred = cv2.GaussianBlur(base_image, BLUR_KERNEL, BLUR_SIGMAX)  # blur to have smoother edges
    edges = cv2.Canny(blurred, CANNY_LOW_THRESHOLD, CANNY_HIGH_THRESHOLD)

    # some morphology transformations to have continuous edge
    dilated = cv2.dilate(edges, DILATION_KERNEL, iterations=MORPHOLOGY_ITERATIONS)
    erosion = cv2.erode(dilated, EROSION_KERNEL, iterations=MORPHOLOGY_ITERATIONS)
    erosion = cv2.GaussianBlur(erosion, BLUR_KERNEL, BLUR_SIGMAX)
    erosion = cv2.threshold(erosion, THRESHOLD, THRESHOLD_MAX_VALUE, cv2.THRESH_BINARY)[1]

    return erosion


def get_biggest_contour(edges):
    contours = cv2.findContours(edges.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)[-2]

    # if the threshold was too small it could be more than one detected contours (however they are nearly identical)
    # than we want only the biggest one
    if len(contours) > 1:
        contours = list(sorted(contours, key=lambda contour: cv2.contourArea(contour), reverse=True))

    return contours[0]


def find_centroid(biggest_contour):
    moments = cv2.moments(biggest_contour)

    centroid_x = int(moments["m10"] / moments["m00"])
    centroid_y = int(moments["m01"] / moments["m00"])

    return centroid_x, centroid_y


def fill_r_table(edges, centroid_coordinates, biggest_contour):
    centroid_x, centroid_y = centroid_coordinates

    sobel_x = cv2.Sobel(edges, cv2.CV_64F, 1, 0, ksize=5)
    sobel_y = cv2.Sobel(edges, cv2.CV_64F, 0, 1, ksize=5)

    r_table = collections.defaultdict(list)

    for i in biggest_contour:
        x = i[0][0]
        y = i[0][1]
        gradient_x = sobel_x[y][x]
        gradient_y = sobel_y[y][x]
        gradient_angle = round(math.atan2(gradient_y, gradient_x) * 180 / math.pi)
        displacement_vector = (centroid_x - x, centroid_y - y)
        r_table[gradient_angle].append(displacement_vector)

    return r_table


def get_r_table(base_image_name):
    edges = preprocess_edges(base_image_name)
    biggest_contour = get_biggest_contour(edges)
    centroid_x, centroid_y = find_centroid(biggest_contour)
    r_table = fill_r_table(edges, (centroid_x, centroid_y), biggest_contour)
    return r_table


def main():
    r_table = get_r_table("base.jpg")


if __name__ == '__main__':
    main()
