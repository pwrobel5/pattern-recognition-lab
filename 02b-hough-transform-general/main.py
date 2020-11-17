import collections
import math

import cv2
import numpy as np
from scipy.signal import argrelextrema

CANNY_LOW_THRESHOLD = 100
CANNY_HIGH_THRESHOLD = 250
BLUR_KERNEL = 21, 21
BLUR_SIGMAX = 1
DILATION_KERNEL = (11, 11)
EROSION_KERNEL = (8, 8)
MORPHOLOGY_ITERATIONS = 1
THRESHOLD = 75
THRESHOLD_MAX_VALUE = 255
MINIMAL_ANGLE = -180
MAXIMAL_ANGLE = 180
ANGLE_STEP = 10
COORDINATE_STEP = 10
RESIZE_FACTOR = 8
VOTE_THRESHOLD = 160
CIRCLE_RADIUS = 3
CIRCLE_COLOR = (255, 0, 0)
CIRCLE_THICKNESS = -1
CONTOUR_INDEX = 0
CONTOUR_COLOR = (0, 255, 0)
CONTOUR_THICKNESS = 1


def preprocess_edges(base_image_name):
    base_image = cv2.imread(base_image_name)
    size_y, size_x, _ = base_image.shape
    resized_y, resized_x = int(size_y / RESIZE_FACTOR), int(size_x / RESIZE_FACTOR)
    base_image = cv2.resize(base_image, (resized_x, resized_y))
    blurred = cv2.GaussianBlur(base_image, BLUR_KERNEL, BLUR_SIGMAX)  # blur to have smoother edges
    edges = cv2.Canny(blurred, CANNY_LOW_THRESHOLD, CANNY_HIGH_THRESHOLD)

    # some morphology transformations to have continuous edges
    dilated = cv2.dilate(edges, DILATION_KERNEL, iterations=MORPHOLOGY_ITERATIONS)
    erosion = cv2.erode(dilated, EROSION_KERNEL, iterations=MORPHOLOGY_ITERATIONS)
    erosion = cv2.GaussianBlur(erosion, BLUR_KERNEL, BLUR_SIGMAX)
    erosion = cv2.threshold(erosion, THRESHOLD, THRESHOLD_MAX_VALUE, cv2.THRESH_BINARY)[1]

    cv2.imwrite("eroded_{}".format(base_image_name), erosion)

    return erosion


def get_biggest_contour(edges):
    contours = cv2.findContours(edges.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)[-2]

    # if the threshold was too small it could be more than one detected contours (however they are nearly identical)
    # then we want only the biggest one
    if len(contours) > 1:
        contours = list(sorted(contours, key=lambda contour: cv2.contourArea(contour), reverse=True))

    return contours[0]


def find_centroid(biggest_contour):
    moments = cv2.moments(biggest_contour)

    centroid_x = int(moments["m10"] / moments["m00"])
    centroid_y = int(moments["m01"] / moments["m00"])

    return centroid_x, centroid_y


def get_gradients(edges):
    sobel_x = cv2.Sobel(edges, cv2.CV_64F, 1, 0, ksize=5)
    sobel_y = cv2.Sobel(edges, cv2.CV_64F, 0, 1, ksize=5)
    return sobel_x, sobel_y


def get_gradient_angle(sobel, x, y):
    sobel_x, sobel_y = sobel
    gradient_x = sobel_x[y][x]
    gradient_y = sobel_y[y][x]
    gradient_angle = round(math.atan2(gradient_y, gradient_x) * 180 / math.pi)

    return gradient_angle


def fill_r_table(edges, centroid_coordinates, biggest_contour):
    centroid_x, centroid_y = centroid_coordinates
    sobel_x, sobel_y = get_gradients(edges)
    r_table = collections.defaultdict(list)

    for i in biggest_contour:
        x = i[0][0]
        y = i[0][1]
        gradient_angle = get_gradient_angle((sobel_x, sobel_y), x, y)
        displacement_vector = (centroid_x - x, centroid_y - y)
        r_table[gradient_angle].append(displacement_vector)

    return r_table


def get_r_table_and_base_contour(base_image_name):
    edges = preprocess_edges(base_image_name)
    biggest_contour = get_biggest_contour(edges)
    centroid_x, centroid_y = find_centroid(biggest_contour)
    r_table = fill_r_table(edges, (centroid_x, centroid_y), biggest_contour)
    base_contour = biggest_contour + np.array([-centroid_x, -centroid_y])

    return r_table, base_contour


def perform_hough_transform(edges, r_table):
    sobel_x, sobel_y = get_gradients(edges)
    edges_y, edges_x = edges.shape
    max_y = round(edges_y / COORDINATE_STEP)
    max_x = round(edges_x / COORDINATE_STEP)
    max_angle = round((MAXIMAL_ANGLE - MINIMAL_ANGLE) / ANGLE_STEP) + 1
    voting_space_dimension = (max_y, max_x, max_angle)

    voting_space = np.zeros(voting_space_dimension, dtype=np.uint8)

    for y in range(edges_y):
        for x in range(edges_x):
            if edges[y][x] > 0:
                gradient_angle = get_gradient_angle((sobel_x, sobel_y), x, y)
                displacements = r_table[gradient_angle]

                for vector in displacements:
                    x_displacement, y_displacement = vector

                    for angle in range(MINIMAL_ANGLE, MAXIMAL_ANGLE + 1, ANGLE_STEP):
                        radian_angle = math.radians(angle)

                        x_centroid = round(
                            (x + x_displacement * math.cos(radian_angle) - y_displacement * math.sin(
                                radian_angle)) / COORDINATE_STEP)

                        y_centroid = round(
                            (y + x_displacement * math.sin(radian_angle) + y_displacement * math.cos(
                                radian_angle)) / COORDINATE_STEP)

                        if 0 <= x_centroid < max_x and 0 <= y_centroid < max_y:
                            angle_index = round((angle - MINIMAL_ANGLE) / ANGLE_STEP)

                            directions = (-1, 1)
                            modifiers = [(0, 0, 0)] + \
                                        [(i, 0, 0) for i in directions] + \
                                        [(0, j, 0) for j in directions] + \
                                        [(0, 0, k) for k in directions]

                            for modifier in modifiers:
                                x_modifier, y_modifier, angle_modifier = modifier
                                x_mark = x_centroid + x_modifier
                                y_mark = y_centroid + y_modifier
                                angle_mark = angle_index + angle_modifier

                                if 0 <= x_mark < max_x and 0 <= y_mark < max_y and 0 <= angle_mark < max_angle:
                                    voting_space[y_mark][x_mark][angle_mark] += 1

    return voting_space


def cartesian_to_polar(x, y):
    theta = np.arctan2(y, x)
    rho = np.hypot(x, y)
    return theta, rho


def polar_to_cartesian(theta, rho):
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    return x, y


def rotate_contour(contour, angle):
    result = contour.copy()
    coordinates = contour[:, 0, :]
    xs, ys = coordinates[:, 0], coordinates[:, 1]
    thetas, rhos = cartesian_to_polar(xs, ys)

    thetas = np.rad2deg(thetas)
    thetas = (thetas + angle) % 360
    thetas = np.deg2rad(thetas)

    xs, ys = polar_to_cartesian(thetas, rhos)

    result[:, 0, 0] = xs
    result[:, 0, 1] = ys
    result = result.astype(np.int32)

    return result


def main():
    r_table, base_contour = get_r_table_and_base_contour("base.jpg")
    image_edges = preprocess_edges("picture.jpg")
    voting_space = perform_hough_transform(image_edges, r_table)

    print("Voting space ready")

    peaks_x = np.array(argrelextrema(voting_space, np.greater, axis=0))
    peaks_y = np.array(argrelextrema(voting_space, np.greater, axis=1))
    peaks_angle = np.array(argrelextrema(voting_space, np.greater, axis=2))

    stacked = np.vstack((peaks_x.transpose(), peaks_y.transpose(), peaks_angle.transpose()))

    elements, counts = np.unique(stacked, axis=0, return_counts=True)
    coords = elements[np.where(counts == 3)[0]]

    values = voting_space[coords[:, 0], coords[:, 1], coords[:, 2]]
    visualization = cv2.cvtColor(image_edges, cv2.COLOR_GRAY2RGB)
    for coord_element, value in zip(coords, values):
        if value >= VOTE_THRESHOLD:
            y, x, angle = coord_element
            angle = angle * ANGLE_STEP + MINIMAL_ANGLE
            y = y * COORDINATE_STEP
            x = x * COORDINATE_STEP

            cv2.circle(visualization, (x, y), CIRCLE_RADIUS, CIRCLE_COLOR, CIRCLE_THICKNESS)
            visualization[y][x][0] = value

            contour = rotate_contour(base_contour, angle)
            contour = contour + np.array([x, y])
            cv2.drawContours(visualization, [contour], CONTOUR_INDEX, CONTOUR_COLOR, CONTOUR_THICKNESS)

    cv2.imwrite("visualization.png", visualization)


if __name__ == '__main__':
    main()
