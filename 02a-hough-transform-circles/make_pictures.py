import numpy as np
import cv2
import random
from constants import *


def get_basic_picture():
    image = np.zeros(IMAGE_SIZE, np.uint8)

    for coords, color in zip(CIRCLE_COORDINATES, CIRCLE_COLORS):
        cv2.circle(image, coords, CIRCLE_RADIUS, color)

    return image


def save_basic_picture():
    image = get_basic_picture()
    cv2.imwrite("pictures/basic.png", image)


def save_differing_radius():
    random.seed(RANDOM_SEED)
    image = np.zeros(IMAGE_SIZE, np.uint8)

    for coords, color in zip(CIRCLE_COORDINATES, CIRCLE_COLORS):
        radius = CIRCLE_RADIUS + random.randrange(RANDOM_RANGE_MIN, RANDOM_RANGE_MAX)
        cv2.circle(image, coords, radius, color)

    cv2.imwrite("pictures/differing_radius.png", image)


def save_elliptic():
    image = np.zeros(IMAGE_SIZE, np.uint8)
    axes = (CIRCLE_RADIUS, CIRCLE_RADIUS + ELLIPTIC_DIFFER)

    for coords, color in zip(CIRCLE_COORDINATES, CIRCLE_COLORS):
        cv2.ellipse(image, coords, axes, ELLIPSE_ANGLE, ELLIPSE_START_ANGLE, ELLIPSE_END_ANGLE, color)

    cv2.imwrite("pictures/ellipses.png", image)


def save_blurred():
    image = get_basic_picture()
    image = cv2.GaussianBlur(image, GAUSSIAN_KERNEL_SIZE, GAUSSIAN_SIGMA_X)
    cv2.imwrite("pictures/blurred.png", image)


def save_additional_shapes():
    image = get_basic_picture()
    cv2.rectangle(image, RECTANGLE_START_POINT, RECTANGLE_END_POINT, RECTANGLE_COLOR)

    cv2.line(image, TRIANGLE_VERTEX_1, TRIANGLE_VERTEX_2, TRIANGLE_COLOR)
    cv2.line(image, TRIANGLE_VERTEX_2, TRIANGLE_VERTEX_3, TRIANGLE_COLOR)
    cv2.line(image, TRIANGLE_VERTEX_3, TRIANGLE_VERTEX_1, TRIANGLE_COLOR)

    cv2.imwrite("pictures/additional_shapes.png", image)


def save_crossing_circles():
    image = np.zeros(IMAGE_SIZE, np.uint8)

    for coords, color in zip(CROSSING_CIRCLE_COORDINATES, CIRCLE_COLORS):
        cv2.circle(image, coords, CIRCLE_RADIUS, color)

    cv2.imwrite("pictures/crossing_circles.png", image)


def save_salt_and_pepper():
    image = get_basic_picture()

    uniform_noise = np.zeros(IMAGE_SIZE, np.uint8)
    cv2.randu(uniform_noise, 0, 255)
    _, impulse_noise = cv2.threshold(uniform_noise, 250, 255, cv2.THRESH_BINARY)
    impulse_noise = (0.5 * impulse_noise).astype(np.uint8)
    image = cv2.add(image, impulse_noise)

    cv2.imwrite("pictures/salt_and_pepper.png", image)


def save_distorted():
    image = get_basic_picture()

    for _ in range(RESIZE_ITERATIONS):
        image = cv2.resize(image, SMALL_SIZE)
        image = cv2.resize(image, IMAGE_SIZE)

    cv2.imwrite("pictures/distorted.png", image)


if __name__ == '__main__':
    save_basic_picture()
    save_differing_radius()
    save_elliptic()
    save_blurred()
    save_additional_shapes()
    save_crossing_circles()
    save_salt_and_pepper()
    save_distorted()
