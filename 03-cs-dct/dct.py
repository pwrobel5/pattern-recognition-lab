import os
import cv2
import numpy as np
import math

FREQUENCY_BORDER = 0.3
MIDDLE_BORDER = 0.2


def read_image(folder_path, file_name):
    image_path = os.path.join(folder_path, file_name)
    return cv2.imread(image_path)


def preprocess_image(image):
    shape = image.shape
    # shape has reversed order than that needed for resize
    new_shape = (int(shape[1] / 10 * 1.5), int(shape[0] / 10 * 1.5))
    image = cv2.resize(image, new_shape)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return np.float32(image)


def get_image_for_analysis(folder_path, file_name):
    image = read_image(folder_path, file_name)
    return preprocess_image(image)


def leave_only_low_frequencies(dct):
    shape = dct.shape
    border_row = int(shape[0] * FREQUENCY_BORDER)
    border_column = int(shape[1] * FREQUENCY_BORDER)
    for i in range(0, shape[0]):
        for j in range(0, shape[1]):
            if i >= border_row or j >= border_column:
                dct[i][j] = 0.0


def remove_low_frequencies(dct):
    shape = dct.shape
    border_row = int(shape[0] * FREQUENCY_BORDER)
    border_column = int(shape[1] * FREQUENCY_BORDER)
    for i in range(0, border_row):
        for j in range(0, border_column):
            dct[i][j] = 0.0


def decrease_by_distance(dct):
    shape = dct.shape

    def distance(x, y):
        return math.sqrt(x ** 2 + y ** 2)

    for i in range(0, shape[0]):
        for j in range(0, shape[1]):
            if i != 0 or j != 0:
                dct[i][j] = dct[i][j] / distance(i, j)


def remove_middle_frequencies(dct):
    shape = dct.shape
    middle_row_begin = int(shape[0] / 2 - MIDDLE_BORDER * shape[0])
    middle_row_end = int(shape[0] / 2 + MIDDLE_BORDER * shape[0])
    middle_column_begin = int(shape[1] / 2 - MIDDLE_BORDER * shape[1])
    middle_column_end = int(shape[1] / 2 + MIDDLE_BORDER * shape[1])

    for i in range(middle_row_begin, middle_row_end):
        for j in range(middle_column_begin, middle_column_end):
            dct[i][j] = 0.0


def average_dcts(dct1, dct2):
    averaged = np.add(dct1, dct2) / 2
    cv2.imwrite("averaged.jpg", cv2.idct(averaged))
    cv2.imwrite("averaged_dct.jpg", averaged)


def manipulate_dct(dct, function, output_name, dct_output_name):
    function(dct)
    cv2.imwrite(output_name, cv2.idct(dct))
    cv2.imwrite(dct_output_name, np.abs(dct))


def main():
    dcts = {}
    for folder, _, files in os.walk("pictures/"):
        for picture in files:
            image = get_image_for_analysis(folder, picture)
            cv2.imwrite("preprocessed/" + picture, image)
            dct = cv2.dct(image)
            dcts[picture] = dct
            cv2.imwrite("dct/" + picture, np.abs(dct))

            manipulate_dct(dct.copy(), leave_only_low_frequencies, "low_frequencies/" + picture, "dct_low/" + picture)
            manipulate_dct(dct.copy(), remove_low_frequencies, "high_frequencies/" + picture, "dct_high/" + picture)
            manipulate_dct(dct.copy(), decrease_by_distance, "distance/" + picture, "dct_distance/" + picture)
            manipulate_dct(dct.copy(), remove_middle_frequencies, "removed_middle/" + picture, "dct_middle/" + picture)

    average_dcts(dcts["lenin.jpg"], dcts["szpiglasowy_wierch.jpg"])


if __name__ == '__main__':
    main()
