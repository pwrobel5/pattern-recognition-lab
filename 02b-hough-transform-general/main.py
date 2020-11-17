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


def find_centroid(base_file_name):
    image = cv2.imread(base_file_name)
    blurred = cv2.GaussianBlur(image, BLUR_KERNEL, BLUR_SIGMAX)  # blur to have smoother edges

    edges = cv2.Canny(blurred, CANNY_LOW_THRESHOLD, CANNY_HIGH_THRESHOLD)

    # some morphology transformations to have continuous edge
    dilated = cv2.dilate(edges, DILATION_KERNEL, iterations=MORPHOLOGY_ITERATIONS)
    erosion = cv2.erode(dilated, EROSION_KERNEL, iterations=MORPHOLOGY_ITERATIONS)
    erosion = cv2.GaussianBlur(erosion, BLUR_KERNEL, BLUR_SIGMAX)
    erosion = cv2.threshold(erosion, THRESHOLD, THRESHOLD_MAX_VALUE, cv2.THRESH_BINARY)[1]
    contours = cv2.findContours(erosion.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)[-2]

    # if the threshold was too small it could be more than one detected contours (however they are nearly identical)
    # than we want only the biggest one
    if len(contours) > 1:
        contours = list(sorted(contours, key=lambda contour: cv2.contourArea(contour), reverse=True))

    biggest_contour = contours[0]
    moments = cv2.moments(biggest_contour)

    centroid_x = int(moments["m10"] / moments["m00"])
    centroid_y = int(moments["m01"] / moments["m00"])

    return centroid_x, centroid_y


if __name__ == '__main__':
    find_centroid("base.jpg")
