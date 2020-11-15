from imutils import paths
import requests
import cv2
import os


RESIZE_DIMENSION = (400, 400)


def download_from_file(file_path, output_path):
    url_file = open(file_path, "r")
    total = 0
    rows = url_file.read().strip().split("\n")

    for url in rows:
        try:
            request = requests.get(url, timeout=60)

            output_file_path = os.path.join(output_path, "{}.jpg".format(str(total).zfill(3)))
            output_file = open(output_file_path, "wb")
            output_file.write(request.content)
            output_file.close()

            print("[INFO] downloaded: {}".format(output_file_path))
            total += 1
        except:
            print("[INFO] error downloading {}...skipping".format(url))

    url_file.close()


def check_and_resize(output_path):
    for image_path in paths.list_images(output_path):
        delete = False

        try:
            image = cv2.imread(image_path)

            if image is None:
                delete = True
            else:
                image = cv2.resize(image, RESIZE_DIMENSION)
                cv2.imwrite(image_path, image)
        except:
            print("Error with opening file {}".format(image_path))
            delete = True

        if delete:
            print("[INFO] deleting {}".format(image_path))
            os.remove(image_path)


if __name__ == '__main__':
    path = "./urls/"
    output_path = "./pictures/"
    url_file_names = [file_name for file_name in os.listdir(path)]

    for url_file_name in url_file_names:
        file_path = os.path.join(path, url_file_name)
        output_path = os.path.join(output_path, url_file_name.split(".")[0])
        download_from_file(file_path, output_path)
        check_and_resize(output_path)
