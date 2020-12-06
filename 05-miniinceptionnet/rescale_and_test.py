import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow.keras.models as models

from images.images import get_images

LABEL_NAMES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


if __name__ == '__main__':
    model_name = "miniinception_5"

    chosen_images, chosen_labels = get_images(3)
    resized_images = []
    for image in chosen_images:
        resized_images.append(cv2.resize(image, (36, 28)))

    for i in range(0, 3):
        plt.figure()
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(resized_images[i])
        plt.xlabel(chosen_labels[i])
        plt.savefig("cifar-{}.png".format(i))

    model = models.load_model(model_name)
    for image in resized_images:
        image = np.expand_dims(image, axis=0)
        pred_vec = model.predict(image, batch_size=1)
        prediction = np.argmax(pred_vec)
        print(LABEL_NAMES[prediction])
