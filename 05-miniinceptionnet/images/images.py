import random

import tensorflow.keras.datasets as datasets


def get_images(n):
    random.seed(100)
    (train_images, train_labels), _ = datasets.cifar10.load_data()
    class_names = ["airplane", "automobile", "bird", "cat", "deer",
                   "dog", "frog", "horse", "ship", "truck"]

    chosen_images = []
    chosen_labels = []

    for i in range(0, n):
        index = random.randrange(0, len(train_images))
        chosen_image = train_images[index]
        chosen_images.append(chosen_image)
        chosen_labels.append(class_names[train_labels[index][0]])

    return chosen_images, chosen_labels
