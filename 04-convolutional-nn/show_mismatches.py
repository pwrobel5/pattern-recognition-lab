import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from constants import *

SHOWN_MISMATCHES = 3


def show_mismatches(data_directory_name, model_name):
    model = tf.keras.models.load_model(model_name)
    data_directory = os.path.join(DATA_DIRECTORY, data_directory_name)
    train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        data_directory,
        validation_split=VALIDATION_SPLIT,
        subset="training",
        seed=SPLIT_SEED,
        image_size=PICTURE_DIMENSION,
        label_mode="categorical",
        batch_size=BATCH_SIZE
    )
    validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        data_directory,
        validation_split=VALIDATION_SPLIT,
        subset="validation",
        seed=SPLIT_SEED,
        image_size=PICTURE_DIMENSION,
        label_mode="categorical",
        batch_size=BATCH_SIZE
    )

    image_batch, label_batch = validation_dataset.as_numpy_iterator().next()
    predictions = model.predict_on_batch(image_batch)

    predictions = [np.where(pred_element == max(pred_element))[0][0] for pred_element in predictions]
    labels = [np.where(label_element == 1)[0][0] for label_element in label_batch]
    class_names = train_dataset.class_names

    mismatched = []
    for index, (predicted, real) in enumerate(zip(predictions, labels)):
        if predicted != real:
            mismatched.append(index)

    plt.figure(figsize=(10, 10))
    for i in range(SHOWN_MISMATCHES):
        ax = plt.subplot(1, SHOWN_MISMATCHES, i + 1)
        index = mismatched[i]
        plt.imshow(image_batch[index].astype("uint8"))
        plt.title("{} ({})".format(class_names[predictions[index]], class_names[labels[index]]))
        plt.axis("off")

    plt.savefig("mismatched/{}.png".format(data_directory_name))


if __name__ == '__main__':
    show_mismatches("3_class", "models/50_0")
    show_mismatches("4_class", "models/50_0_4_class")
