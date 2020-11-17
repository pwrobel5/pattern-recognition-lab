import os

import tensorflow as tf

from constants import *

EPOCHS_ITERATIONS = ((10, 4), (50, 1))


def train_nn(initial_epochs, iteration):
    data_directory = os.path.join(DATA_DIRECTORY, "3_class")
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

    AUTOTUNE = tf.data.experimental.AUTOTUNE
    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)

    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
    IMAGE_SHAPE = PICTURE_DIMENSION + (3,)
    base_model = tf.keras.applications.MobileNetV2(input_shape=IMAGE_SHAPE,
                                                   include_top=False,
                                                   weights="imagenet")

    base_model.trainable = False
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    prediction_layer_out = tf.keras.layers.Dense(3, activation="softmax")

    inputs = tf.keras.Input(shape=IMAGE_SHAPE)
    x = preprocess_input(inputs)
    x = base_model(x, training=False)
    x = global_average_layer(x)
    outputs = prediction_layer_out(x)
    model = tf.keras.Model(inputs, outputs)
    base_learning_rate = 0.0001
    model.compile(optimizer=tf.keras.optimizers.SGD(lr=base_learning_rate),
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=["accuracy"])

    output_accuracy_name = "results/accuracy_{}_{}.txt".format(initial_epochs, iteration)
    output_accuracy = open(output_accuracy_name, "w")

    loss_initial, accuracy_initial = model.evaluate(validation_dataset)
    output_accuracy.write("{} {}\n".format(0, accuracy_initial))

    history = model.fit(train_dataset,
                        epochs=initial_epochs,
                        validation_data=validation_dataset)

    model_output_name = "models/{}_{}".format(initial_epochs, iteration)
    model.save(model_output_name)

    accuracy = history.history["val_accuracy"]
    for index, value in enumerate(accuracy):
        output_accuracy.write("{} {}\n".format(index + 1, value))

    output_accuracy.close()


if __name__ == '__main__':
    for epochs_number, iterations in EPOCHS_ITERATIONS:
        for iteration_index in range(0, iterations):
            train_nn(epochs_number, iteration_index)
