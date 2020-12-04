import logging
import time
from statistics import mean

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import Callback

from .model import *


class TimeHistory(Callback):
    def on_train_begin(self, logs=None):
        self.times = []

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        self.times.append(time.time() - self.epoch_time_start)


if __name__ == '__main__':
    matplotlib.use("Agg")
    logging.getLogger("tensorflow").setLevel(logging.CRITICAL)

    INIT_LR = 1e-2
    BATCH_SIZE = 128
    NUM_EPOCHS = 60

    labelNames = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    print("[INFO] loading CIFAR-10 dataset...")
    ((train_x, train_y), (test_x, test_y)) = cifar10.load_data()

    # scale the data to the range [0, 1]
    train_x = train_x.astype("float32") / 255.0
    test_x = test_x.astype("float32") / 255.0

    # convert the labels from integers to vectors
    label_binarizer = LabelBinarizer()
    train_y = label_binarizer.fit_transform(train_y)
    test_y = label_binarizer.fit_transform(test_y)

    # construct the image generator for data augmentation
    augmenter = ImageDataGenerator(rotation_range=18, zoom_range=0.15,
                                   width_shift_range=0.2, height_shift_range=-.2, shear_range=0.15,
                                   horizontal_flip=True, fill_mode="nearest")

    model = mini_inception_net_functional(32, 32, 3, len(labelNames))

    # initialize the optimizer and compile the model
    optimizer = SGD(lr=INIT_LR, momentum=0.9, decay=INIT_LR / NUM_EPOCHS)
    print("[INFO] training network...")
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    model.summary()

    # train the network
    time_callback = TimeHistory()
    history = model.fit_generator(
        augmenter.flow(train_x, train_y, batch_size=BATCH_SIZE),
        validation_data=(test_x, test_y),
        steps_per_epoch=train_x.shape[0] // BATCH_SIZE,
        epochs=NUM_EPOCHS,
        verbose=1,
        callbacks=[time_callback]
    )

    # save model
    model.save("miniinception_1")

    # evaluate the network
    print("[INFO] evaluating network...")
    predictions = model.predict(test_x, batch_size=BATCH_SIZE)
    print(classification_report(test_y.argmax(axis=1),
                                predictions.argmax(axis=1),
                                target_names=labelNames))

    train_losses = history.history["loss"]
    val_losses = history.history["val_loss"]
    train_accuracies = history.history["accuracy"]
    val_accuracies = history.history["val_accuracy"]
    zipped_results = zip(train_losses, train_accuracies, val_losses, val_accuracies)

    output_accuracy = open("accuracy_1.txt", "w")
    for index, train_loss, train_accuracy, val_loss, val_accuracy in enumerate(zipped_results):
        output_accuracy.write("{} {} {} {} {}\n".format(index + 1, train_loss, train_accuracy, val_loss, val_accuracy))
    output_accuracy.close()

    # determine the number of epochs and then construct the plot title
    N = np.arange(0, NUM_EPOCHS)
    title = "Loss/Accuracy bez modyfikacji sieci"

    # plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(N, history.history["loss"], label="Loss (zbiór uczący)")
    plt.plot(N, history.history["val_loss"], label="Loss (zbiór walidacyjny)")
    plt.plot(N, history.history["accuracy"], label="Accuracy (zbiór uczący)")
    plt.plot(N, history.history["val_accuracy"], label="Accuracy (zbiór walidacyjny)")
    plt.title(title)
    plt.xlabel("Epoka")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.savefig("miniinception_1.png")

    average_epoch_time = mean(time_callback.times)
    print("Average epoch training time: {} s".format(average_epoch_time))
