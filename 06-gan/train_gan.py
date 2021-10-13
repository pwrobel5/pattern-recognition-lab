import glob
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow.keras as keras
import tensorflow.keras.activations as activations
import tensorflow.keras.layers as layers
import tensorflow.keras.optimizers as optimizers

INPUT_SIZE = (32, 32)
GENERATOR_INPUT_SIZE = 100
BATCH_SIZE = 32
DISC_UPDATES = 1
GEN_UPDATES = 1
READ_MODEL = False

PROGRESS_INTERVAL = 2000
SAMPLES_DIR = "visualisation"


def add_encoder_block(x, channels, kernel_size=(4, 4), stride=2):
    x = layers.Conv2D(channels, kernel_size, strides=stride, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.3)(x)
    return x


def add_decoder_block(x, channels, kernel_size=(4, 4), stride=2):
    x = layers.Conv2DTranspose(channels, kernel_size, strides=stride, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.3)(x)
    return x


def build_discriminator():
    input_layer = layers.Input(shape=INPUT_SIZE + (3,))

    x = add_encoder_block(input_layer, 32)
    x = add_encoder_block(x, 64)
    x = add_encoder_block(x, 128)
    x = add_encoder_block(x, 128, kernel_size=(2, 2))

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inputs=input_layer, outputs=x)


def build_generator(generator_input_size):
    input_layer = layers.Input(shape=(generator_input_size,))
    x = layers.Dense(2 * 2 * 256, input_dim=generator_input_size)(input_layer)
    x = layers.Reshape(target_shape=(2, 2, 256))(x)

    x = add_decoder_block(x, 128, (4, 4), 2)
    x = add_decoder_block(x, 64, (4, 4), 2)
    x = add_decoder_block(x, 32, (4, 4), 2)

    x = layers.Conv2DTranspose(3, (2, 2), strides=2)(x)
    x = activations.tanh(x)

    return keras.Model(inputs=input_layer, outputs=x)


def construct_models(verbose=False):
    discriminator = build_discriminator()
    discriminator.compile(loss="binary_crossentropy",
                          optimizer=optimizers.Adam(lr=0.0002), metrics=["mae"])

    generator = build_generator(GENERATOR_INPUT_SIZE)

    gan = keras.Sequential()
    gan.add(generator)
    gan.add(discriminator)
    discriminator.trainable = False
    gan.compile(loss="binary_crossentropy",
                optimizer=optimizers.Adam(lr=0.0002), metrics=["mae"])
    if verbose:
        generator.summary()
        discriminator.summary()
        gan.summary()

    return discriminator, generator, gan


def run_training(pictures, discriminator, generator, gan, start_it=0, num_epochs=1000):
    avg_loss_discriminator = []
    avg_loss_generator = []
    total_it = start_it

    for epoch in range(num_epochs):
        loss_discriminator = []
        loss_generator = []
        for it in range(200):
            # Update discriminator
            images_real = pictures[np.random.randint(0, pictures.shape[0], size=BATCH_SIZE)]

            # Generate fake examples
            noise = np.random.randn(BATCH_SIZE, GENERATOR_INPUT_SIZE)
            images_fake = generator.predict(noise)

            d_loss_real = discriminator.train_on_batch(images_real, np.ones([BATCH_SIZE]))[1]
            d_loss_fake = discriminator.train_on_batch(images_fake, np.zeros([BATCH_SIZE]))[1]

            if total_it % PROGRESS_INTERVAL == 0:
                noise = np.random.randn(1, GENERATOR_INPUT_SIZE)
                fake_image = generator.predict(noise)[0]
                plt.imshow(fake_image)
                plt.savefig("./{}/{}.png".format(SAMPLES_DIR, total_it))

            # Update generator
            y = np.ones([BATCH_SIZE, 1])
            noise = np.random.randn(BATCH_SIZE, GENERATOR_INPUT_SIZE)
            loss = gan.train_on_batch(noise, y)[1]

            loss_discriminator.append((d_loss_real + d_loss_fake) / 2.0)
            loss_generator.append(loss)
            total_it += 1

        print("Epoch", epoch)
        avg_loss_discriminator.append(np.mean(loss_discriminator))
        avg_loss_generator.append(np.mean(loss_generator))

    return avg_loss_discriminator, avg_loss_generator


def load_pictures():
    pictures = []
    files = glob.glob("./pictures/raven/*.jpg")

    for file in files:
        image = cv2.imread(file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0
        pictures.append(image)

    pictures = np.array(pictures)
    return pictures


def main():
    if not os.path.isdir(SAMPLES_DIR):
        os.mkdir(SAMPLES_DIR)

    if READ_MODEL:
        discriminator = keras.models.load_model("discriminator.model")
        generator = keras.models.load_model("generator.model")
        gan = keras.models.load_model("gan.model")
    else:
        discriminator, generator, gan = construct_models()

    pictures = load_pictures()

    avg_loss_discriminator, avg_loss_generator = run_training(pictures, discriminator, generator, gan)

    plt.clf()
    plt.plot(range(len(avg_loss_discriminator)), avg_loss_discriminator)
    plt.plot(range(len(avg_loss_generator)), avg_loss_generator)
    plt.legend(["discriminator loss", "generator loss"])
    plt.savefig("losses.png")
    plt.show()

    discriminator.save("discriminator.model")
    generator.save("generator.model")
    gan.save("gan.model")


if __name__ == '__main__':
    main()
