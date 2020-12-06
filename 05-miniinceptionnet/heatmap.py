import matplotlib.pyplot as plt
import numpy as np
import scipy
import tensorflow.keras.models as models

from images.images import get_images

LABEL_NAMES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def get_cam(model, image, weights):
    analysed = np.expand_dims(image, axis=0)
    gap_input, prediction_vector = model.predict(analysed)
    gap_input = np.squeeze(gap_input)
    prediction = np.argmax(prediction_vector)

    scale_factor = 32 / 7
    matrix_for_multiplication = scipy.ndimage.zoom(gap_input, (scale_factor, scale_factor, 1), order=1)
    weights_for_predicted = weights[:, prediction]
    final_output = np.dot(matrix_for_multiplication.reshape((32 * 32, 336)), weights_for_predicted).reshape(32, 32)
    return final_output, prediction


if __name__ == '__main__':
    model_name = "miniinception_5"
    model = models.load_model(model_name)
    weights = model.layers[-2].get_weights()[0]
    model_for_heatmap = models.Model(inputs=model.input, outputs=(model.layers[-6].output, model.layers[-1].output))
    images, labels = get_images(3)

    for i in range(len(images)):
        cam, prediction = get_cam(model_for_heatmap, images[i], weights)
        label = LABEL_NAMES[prediction]

        plt.figure()
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i], alpha=0.5)
        plt.imshow(cam, alpha=0.5)
        plt.xlabel(label)
        plt.savefig("heatmap-{}.png".format(i))
        plt.show()
