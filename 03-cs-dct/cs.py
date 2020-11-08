import enum

import scipy.fftpack as spfft
from pylbfgs import owlqn

from utils import *


class Mode(enum.Enum):
    Random = 1
    Small_image = 2
    Missing_fragment = 3
    Every_nth_sample = 4


mode = Mode.Every_nth_sample

output_directories = {
    Mode.Random: "reconstructed_random/",
    Mode.Small_image: "reconstructed_small_random/",
    Mode.Missing_fragment: "reconstructed_missing_random/",
    Mode.Every_nth_sample: "reconstructed_every_nth/"
}

samples_percentages = (0.01, 0.05, 0.1, 0.25, 0.5)
samples_ns = (100, 10, 5, 3, 2)
division_factor = 20 if mode == Mode.Small_image else 10
multiply_factor = 1.0 if mode == Mode.Small_image else 1.5
x_border_percent = 0.1
y_border_percent = 0.1


def dct2(x):
    return spfft.dct(spfft.dct(x.T, norm="ortho", axis=0).T, norm="ortho", axis=0)


def idct2(x):
    return spfft.idct(spfft.idct(x.T, norm="ortho", axis=0).T, norm="ortho", axis=0)


# based on http://www.pyrunner.com/weblog/2016/05/26/compressed-sensing-python/
def evaluate(x, g, step):
    # determine Ax - b squared
    x2 = x.reshape((nx, ny)).T
    Ax2 = idct2(x2)
    Ax = Ax2.T.flat[ri].reshape(b.shape)
    Axb = Ax - b
    fx = np.sum(np.power(Axb, 2))

    # calculate gradient
    Axb2 = np.zeros(x2.shape)
    Axb2.T.flat[ri] = Axb
    AtAxb2 = 2 * dct2(Axb2)
    AtAxb = AtAxb2.T.reshape(x.shape)
    np.copyto(g, AtAxb)

    return fx


def cut_fragment(image_array):
    for i in range(0, y_border):
        for j in range(0, x_border):
            image_array[i][j] = 0.0


def get_samples(N):
    if mode == Mode.Every_nth_sample:
        samples = []
        for i in range(0, nx, N):
            for j in range(0, ny, N):
                samples.append(i * ny + j)
    else:
        k = round(nx * ny * N)
        samples = np.random.choice(nx * ny, k, replace=False)
        if mode == Mode.Missing_fragment:
            samples = list(filter(lambda x: x % ny >= y_border and x // nx >= x_border, samples))

    return samples


if __name__ == '__main__':
    for folder, _, files in os.walk("pictures/"):
        for picture in files:
            error_output_name = output_directories[mode] + picture.split(".")[0] + ".txt"
            error_output = open(error_output_name, "w")

            image = get_image_for_analysis(folder, picture, division_factor=division_factor,
                                           multiply_factor=multiply_factor)
            ny, nx = image.shape

            if mode == Mode.Missing_fragment:
                image_copy = image.copy()
                x_border = int(nx * x_border_percent)
                y_border = int(ny * y_border_percent)
                cut_fragment(image)
            else:
                image_copy = image

            n_values = samples_ns if mode == Mode.Every_nth_sample else samples_percentages

            for n in n_values:
                ri = get_samples(n)
                b = image.T.flat[ri].astype(float)
                Xat2 = owlqn(nx * ny, evaluate, None, 5)

                Xat = Xat2.reshape(nx, ny).T
                Xa = idct2(Xat)

                image_output_name = picture.split(".")[0] + "_{}.png".format(n)
                cv2.imwrite(output_directories[mode] + image_output_name, Xa)

                difference = np.abs(np.subtract(image_copy, Xa)) / 255
                norm = np.linalg.norm(difference) / (nx * ny)

                if mode == Mode.Every_nth_sample:
                    error_output.write("{} {}\n".format(1.0 / n, norm))
                else:
                    error_output.write("{} {}\n".format(n, norm))

            error_output.close()
