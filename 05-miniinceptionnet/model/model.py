import tensorflow.keras.layers as layers
import tensorflow.keras.models as models

X5_PERCENT = 0.2


def mini_inception_net(width, height, depth, classes, use_dropout=True, separable_convolutions=False, use_5x5=False,
                       use_gap=False):
    def conv_module(x, K, kX, kY, stride, chanDim, padding="same"):
        if separable_convolutions:
            x = layers.SeparableConv2D(K, (kX, kY), strides=stride, padding=padding)(x)
        else:
            x = layers.Conv2D(K, (kX, kY), strides=stride, padding=padding)(x)
        x = layers.BatchNormalization(axis=chanDim)(x)
        x = layers.Activation("relu")(x)

        return x

    def inception_module(x, numK1x1, numK3x3, chanDim):
        if use_5x5:
            num_sum = numK1x1 + numK3x3
            numK5x5 = round(X5_PERCENT * num_sum)
            numK1x1 = round((1 - X5_PERCENT) * numK1x1)
            numK3x3 = num_sum - numK5x5 - numK1x1  # to keep filters number constant

            conv_1x1 = conv_module(x, numK1x1, 1, 1, (1, 1), chanDim)
            conv_3x3 = conv_module(x, numK3x3, 3, 3, (1, 1), chanDim)
            conv_5x5 = conv_module(x, numK5x5, 5, 5, (1, 1), chanDim)
            x = layers.concatenate([conv_1x1, conv_3x3, conv_5x5], axis=chanDim)
        else:
            conv_1x1 = conv_module(x, numK1x1, 1, 1, (1, 1), chanDim)
            conv_3x3 = conv_module(x, numK3x3, 3, 3, (1, 1), chanDim)
            x = layers.concatenate([conv_1x1, conv_3x3], axis=chanDim)

        return x

    def downsample_module(x, K, chanDim):
        conv_3x3 = conv_module(x, K, 3, 3, (2, 2), chanDim, padding="valid")
        pool = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
        x = layers.concatenate([conv_3x3, pool], axis=chanDim)

        return x

    inputShape = (height, width, depth)
    chanDim = -1

    # define the model input and first CONV module
    inputs = layers.Input(shape=inputShape)
    x = conv_module(inputs, 96, 3, 3, (1, 1), chanDim)

    # two Inception modules followed by a downsample module
    x = inception_module(x, 32, 32, chanDim)
    x = inception_module(x, 32, 48, chanDim)
    x = downsample_module(x, 80, chanDim)

    # four Inception modules followed by a downsample module
    x = inception_module(x, 112, 48, chanDim)
    x = inception_module(x, 96, 64, chanDim)
    x = inception_module(x, 80, 80, chanDim)
    x = inception_module(x, 48, 96, chanDim)
    x = downsample_module(x, 96, chanDim)

    # two Inception modules followed by global POOL and dropout
    x = inception_module(x, 176, 160, chanDim)
    x = inception_module(x, 176, 160, chanDim)
    if use_gap:
        x = layers.GlobalAveragePooling2D()(x)
    else:
        x = layers.AveragePooling2D((7, 7))(x)
    if use_dropout:
        x = layers.Dropout(0.5)(x)

    # softmax classifier
    x = layers.Flatten()(x)
    x = layers.Dense(classes)(x)
    x = layers.Activation("softmax")(x)

    # create the model
    model = models.Model(inputs, x, name="miniinceptionnet")

    return model
