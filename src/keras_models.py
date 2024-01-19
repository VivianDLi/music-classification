from datetime import datetime
import tensorflow as tf
from keras import Sequential
from keras.layers import (
    Input,
    Conv2D,
    SeparableConv2D,
    Dense,
    MaxPooling2D,
    GlobalMaxPooling2D,
    BatchNormalization,
)


def get_full_model(n_classes):
    full_model = Sequential(
        [
            Input((84, None, 1)),
            Conv2D(
                32,
                (12, 3),
                1,
                data_format="channels_last",
                activation="relu",
                name="conv_1",
            ),
            BatchNormalization(),
            Conv2D(
                64,
                (13, 3),
                1,
                dilation_rate=(1, 2),
                data_format="channels_last",
                activation="relu",
                name="conv_2",
            ),
            BatchNormalization(),
            MaxPooling2D((1, 2), (1, 2), data_format="channels_last"),
            Conv2D(
                64,
                (13, 3),
                1,
                data_format="channels_last",
                activation="relu",
                name="conv_3",
            ),
            BatchNormalization(),
            Conv2D(
                64,
                3,
                1,
                dilation_rate=(1, 2),
                data_format="channels_last",
                activation="relu",
                name="conv_4",
            ),
            BatchNormalization(),
            MaxPooling2D((1, 2), (1, 2), data_format="channels_last"),
            Conv2D(
                128,
                3,
                1,
                data_format="channels_last",
                activation="relu",
                name="conv_5",
            ),
            BatchNormalization(),
            Conv2D(
                128,
                3,
                1,
                dilation_rate=(1, 2),
                data_format="channels_last",
                activation="relu",
                name="conv_6",
            ),
            BatchNormalization(),
            MaxPooling2D((1, 2), (1, 2), data_format="channels_last"),
            Conv2D(
                256,
                3,
                1,
                data_format="channels_last",
                activation="relu",
                name="conv_7",
            ),
            BatchNormalization(),
            Conv2D(
                256,
                3,
                1,
                dilation_rate=(1, 2),
                data_format="channels_last",
                activation="relu",
                name="conv_8",
            ),
            BatchNormalization(),
            MaxPooling2D((1, 2), (1, 2), data_format="channels_last"),
            Conv2D(
                512,
                3,
                1,
                data_format="channels_last",
                activation="relu",
                name="conv_9",
            ),
            BatchNormalization(),
            Conv2D(
                512,
                3,
                1,
                dilation_rate=(1, 2),
                data_format="channels_last",
                activation="relu",
                name="conv_10",
            ),
            BatchNormalization(),
            GlobalMaxPooling2D(data_format="channels_last"),
            Dense(300, name="representation_layer"),
            Dense(
                n_classes, activation="softmax", name="classification_layer"
            ),
        ]
    )
    return full_model


def get_small_model(n_classes):
    full_model = Sequential(
        [
            Input((84, None, 1)),
            SeparableConv2D(
                (12, 3),
                1,
                depth_multiplier=32,
                data_format="channels_last",
                activation="relu",
                name="conv_1",
            ),
            BatchNormalization(),
            SeparableConv2D(
                (13, 3),
                1,
                depth_multiplier=2,
                dilation_rate=(1, 2),
                data_format="channels_last",
                activation="relu",
                name="conv_2",
            ),
            BatchNormalization(),
            MaxPooling2D((1, 2), (1, 2), data_format="channels_last"),
            SeparableConv2D(
                (13, 3),
                1,
                depth_multiplier=1,
                data_format="channels_last",
                activation="relu",
                name="conv_3",
            ),
            BatchNormalization(),
            SeparableConv2D(
                3,
                1,
                depth_multiplier=1,
                dilation_rate=(1, 2),
                data_format="channels_last",
                activation="relu",
                name="conv_4",
            ),
            BatchNormalization(),
            MaxPooling2D((1, 2), (1, 2), data_format="channels_last"),
            SeparableConv2D(
                3,
                1,
                depth_multiplier=2,
                data_format="channels_last",
                activation="relu",
                name="conv_5",
            ),
            BatchNormalization(),
            SeparableConv2D(
                3,
                1,
                depth_multiplier=1,
                dilation_rate=(1, 2),
                data_format="channels_last",
                activation="relu",
                name="conv_6",
            ),
            BatchNormalization(),
            MaxPooling2D((1, 2), (1, 2), data_format="channels_last"),
            SeparableConv2D(
                3,
                1,
                depth_multiplier=2,
                data_format="channels_last",
                activation="relu",
                name="conv_7",
            ),
            BatchNormalization(),
            SeparableConv2D(
                3,
                1,
                depth_multiplier=1,
                dilation_rate=(1, 2),
                data_format="channels_last",
                activation="relu",
                name="conv_8",
            ),
            BatchNormalization(),
            MaxPooling2D((1, 2), (1, 2), data_format="channels_last"),
            SeparableConv2D(
                3,
                1,
                depth_multiplier=2,
                data_format="channels_last",
                activation="relu",
                name="conv_9",
            ),
            BatchNormalization(),
            SeparableConv2D(
                3,
                1,
                depth_multiplier=1,
                dilation_rate=(1, 2),
                data_format="channels_last",
                activation="relu",
                name="conv_10",
            ),
            BatchNormalization(),
            GlobalMaxPooling2D(data_format="channels_last"),
            Dense(300, name="representation_layer"),
            Dense(
                n_classes, activation="softmax", name="classification_layer"
            ),
        ]
    )
    return full_model


def get_representation_model(full_model, name=""):
    tf.keras.utils.plot_model(
        full_model,
        to_file=f"./models/plots/{name}-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.png",
        expand_nested=True,
        show_shapes=True,
    )
    # Sub Model
    sub_model = tf.keras.Sequential()

    # I will skip inp and d1:
    for layer in full_model.layers[:-1]:
        sub_model.add(layer)
    return sub_model
