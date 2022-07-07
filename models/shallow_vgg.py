import keras
import tensorflow as tf
from tensorflow.keras import layers, metrics
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model


class ShallowVGG(Model):
    def __init__(self, layer1_num: int, layer2_num: int, layer3_num: int):
        super().__init__()
        self.sequence = keras.Sequential()

        for _ in range(layer1_num):
            self.sequence.add(layers.Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding="same"))
            self.sequence.add(layers.BatchNormalization())
            self.sequence.add(layers.ReLU())
        self.sequence.add(layers.MaxPool2D(pool_size=(2, 2), strides=2, padding="same"))

        for _ in range(layer2_num):
            self.sequence.add(layers.Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding="same"))
            self.sequence.add(layers.BatchNormalization())
            self.sequence.add(layers.ReLU())
        self.sequence.add(layers.MaxPool2D(pool_size=(2, 2), strides=2, padding="same"))

        for _ in range(layer3_num):
            self.sequence.add(layers.Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding="same"))
            self.sequence.add(layers.BatchNormalization())
            self.sequence.add(layers.ReLU())
        self.sequence.add(layers.MaxPool2D(pool_size=(2, 2), strides=2, padding="same"))
        self.sequence.add(layers.Flatten())
        self.sequence.add(layers.Dense(1))

    def call(self, inputs, training=None, mask=None):
        return self.sequence(inputs)


def ExtractorVGG(input_shape: tuple, layer1_num: int, layer2_num: int, layer3_num: int, name: str):
    _input = keras.Input(input_shape, name=f"{name}_input")
    output = _input

    for _ in range(layer1_num):
        output = layers.Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding="same")(output)
        output = layers.BatchNormalization()(output)
        output = layers.ReLU()(output)
    output = layers.MaxPool2D(pool_size=(2, 2), strides=2, padding="same")(output)

    for _ in range(layer2_num):
        output = layers.Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding="same")(output)
        output = layers.BatchNormalization()(output)
        output = layers.ReLU()(output)
    output = layers.MaxPool2D(pool_size=(2, 2), strides=2, padding="same")(output)

    for _ in range(layer3_num):
        output = layers.Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding="same")(output)
        output = layers.BatchNormalization()(output)
        output = layers.ReLU()(output)
    output = layers.GlobalAvgPool2D()(output)
    # output = layers.MaxPool2D(pool_size=(2, 2), strides=2, padding="same")(output)
    # output = layers.Flatten(name=f"{name}_output")(output)
    return _input, output


def CompositeVGG():
    stft_input, stft_output = ExtractorVGG(input_shape=(128, 32, 24), layer1_num=1, layer2_num=1, layer3_num=1, name="stft")
    recur_input, recur_output = ExtractorVGG(input_shape=(1024, 1024, 4), layer1_num=1, layer2_num=1, layer3_num=1, name="recur")
    concat = layers.Concatenate()([stft_output, recur_output])

    # concat_output = layers.Dense(units=2, name="concat_output")(concat)
    # body_input = keras.Input((2), name="body_input")
    # body_concat_input = layers.Concatenate()([concat_output, body_input])

    fc1 = layers.Dense(units=128)(concat)
    fc1_act = layers.ReLU()(fc1)
    final_output = layers.Dense(units=1, name="final_output")(fc1_act)
    composite_vgg = keras.Model(inputs=[stft_input, recur_input], outputs=[final_output])

    return composite_vgg


if __name__ == "__main__":
    rand_input = tf.random.normal((1, 128, 32, 24))
    model = ShallowVGG(1, 1, 1)
    print(model(rand_input).shape)
