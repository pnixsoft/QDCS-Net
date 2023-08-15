"""
Note:
    1.This file is used for designing the structure of encoder and decoder.
    2.The neural network structure in this model file is QDCS-Net,
    3.The output of the encoder must be the bitstream.
"""


import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K


#This part realizes the quantization and dequantization operations.
#The output of the encoder must be the bitstream.


def Num2Bit(Num, B):
    Num_ = Num.numpy()
    bit = (np.unpackbits(np.array(Num_, np.uint8), axis=1).reshape(-1, Num_.shape[1], 8)[:, :, 4:]).reshape(-1,
                                                                                                            Num_.shape[
                                                                                                                1] * B)
    bit.astype(np.float32)
    return tf.convert_to_tensor(bit, dtype=tf.float32)


def Bit2Num(Bit, B):
    Bit_ = Bit.numpy()
    Bit_.astype(np.float32)
    Bit_ = np.reshape(Bit_, [-1, int(Bit_.shape[1] / B), B])
    num = np.zeros(shape=np.shape(Bit_[:, :, 1]))
    for i in range(B):
        num = num + Bit_[:, :, i] * 2 ** (B - 1 - i)
    return tf.cast(num, dtype=tf.float32)


@tf.custom_gradient
def QuantizationOp(x, B):
    step = tf.cast((2 ** B), dtype=tf.float32)

    result = tf.cast((tf.round(x * step - 0.5)), dtype=tf.float32)

    result = tf.py_function(func=Num2Bit, inp=[result, B], Tout=tf.float32)

    def custom_grad(dy):
        grad = dy
        return (grad, grad)

    return result, custom_grad


class QuantizationLayer(tf.keras.layers.Layer):
    def __init__(self, B,**kwargs):
        self.B = B
        super(QuantizationLayer, self).__init__()

    def call(self, x):
        return QuantizationOp(x, self.B)

    def get_config(self):
        # Implement get_config to enable serialization. This is optional.
        base_config = super(QuantizationLayer, self).get_config()
        base_config['B'] = self.B
        return base_config

@tf.custom_gradient
def DequantizationOp(x, B):
    x = tf.py_function(func=Bit2Num, inp=[x, B], Tout=tf.float32)
    step = tf.cast((2 ** B), dtype=tf.float32)
    result = tf.cast((x + 0.5) / step, dtype=tf.float32)

    def custom_grad(dy):
        grad = dy * 1
        return (grad, grad)

    return result, custom_grad


class DeuantizationLayer(tf.keras.layers.Layer):
    def __init__(self, B,**kwargs):
        self.B = B
        super(DeuantizationLayer, self).__init__()

    def call(self, x):
        return DequantizationOp(x, self.B)

    def get_config(self):
        base_config = super(DeuantizationLayer, self).get_config()
        base_config['B'] = self.B
        return base_config


def Encoder(x,feedback_bits):
    B=4
    with tf.compat.v1.variable_scope('Encoder'):
        x1 = layers.Conv2D(2, 3, padding = 'SAME',activation=tf.nn.swish)(x)
        x1 = layers.Conv2D(8, 3, padding='SAME', activation=tf.nn.swish)(x1)
        x1 = layers.Conv2D(2, 3, padding = 'SAME',activation=tf.nn.swish)(x1)
        x1 = layers.Flatten()(x1)

        x2 = layers.Conv2D(2, 5, padding='SAME', activation=tf.nn.swish)(x)
        x2 = layers.Conv2D(8, 5, padding='SAME', activation=tf.nn.swish)(x2)
        x2 = layers.Conv2D(2, 5, padding='SAME', activation=tf.nn.swish)(x2)
        x2 = layers.Flatten()(x2)

        x = keras.layers.Add()([x1, x2])

        x = layers.Dense(units=int(feedback_bits/B), activation='sigmoid')(x)
        encoder_output = QuantizationLayer(B)(x)
        print(np.shape(encoder_output))
    return encoder_output
def Decoder(x,feedback_bits):
    B=4
    #print(np.shape(x))   #(none,128)
    #print(np.shape(DeuantizationLayer(B)(x))) # <unknown>

    decoder_input = DeuantizationLayer(B)(x)
    #print(np.shape(decoder_input)) # <unknown>
    x = tf.keras.layers.Reshape((-1, int(feedback_bits/B)))(decoder_input)
    x = layers.Dense(1024, activation='sigmoid')(x)
    x = layers.Reshape((32, 32, 1))(x)
    x = layers.Conv2D(2, 3, padding='SAME', activation=tf.nn.swish)(x)
    x_ini = layers.Conv2D(2, 3, padding='SAME', activation=tf.nn.swish)(x)

    for i in range(3):
        x = layers.Conv2D(2, 3, padding = 'SAME',activation=tf.nn.swish)(x_ini)
        x = layers.Conv2D(8, 3, padding = 'SAME',activation=tf.nn.swish)(x)
        x = layers.Conv2D(16,3, padding = 'SAME',activation=tf.nn.swish)(x)
        x = layers.Conv2D(2, 3, padding = 'SAME',activation=tf.nn.swish)(x)
        x_ini_1 = keras.layers.Add()([x_ini, x])

        x = layers.Conv2D(2, 2, padding='SAME', activation=tf.nn.swish)(x_ini)
        x = layers.Conv2D(8,(1,9), padding='SAME', activation=tf.nn.swish)(x)
        x = layers.Conv2D(16,(9,1), padding='SAME', activation=tf.nn.swish)(x)
        x = layers.Conv2D(2, 2, padding='SAME', activation=tf.nn.swish)(x)
        x_ini_2 = keras.layers.Add()([x_ini, x])

        x_ini = keras.layers.Add()([x_ini_1, x_ini_2])

    x = layers.Conv2D(16,(9,1), padding='SAME', activation=tf.nn.swish)(x_ini)
    x = layers.Conv2D(2, 2, padding='SAME', activation=tf.nn.swish)(x)
    decoder_output = layers.Conv2D(1,1, padding = 'SAME',activation="sigmoid")(x)

    return decoder_output

def NMSE(x, x_hat):
    power = np.sum(abs(x) ** 2, axis=1)
    mse = np.sum(abs(x - x_hat) ** 2, axis=1)
    nmse = np.mean(mse / power)
    return nmse

def Score(NMSE):
    score = 1-NMSE
    return score

# Return keywords of your own custom layers to ensure that model
# can be successfully loaded in test file.
def get_custom_objects():
    return {"QuantizationLayer":QuantizationLayer,"DeuantizationLayer":DeuantizationLayer,'swish':tf.nn.swish,'leaky_relu':tf.nn.leaky_relu}
