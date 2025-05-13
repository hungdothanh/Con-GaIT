

import tensorflow as tf

from utils import *


Precision = tf.keras.metrics.Precision
Recall = tf.keras.metrics.Recall
Adam = tf.keras.optimizers.Adam
Model = tf.keras.models.Model
load_model = tf.keras.models.load_model
Input = tf.keras.layers.Input
Conv1D = tf.keras.layers.Conv1D
MaxPooling1D = tf.keras.layers.MaxPooling1D
LSTM = tf.keras.layers.LSTM
Flatten = tf.keras.layers.Flatten
Dense = tf.keras.layers.Dense
Dropout = tf.keras.layers.Dropout
l2 = tf.keras.regularizers.l2


def build_cnn_lstm_model(input_shape, num_classes=4):
    """Build the CNN-LSTM model as described in Table 5 of the paper."""
    inputs = Input(shape=input_shape)
    
    # 1st Convolution + Pooling
    x = Conv1D(filters=12, kernel_size=1, activation='relu', strides=1, padding='same', kernel_regularizer=l2(0.01))(inputs)
    x = MaxPooling1D(pool_size=2, strides=2)(x)
    
    # 2nd Convolution + Pooling
    x = Conv1D(filters=24, kernel_size=3, activation='relu', strides=1, padding='same', kernel_regularizer=l2(0.01))(x)
    x = MaxPooling1D(pool_size=2, strides=2)(x)
    
    # 3rd Convolution + Pooling
    x = Conv1D(filters=48, kernel_size=3, activation='relu', strides=1, padding='same', kernel_regularizer=l2(0.01))(x)
    x = MaxPooling1D(pool_size=2, strides=2)(x)
    
    # 4th Convolution + Pooling
    x = Conv1D(filters=96, kernel_size=3, activation='relu', strides=1, padding='same', kernel_regularizer=l2(0.01))(x)
    x = MaxPooling1D(pool_size=2, strides=2)(x)
    
    # LSTM layers
    x = LSTM(50, return_sequences=True)(x)
    x = Dropout(0.2)(x)
    x = LSTM(50)(x)
    x = Dropout(0.2)(x)
   
    x = Dense(100, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = Dense(25, kernel_regularizer=l2(0.01))(x)
    
    # Output layer
    outputs = Dense(num_classes, activation='softmax')(x)
    

    model = Model(inputs, outputs)

    return model
