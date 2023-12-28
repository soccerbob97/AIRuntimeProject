import tensorflow as tf
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_gnn as tfgnn
import tensorflow_ranking as tfr
from tensorflow.keras.layers import Embedding, Dense, Dropout, Flatten

"""
MAKE SURE TO RUN ON RAW INPUTS (NOT GRAPH_TENSOR TYPE INPUTS IN KAGGLE)
"""


class BasicNeuralNet(tf.keras.Model):
    def __init__(self, input_shape, output_dim):
        super().__init__()
        self.flatten = Flatten(input_shape=input_shape)
        self.dense1 = Dense(128, activation='relu')
        self.dropout1 = Dropout(0.5)
        self.dense2 = Dense(64, activation='relu')
        self.dropout2 = Dropout(0.5)
        self.output_layer = Dense(output_dim, activation='linear')  # Change to 'softmax' for classification

    def forward(self, inputs):
        x = self.flatten(inputs)
        x = self.dense1(x)
        x = self.dropout1(x)
        x = self.dense2(x)
        x = self.dropout2(x)
        return self.output_layer(x)

    def call(self, inputs):
        return self.forward(inputs)