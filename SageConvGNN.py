import tensorflow as tf
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_gnn as tfgnn
import tensorflow_ranking as tfr

from tensorflow.keras import Sequential
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Embedding, Dense, Dropout, Flatten
import dgl.nn.tensorflow as dglnn
import tensorflow_gnn as tfgnn 

class CustomNormLayer(layers.Layer):
    def __init__(self, units):
        super(MyNormLayer, self).__init__()
        self.units = units

    def build(self, input_shape):
        self.scale = self.add_weight("scale", shape=[self.units,], initializer="ones")
        self.offset = self.add_weight("offset", shape=[self.units,], initializer="zeros")

    def call(self, inputs):
        mean, variance = tf.nn.moments(inputs, axes=-1, keepdims=True)
        return tf.nn.batch_normalization(inputs, mean, variance, self.offset, self.scale, 1e-6)

#following model uses sage conv layers (instead of plain conv layers)
class SAGEConvGNN(tf.keras.Model):
    def __init__(self, cfg, feat_stat):
        super(Net, self).__init__()
        self.node_feat_mean = tf.Variable(feat_stat['node_feat_mean'], dtype=tf.float32)
        self.node_feat_std = tf.Variable(feat_stat['node_feat_std'], dtype=tf.float32)

        hidden_dim = 256
        node_embed_dim = 64

        self.node_embed_dim = layers.Embedding(OP_SIZE, node_embed_dim, embeddings_constraint='max_norm')

        self.gconv = [
            dglnn.SAGEConv(hidden_dim, hidden_dim, aggregator_type='mean'),
            dglnn.SAGEConv(hidden_dim, hidden_dim, aggregator_type='mean'),
            dglnn.SAGEConv(hidden_dim, hidden_dim, aggregator_type='mean'),
            dglnn.SAGEConv(hidden_dim, hidden_dim, aggregator_type='mean')
        ]
        self.norm = [MyNormLayer(hidden_dim) for _ in range(4)]

        self.end_mlp = tf.keras.Sequential([
            layers.Dense(hidden_dim, use_bias=False),
            MyNormLayer(hidden_dim),
            layers.Activation('relu'),
            layers.Dense(hidden_dim // 2, use_bias=False),
            MyNormLayer(hidden_dim // 2),
            layers.Activation('relu'),
        ])

        self.predict = layers.Dense(1)

    def _node_level_forward(self, node_features, config_features, graph, num_configs):
        # Combine node and config features (adjust this based on your needs)
        combined_features = tf.concat([node_features, config_features], axis=-1)

        # Normalize combined features
        combined_features = (combined_features - self.node_feat_mean) / self.node_feat_std

        # Pass through graph convolutional layers and normalizations
        x = combined_features
        for conv, norm in zip(self.gconv, self.norm):
            x = conv(graph, x)
            x = norm(x)
            x = tf.nn.relu(x)

        # Apply end MLP
        x = self.end_mlp(x)

        return x

    def forward(self, graph: tfgnn.GraphTensor, num_configs: int, backprop: bool = True) -> tf.Tensor:
        graph = self.op_embedding(graph)

        config_features = graph.node_sets['nconfig']['feats']
        op_features = tf.concat([graph.node_sets['op']['feats'], graph.node_sets['op']['op_e']], axis=-1)

        if backprop:
            x_full = self._node_level_forward(
                node_features=tf.stop_gradient(op_features),
                config_features=tf.stop_gradient(config_features),
                graph=graph, num_configs=num_configs)

            x_backprop = self._node_level_forward(
                node_features=op_features,
                config_features=config_features,
                graph=graph, num_configs=num_configs)

            is_selected = tf.expand_dims(tf.expand_dims(graph.node_sets['op']['selected'], -1), -1)
            x = tf.where(is_selected, x_backprop, x_full)
        else:
            x = self._node_level_forward(
                node_features=op_features,
                config_features=config_features,
                graph=graph, num_configs=num_configs)

        return x

    def call(self, graph: tfgnn.GraphTensor, training: bool = False) -> tf.Tensor:
        return self.forward(graph, self.num_configs)