import tensorflow as tf
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_gnn as tfgnn
from utils import _OpEmbedding, _mlp

class VanillaGNNModel(tf.keras.Model):
    """
    A Graph Neural Network (GNN) with residual connections. This model
    operates on graph structured data and leverages adjacency information
    for making predictions.
    """
    def __init__(
        self, num_configs: int, num_ops: int, op_embed_dim: int = 32,
        num_gnns: int = 2, mlp_layers: int = 2, hidden_activation: str = 'leaky_relu',
        hidden_dim: int = 32, reduction: str = 'sum'):
        super().__init__()
        self.num_configs = num_configs
        self.num_ops = num_ops
        self.op_embedding = _OpEmbedding(num_ops, op_embed_dim)
        self.prenet = _mlp([hidden_dim] * mlp_layers, hidden_activation)
        self.gc_layers = [ _mlp([hidden_dim] * mlp_layers, hidden_activation) for _ in range(num_gnns) ]
        self.postnet = _mlp([hidden_dim, 1], hidden_activation, use_bias=False)

    def call(self, graph: tfgnn.GraphTensor, training: bool = False) -> tf.Tensor:
        return self.forward(graph, self.num_configs)

    def _node_level_forward(
        self, node_features: tf.Tensor, config_features: tf.Tensor,
        graph: tfgnn.GraphTensor, num_configs: int, edgeset_prefix: str = '') -> tf.Tensor:
        
        adj_op_op = AdjacencyMultiplier(graph, edgeset_prefix + 'feed')  # op->op
        adj_config = AdjacencyMultiplier(graph, edgeset_prefix + 'config')  # nconfig->op

        adj_op_op_hat = (adj_op_op + adj_op_op.transpose()).add_eye().normalize_symmetric()

        node_features_stacked = tf.stack([node_features] * num_configs, axis=1)
        config_features_scaled = 100 * (adj_config @ config_features)
        combined_features = tf.concat([config_features_scaled, node_features_stacked], axis=-1)
        combined_features = self.prenet(combined_features)
        combined_features = tf.nn.leaky_relu(combined_features)

        for layer in self.gc_layers:
            y = tf.concat([config_features_scaled, combined_features], axis=-1)
            y = tf.nn.leaky_relu(layer(adj_op_op_hat @ y))
            combined_features += y

        return combined_features

    def forward(
        self, graph: tfgnn.GraphTensor, num_configs: int, backprop: bool = True) -> tf.Tensor:
        
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
                graph=graph, num_configs=num_configs, edgeset_prefix='sampled_')

            is_selected = tf.expand_dims(tf.expand_dims(graph.node_sets['op']['selected'], -1), -1)
            x = tf.where(is_selected, x_backprop, x_full)
        else:
            x = self._node_level_forward(
                node_features=op_features,
                config_features=config_features,
                graph=graph, num_configs=num_configs)

        adj_config = AdjacencyMultiplier(graph, 'config')
        config_feats = (adj_config.transpose() @ x)

        # Global pooling
        adj_pool_op_sum = AdjacencyMultiplier(graph, 'g_op').transpose()
        adj_pool_op_mean = adj_pool_op_sum.normalize_right()
        adj_pool_config_sum = AdjacencyMultiplier(graph, 'g_config').transpose()

        x = self.postnet(tf.concat([
            adj_pool_op_mean @ x,
            tf.nn.l2_normalize(adj_pool_op_sum @ x, axis=-1),
            tf.nn.l2_normalize(adj_pool_config_sum @ config_feats, axis=-1),
        ], axis=-1))

        return tf.squeeze(x, -1)