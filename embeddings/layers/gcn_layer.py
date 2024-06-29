import keras.backend
from keras import backend as K
from keras.layers import Layer
from keras import activations
import tensorflow as tf


class GCNLayer(Layer):

    def __init__(self, output_dim, adj=None, activation=None, **kwargs):
        self.output_dim = output_dim
        self.adj = adj
        self.activation = activations.get(activation)

        super(GCNLayer, self).__init__(**kwargs)

    def build(self, input_shapes):
        input_shape, adj_shape = input_shapes
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[2], self.output_dim),
                                      initializer='normal', trainable=True)
        super(GCNLayer, self).build(input_shapes)

    def call(self, input_data):
        data, adj = input_data
        adj = adj[0]
        adj = tf.sparse.from_dense(adj)
        data = data[0]
        af = K.dot(adj, data)
        output = K.dot(af, self.kernel)
        output = keras.backend.expand_dims(output, axis=0)
        return self.activation(output)

    def compute_output_shape(self, input_shapes):
        input_shape , adj_shape = input_shapes
        return 1, input_shape[1], self.output_dim


