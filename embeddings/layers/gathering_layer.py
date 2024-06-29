import tensorflow as tf
from keras.layers import Layer
import warnings


class GatherIndices(Layer):

    def __init__(self, axis=None, batch_dims=1, **kwargs):
        super().__init__(**kwargs)
        self._axis = axis
        self._batch_dims = batch_dims

    def get_config(self):
        config = super().get_config()
        config.update(axis=self._axis, batch_dims=self._batch_dims)
        return config

    def compute_output_shape(self, input_shapes):
        data_shape, indices_shape = input_shapes
        return data_shape[0], indices_shape[1], data_shape[2]

    def call(self, inputs):
        data, indices = inputs
        return tf.gather(data, indices, axis=self._axis, batch_dims=self._batch_dims)


def deprecated_model_function(function, old_name):
    def _function_wrapper(*args, **kwargs):
        """Deprecated: use :meth:`in_out_tensors`."""

        warnings.warn(
            f"The '{old_name}' method is deprecated, use 'in_out_tensors' instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        return function(*args, **kwargs)

    return _function_wrapper
