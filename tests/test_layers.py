import numpy as np
from keras.layers import Input
from keras.models import Model
from keras_han.layers import AttentionLayer


def test_compute_output_shape():
    """The layer should remove the time dimension from the output size"""
    layer = AttentionLayer()
    out_shape = layer.compute_output_shape((None, 3, 2))
    assert out_shape == (None, 2)


def test_call_output_shape():
    """The shape ouf the output tensor should match compute_output_shape"""
    layer = AttentionLayer()
    input_tensor = Input((3, 2))
    out_tensor = layer(input_tensor)
    out_shape = tuple(out_tensor.shape.as_list())

    assert out_shape == (None, 2)


def test_attention():
    """The attention weights should be probabilities and hence sum to one"""
    input_tensor = Input((3, 2))
    layer = AttentionLayer()
    output_tensor = layer(input_tensor)
    temp_model = Model(inputs=[input_tensor], outputs=[output_tensor])

    data = np.array([[[0.1, 0.2],
                     [0.2, 0.2],
                     [0.1, 0.2]],
                    [[1, 0.1],
                     [1, 0.1],
                     [1, 0.1]]], dtype='float32')

    predictions = temp_model(data)

    # We can do some validations based on the exact specification
    # of the input data

    # Validate that internally the attention mechanism uses a probability
    # mechanism that sums to one
    np.testing.assert_almost_equal(predictions[1, 0], 1)

    # Validate that attention is applied in the time directions
    np.testing.assert_almost_equal(predictions[1, 1], 0.1)
