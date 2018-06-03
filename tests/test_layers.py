import unittest
import numpy as np
from keras.layers import Input
from keras.models import Model
from keras_han.layers import AttentionLayer

class TestAttentionLayer(unittest.TestCase):
    def setUp(self):
        self.data = np.array([
            [[0.1, 0.2],
             [0.2, 0.2],
             [0.1, 0.2]],
            [[1, 0.1],
             [1, 0.1],
             [1, 0.1]]
        ], dtype='float32')

        self.input_layer = Input((3,2))
        self.layer = AttentionLayer()

    def test_compute_output_shape(self):
        """The layer should remove the time dimension from the output size"""
        out_shape = self.layer.compute_output_shape((None, 3, 2))

        self.assertEqual(
            out_shape, (None, 2),
            "Timedimension should be removed from the output shape"
        )

    def test_call_output_shape(self):
        """The shape ouf the output tensor should match compute_output_shape"""
        out_tensor = self.layer(self.input_layer)
        out_shape = tuple(out_tensor.shape.as_list())

        self.assertEqual(
            out_shape, (None, 2),
            "Out shape should not contain a time dimension"
        )

    def test_attention(self):
        """The attention weights should be probabilities and hence sum to one"""
        model_output = self.layer(self.input_layer)
        temp_model = Model(inputs=[self.input_layer], outputs=[model_output])

        predictions = temp_model.predict(self.data)

        # We can do some validations based on the exact specification
        # of the input data
        self.assertAlmostEqual(
            predictions[1][0], 1, places=5,
            msg="Probabilties should sum to one"
        )

        self.assertAlmostEqual(
            predictions[1][1], 0.1, places=5,
            msg="Probabilities should be applied in the time direction"
        )

    def tearDown(self):
        del self.data
        del self.layer