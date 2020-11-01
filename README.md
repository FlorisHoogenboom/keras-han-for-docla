# keras-han-for-docla
<img align="right" width="300" src="http://digitaldreamworks.nl/misc/images/han.png">

This repository contains a Keras implementation of the network presented in [Hierarchical Attention Networks for Document Classification](http://www.cs.cmu.edu/~./hovy/papers/16HLT-hierarchical-attention-networks.pdf) by Zichao Yang, Diyi Yang, Chris Dyer, Xiaodong He, Alex Smola, and Eduard Hovy (2016). The implementation in this repository should be fully backend agnostic.

# Short description of the network
The idea of this network is to summarize a whole text into a single vector by first summarizing each sentence into a vector and subsequently summarizing these representations of sentences. For a technical description of the network we refer to the aforementioned paper.

# How to use
First clone this repository and run `pip install .` in the root of this repository. Next, you can import the model simply by running
```python
from keras_han.model import HAN
```
and instantiate and use it by
```python
han = HAN(
    max_words=100, max_sentences=15, output_size=2,
    embedding_matrix={your embedding matrix}
)

han.summary()
```
This should output something like
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 15, 100)           0         
_________________________________________________________________
word_encoder (TimeDistribute (None, 15, 100, 200)      82404     
_________________________________________________________________
word_attention (TimeDistribu (None, 15, 200)           20100     
_________________________________________________________________
sentence_encoder (Model)     (None, 15, 200)           240800    
_________________________________________________________________
sentence_attention (Attentio (None, 200)               20100     
_________________________________________________________________
class_prediction (Dense)     (None, 2)                 402       
=================================================================
Total params: 363,806
Trainable params: 363,802
Non-trainable params: {pretrained params in your embedding matrix}
_________________________________________________________________
```
To instantiate the model you need to provide a few parameters. Here we list only the important ones, for the full list of parameters please see the docsting of the `HAN` class.

- __max_words__: The maximum number of words per sentence you wish to allow
- __max_sentences__: The maximum number of sentences you wish to allow for one example
- __output_size__: The number of output classes
- __embedding_matrix__: The word embedding matrix you wish to use in the network's word encoder.

Once you have instantiated the `HAN` class, you can simply call fit, predict and evaluate on this object. Actually, your `HAN` instance is just a Keras model so you can use all features Keras offers.

## How to structure your data
Since this model needs to distinguish between sentences, the input format of this model is a bit different from most models you may be used to in NLP. Your data should be structured into a 3d-tensor with dimensions `(num_obs, max_sentences, max_words)`. Each entry in this matrix should be a representation of a single token. This matrix may be zero-padded to allow for sorter sentences/texts.

## Gettting the attention weights
One of the best features of models with attention is that you can better understand what "drives" the model. The implementation of the `HAN` model given here has support for visualizing the sentence attention weights. For this simply call `han.predict_sentence_attention(X)` with your input data. This returns a 2d-numpy array of dimensions `(num_samples, max_sentences)`. To give you an idea, the example below shows the attentions per sentence for a short IMDB review.
```
this movie is full of references.  --  Attention: 0.13
like mad max ii, the wild one and many others.  --  Attention: 0.06
the ladybug´s face it´s a clear reference (or tribute) to peter lorre.  --  Attention: 0.07
this movie is a masterpiece.  --  Attention: 0.31
we´ll talk much more about in the future.  --  Attention: 0.08
```

To generate word attention weights, simply call `han.predict_word_attention(X)` with your input data. This returns a 3d-numpy array of dimensions `(num_samples, max_sentences, max_words)`. To give you an idea, the example below shows the word attention weights within brackets for a short IMDB review.
```
this (0.17) movie (0.2) is (0.08) full (0.32) of (0.02) references (0.21).
```

# Examples
In `/examples` you can find an example script how to apply this model on IMDB's review data. For this two files need to be downloaded.
The embeddings can be downloaded [here](https://nlp.stanford.edu/projects/glove/) and the data to IMDB review data can be found
[here](https://www.kaggle.com/c/word2vec-nlp-tutorial/data).

# Loading a saved model
The network can simply be saved like any Keras model (e.g. using training callbacks or simply by calling `han.save(...)` on your model). To load a saved model you need to provide Keras with the custom model and layers provided in this repo.

```python
from keras_han.model import HAN
from keras_han.layers import AttentionLayer
from keras.models import load_model

han = load_model({file_path}, custom_objects={
    'HAN': HAN,
    'AttentionLayer': AttentionLayer
})
```

# Example use cases
The examples attached to this repo contain a simple application on a sentiment classification problem. Also, I've applied this network to classify parts of legal texts (my motivation for implementing this). I would love to hear your use cases.

# Tests
To run tests make sure `nose` is installed. You can kick-off the test suite with
```
nosetests --verbose -w ./tests/
```
