# RNN only numpy

import numpy as np
from datasets import load_dataset
import string
from collections import Counter
from nltk.corpus import stopwords

# Hidden state H_t is an element of R^(n x h).
# Input I_t is an element of R^(n x d).
# n is the number of samples
# d is the number of inputs at each sample
# h is the number of hidden units
# W_xh is an element of R^(d x h)
# W_hh is an element of R^(h x h)
# b_h is an element of R^(1 x h)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def crossentropy_loss(x, y):
    cel = -np.sum(y * np.log(x + 1e-9)) / x.shape[0]
    return cel

class RNN:
    def __init__(self, inputs_per_sample, hidden_size, output_size):
        self.RNNLayer = RNNLayer(inputs_per_sample, hidden_size)
        self.OutLayer = OutLayer(hidden_size, output_size)
        self.steps = 0

    def forward(self, input):
        f1 = self.RNNLayer.forward(input)
        f2 = self.OutLayer.forward(f1)
        return f2

class RNNLayer:
    # TANH activation function
    def __init__(self, inputs_per_sample, hidden_size):
        # Xavier (Glorot) Weight Initialization
        self.weights_xh = np.random.randn(inputs_per_sample, hidden_size) * np.sqrt(2. / (inputs_per_sample + hidden_size))
        self.weights_hh = np.random.randn(hidden_size, hidden_size) * np.sqrt(2. / (hidden_size + hidden_size))
        self.bias = np.zeros((hidden_size,))
        self.prev = np.zeros((hidden_size,))
    def forward(self, input):
        f1 = np.matmul(input, self.weights_xh) + np.matmul(self.prev, self.weights_hh) + self.bias
        f2 = np.tanh(f1)
        self.prev = f2
        return f2

class OutLayer:
    # Sigmoid activation function
    def __init__(self, hidden_size, output_size):
        # Xavier (Glorot) Weight Initialization
        self.weights = np.random.randn(hidden_size, output_size) * np.sqrt(2. / (hidden_size + output_size))
        self.bias = np.zeros((output_size,))

    def forward(self, input):
        f1 = np.matmul(input, self.weights) + self.bias
        f2 = sigmoid(f1)
        return f2

# Convert sentence to tokens
def tokenize(a):
    # Remove punctation
    a = [''.join([char for char in a if char not in string.punctuation]) for b in a]
    # Convert to lowercase
    a = [b.lower() for b in a]
    # Tokenize
    tokens = [b.split() for b in a]

    return tokens

if __name__ == "__main__":
    # Input is 2 x 2, 2 samples, 2 inputs each sample
    # Input = [.5, .5]
    #         [.5, .5]
    imdb = load_dataset("imdb")

    train_data = imdb["train"]
    test_data = imdb["test"]

    reviews = train_data["text"]
    labels = train_data["label"]

    embedding_dim = 128
    hidden_size = 256
    output_size = 1

    rnn = RNN(embedding_dim, hidden_size, output_size)

    steps = 10000
    print(reviews[0])
    a = tokenize(reviews[0])
    print(len(a))
    #for i in range(steps):
