# RNN only numpy

import numpy as np
from datasets import load_dataset
import string
import random


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
    print(x[0], x[1], x[2], x[3])
    print(y[0], y[1], y[2], y[3])
    cel = -np.sum(y * np.log(x + 1e-9)) / x.shape[0]
    return cel


def calculate_accuracy(predictions, labels):
    # Assuming your model outputs predictions as probabilities, you might need to convert them to binary output
    binary_predictions = [1 if p >= 0.5 else 0 for p in predictions]

    correct_predictions = sum([1 if pred == true else 0 for pred, true in zip(binary_predictions, labels)])
    accuracy = correct_predictions / len(labels)

    return accuracy

class RNN():
    def __init__(self, inputs_per_sample, hidden_size, output_size, lr):
        self.RNNLayer = RNNLayer(inputs_per_sample, hidden_size, batch_size)
        self.OutLayer = OutLayer(hidden_size, output_size)
        self.steps = 0
        self.index = 0
        self.preds = None
        self.labels = None
        self.inputs = None
        self.lr = lr

        self.grad_sum_weights_xh = np.zeros_like(self.RNNLayer.weights_xh)
        self.grad_sum_weights_hh = np.zeros_like(self.RNNLayer.weights_hh)
        self.grad_sum_bias_h = np.zeros_like(self.RNNLayer.bias)
        self.grad_sum_weights_o = np.zeros_like(self.OutLayer.weights)
        self.grad_sum_bias_o = np.zeros_like(self.OutLayer.bias)

    def forward(self, input):
        f1 = self.RNNLayer.forward(input)
        f2 = np.asarray(self.OutLayer.forward(f1))
        self.input = np.array(input)
        self.preds = f2
        return f2

    def get_next_batch(self, padded, labels, batch_size):

        if self.index + batch_size > len(padded):  # Reset the index if we've exhausted the dataset
            self.index = 0

        batch_data = np.atleast_2d(list(padded[self.index : self.index + batch_size]))
        batch_labels = np.transpose(np.atleast_2d(labels[self.index : self.index + batch_size]))

        self.labels = batch_labels
        self.index += batch_size

        return batch_data, batch_labels

    def compute_gradients(self):
        # derivative of loss function w.r.t the output layer
        grad_output = np.subtract(self.preds, self.labels)

        # derivative of loss function w.r.t the hidden layer output
        grad_hidden = grad_output.dot(self.OutLayer.weights.T) * (1 - np.power(self.RNNLayer.prev, 2))

        # derivative of loss function w.r.t weights and biases in RNN layer
        grad_weights_xh = self.input.T.dot(grad_hidden)
        grad_weights_hh = self.RNNLayer.prev.T.dot(grad_hidden)
        grad_bias_h = np.sum(grad_hidden, axis=0)

        # derivative of loss function w.r.t weights and biases in Out layer
        grad_weights_o = self.RNNLayer.prev.T.dot(grad_output)
        grad_bias_o = np.sum(grad_output, axis=0)

        self.grad_sum_weights_xh += grad_weights_xh
        self.grad_sum_weights_hh += grad_weights_hh
        self.grad_sum_bias_h += grad_bias_h
        self.grad_sum_weights_o += grad_weights_o
        self.grad_sum_bias_o += grad_bias_o



    def update_gradients(self):
        # update weights & biases using accumulated gradients
        self.RNNLayer.weights_xh -= self.lr * self.grad_sum_weights_xh
        self.RNNLayer.weights_hh -= self.lr * self.grad_sum_weights_hh
        self.RNNLayer.bias -= self.lr * self.grad_sum_bias_h

        self.OutLayer.weights -= self.lr * self.grad_sum_weights_o
        self.OutLayer.bias -= self.lr * self.grad_sum_bias_o

        # reset gradients after each update
        self.grad_sum_weights_xh.fill(0)
        self.grad_sum_weights_hh.fill(0)
        self.grad_sum_bias_h.fill(0)
        self.grad_sum_weights_o.fill(0)
        self.grad_sum_bias_o.fill(0)


class RNNLayer:
    # TANH activation function
    def __init__(self, inputs_per_sample, hidden_size, batch_size):
        # Xavier (Glorot) Weight Initialization
        self.weights_xh = np.random.randn(inputs_per_sample, hidden_size) * np.sqrt(2. / (inputs_per_sample + hidden_size))
        self.weights_hh = np.random.randn(hidden_size, hidden_size) * np.sqrt(2. / (hidden_size + hidden_size))
        self.bias = np.zeros((1, hidden_size))
        self.prev = np.zeros((batch_size, hidden_size))
    def forward(self, input):
        f1 = np.add(np.matmul(input, self.weights_xh), np.matmul(self.prev, self.weights_hh))
        f1 = np.add(f1, self.bias)
        f2 = np.tanh(f1)
        self.prev = f2
        # Output of RNNLayer is batch_size x hidden_size
        return f2

    def zero_hidden(self):
        self.prev = np.zeros((batch_size, hidden_size))

class OutLayer:
    # Sigmoid activation function
    def __init__(self, hidden_size, output_size):
        # Xavier (Glorot) Weight Initialization
        self.weights = np.random.randn(hidden_size, output_size) * np.sqrt(2. / (hidden_size + output_size))
        self.bias = np.zeros((output_size, 1))

    def forward(self, input):
        f1 = np.add(np.matmul(input, self.weights), self.bias)
        f2 = sigmoid(f1)
        # Output of OutLayer is batch_size x 1
        return f2

# Convert sentence to tokens
def tokenize(a):
    # Convert to lowercase and remove punctuation
    a = [b.lower().translate(str.maketrans('', '', string.punctuation)) for b in a]
    # Tokenize
    tokens = [b.split() for b in a]
    return tokens

def build_vocab(tokenized):
    # Flatten the list of reviews and create a set of unique words
    vocabulary = set(word for a in tokenized for word in a)
    # Start indexing from 1, as 0 is typically reserved for padding
    word_to_int = {word: i+1 for i, word in enumerate(vocabulary)}
    return word_to_int

def encode(tokenized, word_to_int):
    encoded = []
    for review in tokenized:
        encoded.append([word_to_int[word] for word in review])
    return encoded

def pad(encoded, max_sequence_length):
    padded_sequences = []
    for sequence in encoded:
        # Calculate the number of zeros needed for padding
        num_padding = max_sequence_length - len(sequence)

        # If the current sequence is shorter, append the necessary zeros
        if num_padding > 0:
            padded_sequence = sequence + [0] * num_padding
        else:
            padded_sequence = sequence

        # Debugging line to verify that all sequences are of the correct length
        assert len(padded_sequence) == max_sequence_length, f"Sequence length: {len(padded_sequence)}, expected: {max_sequence_length}"

        # Add the padded (or original, if no padding was needed) sequence to the new list
        padded_sequences.append(padded_sequence)



    return padded_sequences

def get_test_data(max_sequence_length):
    imdb = load_dataset("imdb")
    test_data = imdb["test"]
    test_reviews = test_data["text"]
    test_labels = test_data["label"]

    tokenized_reviews = tokenize(test_reviews)
    # Must sort reviews based on length to trim the longest 5% and to assign a max_sequence_length
    sorted_indices = sorted(range(len(tokenized_reviews)), key=lambda x: len(tokenized_reviews[x]), reverse=True)
    sorted_reviews = [tokenized_reviews[i] for i in sorted_indices]
    sorted_labels = [test_labels[i] for i in sorted_indices]

    trim_index = int(len(sorted_reviews) * .05)
    sorted_reviews = sorted_reviews[trim_index:]
    sorted_labels = sorted_labels[trim_index:]  # Make sure to trim labels as well

    word_to_int = build_vocab(sorted_reviews)
    encoded_reviews = encode(sorted_reviews, word_to_int)

    padded = pad(encoded_reviews, max_sequence_length)
    padded = padded[:len(padded) - (len(padded) % batch_size)]

    # Must shuffle now for batching
    combined = list(zip(padded, sorted_labels))
    random.shuffle(combined)
    padded, labels = zip(*combined)

    return padded, test_labels

if __name__ == "__main__":
    imdb = load_dataset("imdb")
    train_data = imdb["train"]
    test_data = imdb["test"]
    reviews = train_data["text"]
    labels = train_data["label"]

    hidden_size = 256
    output_size = 1
    steps_per_epoch = 64
    lr = 0.01
    batch_size = 64
    epochs = 1

    tokenized_reviews = tokenize(reviews)
    # Must sort reviews based on length to trim the longest 5% and to assign a max_sequence_length
    sorted_indices = sorted(range(len(tokenized_reviews)), key=lambda x: len(tokenized_reviews[x]), reverse=True)
    sorted_reviews = [tokenized_reviews[i] for i in sorted_indices]
    sorted_labels = [labels[i] for i in sorted_indices]

    trim_index = int(len(sorted_reviews) * .05)
    sorted_reviews = sorted_reviews[trim_index:]
    sorted_labels = sorted_labels[trim_index:]  # Make sure to trim labels as well
    max_sequence_length = len(sorted_reviews[0])

    word_to_int = build_vocab(sorted_reviews)
    encoded_reviews = encode(sorted_reviews, word_to_int)

    padded = pad(encoded_reviews, max_sequence_length)
    padded = padded[:len(padded) - (len(padded) % batch_size)]

    # Must shuffle now for batching
    combined = list(zip(padded, sorted_labels))
    random.shuffle(combined)
    padded, labels = zip(*combined)

    rnn = RNN(max_sequence_length, hidden_size, output_size, lr)


    for i in range(epochs):
        bptt_loss = 0.0
        for j in range(steps_per_epoch):
            batch_data, batch_labels = rnn.get_next_batch(padded, labels, batch_size)

            preds = rnn.forward(batch_data)
            loss = crossentropy_loss(preds, batch_labels)
            bptt_loss += loss
            rnn.compute_gradients()
        # Zero the hidden state for the RNN layer
        rnn.RNNLayer.zero_hidden()
        rnn.update_gradients()
        # Must shuffle data after every epoch
        combined = list(zip(padded, labels))
        random.shuffle(combined)
        padded, labels = zip(*combined)

    test_data, test_labels = get_test_data(max_sequence_length)
    test_data, test_labels = rnn.get_next_batch(test_data, test_labels, batch_size)
    print(test_data.shape)
    test_preds = rnn.forward(test_data)

    test_accuracy = calculate_accuracy(test_preds, test_labels)
    print(f"Test Accuracy: {test_accuracy}")




