
import numpy as np
from simpleNet import Moduel, losses, layers, optims

# params

batch_size = 64
epochs = 5
latent_dim = 256
num_samples = 10000
data_path = 'examples/datasets/fra.txt'

# handle data

input_texts = []
target_texts = []
input_characters = set()
target_characters = set()
with open(data_path, 'r', encoding='utf-8') as f:
    lines = f.read().split('\n')
for line in lines[: min(num_samples, len(lines) - 1)]:
    input_text, target_text, _ = line.split('\t')
    # We use "tab" as the "start sequence" character
    # for the targets, and "\n" as "end sequence" character.
    target_text = '\t' + target_text + '\n'
    input_texts.append(input_text)
    target_texts.append(target_text)
    for char in input_text:
        if char not in input_characters:
            input_characters.add(char)
    for char in target_text:
        if char not in target_characters:
            target_characters.add(char)

input_characters = sorted(list(input_characters))
target_characters = sorted(list(target_characters))
num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(target_characters)
max_encoder_seq_length = max([len(txt) for txt in input_texts])
max_decoder_seq_length = max([len(txt) for txt in target_texts])

print('Number of samples:', len(input_texts))
print('Number of unique input tokens:', num_encoder_tokens)
print('Number of unique output tokens:', num_decoder_tokens)
print('Max sequence length for inputs:', max_encoder_seq_length)
print('Max sequence length for outputs:', max_decoder_seq_length)


input_token_index = dict(
    [(char, i) for i, char in enumerate(input_characters)])
target_token_index = dict(
    [(char, i) for i, char in enumerate(target_characters)])

encoder_input_data = np.zeros(
    (len(input_texts), max_encoder_seq_length, num_encoder_tokens),
    dtype='float32')
decoder_input_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')
decoder_target_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')

for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    for t, char in enumerate(input_text):
        encoder_input_data[i, t, input_token_index[char]] = 1.
    encoder_input_data[i, t + 1:, input_token_index[' ']] = 1.
    for t, char in enumerate(target_text):
        # decoder_target_data is ahead of decoder_input_data by one timestep
        decoder_input_data[i, t, target_token_index[char]] = 1.
        if t > 0:
            # decoder_target_data will be ahead by one timestep
            # and will not include the start character.
            decoder_target_data[i, t - 1, target_token_index[char]] = 1.
    decoder_input_data[i, t + 1:, target_token_index[' ']] = 1.
    decoder_target_data[i, t:, target_token_index[' ']] = 1.


# 数据gen
class Gen:
    def __init__(self, encoder_input, decoder_input, decoder_target, batch_size):
        self.batch_size = batch_size
        self.x1 = encoder_input
        self.x2 = decoder_input
        self.y = decoder_target
        self.inds = np.arange(encoder_input.shape[0])
        self.end_epoch()

    def next_batch(self, ind:int):
        left = ind * self.batch_size
        right = (ind+1) * self.batch_size
        right = min(right, self.inds.shape[0])

        batch_inds = self.inds[left:right]

        return (self.x1[batch_inds], self.x2[batch_inds]), self.y[batch_inds]

    def __len__(self):
        import math
        return math.ceil(self.inds.shape[0] / self.batch_size)

    def end_epoch(self):
        np.random.shuffle(self.inds)

    def totol_num(self):
        return self.inds.shape[0]

(x1, x2), y = Gen(encoder_input_data, decoder_input_data, decoder_target_data, batch_size).next_batch(0)
print("x1: %s, x2: %s, y: %s" % (str(x1.shape), str(x2.shape), str(y.shape)))

# train model

encoder = layers.LSTM(num_encoder_tokens, latent_dim)
decoder = layers.LSTM(num_decoder_tokens, latent_dim)
dense = layers.Dense(latent_dim, num_decoder_tokens)

class TrainModel(Moduel):
    def __init__(self):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.dense = dense
        # self.softmax = layers.Softmax()


    def forwards(self, x1, x2):

        _, (h, s) = self.encoder(x1)
        y, (_, _) = self.decoder(x2, (h, s))
        y = self.dense(y)
        # y = self.softmax(y)

        return y

    def backwards(self, da):

        # da = self.softmax.backwards(da)
        da = self.dense.backwards(da)
        da, (dh0, ds0) = self.decoder.backwards(da, None)
        dx = self.encoder.backwards(None, (dh0, ds0))
        return dx

# train

net = TrainModel()
net.summary()
dataGen = Gen(encoder_input_data, decoder_input_data, decoder_target_data, 2)

(x1, x2), y = dataGen.next_batch(0)

from simpleNet.utils.grad_check import grad_check

grad_check(net, (x1, x2))