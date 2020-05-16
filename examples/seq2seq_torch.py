import numpy as np
import torch
import torch.nn as nn

# params

batch_size = 64
epochs = 20
latent_dim = 256
num_samples = 10000
data_path = 'datasets/fra.txt'

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
# decoder_target_data = np.zeros(
#     (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
#     dtype='float32')

decoder_target_data = np.ones(
    (len(input_texts), max_decoder_seq_length),
    dtype='float32') * target_token_index[" "]

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
            # decoder_target_data[i, t - 1, target_token_index[char]] = 1.
            decoder_target_data[i, t - 1] = target_token_index[char]
    decoder_input_data[i, t + 1:, target_token_index[' ']] = 1.
    # decoder_target_data[i, t:, target_token_index[' ']] = 1.


# 数据gen
class Gen:
    def __init__(self, encoder_input, decoder_input, decoder_target, batch_size):
        self.batch_size = batch_size
        self.x1 = encoder_input
        self.x2 = decoder_input
        self.y = decoder_target
        self.inds = np.arange(encoder_input.shape[0])
        self.end_epoch()

    def next_batch(self, ind: int):
        left = ind * self.batch_size
        right = (ind + 1) * self.batch_size
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


class TrainModel(nn.Module):

    def __init__(self):
        super(TrainModel, self).__init__()
        self.lstm1 = nn.LSTM(num_encoder_tokens, latent_dim)
        self.lstm2 = nn.LSTM(num_decoder_tokens, latent_dim)
        self.dense = nn.Linear(latent_dim, num_decoder_tokens)
        self.dp = nn.Dropout(0.1)

    def forward(self, x1, x2):
        _, (h_last, s_last) = self.lstm1(x1)
        y, (_, _) = self.lstm2(x2, (h_last, s_last))
        # y: [len, batch, dim]
        y = y.transpose(0, 1) # [batch, len, dim]
        y = self.dense(y) # [batch, len, dim2]
        # y = self.dp(y)
        return y


net = TrainModel()
net.cuda()
print(net)

validation_split = 0.1
num_val = int(encoder_input_data.shape[0] * validation_split)
num_train = encoder_input_data.shape[0] - num_val
gen_train = Gen(encoder_input_data[:num_train], decoder_input_data[:num_train], decoder_target_data[:num_train],
                batch_size=batch_size)
gen_val = Gen(encoder_input_data[num_train:], decoder_input_data[num_train:], decoder_target_data[num_train:],
              batch_size=batch_size)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.RMSprop(net.parameters(), lr=0.001, alpha=0.9, eps=1e-07)
line_len = torch.Tensor([max_decoder_seq_length]).cuda()


import math
# class custom_loss:
#     def __call__(self, y_pred, y_true):
#         # y_pred [batch, class, len]
#         # y_true [batch, len] -> [batch, class, len]
#         shape = y_true.shape
#         y_true = y_true.view(shape[0], 1, shape[1])
#         y_true_oh = torch.FloatTensor(y_pred.shape).cuda()
#         y_true_oh.zero_()
#         y_true_oh.scatter_(1, y_true, 1)
#
#         exp = torch.exp(y_pred - torch.max(y_pred, dim=1, keepdim=True)[0])
#
#         y_pred_softmax = exp / torch.sum(exp, dim=1, keepdim=True)
#         y_pred_softmax.clamp_(1e-12, 1.0 - 1e-12)
#         y_pred_log = torch.log(y_pred_softmax)
#         #
#         shape = y_pred.shape
#         l = torch.sum(- y_true_oh * y_pred_log) / (shape[0] * shape[2])
#         return l
#
# criterion = custom_loss()


import math

class custom_loss:
    def __call__(self, y_pred, y_true):
        # y_pred [batch, class, len]
        # y_true [batch, len] -> [batch, class, len]
        shape = y_true.shape
        y_true = y_true.view(shape[0], 1, shape[1])

        left = - torch.sum(torch.gather(y_pred, 1, y_true))

        right_batch = torch.sum(torch.exp(y_pred), dim=1)
        right_log_batch = torch.log(right_batch)
        right = torch.sum(right_log_batch)

        if math.isnan(right):
            print("aaaaa")

        shape = y_pred.shape

        l = (left + right) / (shape[0] * shape[2])
        return l

criterion = custom_loss()


for i in range(epochs):
    for j in range(len(gen_train)):
        (x1, x2), y = gen_train.next_batch(j)
        x1 = torch.Tensor(x1).cuda().transpose(0, 1)
        x2 = torch.Tensor(x2).cuda().transpose(0, 1)
        y = torch.Tensor(y).cuda().long().transpose(0, 1)

        y_pred = net(x1, x2) # [batch, len, dim]

        y_pred_ = y_pred.transpose(1, 2) # [batch, dim, len]
        y = y.transpose(0, 1) # [batch, len]
        loss = criterion(y_pred_, y)
        loss.backward()
        optimizer.step()

        if j % 10 == 0:
            total_num = np.prod(y.shape)
            y_pred_char = torch.argmax(torch.softmax(y_pred, dim=2), dim=-1, keepdim=False)  # [batch, len]
            right_char = y_pred_char == y  # [batch, len]
            batch_right = torch.sum(right_char, dim=-1, keepdim=False) == line_len
            num_batch_right = torch.sum(batch_right).cpu().detach().numpy()
            acc = num_batch_right / y_pred.shape[0]
            l_ = loss.cpu().detach().numpy()
            print("%d/%d: loss %.05f acc %.03f" % (j, len(gen_train), l_, acc))

    # val
    total_right = 0
    total_num = 0
    for k in range(len(gen_val)):
        (x1, x2), y = gen_val.next_batch(k)
        x1 = torch.Tensor(x1).cuda().transpose(0, 1)
        x2 = torch.Tensor(x2).cuda().transpose(0, 1)
        y = torch.Tensor(y).cuda().long().transpose(0, 1)
        y_pred = net(x1, x2)

        y_pred_ = y_pred.transpose(1, 2)  # [batch, dim, len]
        y = y.transpose(0, 1)  # [batch, len]

        loss = criterion(y_pred_, y)

        y_pred_char = torch.argmax(torch.softmax(y_pred, dim=2), dim=-1, keepdim=False)  # [batch, len]
        right_char = y_pred_char == y  # [batch, len]
        batch_right = torch.sum(right_char, dim=-1, keepdim=False) == line_len
        total_right += torch.sum(batch_right).cpu().detach().numpy()
        total_num += y_pred.shape[0]

    acc = total_right / float(total_num)
    print("batch %d/%d: acc %.03f  --------------" % (i+1, epochs, acc))
