import torch

a = torch.nn.LSTM(4, hidden_size=3)
x = torch.rand((2, 5, 4))

y = a(x)

for i in y:
    print(i.shape)