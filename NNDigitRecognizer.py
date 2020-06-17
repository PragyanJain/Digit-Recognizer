from importlib import reload
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data.sampler import SubsetRandomSampler
from torch.autograd import Variable
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import pandas

WIDTH = 28

class Data:
    def __init__(self, values, labels):
        self.values = values
        self.labels = labels

def csv_to_matrix(path):
    data = pandas.read_csv(path, delimiter = ',')
    mat = data.to_numpy()
    mat = mat.astype(np.float32)
    values = mat[:, 1:]
    labels = mat[:, 0:1].astype(np.int64)
    labels = labels.T[0]
    values = values.reshape(data.shape[0], 1, WIDTH, WIDTH)
    values = torch.from_numpy(values)
    return Data(values, labels)

class BigNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size = 5, stride = 1, padding = 2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace = True),
            nn.Conv2d(16, 16, kernel_size = 5, stride = 1, padding = 2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.Conv2d(16, 32, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace = True),
            nn.Conv2d(32, 32, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 2, stride = 2)
        )

        self.linear_layers = nn.Sequential(
            nn.Linear(32 * 7 * 7, 32 * 7 * 7),
            nn.Linear(32 * 7 * 7, 10)
        )

    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(-1, 32 * 7 * 7)
        x = self.linear_layers(x)
        return x

class SmallNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn_layers = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size = 5, stride = 1, padding = 2),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.Conv2d(4, 8, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 2, stride = 2)
        )

        self.linear_layers = nn.Sequential(
            nn.Linear(8 * 7 * 7, 8 * 7 * 7),
            nn.Linear(8 * 7 * 7, 10)
        )

    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(-1, 8 * 7 * 7)
        x = self.linear_layers(x)
        return x

def plot_curve(rates):
    train_rates = rates[0]
    val_rates = rates[1]
    plt.title("Train vs Validation Success Rate")
    n = len(train_rates) # number of epochs
    plt.plot(range(1,n+1), train_rates, label="Train")
    plt.plot(range(1,n+1), val_rates, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Success Rate")
    plt.legend(loc='best')
    plt.show()

def evaluate(net, data):
    x = data.values
    out = net(x)
    out = out.detach().numpy()
    guess = np.apply_along_axis(np.argmax, 1, out)
    return success_rate(guess, data)

def success_rate(output, data):
    right = np.where(output == data.labels, 1, 0)
    return sum(right) / len(data.labels)

def train_net(net, n_epochs, learning_rate, data_train = None, data_val = None):
    if data_train is None:
        data_train = csv_to_matrix('train.csv')
    if data_val is None:
        data_val = data_train
    x_train = data_train.values
    y_train = torch.from_numpy(data_train.labels)
    x_val = data_val.values
    y_val = torch.from_numpy(data_val.labels)

    #net = net.float()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)

    #Use GPU if available
    if torch.cuda.is_available():
        net = net.cuda()
        criterion = criterion.cuda()

    train_rates = []
    val_rates = []
    for epoch in range(n_epochs):
        net.train()

        x_train_v, y_train_v = Variable(x_train), Variable(y_train)
        x_val_v, y_val_v = Variable(x_val), Variable(y_val)

        if torch.cuda.is_available():
            x_train_v, y_train_v = x_train_v.cuda(), y_train_v.cuda()
            x_val_v, y_val_v = x_val_v.cuda(), y_val_v.cuda()

        optimizer.zero_grad()

        output_train = net(x_train_v)
        loss_train = criterion(output_train, y_train_v)

        loss_train.backward()
        optimizer.step()

        output_train = np.apply_along_axis(np.argmax, 1, output_train.detach().numpy())
        output_val = net(x_val_v)
        output_val = np.apply_along_axis(np.argmax, 1, output_val.detach().numpy())

        train_rates.append(success_rate(output_train, data_train))
        val_rates.append(success_rate(output_val, data_val))

        print('Finished epoch {}'.format(epoch))
    return train_rates, val_rates
