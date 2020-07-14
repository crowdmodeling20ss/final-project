import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Eueler(nn.Module):
    def __init__(self, input_size=3, output_size=2, hidden_size=64):
        """

        :param input_size:
        :param output_size:
        :param hidden_size:
        """
        super(Eueler, self).__init__()
        self.M = input_size
        self.D = output_size
        self.H = hidden_size

        self.linear_input = nn.Linear(self.M, self.H, bias=True)
        self.linear_hidden1 = nn.Linear(self.H, self.H, bias=True)
        self.linear_hidden2 = nn.Linear(self.H, self.H, bias=True)
        self.linear_hidden3 = nn.Linear(self.H, self.H, bias=True)
        self.linear_hidden4 = nn.Linear(self.H, self.H, bias=True)
        self.linear_output = nn.Linear(self.H, self.D, bias=True)

    def forward(self, X, dt):
        """
        param X: (N,M) Initial points with corresponding operating parameters
        param t: Delta(t)
        """
        # red-box
        f = self.linear_input(X)
        f = self.linear_hidden1(F.relu(f))
        f = self.linear_hidden2(F.relu(f))
        f = self.linear_hidden3(F.relu(f))
        f = self.linear_hidden4(F.relu(f))
        f = self.linear_output(f)

        # green-box
        x_1 = X[:, :self.D] + f * (torch.from_numpy(np.ones_like(f.shape)) * dt).to(device)

        return x_1
