import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RungeKuttaCompact(nn.Module):
    def __init__(self, input_size=3, output_size=2, hidden_size=64):
        """

        :param input_size:
        :param output_size:
        :param hidden_size:
        """

        super(RungeKuttaCompact, self).__init__()
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

        :param X: 3 input neurons (a two dimensional state(x_0) and a single operating parameter(alpha))
        :param dt: time step
        :return:
        """

        # Red box
        def red_box(X):
            f = self.linear_input(X)
            f = self.linear_hidden1(F.relu(f))
            f = self.linear_hidden2(F.relu(f))
            f = self.linear_hidden3(F.relu(f))
            f = self.linear_hidden4(F.relu(f))
            f = self.linear_output(f)

            return f

        y_pred = red_box(X)  # NxM => NxD
        h = (torch.from_numpy(np.ones_like(y_pred.shape)) * dt).to(device)

        k1 = torch.zeros_like(X).to(device)  # to add k1 onto x, they need to be same size though
        k1[:, :y_pred.shape[1]] = y_pred * h
        # Add it to x and get another prediction, this time for k2
        y_pred = red_box(X + k1 / 2)  # NxM => NxD
        k2 = torch.zeros_like(X).to(device)  # to add k1 onto x, they need to be same size though
        k2[:, :y_pred.shape[1]] = y_pred * h
        # k3:
        y_pred = red_box(X + k2 / 2)  # NxM => NxD
        k3 = torch.zeros_like(X).to(device)  # to add k1 onto x, they need to be same size though
        k3[:, :y_pred.shape[1]] = y_pred * h
        # k4:
        y_pred = red_box(X + k3).to(device)  # NxM => NxD
        k4 = torch.zeros_like(X)  # to add k1 onto x, they need to be same size though
        k4[:, :y_pred.shape[1]] = y_pred * h

        x_1 = X[:, :self.D] + 1 / 6 * (k1[:, :self.D] + 2 * k2[:, :self.D] + 2 * k3[:, :self.D] + k4[:, :self.D])

        return x_1
