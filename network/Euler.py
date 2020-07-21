import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Eueler(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, activation='relu'):
        """

        :param input_size:
        :param hidden_sizes: Number of neurons per layer
        :param output_size:
        :param activation:
        """
        super(Eueler, self).__init__()
        self.M = input_size
        self.D = output_size

        activations = nn.ModuleDict([
            ['lrelu', nn.LeakyReLU()],
            ['relu', nn.ReLU()]
        ])

        self.neuron_sizes = [self.M, *hidden_sizes]
        self.hidden_layers = nn.Sequential(*[
            nn.Sequential(
                nn.Linear(in_f, out_f, bias=True),
                activations[activation]
            )
            for in_f, out_f in zip(self.neuron_sizes, self.neuron_sizes[1:])])
        self.linear_output = nn.Linear(hidden_sizes[-1], self.D, bias=True)

    def forward(self, X, dt):
        """
        param X: (N,M) Initial points with corresponding operating parameters
        param t: Delta(t) time step
        """
        # red-box
        f = self.hidden_layers(X)
        f = self.linear_output(f)

        # green-box
        x_1 = X[:, :self.D] + f * (torch.from_numpy(np.ones_like(f.shape)) * dt).to(device)

        return x_1, f
