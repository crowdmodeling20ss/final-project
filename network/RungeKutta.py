import torch
import torch.nn as nn
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RungeKutta(nn.Module):
    def __init__(self, input_size=3, hidden_sizes=[64], output_size=2, activation='relu'):
        """
        :param input_size:
        :param hidden_sizes: Number of neurons per layer
        :param output_size:
        :param activation:
        """

        super(RungeKutta, self).__init__()
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
        def frwd(X, dt):
            f = self.hidden_layers(X)
            f = self.linear_output(f)

            f = f * (torch.from_numpy(np.ones_like(f.shape)) * dt).to(device)

            return f

        # find k1,
        y_pred = frwd(X, dt)  # NxM => NxD
        k1 = torch.zeros_like(X)  # to add k1 onto x, they need to be same size though
        k1[:, :y_pred.shape[1]] = y_pred
        # Add it to x and get another prediction, this time for k2
        y_pred = frwd(X + k1 / 2, dt)  # NxM => NxD
        k2 = torch.zeros_like(X)  # to add k1 onto x, they need to be same size though
        k2[:, :y_pred.shape[1]] = y_pred
        # k3:
        y_pred = frwd(X + k2 / 2, dt)  # NxM => NxD
        k3 = torch.zeros_like(X)  # to add k1 onto x, they need to be same size though
        k3[:, :y_pred.shape[1]] = y_pred
        # k4:
        y_pred = frwd(X + k3, dt)  # NxM => NxD
        k4 = torch.zeros_like(X)  # to add k1 onto x, they need to be same size though
        k4[:, :y_pred.shape[1]] = y_pred

        prediction = X + 1 / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        return prediction[:, :y_pred.shape[1]], frwd(X, dt)
