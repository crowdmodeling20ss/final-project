import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from network.Euler import Eueler
from network.EulerVector import EulerVector
from network.RungeKutta import RungeKutta
from network.RungeKuttaCompact import RungeKuttaCompact

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def euler_model(input_size=3, output_size=2, hidden_size=128, vector_field=False):
    if vector_field:
        model = EulerVector(input_size=input_size, output_size=output_size, hidden_size=hidden_size).to(device)
    else:
        model = Eueler(input_size=input_size, output_size=output_size, hidden_size=hidden_size).to(device)
    loss_fn = nn.MSELoss()
    learning_rate = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    return model, loss_fn, optimizer


def rungekutta_model(input_size=3, output_size=2, hidden_size=128):
    model = RungeKutta(input_size=input_size, output_size=output_size, hidden_size=hidden_size).to(device)
    loss_fn = nn.MSELoss()
    learning_rate = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    return model, loss_fn, optimizer


def rungekutta_model_compact(input_size=3, output_size=2, hidden_size=128):
    model = RungeKuttaCompact(input_size=input_size, output_size=output_size, hidden_size=hidden_size).to(device)
    loss_fn = nn.MSELoss()
    learning_rate = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    return model, loss_fn, optimizer


"""
FULL TRAINING PROCEDURE FOR EULER
ASSUME WE HAVE `BIFURCATION_DATASET`
ALL OTHER OPERATIONS WILL BE DONE IN THIS BLOCK FOR TRAINING
"""


def train_euler(BIFURCATION_DATASET, epochs=1000, H=128, vector_field=False):
    import time
    print("BIFURCATION_DATASET.shape {}".format(BIFURCATION_DATASET.shape))
    N = len(BIFURCATION_DATASET)  # Size of dataset

    M = 3  # input_size
    D = 2  # output_size
    H = H  # hidden_size
    model, loss_fn, optimizer = euler_model(M, D, H,
                                            vector_field)  # defaults: input_size=3, output_size=2, hidden_size=128

    num_train, num_val, num_test = int(N * 0.6), int(N * 0.3), int(N * 0.1)

    np.random.seed(0)
    indices = np.random.permutation(N)
    train_idx, val_idx, test_idx = indices[:num_train], indices[num_train:num_train + num_val], indices[
                                                                                                num_train + num_val:]
    # print("BIFURCATION_DATASET.shape {}".format(BIFURCATION_DATASET.shape))
    batch_size = 32
    batch_num = np.math.floor(N / batch_size)
    # print("train_idx {}".format(train_idx))
    batched_train_idxs = np.array_split(train_idx, batch_num)
    # print("batched_train_idxs.shape {}".format(batched_train_idxs))
    X = BIFURCATION_DATASET[:, :M]
    X_1 = BIFURCATION_DATASET[:, M:]

    dt = torch.tensor(0.1).to(device).float()
    for t in range(epochs):
        train_epoch_loss = 0.0
        seconds = time.time()
        for batch_idx in batched_train_idxs:
            x_0_with_alpha = X[batch_idx].float()
            x_1 = X_1[batch_idx].float()

            y_pred = model(x_0_with_alpha, dt)  # NxM => NxD

            # Compute and print loss.
            loss = loss_fn(y_pred, x_1)
            train_epoch_loss += loss

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()
        train_epoch_loss /= len(train_idx)
        validation_loss = loss_fn(model(X[val_idx].float(), dt), X_1[val_idx].float())
        print('(Epoch %d / %d, seconds: %d) train loss: %f validation loss: %f' % (
            t + 1, epochs, (time.time() - seconds), train_epoch_loss, validation_loss))
    return model


"""
FULL TRAINING PROCEDURE FOR EULER
ASSUME WE HAVE `BIFURCATION_DATASET`
ALL OTHER OPERATIONS WILL BE DONE IN THIS BLOCK FOR TRAINING
"""


def train_rungekutta(BIFURCATION_DATASET, epochs=1000, H=128):
    import time
    print("BIFURCATION_DATASET.shape {}".format(BIFURCATION_DATASET.shape))
    N = len(BIFURCATION_DATASET)  # Size of dataset

    M = 3  # input_size
    D = 2  # output_size
    H = H  # hidden_size
    model, loss_fn, optimizer = rungekutta_model(M, D, H)  # defaults: input_size=3, output_size=2, hidden_size=128

    num_train, num_val, num_test = int(N * 0.6), int(N * 0.3), int(N * 0.1)

    np.random.seed(0)
    indices = np.random.permutation(N)
    train_idx, val_idx, test_idx = indices[:num_train], indices[num_train:num_train + num_val], indices[
                                                                                                num_train + num_val:]
    # print("BIFURCATION_DATASET.shape {}".format(BIFURCATION_DATASET.shape))
    batch_size = 32
    batch_num = np.math.floor(N / batch_size)
    # print("train_idx {}".format(train_idx))
    batched_train_idxs = np.array_split(train_idx, batch_num)
    # print("batched_train_idxs.shape {}".format(batched_train_idxs))
    X = BIFURCATION_DATASET[:, :M]
    X_1 = BIFURCATION_DATASET[:, M:]

    dt = torch.tensor(0.1).to(device).float()
    for t in range(epochs):
        train_epoch_loss = 0.0
        seconds = time.time()
        for batch_idx in batched_train_idxs:
            x_0_with_alpha = X[batch_idx].float()
            x_1 = X_1[batch_idx].float()

            # find k1,
            y_pred = model(x_0_with_alpha, dt)  # NxM => NxD
            k1 = torch.zeros_like(x_0_with_alpha)  # to add k1 onto x, they need to be same size though
            k1[:, :y_pred.shape[1]] = y_pred
            # Add it to x and get another prediction, this time for k2
            y_pred = model(x_0_with_alpha + k1, dt)  # NxM => NxD
            k2 = torch.zeros_like(x_0_with_alpha)  # to add k1 onto x, they need to be same size though
            k2[:, :y_pred.shape[1]] = y_pred
            # k3:
            y_pred = model(x_0_with_alpha + k2, dt)  # NxM => NxD
            k3 = torch.zeros_like(x_0_with_alpha)  # to add k1 onto x, they need to be same size though
            k3[:, :y_pred.shape[1]] = y_pred
            # k4:
            y_pred = model(x_0_with_alpha + k3, dt)  # NxM => NxD
            k4 = torch.zeros_like(x_0_with_alpha)  # to add k1 onto x, they need to be same size though
            k4[:, :y_pred.shape[1]] = y_pred

            prediction = x_0_with_alpha + 1 / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            # Compute and print loss.
            loss = loss_fn(prediction[:, :y_pred.shape[1]], x_1)
            train_epoch_loss += loss

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()
        train_epoch_loss /= len(train_idx)
        validation_loss = loss_fn(model(X[val_idx].float(), dt), X_1[val_idx].float())
        print('(Epoch %d / %d, seconds: %d) train loss: %f validation loss: %f' % (
            t + 1, epochs, (time.time() - seconds), train_epoch_loss, validation_loss))
    return model


def train_rungekutta_compact(BIFURCATION_DATASET, epochs=1000, H=128):
    import time
    print("BIFURCATION_DATASET.shape {}".format(BIFURCATION_DATASET.shape))
    N = len(BIFURCATION_DATASET)  # Size of dataset

    M = 3  # input_size
    D = 2  # output_size
    H = H  # hidden_size
    model, loss_fn, optimizer = rungekutta_model_compact(M, D, H)  # defaults: input_size=3, output_size=2, hidden_size=128

    num_train, num_val, num_test = int(N * 0.6), int(N * 0.3), int(N * 0.1)

    np.random.seed(0)
    indices = np.random.permutation(N)
    train_idx, val_idx, test_idx = indices[:num_train], indices[num_train:num_train + num_val], indices[
                                                                                                num_train + num_val:]
    # print("BIFURCATION_DATASET.shape {}".format(BIFURCATION_DATASET.shape))
    batch_size = 32
    batch_num = np.math.floor(N / batch_size)
    # print("train_idx {}".format(train_idx))
    batched_train_idxs = np.array_split(train_idx, batch_num)
    # print("batched_train_idxs.shape {}".format(batched_train_idxs))
    X = BIFURCATION_DATASET[:, :M]
    X_1 = BIFURCATION_DATASET[:, M:]

    dt = torch.tensor(0.1).to(device).float()
    for t in range(epochs):
        train_epoch_loss = 0.0
        seconds = time.time()
        for batch_idx in batched_train_idxs:
            x_0_with_alpha = X[batch_idx].float()
            x_1 = X_1[batch_idx].float()

            y_pred = model(x_0_with_alpha, dt)

            # Compute and print loss.
            loss = loss_fn(y_pred, x_1)
            train_epoch_loss += loss

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()
        train_epoch_loss /= len(train_idx)
        validation_loss = loss_fn(model(X[val_idx].float(), dt), X_1[val_idx].float())
        print('(Epoch %d / %d, seconds: %d) train loss: %f validation loss: %f' % (
            t + 1, epochs, (time.time() - seconds), train_epoch_loss, validation_loss))
    return model
