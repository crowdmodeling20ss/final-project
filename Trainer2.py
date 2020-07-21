import time

import torch
import torch.nn as nn
import numpy as np

from Util import plot_loss_curve

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model(DATASET, model, input_size=3, epochs=1000, learning_rate=1e-4, batch_size=32, dt=0.1):
    N = len(DATASET)
    M = input_size

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    num_train, num_val, num_test = int(N * 0.6), int(N * 0.3), int(N * 0.1)

    # Split data sets into train, validation and test
    np.random.seed(0)
    indices = np.random.permutation(N)
    train_idx, val_idx, test_idx = indices[:num_train], indices[num_train:num_train + num_val], indices[
                                                                                                num_train + num_val:]
    batch_num = np.math.floor(N / batch_size)
    batched_train_idxs = np.array_split(train_idx, batch_num)
    X_0 = DATASET[:, :M]  # Initial values
    X_1 = DATASET[:, M:]  # Ground truth values

    train_loss_arr = []
    val_loss_arr = []
    dt = torch.tensor(dt).to(device).float()
    for t in range(epochs):
        train_epoch_loss = 0.0
        seconds = time.time()
        for batch_idx in batched_train_idxs:
            x_0_with_alpha = X_0[batch_idx].float()
            x_1 = X_1[batch_idx].float()

            optimizer.zero_grad()
            x_1_pred, _ = model(x_0_with_alpha, dt)  # NxM => NxD

            # Compute and print loss.
            loss = loss_fn(x_1_pred, x_1)
            train_epoch_loss += loss
            loss.backward()
            optimizer.step()

        train_epoch_loss /= len(train_idx)
        train_loss_arr.append(train_epoch_loss)

        # Check validation loss
        model.eval()
        with torch.no_grad():
            batch_val_num = np.math.floor(N / batch_size)
            batched_val_idxs = np.array_split(val_idx, batch_val_num)
            validation_loss = 0.0
            for batch_idx in batched_val_idxs:
                x_1_pred, _ = model(X_0[batch_idx].float(), dt)
                loss = loss_fn(x_1_pred, X_1[batch_idx].float())
                validation_loss += loss
            validation_loss /= len(val_idx)
            val_loss_arr.append(validation_loss)
            print('(Epoch %d / %d, seconds: %d) train loss: %f validation loss: %f' % (
                t + 1, epochs, (time.time() - seconds), train_epoch_loss, validation_loss))
        model.train()

    plot_loss_curve(train_loss_arr, val_loss_arr)
    return model
