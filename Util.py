import numpy as np
import torch
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def visualize_trajectories(trajectories, point, alphas):
    fig, axs = plt.subplots(1, len(trajectories), figsize=(3 * len(alphas), 3))
    for i, trajectory in enumerate(trajectories):
        axs[i].scatter(trajectory[:, 0], trajectory[:, 1], linewidth=1, s=1, color="teal")
        axs[i].set_title('Point: {}, alpha: {}'.format(point, alphas[i]))
        axs[i].set_xlim([-2, 2])
        axs[i].set_ylim([-2, 2])
    plt.show()


def plot_phase_portrait(ax, pred, alpha):
    ax.streamplot(*pred, color='dodgerblue', linewidth=1)
    ax.set_title("alpha = {}".format(alpha))
    ax.set_xlim([-4, 4])
    ax.set_ylim([-4, 4])


def get_predictions(alpha, model):
    x = np.arange(-4, 4, 0.01)
    x1, x2 = np.meshgrid(x, x)
    X_0 = np.vstack((x1.flatten(), x2.flatten())).T

    A = np.array([[alpha] * len(X_0)]).T
    X = np.append(X_0, A, axis=1)
    X = torch.from_numpy(X).float().to(device)  # use np.column_stack([X_0, alpha]) instead

    dt = torch.tensor(0.1).to(device).float()
    _, f = model(X, dt).cpu().detach().numpy()
    y1 = f[:, 0].reshape(x1.shape)
    y2 = f[:, 1].reshape(x2.shape)

    return (x1, x2, y1, y2)


def get_predictions_runge_kutta(alpha, model):
    x = np.arange(-2, 2, 0.01)
    x1, x2 = np.meshgrid(x, x)
    X_0 = np.vstack((x1.flatten(), x2.flatten())).T

    A = np.array([[alpha] * len(X_0)]).T
    X = np.append(X_0, A, axis=1)
    X = torch.from_numpy(X).float().to(device)  # use np.column_stack([X_0, alpha]) instead

    dt = torch.tensor(0.1).to(device).float()

    # find k1,
    y_pred = model(X, dt)  # NxM => NxD
    k1 = torch.zeros_like(X)  # to add k1 onto x, they need to be same size though
    k1[:, :y_pred.shape[1]] = y_pred
    # Add it to x and get another prediction, this time for k2
    y_pred = model(X + k1, dt)  # NxM => NxD
    k2 = torch.zeros_like(X)  # to add k1 onto x, they need to be same size though
    k2[:, :y_pred.shape[1]] = y_pred
    # k3:
    y_pred = model(X + k2, dt)  # NxM => NxD
    k3 = torch.zeros_like(X)  # to add k1 onto x, they need to be same size though
    k3[:, :y_pred.shape[1]] = y_pred
    # k4:
    y_pred = model(X + k3, dt)  # NxM => NxD
    k4 = torch.zeros_like(X)  # to add k1 onto x, they need to be same size though
    k4[:, :y_pred.shape[1]] = y_pred

    prediction = X + 1 / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    # Compute and print loss.
    y_pred = prediction[:, :y_pred.shape[1]].cpu().detach().numpy()

    # y_pred = model(X, dt).cpu().detach().numpy()
    y1 = y_pred[:, 0].reshape(x1.shape)
    y2 = y_pred[:, 1].reshape(x2.shape)

    return (x1, x2, y1, y2)


def phase_check(model):
    # alphas = np.random.uniform(-2,2, 10)
    alphas = [-1.0, 0.0, 1.0, 2.0]
    fig, axs = plt.subplots(2, len(alphas), figsize=(16, 8))

    print("FIRST ROW: X_1 is direct output of the Euler Network")
    print("SECOND ROW: X_1 is (pred-x_0)/dt so it is vector field")

    for i, a in enumerate(alphas):
        pred = get_predictions(a, model)
        x1, x2, y1, y2 = pred
        new_pred = (x1, x2, (y1 - x1) / 0.1, (y2 - x2) / 0.1)
        plot_phase_portrait(axs[0][i], pred, a)
        plot_phase_portrait(axs[1][i], new_pred, a)

    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.show()


def phase_check_runge(model):
    # alphas = np.random.uniform(-2,2, 10)
    alphas = [-1.0, 0.0, 1.0, 2.0]
    fig, axs = plt.subplots(2, len(alphas), figsize=(16, 8))

    print("FIRST ROW: X_1 is direct output of the Euler Network")
    print("SECOND ROW: X_1 is (pred-x_0)/dt so it is vector field")

    for i, a in enumerate(alphas):
        pred = get_predictions_runge_kutta(a, model)
        x1, x2, y1, y2 = pred
        new_pred = (x1, x2, (y1 - x1) / 0.1, (y2 - x2) / 0.1)
        plot_phase_portrait(axs[0][i], pred, a)
        plot_phase_portrait(axs[1][i], new_pred, a)

    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.show()


def plot_loss_curve(train_loss_arr, val_loss_arr):
    plt.plot(train_loss_arr, color='blue')
    plt.plot(val_loss_arr, color='orange')
    plt.xlabel("Number of epoch")
    plt.ylabel("Loss")
    plt.show()
