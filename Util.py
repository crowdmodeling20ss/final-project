from scipy.integrate import solve_ivp
import numpy as np
import torch
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def exercise_3_vector_field(t, x, alpha):
    """
    Hobf bifurcation for a=1

    Computes next state of given function based on the previous x value.

    :param t: One-dimensional independent variable (time)
    :param x: State of the function.
    :return: New state of the function.
    """
    v = [(alpha * x[0] - x[1] - x[0] * (x[0] ** 2 + x[1] ** 2)),
         (x[0] + alpha * x[1] - x[1] * (x[0] ** 2 + x[1] ** 2))]

    return v


def vector_field_ellipse(t, x, alpha):
    v = [-x[1], x[0] / 2]
    return v


def vector_field_spiral(t, x, alpha):
    v = [-0.1 * x[0] - 0.5 * x[1], 0.5 * x[0] - 0.1 * x[1]]
    return v


def generate_trajectory_by_ivp(start_point, fun=exercise_3_vector_field, alpha=None, duration=100):
    """

    :param start_point: Array [x, y]
    :param alpha:
    :param duration:
    :return: tensor (N, M)
    """
    t = 0
    if alpha is None:
        sol = solve_ivp(fun, [t, t + duration + 1], np.array(start_point))

        x_0 = sol.y[:, :-1].T
        x_1 = sol.y[:, 1:].T

        x_0_train = torch.from_numpy(x_0).float()
        x_1_train = torch.from_numpy(x_1).float()

        return torch.cat([x_0_train, x_1_train], dim=1)
    else:
        sol = solve_ivp(fun, [t, t + duration + 1], np.array(start_point), args=(alpha,))
        x_0 = sol.y[:, :-1].T
        x_1 = sol.y[:, 1:].T
        ax = np.array([[alpha] * len(x_0)]).T
        x_0_train = torch.from_numpy(np.append(x_0, ax, axis=1)).float()
        x_1_train = torch.from_numpy(np.append(x_1, ax, axis=1)).float()

        return torch.cat([x_0_train, x_1_train[:, :2]], dim=1)


def generate_trajectory_by_euler(start_point, fun=exercise_3_vector_field, alpha=-1, duration=100,
                                 time_step=0.001):
    if alpha is None:
        M=4
    else:
        M=5
    points = np.empty((1, M))
    time = 0
    p = np.array(start_point)

    while time < duration:
        if alpha is None:
            f = fun(time_step, x=p)
            x_0_alpha = p
        else:
            f = fun(time_step, x=p, alpha=alpha)
            x_0_alpha = np.append(p, alpha)
        x_1 = p + np.array(f) * time_step

        row = np.append(x_0_alpha, x_1).reshape(1, M)

        points = np.vstack((points, row))

        p = x_1
        time += time_step
    return torch.from_numpy(points).float()


def generate_trajectory_by_euler_netwok(model, start_point, alpha=-1, duration=100,
                                        time_step=0.001):
    points = np.empty((1, 5))
    time = 0
    p = np.array(start_point)

    torch_time_step = torch.tensor(time_step).to(device).float()

    while time < duration:
        x_0_alpha = np.append(p, alpha)
        x_1 = model(torch.from_numpy(np.array(x_0_alpha).reshape(1, 3)).float(), torch_time_step).detach().numpy()

        row = np.append(x_0_alpha, x_1).reshape(1, 5)

        points = np.vstack((points, row))

        p = x_1
        time += time_step
    return torch.from_numpy(points).float()


def generate_trajectory_by_rung4(start_point, fun=exercise_3_vector_field, alpha=-1, duration=100,
                                 time_step=0.001):
    points = np.empty((1, 5))
    time = 0
    p = np.array(start_point)

    while time < duration:
        f = fun(time_step, x=p, alpha=alpha)
        k1 = np.array(f) * time_step

        f = fun(time_step, x=(p + k1 / 2), alpha=alpha)
        k2 = np.array(f) * time_step

        f = fun(time_step, x=(p + k2 / 2), alpha=alpha)
        k3 = np.array(f) * time_step

        f = fun(time_step, x=(p + k3), alpha=alpha)
        k4 = np.array(f) * time_step

        x_1 = p + k1 / 6 + k2 / 3 + k3 / 3 + k4 / 6

        x_0_alpha = np.append(p, alpha)
        row = np.append(x_0_alpha, x_1).reshape(1, 5)

        points = np.vstack((points, row))

        p = x_1
        time += time_step
    return torch.from_numpy(points).float()


def generate_trajectory_by_rung4_network(model, start_point, alpha=-1, duration=100,
                                         time_step=0.001):
    points = np.empty((1, 5))
    time = 0
    p = np.array(start_point)
    torch_time_step = torch.tensor(time_step).to(device).float()

    while time < duration:
        x_0_alpha = np.append(p, alpha)
        f = model(torch.from_numpy(x_0_alpha.reshape(1, 3)).float(), torch_time_step).detach().numpy()
        k1 = f * time_step

        f = model(torch.from_numpy((x_0_alpha + k1 / 2).reshape(1, 3)).float(), torch_time_step).detach().numpy()
        k2 = f * time_step

        f = model(torch.from_numpy((x_0_alpha + k2 / 2).reshape(1, 3)).float(), torch_time_step).detach().numpy()
        k3 = np.array(f) * time_step

        f = model(torch.from_numpy((x_0_alpha + k3).reshape(1, 3)).float(), torch_time_step).detach().numpy()
        k4 = np.array(f) * time_step

        x_1 = p + k1 / 6 + k2 / 3 + k3 / 3 + k4 / 6

        x_0_alpha = np.append(p, alpha)
        row = np.append(x_0_alpha, x_1).reshape(1, 5)

        points = np.vstack((points, row))

        p = x_1
        time += time_step
    return torch.from_numpy(points).float()


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
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])


def get_predictions(alpha, model):
    x = np.arange(-2, 2, 0.01)
    x1, x2 = np.meshgrid(x, x)
    X_0 = np.vstack((x1.flatten(), x2.flatten())).T

    A = np.array([[alpha] * len(X_0)]).T
    X = np.append(X_0, A, axis=1)
    X = torch.from_numpy(X).float().to(device)  # use np.column_stack([X_0, alpha]) instead

    dt = torch.tensor(0.1).to(device).float()
    y_pred = model(X, dt).cpu().detach().numpy()
    y1 = y_pred[:, 0].reshape(x1.shape)
    y2 = y_pred[:, 1].reshape(x2.shape)

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
