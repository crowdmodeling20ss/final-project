import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from Util import *

device_str = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_str)


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


def vector_field_ellipse(t, x):
    v = [-x[1], x[0] / 2]
    return v


def vector_field_spiral(t, x):
    v = [-0.1 * x[0] - 0.5 * x[1], 0.5 * x[0] - 0.1 * x[1]]
    return v


def generate_trajectory_by_ivp(start_point, fun=exercise_3_vector_field, alpha=None, duration=100):
    """

    :param start_point: Array [x, y]
    :param alpha:
    :param duration:
    :return:
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


def generate_trajectory_by_euler(start_point, fun=exercise_3_vector_field, alpha=None, duration=100,
                                 time_step=0.001):
    if alpha is None:
        M = 4
    else:
        M = 5
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


def generate_trajectory_by_euler_netwok(model, start_point, alpha=None, duration=100,
                                        time_step=0.001):
    if alpha is None:
        M = 4
    else:
        M = 5
    points = np.empty((1, M))
    time = 0
    p = np.array(start_point)

    torch_time_step = torch.tensor(time_step).to(device).float()

    if alpha is None:
        while time < duration:
            x_0 = p
            x_1, _ = model(torch.from_numpy(np.array(x_0).reshape(1, 2)).float(), torch_time_step)
            x_1 = x_1.detach().numpy()

            row = np.append(x_0, x_1).reshape(1, 4)

            points = np.vstack((points, row))

            p = x_1
            time += time_step
    else:
        while time < duration:
            x_0_alpha = np.append(p, alpha)
            x_1, _ = model(torch.from_numpy(np.array(x_0_alpha).reshape(1, 3)).float(), torch_time_step)
            x_1 = x_1.detach().numpy()

            row = np.append(x_0_alpha, x_1).reshape(1, 5)

            points = np.vstack((points, row))

            p = x_1
            time += time_step
    return torch.from_numpy(points).float()


def generate_trajectory_by_rung4(start_point, fun=exercise_3_vector_field, alpha=None, duration=100,
                                 time_step=0.001):
    if alpha is None:
        M = 4
    else:
        M = 5
    points = np.empty((1, M))
    time = 0
    p = np.array(start_point)

    while time < duration:
        if alpha is None:
            f = fun(time_step, x=p)
            k1 = np.array(f) * time_step

            f = fun(time_step, x=(p + k1 / 2))
            k2 = np.array(f) * time_step

            f = fun(time_step, x=(p + k2 / 2))
            k3 = np.array(f) * time_step

            f = fun(time_step, x=(p + k3))
            k4 = np.array(f) * time_step
        else:
            f = fun(time_step, x=p, alpha=alpha)
            k1 = np.array(f) * time_step

            f = fun(time_step, x=(p + k1 / 2), alpha=alpha)
            k2 = np.array(f) * time_step

            f = fun(time_step, x=(p + k2 / 2), alpha=alpha)
            k3 = np.array(f) * time_step

            f = fun(time_step, x=(p + k3), alpha=alpha)
            k4 = np.array(f) * time_step

        x_1 = p + k1 / 6 + k2 / 3 + k3 / 3 + k4 / 6

        if alpha is None:
            x_0_alpha = p
        else:
            x_0_alpha = np.append(p, alpha)

        row = np.append(x_0_alpha, x_1).reshape(1, M)

        points = np.vstack((points, row))

        p = x_1
        time += time_step
    return torch.from_numpy(points).float()


# Since we embed integration into network, we don't need this function anymore.
def generate_trajectory_by_rung4_network(model, start_point, alpha=None, duration=100,
                                         time_step=0.001):
    if alpha is None:
        M = 4
    else:
        M = 5
    points = np.empty((1, M))
    time = 0
    p = np.array(start_point)
    torch_time_step = torch.tensor(time_step).to(device).float()

    if alpha is None:
        while time < duration:
            x_0 = p
            f = model(torch.from_numpy(x_0.reshape(1, 2)).float(), torch_time_step).detach().numpy()
            k1 = f * time_step

            f = model(torch.from_numpy((x_0 + k1 / 2).reshape(1, 2)).float(), torch_time_step).detach().numpy()
            k2 = f * time_step

            f = model(torch.from_numpy((x_0 + k2 / 2).reshape(1, 2)).float(), torch_time_step).detach().numpy()
            k3 = np.array(f) * time_step

            f = model(torch.from_numpy((x_0 + k3).reshape(1, 2)).float(), torch_time_step).detach().numpy()
            k4 = np.array(f) * time_step

            x_1 = p + k1 / 6 + k2 / 3 + k3 / 3 + k4 / 6

            x_0_alpha = np.append(p, alpha)
            row = np.append(x_0_alpha, x_1).reshape(1, M)

            points = np.vstack((points, row))

            p = x_1
            time += time_step
    else:
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


def generate_data_for_single_parameter_dynamical_system(num_points, fun=exercise_3_vector_field, duration=100,
                                                        show_plot=True):
    """
    The purpose of this function is generating trajectories for a dynamical system which given by 'fun' function.
    1. Generate random points as initial points and system parameter 'alpha' from a uniform distribution.
    2. Starting from generated initial points, generate trajectories using scipy.solve_ivp method for 'duration' seconds.
    3. Return data in the following form:
        (N, M)
        N : Total number of points generated in the trajectories
        D : Dimension of the state variables
        M : =(D*2 + 1) Combination of first 2 columns for initial state variables, 1 column for alpha values, and last
        2 columns for corresponding target values

    :param num_points: Number of points that will be generated for training
    :param fun: Explicit function of the dynamic system
    :param duration: Duration of integrating the expilict function with solve_ivp method.
    :param show_plot:
    :return:
    """
    x = np.random.uniform(-2, 2, num_points)
    y = np.random.uniform(-2, 2, num_points)
    alphas = np.arange(-2, 2.1, 0.05)

    if show_plot:
        fig, ax = plt.subplots()
        ax.scatter(x, y)
        ax.set_title("{} points are generated.".format(num_points))
        plt.show()

    point_num = 0

    DATAPOINTS = None

    for alpha in alphas:
        for i, j in tuple(zip(x, y)):
            trajectory = generate_trajectory_by_ivp([i, j], fun, alpha, duration)
            if DATAPOINTS is None:
                DATAPOINTS = trajectory
            else:
                DATAPOINTS = torch.cat([DATAPOINTS, trajectory], dim=0)

        print("({}) points are generated for alpha value ({})".format(len(DATAPOINTS) - point_num, alpha))
        point_num = len(DATAPOINTS)

    return DATAPOINTS


def generate_data_for_non_parameterized_dynamical_system(num_points, fun=vector_field_ellipse, duration=100,
                                                         show_plot=True):
    """
    The purpose of this function is generating trajectories for a dynamical system which given by 'fun' function.
    1. Generate random points as initial points
    2. Starting from generated initial points, generate trajectories using scipy.solve_ivp method for 'duration' seconds.
    3. Return data in the following form:
        (N, M)
        N : Total number of points generated in the trajectories
        D : Dimension of the state variables
        M : = (D*2) Combination of first 2 columns for initial state variables and last 2 columns for corresponding
        target values

    :param num_points: Number of points that will be generated for training
    :param fun: Explicit function of the dynamic system
    :param duration: Duration of integrating the explicit function with solve_ivp method.
    :param show_plot:
    :return:
    """
    x = np.random.uniform(-2, 2, num_points)
    y = np.random.uniform(-2, 2, num_points)

    if show_plot:
        fig, ax = plt.subplots()
        ax.scatter(x, y)
        ax.set_title("{} points are generated.".format(num_points))
        plt.show()

    DATAPOINTS = None

    for i, j in tuple(zip(x, y)):
        trajectory = generate_trajectory_by_ivp([i, j], fun, None, duration)
        if DATAPOINTS is None:
            DATAPOINTS = trajectory
        else:
            DATAPOINTS = torch.cat([DATAPOINTS, trajectory], dim=0)

    print("({}) points are generated".format(len(DATAPOINTS)))

    return DATAPOINTS


def get_data_for_single_parameter_dynamical_system(num_points, file, fun=exercise_3_vector_field, duration=100,
                                                   create=False):
    if create:
        data = generate_data_for_single_parameter_dynamical_system(num_points, fun, duration).to(device)
        torch.save(data, file)
    else:
        data = torch.load(file, map_location=torch.device(device_str))
    return data


def get_data_for_non_parameterized_dynamical_system(num_points, file, fun=vector_field_ellipse, duration=100,
                                                    create=False):
    if create:
        data = generate_data_for_non_parameterized_dynamical_system(num_points, fun, duration).to(device)
        torch.save(data, file)
    else:
        data = torch.load(file, map_location=torch.device(device_str))
    return data
