import matplotlib.pyplot as plt
from Util import *

device_str = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_str)


def generate_data_from_2d_space(num_points, duration=100, show_plot=True):
    x = np.random.uniform(-2, 2, num_points)
    y = np.random.uniform(-2, 2, num_points)
    alphas = np.arange(-2, 2.1, 0.1)

    if show_plot:
        fig, ax = plt.subplots()
        ax.scatter(x, y)
        ax.set_title("{} points are generated.".format(num_points))
        plt.show()

    point_num = 0

    DATAPOINTS = None

    for alpha in alphas:
        for i, j in tuple(zip(x, y)):
            trajectory = generate_trajectory_by_ivp([i, j], alpha, duration)
            if DATAPOINTS is None:
                DATAPOINTS = trajectory
            else:
                DATAPOINTS = torch.cat([DATAPOINTS, trajectory], dim=0)

        print("({}) points are generated for alpha value ({})".format(len(DATAPOINTS) - point_num, alpha))
        point_num = len(DATAPOINTS)

    return DATAPOINTS


def get_data(create=False):
    if create == True:
        # LOAD FROM DATASET. data is generated in cuda from. use "map_location=torch.device('cpu')" if you want to load into cpu
        '''
          alphas are between -2 and 2 in the generated dataset.
        '''
        BIFURCATION_DATASET = generate_data_from_2d_space(75).to(
            device)  # torch.load('data_tensor.pt', map_location=torch.device('cpu')) #
        # BIFURCATION_DATASET = BIFURCATION_DATASET.to('cpu')

        # Save the dataset into a file so that you dont have to generate it again:
        torch.save(BIFURCATION_DATASET, "data_tensor.pt")
    else:
        BIFURCATION_DATASET = torch.load("data_tensor.pt", map_location=torch.device(device_str))
    return BIFURCATION_DATASET
