import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from scipy.spatial import distance_matrix

import variables as v
import wandb
from qfunction import QFunction
from qnet import QNet


def get_graph_mat(n=10, size=1):
    """ Throws n nodes uniformly at random on a square, and build a (fully connected) graph.
        Returns the (N, 2) coordinates matrix, and the (N, N) matrix containing pairwise euclidean distances.
    """
    coords = size * np.random.uniform(size=(n, 2))
    dist_mat = distance_matrix(coords, coords)
    return coords, dist_mat


def plot_graph(coords, mat):
    """ Utility function to plot the fully connected graph
    """
    n = len(coords)

    plt.scatter(coords[:, 0], coords[:, 1], s=[50 for _ in range(n)])
    for i in range(n):
        for j in range(n):
            if j < i:
                plt.plot([coords[i, 0], coords[j, 0]], [coords[i, 1], coords[j, 1]], 'b', alpha=0.7)


def state2tens(state):
    """ Creates a Pytorch tensor representing the history of visited nodes, from a (single) state tuple.

        Returns a (Nx5) tensor, where for each node we store whether this node is in the sequence,
        whether it is first or last, and its (x,y) coordinates.
    """
    solution = set(state.partial_solution)
    sol_last_node = state.partial_solution[-1] if len(state.partial_solution) > 0 else -1
    sol_first_node = state.partial_solution[0] if len(state.partial_solution) > 0 else -1
    coords = state.coords
    nr_nodes = coords.shape[0]

    xv = [[(1 if i in solution else 0),
           (1 if i == sol_first_node else 0),
           (1 if i == sol_last_node else 0),
           coords[i, 0],
           coords[i, 1]
           ] for i in range(nr_nodes)]

    return torch.tensor(xv, dtype=torch.float32, requires_grad=False, device=v.device)


def total_distance(solution, W):
    if len(solution) < 2:
        return 0  # there is no travel

    total_dist = 0
    for i in range(len(solution) - 1):
        total_dist += W[solution[i], solution[i + 1]].item()

    # if this solution is "complete", go back to initial point
    if len(solution) == W.shape[0]:
        total_dist += W[solution[-1], solution[0]].item()

    return total_dist


def is_state_final(state):
    return len(set(state.partial_solution)) == state.W.shape[0]


def get_next_neighbor_random(state):
    solution, W = state.partial_solution, state.W

    if len(solution) == 0:
        return random.choice(range(W.shape[0]))
    already_in = set(solution)
    candidates = list(filter(lambda n: n.item() not in already_in, W[solution[-1]].nonzero()))
    if len(candidates) == 0:
        return None
    return random.choice(candidates).item()


def init_model(fname=None):
    """ Create a new model. If fname is defined, load the model from the specified file.
    """
    Q_net = QNet(v.EMBEDDING_DIMENSIONS, T=v.EMBEDDING_ITERATIONS_T).to(v.device)
    optimizer = optim.Adam(Q_net.parameters(), lr=v.INIT_LR)
    lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=v.LR_DECAY_RATE)

    if fname is not None:
        checkpoint = torch.load(fname)
        Q_net.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

    Q_func = QFunction(Q_net, optimizer, lr_scheduler)
    return Q_func, Q_net, optimizer, lr_scheduler


def checkpoint_model(model, optimizer, lr_scheduler, loss, episode, avg_length):
    if not os.path.exists(v.FOLDER_NAME):
        os.makedirs(v.FOLDER_NAME)

    fname = os.path.join(v.FOLDER_NAME, 'ep_{}'.format(episode))
    fname += '_length_{}'.format(avg_length)
    fname += '.tar'

    torch.save({
        'episode': episode,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'loss': loss,
        'avg_length': avg_length
    }, fname)

    wandb.save(fname)


def _moving_avg(x, N=10):
    return np.convolve(np.array(x), np.ones((N,)) / N, mode='valid')


def plot_it(losses, path_lengths):
    plt.figure(figsize=(8, 5))
    plt.semilogy(_moving_avg(losses, 100))
    plt.ylabel('loss')
    plt.xlabel('training iteration')

    plt.figure(figsize=(8, 5))
    plt.plot(_moving_avg(path_lengths, 100))
    plt.ylabel('average length')
    plt.xlabel('episode')
    plt.show()


def plot_solution(coords, mat, solution, title):
    """
    A function to plot solutions
    """
    plt.figure()
    plt.scatter(coords[:, 0], coords[:, 1])
    n = len(coords)

    for idx in range(n - 1):
        i, next_i = solution[idx], solution[idx + 1]
        plt.plot([coords[i, 0], coords[next_i, 0]], [coords[i, 1], coords[next_i, 1]], 'k', lw=2, alpha=0.8)

    i, next_i = solution[-1], solution[0]
    plt.plot([coords[i, 0], coords[next_i, 0]], [coords[i, 1], coords[next_i, 1]], 'k', lw=2, alpha=0.8)
    plt.plot(coords[solution[0], 0], coords[solution[0], 1], 'x', markersize=10)
    plt.title(title)
    plt.show()


def download_wandb_models_and_return_list(project_name, run_name):
    # Initialize Wandb API
    api = wandb.Api()
    path = f"{project_name}/{run_name}"
    # Get the run
    run = api.run(path)

    # List all files in the run
    all_files = run.files()

    # Filter out the .tar files (models)
    model_files = [f for f in all_files if f.name.endswith('.tar')]

    # Download the models to a desired folder
    download_folder = "downloaded_models"
    if not os.path.exists(download_folder):
        os.makedirs(download_folder)

    for file in model_files:
        # Check if the file already exists
        if os.path.exists(os.path.join(download_folder, file.name)):
            print(f"File {file.name} already exists in {download_folder} (skipping)")
            continue
        file.download(replace=True, root=download_folder)
        print(f"Downloaded {file.name} to {download_folder}")

    return all_files
