import os
import random

import matplotlib.pyplot as plt
import torch

import variables as v
from helpers import get_graph_mat, init_model, state2tens, is_state_final, total_distance, \
    plot_solution

""" Get file with smallest distance
"""
all_lengths_fnames = [f for f in os.listdir(v.FOLDER_NAME) if f.endswith('.tar')]
shortest_fname = sorted(all_lengths_fnames, key=lambda s: float(s.split('.tar')[0].split('_')[-1]))[0]
print('shortest avg length found: {}'.format(shortest_fname.split('.tar')[0].split('_')[-1]))

""" Load checkpoint
"""
Q_func, Q_net, optimizer, lr_scheduler = init_model(os.path.join(v.FOLDER_NAME, shortest_fname))

""" Generate example solutions
"""
NR_NODES = 10
for sample in range(10):
    coords, W_np = get_graph_mat(n=NR_NODES)
    W = torch.tensor(W_np, dtype=torch.float32, requires_grad=False, device=v.device)

    solution = [random.randint(0, NR_NODES - 1)]
    current_state = v.State(partial_solution=solution, W=W, coords=coords)

    while not is_state_final(current_state):
        current_state_tsr = state2tens(current_state)
        next_node, est_reward = Q_func.get_best_action(current_state_tsr, current_state)

        solution = solution + [next_node]
        current_state = v.State(partial_solution=solution, W=W, coords=coords)
    plt.close()
    plt.figure()
    plot_solution(coords, W, solution)
    plt.title('model / len = {}'.format(total_distance(solution, W)))

    # for comparison, plot a random solution
    plt.figure()
    random_solution = list(range(NR_NODES))
    plot_solution(coords, W, random_solution)
    plt.title('random / len = {}'.format(total_distance(random_solution, W)))
    plt.show()
