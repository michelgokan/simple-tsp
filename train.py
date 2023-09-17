import random
from collections import namedtuple

import numpy as np
import torch
import wandb

import variables as v
from helpers import get_graph_mat, init_model, state2tens, is_state_final, get_next_neighbor_random, total_distance, \
    checkpoint_model, plot_it, plot_graph
from memory import Memory
from qnet import QNet

coords, W_np = get_graph_mat(n=10)
plot_graph(coords, W_np)

""" See what the model returns
"""
model = QNet(3, T=1).to(v.device)
coords, W_np = get_graph_mat(n=10)
W = torch.tensor(W_np, dtype=torch.float32, device=v.device)
xv = torch.rand((1, W.shape[0], 5)).to(v.device)  # random node state
Ws = W.unsqueeze(0)

y = model(xv, Ws)
print('model output: {}'.format(y))

# Note: we store state tensors in experience to compute these tensors only once later on
Experience = namedtuple('Experience', ('state', 'state_tsr', 'action', 'reward', 'next_state', 'next_state_tsr'))

# seed everything for reproducible results first:
torch.manual_seed(v.SEED)
np.random.seed(v.SEED)
random.seed(v.SEED)

# Create module, optimizer, LR scheduler, and Q-function
Q_func, Q_net, optimizer, lr_scheduler = init_model()

# Create memory
memory = Memory(v.MEMORY_CAPACITY)

# Storing metrics about training:
found_solutions = dict()  # episode --> (coords, W, solution)
losses = []
path_lengths = []

# keep track of median path length for model checkpointing
current_min_med_length = float('inf')
# start a new wandb run to track this script

if v.SAVE_IN_WANDB:
    wandb.init(project=v.WANDB_PROJECT_NAME, name=v.WANDB_RUN_NAME, config=v.WANDB_CONFIG)

for episode in range(v.NR_EPISODES):
    # sample a new random graph
    coords, W_np = get_graph_mat(n=v.NR_NODES)
    W = torch.tensor(W_np, dtype=torch.float32, requires_grad=False, device=v.device)

    # current partial solution - a list of node index
    solution = [random.randint(0, v.NR_NODES - 1)]

    # current state (tuple and tensor)
    current_state = v.State(partial_solution=solution, W=W, coords=coords)
    current_state_tsr = state2tens(current_state)

    # Keep track of some variables for insertion in replay memory:
    states = [current_state]
    states_tsrs = [current_state_tsr]  # we also keep the state tensors here (for efficiency)
    rewards = []
    actions = []

    # current value of epsilon
    epsilon = max(v.MIN_EPSILON, (1 - v.EPSILON_DECAY_RATE) ** episode)

    nr_explores = 0
    t = -1
    while not is_state_final(current_state):
        t += 1  # time step of this episode

        if epsilon >= random.random():
            # explore
            next_node = get_next_neighbor_random(current_state)
            nr_explores += 1
        else:
            # exploit
            next_node, est_reward = Q_func.get_best_action(current_state_tsr, current_state)
            if episode % 50 == 0:
                print('Ep {} | current sol: {} / next est reward: {}'.format(episode, solution, est_reward))

        next_solution = solution + [next_node]

        # reward observed for taking this step
        reward = -(total_distance(next_solution, W) - total_distance(solution, W))

        next_state = v.State(partial_solution=next_solution, W=W, coords=coords)
        next_state_tsr = state2tens(next_state)

        # store rewards and states obtained along this episode:
        states.append(next_state)
        states_tsrs.append(next_state_tsr)
        rewards.append(reward)
        actions.append(next_node)

        # store our experience in memory, using n-step Q-learning:
        if len(solution) >= v.N_STEP_QL:
            memory.remember(Experience(state=states[-v.N_STEP_QL],
                                       state_tsr=states_tsrs[-v.N_STEP_QL],
                                       action=actions[-v.N_STEP_QL],
                                       reward=sum(rewards[-v.N_STEP_QL:]),
                                       next_state=next_state,
                                       next_state_tsr=next_state_tsr))

        if is_state_final(next_state):
            for n in range(1, v.N_STEP_QL):
                memory.remember(Experience(state=states[-n],
                                           state_tsr=states_tsrs[-n],
                                           action=actions[-n],
                                           reward=sum(rewards[-n:]),
                                           next_state=next_state,
                                           next_state_tsr=next_state_tsr))

        # update state and current solution
        current_state = next_state
        current_state_tsr = next_state_tsr
        solution = next_solution

        # take a gradient step
        loss = None
        if len(memory) >= v.BATCH_SIZE and len(memory) >= 2000:
            experiences = memory.sample_batch(v.BATCH_SIZE)

            batch_states_tsrs = [e.state_tsr for e in experiences]
            batch_Ws = [e.state.W for e in experiences]
            batch_actions = [e.action for e in experiences]
            batch_targets = []

            for i, experience in enumerate(experiences):
                target = experience.reward
                if not is_state_final(experience.next_state):
                    _, best_reward = Q_func.get_best_action(experience.next_state_tsr,
                                                            experience.next_state)
                    target += v.GAMMA * best_reward
                batch_targets.append(target)

            # print('batch targets: {}'.format(batch_targets))
            loss = Q_func.batch_update(batch_states_tsrs, batch_Ws, batch_actions, batch_targets)
            losses.append(loss)

            """ Save model when we reach a new low average path length
            """
            med_length = np.median(path_lengths[-100:])
            if med_length < current_min_med_length:
                current_min_med_length = med_length
                checkpoint_model(Q_net, optimizer, lr_scheduler, loss, episode, med_length)

    length = total_distance(solution, W)
    path_lengths.append(length)
    wandb.log(
        {"loss": (-1 if loss is None else loss), "median length": np.median(path_lengths[-50:]), "length": length})

    if episode % 10 == 0:
        print('Ep %d. Loss = %.3f / median length = %.3f / last = %.4f / epsilon = %.4f / lr = %.4f' % (
            episode, (-1 if loss is None else loss), np.median(path_lengths[-50:]), length, epsilon,
            Q_func.optimizer.param_groups[0]['lr']))
        a = W
        found_solutions[episode] = (W.clone(), coords.copy(), [n for n in solution])

wandb.finish()
plot_it(losses, path_lengths)
