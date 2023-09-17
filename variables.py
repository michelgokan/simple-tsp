from collections import namedtuple

import torch
import wandb
import configparser

""" Note: the code is not optimized for GPU
"""
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

SEED = 1  # A seed for the random number generator

# Graph
NR_NODES = 10  # Number of nodes N
EMBEDDING_DIMENSIONS = 5  # Embedding dimension D
EMBEDDING_ITERATIONS_T = 1  # Number of embedding iterations T

# Learning
NR_EPISODES = 4001
MEMORY_CAPACITY = 10000
N_STEP_QL = 2  # Number of steps (n) in n-step Q-learning to wait before computing target reward estimate
BATCH_SIZE = 16

GAMMA = 0.9
INIT_LR = 5e-3
LR_DECAY_RATE = 1. - 2e-5  # learning rate decay

MIN_EPSILON = 0.1
EPSILON_DECAY_RATE = 6e-4  # epsilon decay

FOLDER_NAME = './models'  # where to checkpoint the best models

State = namedtuple('State', ('W', 'coords', 'partial_solution'))

# Initialize the configparser and read from config.ini
config = configparser.ConfigParser()
config.read('config.ini')
WANDB_RUN_ID = config['DEFAULT']['WANDB_RUN_ID']
WANDB_RUN_NAME = config['DEFAULT']['WANDB_RUN_NAME']
WANDB_PROJECT_NAME = config['DEFAULT']['WANDB_PROJECT_NAME']
WANDB_CONFIG = {
    "learning_rate": INIT_LR,
    "batch_size": BATCH_SIZE,
    "architecture": "DQN",
    "dataset": "",
    "epochs": NR_EPISODES,
}

# If WANDB_RUN_ID empty then set LOAD_FROM_WANDB_OR_LOCALLY to locally
if WANDB_RUN_ID == "" or WANDB_PROJECT_NAME == "":
    LOAD_FROM_WANDB_OR_LOCALLY = "locally"
else:
    LOAD_FROM_WANDB_OR_LOCALLY = "wandb"

SAVE_IN_WANDB = True
