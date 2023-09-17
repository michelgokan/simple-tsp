# Simple TSP using PyTorch
### Introduction
This repository is a revised debugger-ready fork of 
'Appendix repository for Medium article ["Routing Traveling Salesmen on Random Graphs using Reinforcement Learning"](https://medium.com/unit8-machine-learning-publication/routing-traveling-salesmen-on-random-graphs-using-reinforcement-learning-in-pytorch-7378e4814980)'. 

The majority of the code has been copied and pasted from the original Jupyter notebook found in the original repository. Full credit for these portions of the code should be attributed to the original author, @hrzn.

The current status of this project is geared towards my personal experiments and should not be regarded as production-ready in any way. If you wish to comprehend the Medium article, I recommend examining the original repository rather than this one.

### Execution
If you make a mistake (!) and decided to use this repository (you probably shouldn't), follow these steps:

* copy/paste config.ini.sample into config.ini and replace variables in it
* run followings:
  ```bash
  pip3.11 install -r requirements.txt 
    ```
* Follow steps in ["Weights & Biases quickstart page"](https://docs.wandb.ai/quickstart)
    ```bash
  wandb login
  ```
* To run the training:
  ```
  python3.11 train.py
  ```
* To use the trained model to find the shortest path:
  ```bash
  python3.11 execute.py 
  ```
### Compatibility
> I utilized Python version 3.11.15 in my local environment. I strongly recommend using this version or a higher one, as I encountered difficulties when attempting to run the code with older Python versions, e.g., 3.11.12 didn't work because of a compatibility error.
