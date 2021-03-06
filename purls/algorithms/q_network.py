import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gym.spaces.discrete import Discrete
from scipy.signal import savgol_filter

from gym_minigrid.envs import MiniGridEnv
from gym_minigrid.wrappers import FullyObsWrapper
from purls.algorithms.base import ReinforcementLearningAlgorithm
from purls.utils.logs import debug, info, success

# import adabound - if you want to experiment with (https://github.com/Luolc/AdaBound)

DIRECTIONS = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def preprocess_obs(obs, in_features, discrete=False):
    """
    Very similar to the preprocess_obs method in q_table. Main difference is
    that we want to return a onehot encoded vector here instead of an int.
    """
    onehot = torch.zeros(in_features, dtype=torch.float, device=device)

    # for other gym environments like FrozenLake-v0
    if discrete:
        state = obs
    # for MiniGrid environments
    else:
        obs = obs.flatten()
        i = np.nonzero(obs == 255)[0][0]
        position = i // 3
        direction = obs[i + 1]
        state = position + in_features // DIRECTIONS * direction

    onehot[state] = 1
    return onehot


class Net(nn.Module):
    def __init__(self, in_features, action_space):
        super(Net, self).__init__()
        self.fully_connected = nn.Linear(in_features, action_space.n, bias=False)

    def forward(self, x):
        x = self.fully_connected(x)
        return x


class q_network(ReinforcementLearningAlgorithm):
    def __init__(self, env, args):
        super().__init__(
            env,
            args,
            # default values for this algorithm
            default_learning_rate=0.1,
            default_discount_factor=0.99,
            default_start_eps=0.5,
            default_end_eps=0.05,
            default_annealing_steps=2500,
            default_num_updates=4000,
        )

        try:
            # for MiniGrid environments
            self.env: MiniGridEnv = FullyObsWrapper(self.env)
            width, height = self.env.observation_space.shape[0:2]
            self.in_features = width * height * DIRECTIONS
            # really Discrete(7) for this env but we don't need the pick up, drop... actions
            self.env.action_space = Discrete(3)
            self.discrete_obs_space = False

        except Exception:
            # for other gym environments like FrozenLake-v0
            if isinstance(self.env.observation_space, Discrete):
                self.in_features = self.env.observation_space.n
                self.discrete_obs_space = True
            # for other enviroments, we don't know how in_features is calculated from the obs space
            else:
                raise RuntimeError(
                    f"Don't know how to handle this observation space{self.env.obeservation_space}"
                )

        self.model = {"q_network": Net(self.in_features, self.env.action_space).to(device)}

    def train(self):
        q_net = self.model["q_network"]
        q_net.train()

        # loss function, could experiment with alternatives like Huber loss (F.smooth_l1_loss) too
        criterion = F.mse_loss
        # optimizer, could experiment with alternatives like AdaBound (adabound.AdaBound) too
        optimizer = optim.SGD(q_net.parameters(), lr=self.lr)

        eps = self.start_eps
        rewards = []

        for i in range(1, self.max_num_updates + 1):
            # reduce chance for random action
            if eps > self.end_eps:
                eps -= self.eps_decay

            if self.seed:
                self.env.seed(self.seed)
            obs = self.env.reset()
            obs = preprocess_obs(obs, self.in_features, self.discrete_obs_space)

            current_reward = 0
            done = False

            while True:
                # get q values
                q = q_net(obs.unsqueeze(0))

                # greedy-epsilon
                if np.random.rand(1) < eps:
                    # sample random action from action space
                    a = self.env.action_space.sample()
                else:
                    with torch.no_grad():
                        # choose action with highest Q value
                        a = q.argmax().item()

                # get next observation, reward and done from environment
                next_obs, reward, done, _ = self.env.step(a)
                next_obs = preprocess_obs(next_obs, self.in_features, self.discrete_obs_space)

                # construct a target (compare this to a label in supervised learning) by taking
                # our current q values and replacing the q value for the action chosen with:
                # the max q value in the next observation * discount factor + the reward
                next_q = q_net(next_obs.unsqueeze(0))
                next_q_max = next_q.max().item()
                target_q = q.detach().clone()  # clone an independant
                target_q[0, a] = next_q_max * self.y + reward

                # compute loss
                loss = criterion(q, target_q)

                # optimize: backprop and update weights
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # update variables for next iteration
                current_reward += reward
                obs = next_obs

                if self.render_interval != 0 and i % self.render_interval == 0:
                    self.env.render()
                    time.sleep(1 / self.fps)

                if done:
                    break

            rewards.append(current_reward)

            if i % 100 == 0:
                debug(f"episode {i:5d} finished - avg. reward: {np.average(rewards[-100:-1]):2f}")

            if self.save_interval != 0 and i % self.save_interval == 0:
                self.save()

        success(f"all {self.max_num_updates:5d} episodes finished!")
        info(f"reward for the final episode: {rewards[-1]:2f}")

        if self.save_interval != 0:
            self.save()

        debug("plotting reward over episodes")
        matplotlib.rcParams["figure.dpi"] = 200
        plt.plot(rewards)
        plt.plot(savgol_filter(rewards, 23, 3), "-r", linewidth=2.0)
        plt.title(self.model_name)
        plt.xlabel("episode")
        plt.ylabel("reward")
        plt.show()

    def visualize(self):
        self.model = self.load()
        q_net = self.model["q_network"]
        q_net.eval()

        for i in range(self.max_num_updates + 1):
            if self.seed:
                self.env.seed(self.seed)
            obs = self.env.reset()
            obs = preprocess_obs(obs, self.in_features, self.discrete_obs_space)
            self.env.render()

            done = False

            time.sleep(0.5)
            while True:
                a = q_net(obs.unsqueeze(0)).argmax().item()
                next_obs, reward, done, _ = self.env.step(a)
                next_obs = preprocess_obs(next_obs, self.in_features, self.discrete_obs_space)
                obs = next_obs

                self.env.render()
                time.sleep(1 / self.fps)

                if done:
                    break
            time.sleep(0.5)
