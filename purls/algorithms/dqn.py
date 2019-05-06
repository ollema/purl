import random
import time
from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gym.spaces.discrete import Discrete
from tensorboardX import SummaryWriter

from gym_minigrid.envs import MiniGridEnv
from purls.algorithms.base import ReinforcementLearningAlgorithm
from purls.utils.logs import debug

BATCH_SIZE = 32
TRAIN_FREQ = 4
WARMUP = 10000
TAU = 0.001
MEMORY_SIZE = 10000
MOMENTUM = 0.95


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


def preprocess_obs(obs):
    # return a torch tensor ~ W x H x C -> C x W x H
    return torch.tensor(np.transpose(obs["image"], (2, 0, 1)), dtype=torch.float, device=device)


class Tracker:
    def __init__(self, writer, debug_every=100):
        self.writer = writer
        self.debug_every = debug_every

    def __enter__(self):
        self.current_time = time.time()
        self.current_episode = 0
        self.total_rewards = []
        # self.writer.flush()
        return self

    def __exit__(self, *args):
        debug("closing tensorboard")
        self.writer.close()

    def push(self, episode, reward, epsilon):
        eps = (episode - self.current_episode) / (time.time() - self.current_time)
        self.current_episode = episode
        self.current_time = time.time()

        self.total_rewards.append(reward)
        mean_reward = np.mean(self.total_rewards[-100:])

        self.writer.add_scalar("epsilon", epsilon, episode)
        self.writer.add_scalar("episodes_per_second", eps, episode)
        self.writer.add_scalar("reward_avg_100", mean_reward, episode)
        self.writer.add_scalar("reward", reward, episode)

        if episode % self.debug_every == 0:
            debug(f"episode {episode:6d} finished - avg. reward: {mean_reward:2f}")


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, 2)
        self.conv2 = nn.Conv2d(16, 32, 2)
        self.conv3 = nn.Conv2d(32, 64, 2)
        self.fc1 = nn.Linear(4 * 4 * 64, 2 * 4 * 64)
        self.fc2 = nn.Linear(2 * 4 * 64, 3)

    def forward(self, x: torch.Tensor):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class dqn(ReinforcementLearningAlgorithm):
    def __init__(self, env, args):
        super().__init__(
            env,
            args,
            # default values for this algorithm
            default_learning_rate=0.001,
            default_discount_factor=0.99,
            default_start_eps=1,
            default_end_eps=0.1,
            default_annealing_steps=25000,
            default_num_updates=110_000,
        )

        # for MiniGrid environments
        self.env: MiniGridEnv = self.env
        # really Discrete(7) for this env but we don't need the pick up, drop... actions
        self.env.action_space = Discrete(3)

        self.model = {"policy_network": Net().to(device), "target_network": Net().to(device)}

    def train(self):
        policy_net = self.model["policy_network"]
        target_net = self.model["target_network"]
        policy_net.train()
        target_net.train()

        criterion = F.mse_loss
        optimizer = optim.RMSprop(policy_net.parameters(), lr=self.lr, momentum=MOMENTUM)
        memory = ReplayMemory(MEMORY_SIZE)
        writer = SummaryWriter(comment=f"-{self.model_name}")
        eps = self.start_eps

        # TODO: use a tracker from utils.trackers!
        with Tracker(writer) as tracker:
            for i in range(1, self.max_num_updates + 1):
                # reduce chance for random action
                if i > WARMUP and eps > self.end_eps:
                    eps -= self.eps_decay

                if self.seed:
                    self.env.seed(self.seed)
                obs = self.env.reset()
                obs = preprocess_obs(obs)

                current_reward = 0
                done = False

                while True:
                    # greedy-epsilon
                    if np.random.rand(1) < eps:
                        # sample random action from action space
                        action = self.env.action_space.sample()
                    else:
                        with torch.no_grad():
                            # choose action with highest Q value
                            action = policy_net(obs.unsqueeze(0)).argmax().item()

                    # get next observation, reward and done from environment
                    next_obs, reward, done, _ = self.env.step(action)
                    next_obs = preprocess_obs(next_obs)

                    memory.push(obs, action, next_obs, reward)

                    if i > WARMUP and i % TRAIN_FREQ == 0:
                        transitions = memory.sample(BATCH_SIZE)
                        batch = Transition(*zip(*transitions))

                        batch_state = torch.stack(batch.state)
                        batch_action = torch.tensor(batch.action, device=device).unsqueeze(1)
                        batch_next_state = torch.stack(batch.next_state)
                        batch_reward = torch.tensor(
                            batch.reward, device=device, dtype=torch.float
                        ).unsqueeze(1)

                        q = target_net(batch_state).gather(1, batch_action)

                        # construct a target (compare this to a label in supervised learning)
                        # max q value in the next observation * discount factor + the reward
                        next_q = target_net(batch_next_state)
                        next_q_max = next_q.max(1)[0].unsqueeze(1)

                        # target_q = q.detach().clone()  # clone an independant
                        target_q = next_q_max * self.y + batch_reward

                        # compute loss
                        loss = criterion(q, target_q)

                        # optimize: backprop and update weights
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                    if i > WARMUP:
                        policy_state = policy_net.state_dict()
                        target_state = target_net.state_dict()
                        for k, v in policy_state.items():
                            target_state[k] = target_state[k] * (1 - TAU) + TAU * v
                        target_net.load_state_dict(target_state)

                        # TODO: Revert if needed - do the below if i % (1/TAU) == 0
                        # target_net.load_state_dict(policy_net.state_dict())

                    # update variables for next iteration
                    current_reward += reward
                    obs = next_obs

                    if i > WARMUP and self.render_interval != 0 and i % self.render_interval == 0:
                        self.env.render()
                        time.sleep(1 / self.fps)

                    if done:
                        break

                tracker.push(episode=i, reward=current_reward, epsilon=eps)

                if self.save_interval != 0 and i % self.save_interval == 0:
                    self.save()

        if self.save_interval != 0:
            self.save()

    def visualize(self):
        pass
