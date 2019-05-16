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

BATCH_SIZE = 4
TRACE_LENGTH = 4
TRAIN_FREQ = 8
WARMUP = 1000
TAU = 0.01
MEMORY_SIZE = 1000
MOMENTUM = 0.99


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
    """Circular buffer of Transitions, used in training."""

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Add Transition to the buffer."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

    def __getitem__(self, key):
        return self.memory[key]


class EpisodeMemory:
    """A circular buffer of complete episodes, i.e. sequences of Transitions."""

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, episode):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = episode
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, trace_len):
        """Returns a batch of traces from randomly selected episodes"""
        episodes = random.sample(self.memory, batch_size)
        traces = []
        for episode in episodes:
            ep_length = len(episode)
            # TODO: Arbitrary trace length
            trace_starting_point = random.randint(0, ep_length - trace_len)
            traces.append(episode[trace_starting_point : trace_starting_point + trace_len])
        return traces

    def __len__(self):
        return len(self.memory)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, 2)
        self.conv2 = nn.Conv2d(16, 32, 2)
        self.conv3 = nn.Conv2d(32, 64, 2)
        self.lstm = nn.LSTMCell(4 * 4 * 64, 2 * 4 * 8)
        self.fc = nn.Linear(64, 3)

    def forward(self, x: torch.Tensor, hidden_states):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.view(x.size(0), -1)

        hidden = self.lstm(x, hidden_states)

        x = hidden[0]
        x = F.relu(self.fc(x))
        return x, hidden


class drqn(ReinforcementLearningAlgorithm):
    def __init__(self, env, args):
        super().__init__(
            env,
            args,
            # default values for this algorithm
            default_learning_rate=0.01,
            default_discount_factor=0.99,
            default_start_eps=1,
            default_end_eps=0.1,
            default_annealing_steps=1000,
            default_num_episodes=110_000,
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
        memory = EpisodeMemory(MEMORY_SIZE)
        writer = SummaryWriter(comment=f"-{self.model_name}")
        eps = self.start_eps
        policy_hidden = (torch.randn(TRACE_LENGTH, 64), torch.randn(TRACE_LENGTH, 64))
        target_hidden = (torch.randn(TRACE_LENGTH, 64), torch.randn(TRACE_LENGTH, 64))

        # For setting max nr of steps lower than MiniGrid
        max_steps = 100

        with Tracker(writer) as tracker:
            for i in range(self.num_episodes + 1):
                # reduce chance for random action
                if i > WARMUP and eps > self.end_eps:
                    eps -= self.eps_decay

                if self.seed:
                    self.env.seed(self.seed)
                obs = self.env.reset()
                obs = preprocess_obs(obs)

                current_reward = 0
                done = False

                # Replaymemory for storing a single episode
                temp_memory = ReplayMemory(max_steps)

                batch_obs = torch.stack(tuple(obs.unsqueeze(0)) * TRACE_LENGTH)
                episode_lstm_state = (torch.randn(TRACE_LENGTH, 64), torch.randn(TRACE_LENGTH, 64))
                step = 0
                while step < max_steps:
                    # greedy-epsilon
                    if np.random.rand(1) < eps:
                        # sample random action from action space
                        action = self.env.action_space.sample()

                    else:
                        with torch.no_grad():
                            # TODO: forward pass seems to always result in same action (zero tensor returned)
                            res = policy_net(batch_obs, episode_lstm_state)
                            episode_lstm_state = res[1]
                            action = res[0][0].argmax().item()

                    # get next observation, reward and done from environment
                    next_obs, reward, done, _ = self.env.step(action)
                    next_obs = preprocess_obs(next_obs)

                    # Push recent Transition to single episode memory
                    temp_memory.push(obs, action, next_obs, reward)

                    # training sequence
                    if i > WARMUP and i % TRAIN_FREQ == 0:
                        traces = memory.sample(BATCH_SIZE, TRACE_LENGTH)

                        # Train for each element in the batch
                        for trace in traces:
                            batch = Transition(*zip(*trace))

                            batch_state = torch.stack(batch.state)
                            batch_action = torch.tensor(batch.action, device=device).unsqueeze(1)
                            batch_next_state = torch.stack(batch.next_state)
                            batch_reward = torch.tensor(
                                batch.reward, device=device, dtype=torch.float
                            ).unsqueeze(1)

                            q, policy_hidden = policy_net(batch_state, policy_hidden)
                            q = q.gather(1, batch_action)

                            # max q value in the next observation * discount factor + the reward
                            next_q, target_hidden = target_net(batch_next_state, target_hidden)
                            next_q_max = next_q.max(1)[0].unsqueeze(1)
                            target_q = next_q_max * self.y + batch_reward

                            # compute loss
                            loss = criterion(q, target_q)

                            # optimize: backprop and update weights
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()

                            # Reset LSTM internal states
                            # TODO: Is this correct?
                            policy_hidden = (
                                torch.randn(TRACE_LENGTH, 64),
                                torch.randn(TRACE_LENGTH, 64),
                            )
                            target_hidden = (
                                torch.randn(TRACE_LENGTH, 64),
                                torch.randn(TRACE_LENGTH, 64),
                            )

                    if i > WARMUP:
                        # Periodically update target net to latest policy net
                        if i % (1 / TAU) == 0:
                            target_net.load_state_dict(policy_net.state_dict())

                    # update variables for next iteration
                    current_reward += reward
                    obs = next_obs
                    step += 1

                    if i > WARMUP and self.render_interval != 0 and i % self.render_interval == 0:
                        self.env.render()
                        time.sleep(1 / self.fps)

                    if done:
                        break

                if reward > 0 and len(temp_memory) > TRACE_LENGTH:
                    memory.push(temp_memory)  # push episode to episode memory

                tracker.push(episode=i, reward=current_reward, epsilon=eps)
                if self.save_interval != 0 and i % self.save_interval == 0:
                    self.save()

        if self.save_interval != 0:
            self.save()

    def visualize(self):
        pass

    def evaluate(self):
        pass
