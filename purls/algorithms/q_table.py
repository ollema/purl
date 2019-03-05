import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from gym.spaces.discrete import Discrete
from scipy.signal import savgol_filter

from gym_minigrid.envs import MiniGridEnv
from gym_minigrid.wrappers import FullyObsWrapper
from purls.algorithms.base import ReinforcementLearningAlgorithm
from purls.utils.logs import debug, info, success

DIRECTIONS = 4


def preprocess_obs(obs, q_table_length, discrete=False):
    """
    The input s is a matrix with the dimensions n x m x 3,
    which, for each x in n, y in m,
    contains: object encoding, direction, state

    In other words, each state has:
    * a certain object type (agent, wall, lava etc...)
    * the direction of the object
    * if it's open/closed/locked (mostly used for doors)

    We are only interested in where our agent is located and what direction
    it is facing. We can determine this by flattening the matrix and looking for
    the element where the object encoding is 255. The index of this element,
    divided by three, will be our position. The very next element will
    be the direction.

    Finally, we can encode this into our new state that is compatible with
    Q-tables by taking our position and adding x * gridsize
    where the x is a direction (0, 1, 2, 3)

    This took a while to figure out!
    """
    if discrete:
        return obs
    flattened_s = obs.flatten()
    i = np.nonzero(flattened_s == 255)[0][0]
    position = i // 3
    direction = flattened_s[i + 1]

    return position + obs.shape[0] * obs.shape[1] * direction


class q_table(ReinforcementLearningAlgorithm):
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
            default_num_episodes=4000,
        )

        try:
            # for MiniGrid environments
            self.env: MiniGridEnv = FullyObsWrapper(self.env)
            width, height = self.env.observation_space.shape[0:2]
            self.q_table_length = width * height * DIRECTIONS
            # really Discrete(7) for this env but we don't need the pick up, drop... actions
            self.env.action_space = Discrete(3)
            self.discrete_obs_space = False

        except Exception:
            # for other gym environments like FrozenLake-v0
            if isinstance(self.env.observation_space, Discrete):
                self.q_table_length = self.env.observation_space.n
                self.discrete_obs_space = True
            # for other enviroments, we don't know how in_features is calculated from the obs space
            else:
                raise RuntimeError(
                    f"Don't know how to handle this obeservation space{self.env.obeservation_space}"
                )

        self.eps_decay = (self.start_eps - self.end_eps) / self.annealing_steps

        self.model = {"q_table": np.zeros([self.q_table_length, self.env.action_space.n])}

    def train(self):
        Q = self.model["q_table"]

        eps = self.start_eps
        rewards = []

        for i in range(self.num_episodes + 1):
            # reduce chance for random action
            if eps > self.end_eps:
                eps -= self.eps_decay

            if self.seed:
                self.env.seed(self.seed)
            obs = self.env.reset()
            obs = preprocess_obs(obs, self.q_table_length, self.discrete_obs_space)

            current_reward = 0
            done = False

            while True:
                # get q values
                q = Q[obs, :]

                # greedy-epsilon
                if np.random.rand(1) < eps:
                    # sample random action from action space
                    a = self.env.action_space.sample()
                else:
                    # choose action with highest Q value
                    a = np.argmax(q)

                # get next observation, reward and done from environment
                next_obs, reward, done, _ = self.env.step(a)
                next_obs = preprocess_obs(next_obs, self.q_table_length, self.discrete_obs_space)

                # construct a target
                next_q_max = np.max(Q[next_obs, :])
                target_q = next_q_max * self.y + reward

                # update q-table with new knowledge
                Q[obs, a] = (1 - self.lr) * Q[obs, a] + self.lr * target_q

                # update variables for next iteration
                current_reward += reward
                obs = next_obs

                if self.render_interval != 0 and i % self.render_interval == 0:
                    self.env.render()
                    time.sleep(1 / self.fps)

                if done:
                    break

            rewards.append(current_reward)

            if i % 100 == 0 and i != 0:
                debug(f"episode {i:5d} finished - avg. reward: {np.average(rewards[-100:-1]):2f}")

            if self.save_interval != 0 and i % self.save_interval == 0:
                self.save()

        success(f"all {self.num_episodes:5d} episodes finished!")
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
        Q = self.model["q_table"]

        while True:
            if self.seed:
                self.env.seed(self.seed)
            s = self.env.reset()
            s = preprocess_obs(s, self.q_table_length, self.discrete_obs_space)
            self.env.render()

            done = False

            time.sleep(0.5)
            while True:
                a = np.argmax(Q[s, :])
                s1, reward, done, _ = self.env.step(a)
                s1 = preprocess_obs(s, self.q_table_length, self.discrete_obs_space)
                s = s1

                self.env.render()
                time.sleep(1 / self.fps)

                if done:
                    break
            time.sleep(0.5)

    def evaluate(self):
        self.model = self.load()
        Q = self.model["q_table"]

        rewards = []

        for i in range(self.num_episodes + 1):
            if self.seed:
                self.env.seed(self.seed)
            s = self.env.reset()
            s = preprocess_obs(s, self.q_table_length, self.discrete_obs_space)

            current_reward = 0
            done = False

            while True:
                a = np.argmax(Q[s, :])
                s1, reward, done, _ = self.env.step(a)
                s1 = preprocess_obs(s, self.q_table_length, self.discrete_obs_space)

                current_reward += reward
                s = s1

                if done:
                    break

            rewards.append(current_reward)

        success(f"all {self.num_episodes:5d} episodes finished!")
        avg_reward = np.average(rewards)
        info(f"average reward: {avg_reward:2f}")
