import time

import numpy as np
from gym_minigrid.wrappers import FullyObsWrapper
from purls.algorithms.base import ReinforcementLearningAlgorithm
from purls.utils.logs import debug, info, success

DIRECTIONS = 4


class QLearningWithTable(ReinforcementLearningAlgorithm):
    def __init__(self, env, args):
        super().__init__(
            args, default_learning_rate=0.8, default_discount_factor=0.95, default_num_episodes=2000
        )
        self.env = FullyObsWrapper(env)

        # we don't need the pick up, drop... actions
        self.action_space_n = self.env.action_space.n - 4
        self.positions = self.env.grid.width * self.env.grid.height

        self.model = {"q_table": np.zeros([(self.positions * DIRECTIONS), self.action_space_n])}

    def train(self):
        Q = self.model["q_table"]

        rewards = []

        for i in range(self.num_episodes + 1):
            if self.seed:
                self.env.seed(self.seed)
            s = self.env.reset()
            s = minigrid_encoding_to_table(s)

            current_reward = 0
            done = False

            j = 0
            while j < self.positions * DIRECTIONS:
                j += 1

                # Choose an action by greedily (with noise) picking from Q table
                a = np.argmax(Q[s, :] + np.random.randn(1, self.action_space_n) * (1.0 / (i + 1)))

                # Get new state, reward and done from environment
                s1, reward, done, _ = self.env.step(a)
                s1 = minigrid_encoding_to_table(s1)

                # Update Q-Table with new knowledge
                Q[s, a] = Q[s, a] + self.lr * (reward + self.y * np.max(Q[s1, :]) - Q[s, a])
                current_reward += reward
                s = s1

                if self.render_interval != 0 and i % self.render_interval == 0:
                    self.env.render()
                    time.sleep(1 / self.fps)

                if done:
                    break

            rewards.append(current_reward)

            if i % 10 == 0:
                debug(f"episode {i:5d} finished - reward: {rewards[-1]:2f}")

            if self.save_interval != 0 and i % self.save_interval == 0:
                self.save()

        success(f"all {self.num_episodes:5d} episodes finished!")
        info(f"reward for the final episode: {rewards[-1]:2f}")

        if self.save_interval != 0:
            self.save()

    def visualize(self):
        self.model = self.load()
        Q = self.model["q_table"]

        while True:
            if self.seed:
                self.env.seed(self.seed)
            s = self.env.reset()
            s = minigrid_encoding_to_table(s)
            self.env.render()

            done = False

            time.sleep(0.5)
            while True:
                a = np.argmax(Q[s, :])
                s1, reward, done, _ = self.env.step(a)
                s1 = minigrid_encoding_to_table(s1)
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
            s = minigrid_encoding_to_table(s)

            current_reward = 0
            done = False

            j = 0
            while j < self.positions * DIRECTIONS:
                j += 1

                a = np.argmax(Q[s, :])
                s1, reward, done, _ = self.env.step(a)
                s1 = minigrid_encoding_to_table(s1)

                current_reward += reward
                s = s1

                if done:
                    break

            rewards.append(current_reward)

        success(f"all {self.num_episodes:5d} episodes finished!")
        avg_reward = np.average(rewards)
        info(f"average reward: {avg_reward:2f}")


def minigrid_encoding_to_table(s):
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
    flattened_s = s.flatten()
    i = np.nonzero(flattened_s == 255)[0][0]
    position = i // 3
    direction = flattened_s[i + 1]

    return position + s.shape[0] * s.shape[1] * direction
