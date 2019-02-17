import time

import numpy as np
from gym_minigrid.wrappers import FullyObsWrapper
from purls.algorithms.base import ReinforcmentLearningAlgorithm
from purls.utils.logs import debug, info, success

DEFAULTS = {
    "learning_rate": 0.8,
    "discount_factor": 0.95,
    "episodes": 2000,
    "fully_obs": "required",
}


class QLearningWithTable(ReinforcmentLearningAlgorithm):
    def __init__(self, env, args):
        super().__init__(env, args, DEFAULTS)

    def train(self):
        env = FullyObsWrapper(self.env)

        # we don't need the pick up, drop... actions
        reduced_action_space = env.action_space.n - 4

        Q = np.zeros([(env.grid.width * env.grid.height * 4), reduced_action_space])

        rewards = []

        for i in range(self.num_episodes + 1):
            if self.seed:
                env.seed(self.seed)
            s = env.reset()
            s = minigrid_encoding_to_table(s)

            current_reward = 0
            done = False

            j = 0
            # TODO: verify that env.grid.width * env.grid.height * 4 is a valid limit
            while j < env.grid.width * env.grid.height * 4:
                j += 1

                # Choose an action by greedily (with noise) picking from Q table
                a = np.argmax(Q[s, :] + np.random.randn(1, reduced_action_space) * (1.0 / (i + 1)))

                # Get new state, reward and done from environment
                s1, reward, done, _ = env.step(a)
                s1 = minigrid_encoding_to_table(s1)

                # Update Q-Table with new knowledge
                Q[s, a] = Q[s, a] + self.lr * (reward + self.y * np.max(Q[s1, :]) - Q[s, a])
                current_reward += reward
                s = s1

                if self.render_interval != 0 and i % self.render_interval == 0:
                    env.render()
                    time.sleep(0.3)

                if done:
                    break

            rewards.append(current_reward)

            if i % 10 == 0:
                debug(f"episode {i:4d} finished - reward: {rewards[-1]:2f}")

            if self.save_interval != 0 and i % self.save_interval == 0:
                self.save(Q)
                debug(f"model saved in models/{self.model_name}.txt")

        success(f"all {self.num_episodes:04d} episodes finished!")
        info(f"reward for the final episode: {rewards[-1]:2f}")

        if self.save_interval != 0:
            info(f"model saved in models/{self.model_name}.txt")
            self.save(Q)

    def save(self, q_table):
        np.savetxt(f"models/{self.model_name}.txt", q_table, delimiter=",")

    def load(self):
        return np.loadtxt(f"models/{self.model_name}.txt", delimiter=",")

    def visualize(self):
        env = FullyObsWrapper(self.env)
        Q = self.load()

        while True:
            if self.seed:
                env.seed(self.seed)
            s = env.reset()
            s = minigrid_encoding_to_table(s)
            env.render()

            done = False

            time.sleep(0.5)
            while True:
                a = np.argmax(Q[s, :])
                s1, reward, done, _ = env.step(a)
                s1 = minigrid_encoding_to_table(s1)

                s = s1

                env.render()
                time.sleep(0.3)

                if done:
                    break
            time.sleep(0.5)

    def evaluate(self):
        env = FullyObsWrapper(self.env)
        Q = self.load()

        rewards = []

        for i in range(self.num_episodes + 1):
            if self.seed:
                env.seed(self.seed)
            s = env.reset()
            s = minigrid_encoding_to_table(s)

            current_reward = 0
            done = False

            j = 0
            # TODO: verify that env.grid.width * env.grid.height * 4 is a valid limit
            while j < env.grid.width * env.grid.height * 4:
                j += 1

                a = np.argmax(Q[s, :])
                s1, reward, done, _ = env.step(a)
                s1 = minigrid_encoding_to_table(s1)

                current_reward += reward
                s = s1

                if done:
                    break

            rewards.append(current_reward)

        success(f"all {self.num_episodes:04d} episodes finished!")
        avg_reward = np.average(rewards)
        info(f"average reward: {avg_reward:2f}")


def minigrid_encoding_to_table(s):
    """
    The input is a matrix with the dimensions n x m x 3,
    which, for each x in n, y in m,
    contains: object encoding, direction, state

    In other words, each state has:
    * a certain object type there
    * the direction of the object
    * if it's open/closed/locked (only used for doors)

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
