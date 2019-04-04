import os
import sys

import gym
import gym_minigrid  # noqa

# from gym_minigrid.wrappers import FullyObsWrapper
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "purls"))
from utils.logs import debug, info, success  # noqa


env = gym.make("MiniGrid-LavaCrossingS11N5-v0")
# env = gym.make("MiniGrid-ObstructedMaze-2Q-v0")
# env = gym.make("MiniGrid-LockedRoom-v0")
# env = gym.make("MiniGrid-SimpleCrossingS9N2-v0")
# env = FullyObsWrapper(env)
env.render()
while True:
    action = input()

    fwd_pos = env.front_pos
    fwd_cell = env.grid.get(*fwd_pos)

    # turn left
    if action == "a":
        obs, reward, done, _ = env.step(0)
    # move forward
    elif action == "s":
        obs, reward, done, _ = env.step(2)
    # turn right
    else:
        obs, reward, done, _ = env.step(1)

    # add negative reward when agent go into lava
    if fwd_cell is not None and fwd_cell.type == "lava" and action == "s":
        reward = -1

    env.render()
    # print(obs)
    print("\n" + "-" * 50, env.step_count, "-" * 50)
    debug(f"Obs:\n{obs['image']}")
    debug(f"Obs shape: {obs['image'].shape}")
    debug(f"Direction: {obs['direction']}")
    info(f"Mission: {obs['mission']}")
    success(f"Reward: {reward}")
    if done:
        print("Completed with final reward of:", reward)
        env.reset()
        env.render()
