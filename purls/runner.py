from os import listdir

import gym
import gym_minigrid  # noqa
from loguru import logger
from purls.algorithms.base import AlgorithmParameterError
from purls.algorithms.q_table import QLearningWithTable
from purls.utils.logs import error

ALGORITHMS = {"q-table": QLearningWithTable}


def run(action, args):
    try:
        if args.algorithm not in ALGORITHMS:
            valid_algorithms = ", ".join(ALGORITHMS.keys())
            error(f"Choose a valid algorithm: {valid_algorithms}")
            return 1
        try:
            env = gym.make(args.environment)
        except gym.error.Error:
            error(f"Choose a valid enviroment!")
            return 1

        algo = ALGORITHMS[args.algorithm](env=env, args=args)

        files = [f.strip(".pt") for f in listdir("models") if f != ".gitignore"]
        if algo.model_name is not None and algo.model_name not in files:
            valid_model_names = ", ".join(files)
            error(f"Choose a valid model name: {valid_model_names}")
            return 1

        with logger.catch(reraise=True):
            if action == "train":
                algo.train()
            if action == "visualize":
                algo.visualize()
            if action == "evaluate":
                algo.evaluate()

    except AlgorithmParameterError as e:
        error(e.msg)
        return 1

    except Exception as e:
        error(e)
        return 1
    return 0
