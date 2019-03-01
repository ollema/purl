import os
from abc import ABC, abstractmethod

from torch import load, save

from purls.utils.logs import debug


class AlgorithmParameterError(Exception):
    def __init__(self, msg):
        self.msg = msg


class ReinforcementLearningAlgorithm(ABC):
    """
    The base class for reinforcement learning algorithms.

    All RL algorithms should extend this class.

    When initializing a sub-class, try:

    def __init__(self, args):
        super().__init__(
            args,
            default_learning_rate=...,
            default_discount_factor=...,
            default_num_episodes=...,
        )

    These default values will likely vary between different RL algorithms,
    hence why they need to be defined separately.
    """

    def __init__(self, args, default_learning_rate, default_discount_factor, default_num_episodes):
        self.lr = getattr(args, "learning_rate", None) or default_learning_rate
        self.y = getattr(args, "discount_factor", None) or default_discount_factor
        self.num_episodes = getattr(args, "episodes", None) or default_num_episodes
        self.render_interval = getattr(args, "render_interval", None) or 0
        self.save_interval = getattr(args, "save_interval", None) or 0
        self.model_name = getattr(args, "model_name", None)
        self.fps = getattr(args, "fps", None) or 2
        self.seed = getattr(args, "seed", None)

        if self.save_interval > 0 and self.model_name is None:
            raise AlgorithmParameterError(
                f"Save interval set to {self.save_interval} but no model name specified!"
            )

        self.model = NotImplemented

    def save(self):
        """
        Save a model. Used by train at the interval specified by save_interval.
        """
        path = f"models/{self.model_name}.pt"
        save(self.model, path)
        debug(f"model saved in {path}")

    def load(self):
        """
        Load a model. Used by visualize and evaluate to load a trained model.
        """
        files = [f.strip(".pt") for f in os.listdir("models") if f != ".gitignore"]
        if self.model_name not in files:
            valid_model_names = ", ".join(files)
            raise AlgorithmParameterError(f"Choose a valid model name: {valid_model_names}")
        path = f"models/{self.model_name}.pt"
        debug(f"model loaded from {path}")
        return load(path)

    @abstractmethod
    def train(self):
        """
        Train a model. Please respect save_interval, render_interval, seed etc.
        """
        pass

    @abstractmethod
    def visualize(self):
        """
        Visualize (render) a model.
        """
        pass

    @abstractmethod
    def evaluate(self):
        """
        Evaluate a model's performance.
        """
        pass
