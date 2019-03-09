import os
import time
from abc import ABC, abstractmethod

from torch import load, save

from purls.utils.logs import debug
from purls.algorithms import AlgorithmError


class ReinforcementLearningAlgorithm(ABC):
    """
    The base class for reinforcement learning algorithms.

    All RL algorithms should extend this class.

    When initializing a sub-class, try:

    def __init__(self, env, args):
        super().__init__(
            env,
            args,
            default_learning_rate=...,
            default_discount_factor=...,
            default_start_eps=...,
            default_end_eps=...,
            default_annealing_steps=...,
            default_num_episodes=...,
        )

    These default values will likely vary between different RL algorithms,
    hence why they need to be defined separately.
    """

    def __init__(
        self,
        env,
        args,
        default_learning_rate,
        default_discount_factor,
        default_start_eps,
        default_end_eps,
        default_annealing_steps,
        default_num_episodes,
    ):
        self.env = env
        self.lr = getattr(args, "learning_rate", None) or default_learning_rate
        self.y = getattr(args, "discount_factor", None) or default_discount_factor
        self.start_eps = getattr(args, "start_eps", None) or default_start_eps
        self.end_eps = getattr(args, "end_eps", None) or default_end_eps
        self.annealing_steps = getattr(args, "annealing_steps", None) or default_annealing_steps
        self.num_episodes = getattr(args, "episodes", None) or default_num_episodes
        self.eps_decay = (self.start_eps - self.end_eps) / self.annealing_steps

        self.render_interval = getattr(args, "render_interval", None) or 0
        self.save_interval = getattr(args, "save_interval", None) or 0
        self.model_name = getattr(args, "model_name", None) or self._default_model_name()
        self.fps = getattr(args, "fps", None) or 2
        self.seed = getattr(args, "seed", None)

        # TODO: make sure this errors out when a subclass does not define self.model
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
        files = [f.rpartition(".pt")[0] for f in os.listdir("models") if f != ".gitignore"]
        if self.model_name not in files:
            valid_model_names = ", ".join(files)
            raise AlgorithmError(f"Choose a valid model name: {valid_model_names}")
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

    def _default_model_name(self):
        return f"{type(self).__name__}__{self.env.spec.id}__{time.strftime('%Y-%m-%d__%H-%M-%S')}"

    subclasses = []

    def __init_subclass__(cls, **kwargs):
        """
        Keep track of all subclasses of ReinforcementLearningAlgorithm
        """
        super().__init_subclass__(**kwargs)
        cls.subclasses.append(cls)
