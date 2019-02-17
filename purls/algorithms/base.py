from abc import ABC, abstractmethod


class AlgorithmParameterError(Exception):
    def __init__(self, msg):
        self.msg = msg


class ReinforcementLearningAlgorithm(ABC):
    """
    The base class for reinforcement learning algorithms.

    All RL algorithms should extend this class.

    When initializing a sub-class, try:

    def __init__(self, env, args):
        super().__init__(env, args, DEFAULTS)

    where DEFAULTS is a dictionary with default values containing keys for
    some parameters like "learning_rate", "discount_factor" and "episodes".

    These default values will likely vary between different RL algorithms,
    hence why they need to be defined separately.

    "fully_obs" is a special default value found in DEFAULTS that tells us if
    that algorithm can be ran with minigrid FullyObsWrapper or not.

    Please specify "fully_obs": "compatible" if the algorithm works with a fully
    observable gridworld.

    Please specify "fully_obs": required" if the algorithm will ONLY work with
    a fully observable gridworld (like Q-table)

    If the algorithm is not compatible with the FulyObsWrapper, you can specify "nope"
    or any other string (except for compatible/required of course!)

    """

    def __init__(self, env, args, defaults):
        self.env = env
        if "learning_rate" in args and args.learning_rate:
            self.lr = args.learning_rate
        else:
            self.lr = defaults["learning_rate"]

        if "discount_factor" in args and args.discount_factor:
            self.y = args.discount_factor
        else:
            self.y = defaults["discount_factor"]

        if "episodes" in args and args.episodes:
            self.num_episodes = args.episodes
        else:
            self.num_episodes = defaults["episodes"]

        if "render_interval" in args:
            self.render_interval = args.render_interval

        if "save_interval" in args:
            self.save_interval = args.save_interval

        if args.model:
            self.model_name = args.model
        else:
            if self.save_interval > 0:
                raise AlgorithmParameterError(
                    f"Save interval set to {self.save_interval} but no model name specified!"
                )
            self.model_name = None

        if args.seed:
            self.seed = args.seed
        else:
            self.seed = None

        if args.fully_obs:
            if defaults["fully_obs"] in ["compatible", "required"]:
                self.fully_observable = True
            else:
                raise AlgorithmParameterError(
                    f"This algorithm is not compatible with the FullyObs wrapper!"
                )
        else:
            if defaults["fully_obs"] != "required":
                self.fully_observable = False
            else:
                raise AlgorithmParameterError(f"This algorithm requires the FullyObs wrapper!")

    @abstractmethod
    def train(self):
        """
        Train a model. Please respect save_interval, render_interval, seed etc.
        """
        pass

    @abstractmethod
    def save(self):
        """
        Save a model. Used by train at the interval specified by save_interval.
        """
        pass

    @abstractmethod
    def load(self):
        """
        Load a model. Used by visualize and evaluate to load a trained model.
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
