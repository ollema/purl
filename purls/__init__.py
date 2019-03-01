import argparse
from sys import stderr

from loguru import logger

from purls.utils.logs import get_format, info, debug, error

from purls.runner import run


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log-time-stamps", action="store_true", default=False)

    subp = p.add_subparsers(dest="subcmd_name")

    p_train = subp.add_parser("train", formatter_class=argparse.RawTextHelpFormatter)
    p_train.add_argument(
        "--algorithm",
        type=str,
        required=True,
        metavar="algo",
        help="str:   reinforcement learning algorithm algo to use.",
    )
    p_train.add_argument(
        "--environment",
        type=str,
        required=True,
        metavar="env",
        help="str:   minigrid environment env to use.",
    )
    p_train.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        metavar="α",
        help="float: learning rate α to use.",
    )
    p_train.add_argument(
        "--discount-factor",
        type=float,
        default=None,
        metavar="γ",
        help="float: discount factor γ to use.",
    )
    p_train.add_argument(
        "--episodes",
        type=int,
        default=None,
        metavar="n",
        help="int:   train model for up to n episodes",
    )
    p_train.add_argument(
        "--render-interval",
        type=int,
        default=None,
        metavar="i",
        help="int:   if i > 0, render every i:th episode",
    )
    p_train.add_argument(
        "--save-interval",
        type=int,
        default=None,
        metavar="j",
        help="int:   if j > 0, save model every j:th episode",
    )
    p_train.add_argument(
        "--model-name",
        type=str,
        default=None,
        metavar="name",
        help="str:   save model as models/<name>.pt when (if) the model is saved",
    )
    p_train.add_argument(
        "--seed", type=int, default=None, metavar="seed", help="int:   seed used for all randomness"
    )
    p_train.add_argument(
        "--fps",
        type=int,
        default=None,
        metavar="fps",
        help="int:   rendering delay = 1/fps + time to compute next action",
    )
    p_train.set_defaults(action="train")

    p_visualize = subp.add_parser("visualize")
    p_visualize.add_argument(
        "--algorithm",
        type=str,
        required=True,
        metavar="algo",
        help="str:   reinforcement learning algorithm algo to use.",
    )
    p_visualize.add_argument(
        "--environment",
        type=str,
        required=True,
        metavar="env",
        help="str:   minigrid environment env to use.",
    )
    p_visualize.add_argument(
        "--model-name",
        type=str,
        default=None,
        metavar="name",
        help="str:   load model from models/<name>.pt",
    )
    p_visualize.add_argument(
        "--seed", type=int, default=None, metavar="seed", help="int:   seed used for all randomness"
    )
    p_visualize.add_argument(
        "--fps",
        type=int,
        default=None,
        metavar="fps",
        help="int:   rendering delay = 1/fps + time to compute next action",
    )
    p_visualize.set_defaults(action="visualize")

    p_evaluate = subp.add_parser("evaluate")
    p_evaluate.add_argument(
        "--algorithm",
        type=str,
        required=True,
        metavar="algo",
        help="str:   reinforcement learning algorithm algo to use.",
    )
    p_evaluate.add_argument(
        "--environment",
        type=str,
        required=True,
        metavar="env",
        help="str:   minigrid environment env to use.",
    )
    p_evaluate.add_argument(
        "--episodes",
        type=int,
        default=None,
        metavar="n",
        help="int:   evaluate model for up to n episodes",
    )
    p_evaluate.add_argument(
        "--model-name",
        type=str,
        default=None,
        metavar="name",
        help="str:   load model from models/<name>.pt",
    )
    p_evaluate.add_argument(
        "--seed", type=int, default=None, metavar="seed", help="int:   seed used for all randomness"
    )
    p_evaluate.set_defaults(action="evaluate")

    args = p.parse_args()

    fmt = get_format(args.log_time_stamps)
    config = {"handlers": [{"sink": stderr, "format": fmt}]}
    logger.configure(**config)

    if not hasattr(args, "action"):
        error("You need to select a subcommand {train, visualize, evaluate}")
        info("\n" + p_train.format_usage() + p_visualize.format_usage() + p_evaluate.format_usage())
        return 1
    try:
        result = run(args.action, args)
        debug(f"{args.subcmd_name} returned {result}")
    except KeyboardInterrupt:
        error("Interrupted by user")
        return 1
    return result


if __name__ == "__main__":
    exit_status = main()
    exit(exit_status)
