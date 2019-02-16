import argparse
from sys import stderr

from loguru import logger

from purls.utils.logs import get_format, info, debug, error

from purls.train import train
from purls.visualize import visualize
from purls.evaluate import evaluate


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log-time-stamps", action="store_true", default=False)

    subp = p.add_subparsers(dest="subcmd_name")

    p_train = subp.add_parser("train")
    p_train.add_argument("-a", "--algorithm", type=str, required=True)
    p_train.add_argument("-e", "--env", type=str, required=True)
    p_train.add_argument("-m", "--model", type=str, default=None)
    p_train.add_argument("-s", "--seed", type=int, default=1)
    p_train.set_defaults(command=train)

    p_visualize = subp.add_parser("visualize")
    p_visualize.add_argument("-m", "--model", type=str, required=True)
    p_visualize.add_argument("-s", "--seed", type=int, default=1)
    p_visualize.set_defaults(command=visualize)

    p_evaluate = subp.add_parser("evaluate")
    p_evaluate.add_argument("-m", "--model", type=str, required=True)
    p_evaluate.add_argument("-s", "--seed", type=int, default=1)
    p_evaluate.set_defaults(command=evaluate)

    args = p.parse_args()

    fmt = get_format(args.log_time_stamps)
    config = {"handlers": [{"sink": stderr, "format": fmt}]}
    logger.configure(**config)

    if not hasattr(args, "command"):
        error("You need to select a subcommand {train, visualize, evaluate}")
        info("\n" + p_train.format_usage() + p_visualize.format_usage() + p_evaluate.format_usage())
        return 1

    try:
        result = args.command(args)
        debug(f"{args.subcmd_name} returned {result}")
    except KeyboardInterrupt:
        error("Interrupted by user")
    return result


if __name__ == "__main__":
    exit_status = main()
    exit(exit_status)
