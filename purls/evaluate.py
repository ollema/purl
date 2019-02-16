from loguru import logger


def evaluate(args):
    try:
        with logger.catch(reraise=True):
            raise NotImplementedError
    except Exception:
        return 1
    return 0
