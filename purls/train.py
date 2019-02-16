from loguru import logger

from purls.utils.logs import info


def train(args):
    try:
        with logger.catch(reraise=True):
            info("not done yet")
            raise NotImplementedError
    except Exception:
        return 1
    return 0
