from loguru import logger

logger.level("DEBUG", no=10, color="<white><cyan>")
logger.level("INFO", no=20, color="<white>")
logger.level("IMPORTANT", no=22, color="<white><bold>")


def get_format(timestamps):
    fmt = "<level>{level: <7}</level> | <level>{message}</level>"

    if timestamps:
        fmt = "{time:MM-DD-YYYY - HH:mm:ss} | " + fmt

    def formatter(record):
        if "important" in record["extra"]:
            final_fmt = "<bold>" + fmt + "</bold>"
        else:
            final_fmt = fmt
        return final_fmt + "\n{exception}"

    return formatter


def debug(message):
    logger.debug(message)


def info(message):
    logger.info(message)


def important(message):
    logger.bind(important=True).info(message)


def success(message):
    logger.success(message)


def warning(message):
    logger.warning(message)


def error(message):
    logger.error(message)
