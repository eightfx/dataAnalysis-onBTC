import logging
from pathlib import Path
root_path = Path(__file__).resolve().parent.parent.parent


def getLogger1(name):
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    handler = logging.FileHandler(root_path / "logs" / "logger1.log")
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(levelname)-9s  %(asctime)s  [%(name)s] %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def errorLogger(module_name):
    logging.basicConfig(level=logging.ERROR)
    logger = logging.getLogger(module_name)
    logger.setLevel(logging.ERROR)
    handler = logging.FileHandler(root_path / "logs" / "errors.log")
    handler.setLevel(logging.ERROR)
    formatter = logging.Formatter(
        '%(levelname)-9s  %(asctime)s  [%(name)s] %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger
