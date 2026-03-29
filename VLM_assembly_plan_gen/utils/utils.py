import logging
import os

import yaml


def read_yaml_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def configure_logging(log_level="INFO", log_file=None):
    """Simple one-time logging configuration."""
    level = getattr(logging, log_level.upper())

    handlers = []
    handlers.append(logging.StreamHandler())
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file, mode="w"))

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s:%(lineno)d: %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
        force=True,  # Override any existing configuration
    )

    # Reduce noise from third-party libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("spiff").setLevel(logging.WARNING)
    logging.getLogger("SpiffWorkflow").setLevel(logging.WARNING)
