import logging

_LOGGER_NAME = "python_dpo"
_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"

_configured = False


def configure_logging(level: str = "INFO") -> None:
    """Configure the python_dpo logger with a single stderr handler."""
    global _configured
    if _configured:
        return

    numeric_level = logging.getLevelName(level.upper())
    if not isinstance(numeric_level, int):
        raise ValueError(f"Unknown log level: {level!r}")

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(fmt=_FORMAT, datefmt=_DATEFMT))

    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(numeric_level)
    logger.addHandler(handler)
    logger.propagate = False

    _configured = True
