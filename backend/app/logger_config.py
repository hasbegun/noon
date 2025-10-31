import logging
from logging.config import dictConfig

def setup_logging():
    """
    Configures the application's logging using a dictionary-based setup.
    This integrates with Uvicorn's default loggers.
    """
    config = {
        "version": 1,
        "disable_existing_loggers": False, # Keep Uvicorn's loggers active
        "formatters": {
            "default": {
                "()": "uvicorn.logging.DefaultFormatter",
                "fmt": "%(levelprefix)s %(asctime)s - %(name)s - %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
        },
        "handlers": {
            "default": {
                "formatter": "default",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stderr", # Log to stderr
            },
        },
        "loggers": {
            "app": {                        # Configure the logger for our specific application
                "handlers": ["default"],
                "level": "INFO",            # Set the default level to INFO
                "propagate": False,
            },
        },
    }
    dictConfig(config)
