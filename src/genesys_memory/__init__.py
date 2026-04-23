"""Genesys Memory — the intelligence layer for AI memory."""
import logging


def configure_logging(level: int = logging.WARNING) -> None:
    """Set up default logging for the genesys_memory package.

    Only adds a handler if the root genesys_memory logger has none,
    so host applications that configure their own logging are unaffected.
    """
    pkg_logger = logging.getLogger("genesys_memory")
    if not pkg_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
        )
        pkg_logger.addHandler(handler)
    pkg_logger.setLevel(level)
