import logging
import sys

def setup_logging(level="INFO"):
    """Configure global logging."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

def get_logger(name: str = __name__) -> logging.Logger:
    """Convenience function to get a logger."""
    return logging.getLogger(name)


