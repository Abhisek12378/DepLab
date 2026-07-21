"""FastAPI transport for the DepLab advisory backend."""

from typing import Any


def create_app(*args: Any, **kwargs: Any) -> Any:
    """Import FastAPI lazily so core DepLab remains dependency-light."""
    from .main import create_app as factory

    return factory(*args, **kwargs)


__all__ = ["create_app"]
