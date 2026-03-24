"""Allow ``python -m src <command>`` from the project root directory.

This makes the package executable as a module:
    python -m src extract --test
    python -m src train
    python -m src infer --mode motion
    python -m src validate
"""
from .main import main

main()
