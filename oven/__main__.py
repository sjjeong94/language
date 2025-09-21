"""
Main entry point for oven package when run as module.

This allows running: python -m oven kernel.py
"""

from .cli import main

if __name__ == "__main__":
    main()
