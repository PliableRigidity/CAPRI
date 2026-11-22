"""MuJoCo-only simulation of the Capri robot and arena, with a tactic-level RL environment.

The package and class names (``autobot``, ``AutoBotSim``, ``AutoBotEnv``) predate the name Capri and are kept:
scripts, tests and saved checkpoints refer to them.
"""

from .core import AutoBotSim

__all__ = ["AutoBotSim"]
