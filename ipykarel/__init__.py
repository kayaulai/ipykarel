"""
ipykarel - a small ASCII-world simulator and matplotlib-based animator
for teaching the classic "Karel the Robot" exercises, usable both inside
Jupyter notebooks and in plain Python scripts.
"""

from .world import (
    KarelWorld,
    KarelError,
    load_ascii_state,
    get_ascii_from_state,
    get_state_from_world,
    activate_world,
    inject_karel_commands,
    run_karel_ipynb_pipeline,
    test_karel,
    DIRECTION_VECTORS,
    DIRECTION_ARROWS,
)

__version__ = "0.1.2"

__all__ = [
    "KarelWorld",
    "KarelError",
    "load_ascii_state",
    "get_ascii_from_state",
    "get_state_from_world",
    "activate_world",
    "inject_karel_commands",
    "run_karel_ipynb_pipeline",
    "test_karel",
    "DIRECTION_VECTORS",
    "DIRECTION_ARROWS",
]
