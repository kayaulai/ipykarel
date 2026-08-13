"""
Core Karel-the-Robot simulation engine.

This module is a packaged version of a prototype originally written for use
inside Jupyter/IPython notebooks. It provides:

- ASCII world loading/serialization helpers
- The `KarelWorld` class (state, movement rules, matplotlib animation)
- Notebook-style helpers (`run_karel_ipynb_pipeline`, `test_karel`) that make
  it easy to run a student's Karel program against a world and visualize it.

A prototype of this code was created with ChatGPT.
"""

from __future__ import annotations

import os
import re
import inspect
import traceback
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.font_manager as fm

try:
    # These are optional: only needed when running inside a notebook.
    from IPython.display import HTML, display, clear_output
    _HAS_IPYTHON = True
except ImportError:  # pragma: no cover - IPython not installed
    _HAS_IPYTHON = False


# --------------------------------------------------------------------------
# Emoji font setup
# --------------------------------------------------------------------------
# Look for a bundled emoji font (e.g. Symbola.ttf) inside this package's
# fonts/ directory. If it isn't present, fall back gracefully to whatever
# system emoji font matplotlib can find, so the package still works without
# requiring a font file to be shipped/downloaded.
_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_FONT_PATH = os.path.join(_PACKAGE_DIR, "fonts", "Symbola.ttf")


def _load_emoji_font(font_path: str | None = None) -> fm.FontProperties | None:
    candidate = font_path or os.environ.get("KAREL_EMOJI_FONT_PATH") or _DEFAULT_FONT_PATH
    if candidate and os.path.isfile(candidate):
        return fm.FontProperties(fname=candidate)
    return None  # matplotlib will fall back to rcParams font list below


emoji_font = _load_emoji_font()

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = [
    "Symbola",
    "Segoe UI Emoji",
    "Noto Sans Emoji",
    "Apple Color Emoji",
    "Noto Emoji",
    "Noto Color Emoji",
    "Twemoji Mozilla",
    "EmojiOne Color",
    "OpenMoji",
    "Arial",
    "Helvetica",
    "DejaVu Sans",
]

# Direction: 0 = East, 1 = North, 2 = West, 3 = South
DIRECTION_VECTORS = [(1, 0), (0, 1), (-1, 0), (0, -1)]
DIRECTION_ARROWS = ["→", "↑", "←", "↓"]


class KarelError(Exception):
    """Custom exception to stop Karel on error."""
    pass


# --------------------------------------------------------------------------
# ASCII world (de)serialization
# --------------------------------------------------------------------------

def load_ascii_state(ascii_lines=None, filepath=""):
    state = dict()

    if ascii_lines is None:
        with open(filepath, 'r') as f:
            ascii_lines = [line.strip() for line in f.readlines()]

    directions = {
        '>': 0,
        '<': 2,
        '^': 1,
        'v': 3,
    }

    state["size"] = (len(ascii_lines[0]), len(ascii_lines))
    state["robot_pos"] = None

    state["treasures"] = set()
    state["wall_cells"] = set()  # set of (x, y) cells Karel cannot enter
    state["beepers"] = defaultdict(int)  # maps (x, y) → count

    for row, line in enumerate(reversed(ascii_lines)):
        for col, char in enumerate(line):
            if char == '#':
                state["wall_cells"].add((col, row))
            elif char == '.':
                continue  # empty cell
            elif char == '$':
                state["treasures"].add((col, row))
            elif char in directions:
                state["robot_pos"] = [col, row]
                state["robot_dir"] = directions[char]
            elif char.isdigit():
                state["beepers"][(col, row)] = int(char)

    if state["robot_pos"] is None:
        raise ValueError("No robot start position found in ASCII world!")

    return state


def get_ascii_from_state(state, as_list=True):
    directions = {
        0: '>',
        2: '<',
        1: '^',
        3: 'v',
    }

    lines = [""] * state["size"][1]
    for row in range(state["size"][1]):
        line_index = state["size"][1] - row - 1
        lines[line_index] = ""
        for col in range(state["size"][0]):
            if (col, row) in state["treasures"]:
                lines[line_index] += "$"
            elif (col, row) in state["wall_cells"]:
                lines[line_index] += "#"
            elif [col, row] == state["robot_pos"]:
                lines[line_index] += directions[state["robot_dir"]]
            elif state["beepers"][(col, row)] > 0:
                lines[line_index] += str(state["beepers"][(col, row)])
            else:
                lines[line_index] += "."

    if not as_list:
        return "\n".join(lines)
    else:
        return lines


def get_state_from_world(world: "KarelWorld"):
    state = dict()
    state["size"] = world.size
    state["robot_pos"] = world.robot_pos
    state["robot_dir"] = world.robot_dir
    state["treasures"] = world.treasures
    state["wall_cells"] = world.wall_cells
    state["beepers"] = world.beepers
    return state


# --------------------------------------------------------------------------
# KarelWorld
# --------------------------------------------------------------------------

class KarelWorld:
    def __init__(self, state=None, size=None):
        if state is None:
            assert size is not None
            self.size = size
            self.robot_pos = [0, 0]
            self.robot_dir = 0  # 0=E, 1=N, 2=W, 3=S
            self.treasures = set()
            self.wall_cells = set()  # set of (x, y) cells Karel cannot enter
            self.beepers = defaultdict(int)  # maps (x, y) → count
        else:
            self.size = state["size"]
            self.robot_pos = state["robot_pos"]
            self.robot_dir = state["robot_dir"]
            self.treasures = state["treasures"]
            self.wall_cells = state["wall_cells"]
            self.beepers = state["beepers"]

        self.frames = []
        self.snapshot("start")
        self.halted = False

        self.initial_robot_pos = self.robot_pos.copy()
        self.initial_robot_dir = self.robot_dir
        self.initial_beepers = self.beepers.copy()

    def set_curr_to_init(self):
        self.robot_pos = self.initial_robot_pos.copy()
        self.robot_dir = self.initial_robot_dir
        self.beepers = self.initial_beepers.copy()
        self.frames = []
        self.snapshot("start")
        self.halted = False

    def reset(self):
        self.robot_pos = self.initial_robot_pos
        self.robot_dir = self.initial_robot_dir
        self.beepers = self.initial_beepers.copy()
        self.frames = []
        self.snapshot("start")
        self.halted = False

    def add_wall_cell(self, x, y):
        """Mark (x, y) as a wall that Karel cannot enter."""
        if 0 <= x < self.size[0] and 0 <= y < self.size[1]:
            self.wall_cells.add((x, y))

    def add_treasure(self, x, y):
        """Mark (x, y) as a treasure that Karel cannot enter."""
        if 0 <= x < self.size[0] and 0 <= y < self.size[1]:
            self.treasures.add((x, y))

    def move(self):
        if self.front_is_blocked():
            raise KarelError("Can't move. Karel is blocked!")

        dx, dy = DIRECTION_VECTORS[self.robot_dir]
        self.robot_pos[0] += dx
        self.robot_pos[1] += dy
        self.snapshot("move")

    def turn_left(self):
        self.robot_dir = (self.robot_dir + 1) % 4
        self.snapshot("turn_left")

    def put_beeper(self):
        pos = tuple(self.robot_pos)
        self.beepers[pos] += 1
        self.snapshot("put beeper")

    def pick_beeper(self):
        pos = tuple(self.robot_pos)
        if self.beepers[pos] > 0:
            self.beepers[pos] -= 1
            if self.beepers[pos] == 0:
                del self.beepers[pos]
            self.snapshot("pick beeper")
        else:
            raise KarelError("Can't pick beeper from empty location")

    def beeper_is_present(self):
        return self.beepers.get(tuple(self.robot_pos), 0) > 0

    def block_is_blocked(self, x, y):
        if not (0 <= x < self.size[0] and 0 <= y < self.size[1]):
            return True
        if (x, y) in self.wall_cells:
            return True
        if (x, y) in self.treasures:
            return True
        return False

    def block_is_clear(self, x, y):
        return not self.block_is_blocked(x, y)

    def front_is_clear(self):
        dx, dy = DIRECTION_VECTORS[self.robot_dir]
        x, y = self.robot_pos
        new_x, new_y = x + dx, y + dy
        return self.block_is_clear(new_x, new_y)

    def right_is_clear(self):
        right_dir = (self.robot_dir - 1) % 4
        dx, dy = DIRECTION_VECTORS[right_dir]
        x, y = self.robot_pos
        new_x, new_y = x + dx, y + dy
        return self.block_is_clear(new_x, new_y)

    def front_is_blocked(self):
        dx, dy = DIRECTION_VECTORS[self.robot_dir]
        x, y = self.robot_pos
        new_x, new_y = x + dx, y + dy
        return self.block_is_blocked(new_x, new_y)

    def front_is_treasure(self):
        dx, dy = DIRECTION_VECTORS[self.robot_dir]
        new_x = self.robot_pos[0] + dx
        new_y = self.robot_pos[1] + dy
        return (new_x, new_y) in self.treasures

    def facing_north(self): return self.robot_dir == 1
    def facing_south(self): return self.robot_dir == 3
    def facing_east(self): return self.robot_dir == 0
    def facing_west(self): return self.robot_dir == 2

    def snapshot(self, action=""):
        frame = {
            "robot": tuple(self.robot_pos),
            "dir": self.robot_dir,
            "beepers": dict(self.beepers),
            "treasures": set(self.treasures),
            "label": action,
        }
        self.frames.append(frame)

    def animate(self, interval=800):
        fig_width = self.size[0] * 0.4
        fig_height = self.size[1] * 0.4
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        ax.set_xlim(0, self.size[0])
        ax.set_ylim(0, self.size[1])

        ax.set_xticks(np.arange(0, self.size[0] + 1), minor=False)
        ax.set_yticks(np.arange(0, self.size[1] + 1), minor=False)
        ax.tick_params(which='major', bottom=False, left=False, labelbottom=False, labelleft=False)

        ax.set_xticks(np.arange(0.5, self.size[0]), minor=True)
        ax.set_yticks(np.arange(0.5, self.size[1]), minor=True)
        ax.set_xticklabels(range(self.size[0]), minor=True)
        ax.set_yticklabels(reversed(range(self.size[1])), minor=True)
        ax.tick_params(which='minor', bottom=True, left=True, labelbottom=True, labelleft=True)
        ax.grid(True)

        ax.set_aspect(1)
        ax.invert_yaxis()

        # Walls
        for x, y in self.wall_cells:
            y_flipped = self.size[1] - 1 - y
            ax.add_patch(plt.Rectangle((x, y_flipped), 1, 1, color="black"))

        # Treasures
        for tx, ty in self.treasures:
            ty_flipped = self.size[1] - 1 - ty
            ax.text(tx + 0.5, ty_flipped + 0.5, "💰", fontsize=16, ha="center",
                     va="center", fontproperties=emoji_font)

        # Karel
        robot_marker = ax.text(
            0.5, self.size[1] - 0.5, "🤖", fontsize=16, ha="center",
            va="center", fontproperties=emoji_font
        )

        # Direction arrow
        dir_marker = ax.text(
            0.5, self.size[1] - 0.5, "", fontsize=12, ha="center",
            va="center", color="blue",
        )

        beeper_markers = {}
        ax.set_title("")

        def update(i):
            frame = self.frames[i]

            x, y = frame["robot"]
            robot_marker.set_position((x + 0.5, self.size[1] - y - 0.5))

            facing = frame["dir"]
            dir_marker.set_text(DIRECTION_ARROWS[facing])
            offsets = {
                0: (+.4, 0),  # E
                1: (0, -.4),  # N
                2: (-.4, 0),  # W
                3: (0, +.4),  # S
            }
            dx, dy = offsets[facing]
            dir_marker.set_position((x + .5 + dx, self.size[1] - y - .5 + dy))

            label_text = ax.set_title(f"Step {i + 1}: {frame['label']}")

            for m in list(beeper_markers.values()):
                m.remove()
            beeper_markers.clear()
            for (bx, by), count in frame["beepers"].items():
                by_flipped = self.size[1] - 1 - by
                b = ax.text(bx + 0.5, by_flipped + 0.5, str(count), fontsize=14,
                             ha='center', va='center', color='green')
                beeper_markers[(bx, by)] = b

            beeper_count = frame["beepers"].get((x, y), 0)
            robot_marker.set_alpha(0.4 if beeper_count > 0 else 1.0)

            return [robot_marker, dir_marker, label_text] + list(beeper_markers.values())

        ani = animation.FuncAnimation(fig, update, frames=len(self.frames),
                                       interval=interval, blit=True, repeat=False)
        plt.close(fig)
        return ani

    def to_ascii(self):
        state = get_state_from_world(self)
        return get_ascii_from_state(state)


# --------------------------------------------------------------------------
# World activation / command injection
# --------------------------------------------------------------------------

def activate_world(ascii_lines=None, filepath=""):
    world = KarelWorld(state=load_ascii_state(ascii_lines, filepath))
    inject_karel_commands(world)
    return world


def inject_karel_commands(world):
    globals()['move'] = world.move
    globals()['turn_left'] = world.turn_left
    globals()['put_beeper'] = world.put_beeper
    globals()['pick_beeper'] = world.pick_beeper
    globals()['beeper_is_present'] = world.beeper_is_present
    globals()['front_is_blocked'] = world.front_is_blocked
    globals()['right_is_clear'] = world.right_is_clear
    globals()['front_is_treasure'] = world.front_is_treasure
    globals()['facing_north'] = world.facing_north
    globals()['facing_south'] = world.facing_south
    globals()['facing_east'] = world.facing_east
    globals()['facing_west'] = world.facing_west


API_FUNCS = [
    "move", "turn_left", "put_beeper", "pick_beeper",
    "beeper_is_present", "front_is_blocked", "front_is_treasure",
    "right_is_clear",
    "facing_north", "facing_south", "facing_east", "facing_west",
]


def run_karel_ipynb_pipeline(
    f, inject_globals=None, ascii_lines=None, filepath="", return_ascii=False, **args
):
    world = activate_world(ascii_lines, filepath)

    if inject_globals is None:
        inject_globals = inspect.currentframe().f_back.f_globals

    inject_globals['_CURRENT_WORLD'] = world

    def move(): inject_globals['_CURRENT_WORLD'].move()
    def turn_left(): inject_globals['_CURRENT_WORLD'].turn_left()
    def put_beeper(): inject_globals['_CURRENT_WORLD'].put_beeper()
    def pick_beeper(): inject_globals['_CURRENT_WORLD'].pick_beeper()
    def beeper_is_present(): return inject_globals['_CURRENT_WORLD'].beeper_is_present()
    def front_is_blocked(): return inject_globals['_CURRENT_WORLD'].front_is_blocked()
    def right_is_clear(): return inject_globals['_CURRENT_WORLD'].right_is_clear()
    def front_is_treasure(): return inject_globals['_CURRENT_WORLD'].front_is_treasure()
    def facing_north(): return inject_globals['_CURRENT_WORLD'].facing_north()
    def facing_south(): return inject_globals['_CURRENT_WORLD'].facing_south()
    def facing_east(): return inject_globals['_CURRENT_WORLD'].facing_east()
    def facing_west(): return inject_globals['_CURRENT_WORLD'].facing_west()

    inject_globals.update({
        'move': move,
        'turn_left': turn_left,
        'put_beeper': put_beeper,
        'pick_beeper': pick_beeper,
        'beeper_is_present': beeper_is_present,
        'front_is_blocked': front_is_blocked,
        'right_is_clear': right_is_clear,
        'front_is_treasure': front_is_treasure,
        'facing_north': facing_north,
        'facing_south': facing_south,
        'facing_east': facing_east,
        'facing_west': facing_west,
    })

    try:
        print("ℹ️ Karel is still running. If this keeps going for minutes, check for infinite loops.",
              end="\r", flush=True)
        f(**args)
        print("ℹ️ Karel is done running; creating animation.                                        ",
              flush=True)
    except Exception as e:
        tb = traceback.extract_tb(e.__traceback__)
        _filename, lineno, _funcname, _text = tb[-1]
        error_message = f"❌ ERROR, line {lineno}: {type(e).__name__}: {e}"
        world.snapshot(error_message)
        print(f"❌ Execution halted: {e}")

    a = world.animate(interval=300)

    if _HAS_IPYTHON:
        clear_output(wait=True)
        display(HTML(a.to_jshtml()))

    if return_ascii:
        end_ascii = world.to_ascii()

    del world
    for name in API_FUNCS:
        inject_globals.pop(name, None)

    if return_ascii:
        return end_ascii
    return a


def test_karel(
    f, inject_globals=None, ascii_lines=None, expected_ascii_lines=None,
    filepath="", expected_filepath="", **args
):
    if inject_globals is None:
        inject_globals = inspect.currentframe().f_back.f_globals

    if expected_ascii_lines is None:
        with open(expected_filepath, 'r') as file:
            expected_ascii_lines = [line.strip() for line in file.readlines()]

    result = run_karel_ipynb_pipeline(
        f=f,
        inject_globals=inject_globals,
        ascii_lines=ascii_lines,
        filepath=filepath,
        return_ascii=True,
        **args
    )

    if result == expected_ascii_lines:
        print("✅ Final world state check passed for current world.")
    else:
        result_nokarel = [re.sub("[<^>v]", ".", x) for x in result]
        expected_ascii_lines_nokarel = [re.sub("[<^>v]", ".", x) for x in expected_ascii_lines]
        if result_nokarel == expected_ascii_lines_nokarel:
            raise KarelError(
                "❌ Beepers are in the right position, but Karel's position and/or "
                "direction is off."
            )
        else:
            raise KarelError("❌ Final world state is incorrect.")
