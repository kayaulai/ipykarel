# karel-robot

A small ASCII-world simulator and matplotlib-based animator for the classic
"Karel the Robot" teaching exercises. Works both inside Jupyter notebooks
(renders an animation with `IPython.display`) and in plain Python scripts
(returns a `matplotlib.animation.FuncAnimation` you can save to a file).

> A prototype of this code was created with ChatGPT.

## Install

```bash
pip install -r requirements.txt
pip install -e .
```

(or, once packaged, `pip install .`)

## Optional: emoji font

The animation renders Karel and treasure as emoji (🤖 / 💰). If your system
doesn't already have a color/emoji font matplotlib can find, drop a
`Symbola.ttf` (or similar) file into `karel/fonts/` before installing, or
point to one at runtime:

```bash
export KAREL_EMOJI_FONT_PATH=/path/to/Symbola.ttf
```

If no emoji font is found, matplotlib falls back to whatever fonts are
available on your system.

## Usage

ASCII world format — one row per line, read bottom-to-top as (x, y):

- `#` wall cell
- `$` treasure
- `0`-`9` beeper count at a cell
- `>` `<` `^` `v` Karel's starting position + facing (E/W/N/S)
- `.` empty cell

```python
from karel import activate_world

world = activate_world(ascii_lines=[
    "......",
    "..$...",
    "......",
    ">.....",
])

world.move()
world.move()
world.turn_left()
world.put_beeper()

ani = world.animate(interval=300)
ani.save("karel_run.gif", writer="pillow")
```

### Inside a Jupyter notebook

```python
from karel import run_karel_ipynb_pipeline

def program():
    move()
    move()
    turn_left()
    put_beeper()

run_karel_ipynb_pipeline(program, ascii_lines=[
    "......",
    "......",
    ">.....",
])
```

### Testing a student's solution against an expected end state

```python
from karel import test_karel

test_karel(
    program,
    ascii_lines=start_world_lines,
    expected_ascii_lines=expected_world_lines,
)
```

## Package layout

```
karel_robot/
├── karel/
│   ├── __init__.py       # public API
│   ├── world.py          # KarelWorld, ASCII (de)serialization, pipeline helpers
│   └── fonts/             # (optional) drop an emoji .ttf here
├── tests/
│   └── test_world.py
├── pyproject.toml
├── requirements.txt
└── README.md
```
