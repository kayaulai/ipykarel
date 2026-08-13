from ipykarel import activate_world, KarelWorld, KarelError


SIMPLE_WORLD = [
    "......",
    "......",
    "......",
    ">.....",
]


def test_load_and_move():
    world = activate_world(ascii_lines=SIMPLE_WORLD)
    assert isinstance(world, KarelWorld)
    assert world.robot_pos == [0, 0]
    assert world.robot_dir == 0  # facing east

    world.move()
    assert world.robot_pos == [1, 0]


def test_turn_left_cycles_direction():
    world = activate_world(ascii_lines=SIMPLE_WORLD)
    start_dir = world.robot_dir
    for _ in range(4):
        world.turn_left()
    assert world.robot_dir == start_dir


def test_put_and_pick_beeper():
    world = activate_world(ascii_lines=SIMPLE_WORLD)
    assert not world.beeper_is_present()
    world.put_beeper()
    assert world.beeper_is_present()
    world.pick_beeper()
    assert not world.beeper_is_present()


def test_pick_beeper_raises_when_empty():
    world = activate_world(ascii_lines=SIMPLE_WORLD)
    try:
        world.pick_beeper()
        assert False, "expected KarelError"
    except KarelError:
        pass


def test_blocked_move_raises():
    world = activate_world(ascii_lines=SIMPLE_WORLD)
    world.add_wall_cell(1, 0)
    try:
        world.move()
        assert False, "expected KarelError"
    except KarelError:
        pass


def test_ascii_round_trip():
    world = activate_world(ascii_lines=SIMPLE_WORLD)
    assert world.to_ascii() == SIMPLE_WORLD
