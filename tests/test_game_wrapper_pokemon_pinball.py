#
# License: See LICENSE.md file
# GitHub: https://github.com/Baekalfen/PyBoy
#

import os
import platform
import sys

import numpy as np
import pytest

from pyboy import PyBoy
from pyboy.plugins.game_wrapper_pokemon_pinball import Pokemon, SpecialMode, Stage
from pyboy.utils import WindowEvent


def test_pokemon_pinball_basics(pokemon_pinball_rom):
    pyboy = PyBoy(
        pokemon_pinball_rom,
        window_type="null",
    )
    pyboy.set_emulation_speed(0)
    assert pyboy.cartridge_title == "POKEPINBALLVPH"

    pokemon_pinball = pyboy.game_wrapper
    pokemon_pinball.start_game()

    assert pokemon_pinball.score == 0
    assert pokemon_pinball.balls_left == 2
    assert pokemon_pinball.current_stage == Stage.RED_BOTTOM.value
    assert pokemon_pinball.special_mode_active == False
    pyboy.stop()


def test_pokemon_pinball_advanced(pokemon_pinball_rom):
    pyboy = PyBoy(pokemon_pinball_rom, window_type="null")
    pyboy.set_emulation_speed(0)
    assert pyboy.cartridge_title == "POKEPINBALLVPH"

    pokemon_pinball = pyboy.game_wrapper
    pokemon_pinball.start_game(stage=Stage.BLUE_BOTTOM)
    pyboy.button_press("a")
    pyboy.button_press("left")
    pyboy.tick(500, render=False)
    pyboy.button_release("left")
    pyboy.button_release("a")
    pyboy.tick(20, render=False)
    pyboy.button_press("left")
    pyboy.button_press("a")
    pyboy.tick(700, render=False)

    assert pokemon_pinball.score == 123200
    assert pokemon_pinball.special_mode == SpecialMode.CATCH.value
    assert pokemon_pinball.current_stage == Stage.BLUE_BOTTOM.value
    assert pokemon_pinball.special_mode_active == True
    assert pokemon_pinball.balls_left == 2

    pyboy.stop()


def test_pokemon_catch_mode(pokemon_pinball_rom):
    pyboy = PyBoy(pokemon_pinball_rom, window_type="null")
    pyboy.set_emulation_speed(0)
    assert pyboy.cartridge_title == "POKEPINBALLVPH"

    pokemon_pinball = pyboy.game_wrapper
    pokemon_pinball.start_game(stage=Stage.RED_BOTTOM)
    pyboy.button_press("a")
    pyboy.button_press("left")
    pyboy.tick(50, render=False)
    pokemon_pinball.start_catch_mode()
    pyboy.tick(270, render=False)
    pyboy.button_release("left")
    pyboy.button_release("a")
    pyboy.button("select")
    pyboy.tick(20, render=False)
    pyboy.button_press("left")
    pyboy.button_press("a")
    pyboy.tick(500, render=False)
    pyboy.button_release("left")
    pyboy.tick(21, render=False)
    pyboy.button_press("left")
    pyboy.tick(700, render=False)

    assert pokemon_pinball.score == 15635100
    assert pokemon_pinball.has_pokemon(Pokemon.BULBASAUR)
    assert pokemon_pinball.has_pokemon(Pokemon.CHARMANDER) == False
    assert pokemon_pinball.get_unique_pokemon_caught() == 1

    pyboy.stop()


def test_pokemon_pinball_game_over(pokemon_pinball_rom):
    pyboy = PyBoy(pokemon_pinball_rom, window_type="null")
    pyboy.set_emulation_speed(0)
    assert pyboy.cartridge_title == "POKEPINBALLVPH"

    pokemon_pinball = pyboy.game_wrapper
    pokemon_pinball.start_game()
    pyboy.button("a")
    for i in range(6176):
        if i % 100 == 0:
            pyboy.button("a")
        pyboy.tick(render=False)

    assert pokemon_pinball.game_over

    pyboy.stop()
