from pyboy import PyBoy
from pyboy.plugins.game_wrapper_pokemon_pinball import Pokemon, Stage
from pyboy.utils import WindowEvent, bcd_to_dec

pyboy = PyBoy("test_roms/secrets/pokemon_pinball.gbc")
pyboy.set_emulation_speed(0)
print(pyboy.cartridge_title)
pinball = pyboy.game_wrapper
pinball.start_game()
pyboy.set_emulation_speed(1)
for i in range(int(5e5)):
    pyboy.tick()
    if i % 20 == 0:
        print(pyboy.game_area().shape)
        print("-----------------")

while True:
    pyboy.tick()
    print(pinball.evolution_success_count)