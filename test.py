from pyboy import PyBoy
from pyboy.plugins.game_wrapper_pokemon_pinball import Pokemon, Stage
from pyboy.utils import WindowEvent, bcd_to_dec

pyboy = PyBoy("pinball.gbc", game_wrapper=True)
pyboy.set_emulation_speed(0)
print(pyboy.cartridge_title)
pinball = pyboy.game_wrapper
pinball.start_game(stage=Stage.BLUE_BOTTOM)
pyboy.set_emulation_speed(1)

pyboy.load_state(open("test.state", "rb"))

addy=0x5946 #0xd946 call PlaySoundEffect
addy+=3 # translates to 0xcd 0xaf 0x4b
bank = 0x3
for i in range(50):
    print(hex(pyboy.memory[bank,addy-25+i]))



input()
while True:
    pyboy.tick()
    print(pinball.evolution_success_count)

addr = 0xd550
addr2 = 0xd54b
addr3 = 0xd579

ADDR_TIMER_SECONDS = 0xd57a
ADDR_TIMER_MINUTES = 0xd57b
ADDR_TIMER_FRAMES = 0xd57c
ADDR_TIMER_RAN_OUT = 0xd57e # 1 = ran out
ADDR_TIMER_PAUSED = 0xd57f # nz = paused
ADDR_TIMER_ACTIVE = 0xd57d # 1 = active
ADDR_D580 = 0xd580

addrx = 0xd962

Pokemon2 = [
    Pokemon.BULBASAUR,
    Pokemon.CHARMANDER,
    Pokemon.SQUIRTLE,
    Pokemon.CATERPIE,
    Pokemon.WEEDLE,
    Pokemon.PIDGEY,
    Pokemon.RATTATA,
    Pokemon.SPEAROW,
    Pokemon.EKANS,
    Pokemon.PIKACHU,
    Pokemon.SANDSHREW,
    Pokemon.NIDORAN_F,
]
pinball.set_unlimited_saver()
old = 0
for pokemon in Pokemon2:
    pinball.start_catch_mode(pokemon, unlimited_time=True)
    while True:
        pyboy.tick()
        x = int.from_bytes(pyboy.memory[addrx:addrx + 1 + 0x96], byteorder="little")
        if x != old:
            print(pokemon.name, " ", hex(x))
            old = x
            break
pyboy.save_state(open("test.state", "wb"))
while True:
    pyboy.tick()
