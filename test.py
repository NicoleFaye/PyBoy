from pyboy import PyBoy
from pyboy.plugins.game_wrapper_pokemon_pinball import Pokemon, Stage
from pyboy.utils import WindowEvent, bcd_to_dec
import os
import signal # Aggressively exit on ctrl+c
signal.signal(signal.SIGINT, lambda sig, frame: os._exit(0))

pyboy = PyBoy("test_roms/secrets/pokemon_pinball.gbc")
pyboy.set_emulation_speed(0)
print(pyboy.cartridge_title)
pinball = pyboy.game_wrapper
pinball.start_game()
pyboy.set_emulation_speed(1)
i = 0
maxVelocity = 0
minVelocity = 0
minX = 0
minY = 0
maxX = 0
maxY = 0
while True:
    i += 1
    pyboy.tick()
    if(i%20 == 0):
        print("Velocity X: ", pinball.ball_x_velocity)
        print("Velocity Y: ", pinball.ball_y_velocity)
        print("Ball X: ", pinball.ball_x)
        print("Ball Y: ", pinball.ball_y)
        print("Max Velocity: ", maxVelocity)
        print("Min Velcotiy: ", minVelocity)
        print("Min X: ", minX)
        print("Min Y: ", minY)
        print("Max X: ", maxX)
        print("Max Y: ", maxY)

    if(maxVelocity < pinball.ball_x_velocity):
        maxVelocity = pinball.ball_x_velocity
    if(maxVelocity < pinball.ball_y_velocity):
        maxVelocity = pinball.ball_y_velocity
    if(minVelocity > pinball.ball_x_velocity):
        minVelocity = pinball.ball_x_velocity
    if(minVelocity > pinball.ball_y_velocity):
        minVelocity = pinball.ball_y_velocity
    if(minX > pinball.ball_x):
        minX = pinball.ball_x
    if(minY > pinball.ball_y):
        minY = pinball.ball_y
    if(maxX < pinball.ball_x):
        maxX = pinball.ball_x
    if(maxY < pinball.ball_y):
        maxY = pinball.ball_y

