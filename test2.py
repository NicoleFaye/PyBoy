from pyboy import PyBoy

debug = False


def printout(pyboy):
    print("------------------------------")
    game_area = pyboy.game_area()
    for i in range(len(game_area)):
        output = "[ "
        for j in range(len(game_area[i])):
            output += str(game_area[i][j]) + " "
        output += "]"
        print(output)


pyboy = PyBoy("pinball.gbc", game_wrapper=True, debug=debug)

pyboy.game_wrapper.start_game()
pyboy.button("a")

for i in range(200):
    pyboy.tick()

printout(pyboy)
input()
pyboy.button_press("a")
pyboy.tick(4)
#pyboy.button_release("a")
pyboy.tick(4)
printout(pyboy)

input()

i = 0
while True:
    pyboy.tick()
    if i % 4 == 0:
        printout(pyboy)
    i += 1

pyboy.stop()
