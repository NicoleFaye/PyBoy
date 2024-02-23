#
# License: See LICENSE.md file
# GitHub: https://github.com/Baekalfen/PyBoy
#
__pdoc__ = {
    "GameWrapperPokemonPinball.cartridge_title": False,
    "GameWrapperPokemonPinball.post_tick": False,
}

import logging
from enum import Enum

from pyboy.utils import WindowEvent, bcd_to_dec

from .base_plugin import PyBoyGameWrapper

logger = logging.getLogger(__name__)

class Stage(Enum):
    RED_TOP = 0
    RED_BOTTOM = 1
    BLUE_TOP = 4
    BLUE_BOTTOM = 5
    GENGAR = 7
    MEWTWO = 9
    MEOWTH = 11
    DIGLETT = 13
    SEEL = 15

class Pokemon(Enum):
    BULBASAUR = 0
    IVYSAUR = 1
    VENUSAUR = 2
    CHARMANDER = 3
    CHARMELEON = 4
    CHARIZARD = 5
    SQUIRTLE = 6
    WARTORTLE = 7
    BLASTOISE = 8
    CATERPIE = 9
    METAPOD = 10
    BUTTERFREE = 11
    WEEDLE = 12
    KAKUNA = 13
    BEEDRILL = 14
    PIDGEY = 15
    PIDGEOTTO = 16
    PIDGEOT = 17
    RATTATA = 18
    RATICATE = 19
    SPEAROW = 20
    FEAROW = 21
    EKANS = 22
    ARBOK = 23
    PIKACHU = 24
    RAICHU = 25
    SANDSHREW = 26
    SANDSLASH = 27
    NIDORAN_F = 28
    NIDORINA = 29
    NIDOQUEEN = 30
    NIDORAN_M = 31
    NIDORINO = 32
    NIDOKING = 33
    CLEFAIRY = 34
    CLEFABLE = 35
    VULPIX = 36
    NINETALES = 37
    JIGGLYPUFF = 38
    WIGGLYTUFF = 39
    ZUBAT = 40
    GOLBAT = 41
    ODDISH = 42
    GLOOM = 43
    VILEPLUME = 44
    PARAS = 45
    PARASECT = 46
    VENONAT = 47
    VENOMOTH = 48
    DIGLETT = 49
    DUGTRIO = 50
    MEOWTH = 51
    PERSIAN = 52
    PSYDUCK = 53
    GOLDUCK = 54
    MANKEY = 55
    PRIMEAPE = 56
    GROWLITHE = 57
    ARCANINE = 58
    POLIWAG = 59
    POLIWHIRL = 60
    POLIWRATH = 61
    ABRA = 62
    KADABRA = 63
    ALAKAZAM = 64
    MACHOP = 65
    MACHOKE = 66
    MACHAMP = 67
    BELLSPROUT = 68
    WEEPINBELL = 69
    VICTREEBEL = 70
    TENTACOOL = 71
    TENTACRUEL = 72
    GEODUDE = 73
    GRAVELER = 74
    GOLEM = 75
    PONYTA = 76
    RAPIDASH = 77
    SLOWPOKE = 78
    SLOWBRO = 79
    MAGNEMITE = 80
    MAGNETON = 81
    FARFETCH_D = 82
    DODUO = 83
    DODRIO = 84
    SEEL = 85
    DEWGONG = 86
    GRIMER = 87
    MUK = 88
    SHELLDER = 89
    CLOYSTER = 90
    GASTLY = 91
    HAUNTER = 92
    GENGAR = 93
    ONIX = 94
    DROWZEE = 95
    HYPNO = 96
    KRABBY = 97
    KINGLER = 98
    VOLTORB = 99
    ELECTRODE = 100
    EXEGGCUTE = 101
    EXEGGUTOR = 102
    CUBONE = 103
    MAROWAK = 104
    HITMONLEE = 105
    HITMONCHAN = 106
    LICKITUNG = 107
    KOFFING = 108
    WEEZING = 109
    RHYHORN = 110
    RHYDON = 111
    CHANSEY = 112
    TANGELA = 113
    KANGASKHAN = 114
    HORSEA = 115
    SEADRA = 116
    GOLDEEN = 117
    SEAKING = 118
    STARYU = 119
    STARMIE = 120
    MR_MIME = 121
    SCYTHER = 122
    JYNX = 123
    ELECTABUZZ = 124
    MAGMAR = 125
    PINSIR = 126
    TAUROS = 127
    MAGIKARP = 128
    GYARADOS = 129
    LAPRAS = 130
    DITTO = 131
    EEVEE = 132
    VAPOREON = 133
    JOLTEON = 134
    FLAREON = 135
    PORYGON = 136
    OMANYTE = 137
    OMASTAR = 138
    KABUTO = 139
    KABUTOPS = 140
    AERODACTYL = 141
    SNORLAX = 142
    ARTICUNO = 143
    ZAPDOS = 144
    MOLTRES = 145
    DRATINI = 146
    DRAGONAIR = 147
    DRAGONITE = 148
    MEWTWO = 149
    MEW = 150


class Maps(Enum):
    PALLET_TOWN = 0
    VIRIDIAN_CITY = 1
    VIRIDIAN_FOREST = 2
    PEWTER_CITY = 3
    MT_MOON = 4
    CERULEAN_CITY = 5
    VERMILION_SEASIDE = 6
    VERMILION_STREETS = 7
    ROCK_MOUNTAIN = 8
    LAVENDER_TOWN = 9
    CELADON_CITY = 10
    CYCLING_ROAD = 11
    FUCHSIA_CITY = 12
    SAFARI_ZONE = 13
    SAFFRON_CITY = 14
    SEAFOAM_ISLANDS = 15
    CINNABAR_ISLAND = 16
    INDIGO_PLATEAU = 17


RedStages = [Stage.RED_TOP, Stage.RED_BOTTOM]
BlueStages = [Stage.BLUE_TOP, Stage.BLUE_BOTTOM]

RedBonusStages = [Stage.DIGLETT, Stage.GENGAR, Stage.MEWTWO]

BlueBonusStages = [Stage.MEOWTH, Stage.SEEL, Stage.MEWTWO]

AllBonusStageValues = []
for stage in RedBonusStages:
    AllBonusStageValues.append(stage.value)
for stage in BlueBonusStages:
    AllBonusStageValues.append(stage.value)


class GameWrapperPokemonPinball(PyBoyGameWrapper):
    cartridge_title = "POKEPINBALLVPH"

    def __init__(self, *args, **kwargs):
        self.shape = (20, 16)
        """The shape of the game area"""
        self.score = 0
        """The score provided by the game"""
        self.balls_left = 0
        """The lives remaining provided by the game"""
        self.game_over = False
        """The game over state"""
        self.saver_active = False
        """The ball saver state"""
        self.ball_type = BallType.POKEBALL.value
        """The current ball type"""
        self.multiplier = 1
        """The current multiplier"""
        self.current_stage = 0
        """The current stage"""
        self.ball_size = 0
        """The current ball size"""
        self.ball_saver_seconds_left = 0
        """The current ball saver seconds left"""
        self.pokedex = [False] * 151
        """The pokedex state"""
        self._unlimited_saver = False
        """The unlimited saver state"""
        
        ##########################
        # Fitness Related Values #
        ##########################

        ######################
        # Evolution tracking #
        ######################
        self.evolution_failure_count = 0
        self.evolution_success_count = 0
        
        ########################
        # Bonus stage tracking #
        ########################
        self.diglett_stages_completed = 0
        self.diglett_stages_visited = 0
        self.gengar_stages_completed = 0
        self.gengar_stages_visited = 0
        self.meowth_stages_completed = 0
        self.meowth_stages_visited = 0
        self.mewtwo_stages_completed = 0
        self.mewtwo_stages_visited = 0
        self.seel_stages_completed = 0
        self.seel_stages_visited = 0

        
        ##########################
        # Pikachu Saver tracking #
        ##########################
        self.pikachu_saver_charge = 0 # range of 0-15
        self.pikachu_saver_increments = 0
        self.pikachu_saver_used = 0
        
        ################
        # Map tracking #
        ################
        self.current_map = 0
        self.map_change_attempts = 0
        self.map_change_successes = 0

        ###########################
        # Pokemon Caught Tracking #
        ###########################
        self.pokemon_caught_in_session = 0
        self.pokemon_seen_in_session = 0

        #########################
        # Ball upgrade Tracking #
        #########################
        self.great_ball_upgrades = 0
        self.ultra_ball_upgrades = 0
        self.master_ball_upgrades = 0

        self._add_hooks()

        super().__init__(*args, game_area_section=(0, 0) + self.shape, game_area_follow_scxy=True, **kwargs)


    def _update_pokedex(self):
        for pokemon in Pokemon:
            self.pokedex[pokemon.value] = self.pyboy.memory[ADDR_POKEDEX + pokemon.value]

    def reset_game(self, timer_div=None):
        """
        After calling `start_game`, use this method to reset the beginning of the game.

        Kwargs:
            timer_div (int): Replace timer's DIV register with this value. Use `None` to randomize.
        """
        PyBoyGameWrapper.reset_game(self, timer_div=timer_div)

        self._set_timer_div(timer_div)

    def has_pokemon(self, pokemon):
        """
        Check if the player has caught the given pokemon

        Args:
            pokemon (Pokemon): The pokemon to check for
        """
        return self.pokedex[pokemon.value] == 2

    def set_unlimited_saver(self, unlimited_saver=True):
        self._unlimited_saver = unlimited_saver

    def _set_stage(self, stage):
        """
        Override rom memory to set the default stage to the desired stage
        Bonus stages require further initialization
        This method should be called before the game starts

        Kwargs:
            stage (Stage): The stage to set the game to.
        """
        if stage is None:
            return
        for i in range(NO_OP_BYTE_WIDTH_STAGE_OVERRIDE):
            # equivalent to no op
            self.pyboy.memory[ADDR_TO_NO_OP_BANK_STAGE_OVERRIDE,ADDR_TO_NO_OP_STAGE_OVERRIDE + i] = 0x00
        # equivalent to ld a, stage.value
        self.pyboy.memory[ADDR_TO_NO_OP_BANK_STAGE_OVERRIDE,ADDR_TO_NO_OP_STAGE_OVERRIDE] = 0b00111110
        self.pyboy.memory[ADDR_TO_NO_OP_BANK_STAGE_OVERRIDE,ADDR_TO_NO_OP_STAGE_OVERRIDE + 1] = stage.value

    def _init_bonus_stage(self, stage):
        # set backup stage if it is a bonus stage
        if stage in RedBonusStages:
            self.pyboy.memory[ADDR_CURRENT_STAGE_BACKUP] = Stage.RED_BOTTOM.value
        elif stage in BlueBonusStages:
            self.pyboy.memory[ADDR_CURRENT_STAGE_BACKUP] = Stage.BLUE_BOTTOM.value

        # do initializations skipped by loading bonus stage instead of main stage
        if stage in RedBonusStages or stage in BlueBonusStages:
            self.pyboy.memory[ADDR_CURRENT_SLOT_FRAME] = CURRENT_SLOT_FRAME_VALUE
            self.pyboy.memory[ADDR_BALLS_LEFT] = 1
            self.pyboy.memory[ADDR_NUM_BALL_LIVES] = 3
            self.pyboy.memory[ADDR_STAGE_COLLISION_STATE] = 0b100
            self.pyboy.memory[ADDR_STAGE_COLLISION_STATE_HELPER] = 0b100


    def start_catch_mode(self, pokemon=Pokemon.BULBASAUR, unlimited_time=False):
        # All values are based on PRET disassembly
        self.pyboy.memory[ADDR_SPECIAL_MODE] = SpecialMode.CATCH.value
        self.pyboy.memory[ADDR_POKEMON_TO_CATCH] = pokemon.value
        self.pyboy.memory[ADDR_SPECIAL_MODE_ACTIVE] = 1
        self.pyboy.memory[ADDR_SPECIAL_MODE_STATE] = 0
        self.pyboy.memory[0xd5c6] = 0
        self.pyboy.memory[ADDR_NUM_MON_HITS] = 0
        self.pyboy.memory[ADDR_NUM_CATCH_TILES_FLIPPED] = 0
        for i in range(TILE_ILLUMINATION_BYTE_WIDTH):
            self.pyboy.memory[ADDR_TILE_ILLUMINATION + i] = 0
        if not unlimited_time:
            self.pyboy.memory[ADDR_TIMER_SECONDS] = 0
            self.pyboy.memory[ADDR_TIMER_MINUTES] = 2
            self.pyboy.memory[ADDR_TIMER_FRAMES] = 0
            self.pyboy.memory[ADDR_TIMER_RAN_OUT] = 0
            self.pyboy.memory[ADDR_TIMER_PAUSED] = 0
            self.pyboy.memory[ADDR_TIMER_ACTIVE] = 1
            self.pyboy.memory[ADDR_D580] = 1

    def enable_evolve_hack(self, unlimited_time=False):
        bank_addr_evo = rom_address_to_bank_and_offset(ROM_ADDR_START_EVOLUTION_METHOD)

        lower_8bits = bank_addr_evo[1] & 0xFF
        upper_8bits = (bank_addr_evo[1]>> 8) & 0xFF

        bank_addr_pause=rom_address_to_bank_and_offset(ROM_ADDR_PAUSE_METHOD_CALL) 

        self.pyboy.memory[rom_address_to_bank_and_offset(ROM_ADDR_PAUSE_BANK)] = bank_addr_evo[0]
        self.pyboy.memory[bank_addr_pause[0], bank_addr_pause[1]] = lower_8bits
        self.pyboy.memory[bank_addr_pause[0], bank_addr_pause[1]+1] = upper_8bits
        if unlimited_time:
            def disable_timer(context):
                context.memory[ADDR_TIMER_ACTIVE] = 0
            bank=4
            offset=0x4d64
            try:
                self.pyboy.hook_register(bank,offset,disable_timer,self.pyboy)
            except:
                pass #hook already exists

    def current_map_completed(self):
        """
        Check if the current map has been completed by catching all available pokemon
        """
        if self.current_stage in RedStages:
            for pokemon in RedStageMapWildMons[self.current_map]:
                if not self.has_pokemon(pokemon):
                    return False
            for pokemon in RedStageMapWildMonsRare[self.current_map]:
                if not self.has_pokemon(pokemon):
                    return False
        elif self.current_stage in BlueStages:
            for pokemon in BlueStageMapWildMons[self.current_map]:
                if not self.has_pokemon(pokemon):
                    return False
            for pokemon in BlueStageMapWildMonsRare[self.current_map]:
                if not self.has_pokemon(pokemon):
                    return False
        return True

    def start_game(self, timer_div=None, stage=None):
        PyBoyGameWrapper.start_game(self, timer_div=timer_div)

        self._set_stage(stage)

        # Random tilemap I observed doesn't change until shortly before input is read
        while True and self.tilemap_background[10, 10] != 269:
            self.pyboy.tick()

        # tick needed count to get to the point where input is read
        for i in range(18):
            self.pyboy.tick()

        # start game
        self.pyboy.send_input(WindowEvent.PRESS_BUTTON_A)
        self.pyboy.tick()
        self.pyboy.send_input(WindowEvent.RELEASE_BUTTON_A)
        # tick count needed to get to the next point where input is read
        for i in range(95):
            self.pyboy.tick()

        self.pyboy.send_input(WindowEvent.PRESS_BUTTON_A)
        self.pyboy.tick()
        self.pyboy.send_input(WindowEvent.RELEASE_BUTTON_A)
        self.pyboy.tick()

        ticks_until_visible = 74
        for i in range(ticks_until_visible):
            self.pyboy.tick()

        ticks_until_input_ready = 4
        for i in range(ticks_until_input_ready):
            self.pyboy.tick()

        #needs to be called after normal initializations, otherwise it will be overwritten
        self._init_bonus_stage(stage)

        self.saved_state.seek(0)
        self.pyboy.save_state(self.saved_state)

        self.game_has_started = True

    def get_unique_pokemon_caught(self):
        """
        Get the number of unique pokemon caught in the current session based off the in game pokedex
        """
        return len([pokemon for pokemon in self.pokedex if pokemon == 2])

    def post_tick(self):
        self._tile_cache_invalid = True
        self._sprite_cache_invalid = True

        self.ball_type = self.pyboy.memory[ADDR_BALL_TYPE]
        self.balls_left = 4 - self.pyboy.memory[ADDR_BALLS_LEFT] + self.pyboy.memory[ADDR_EXTRA_BALLS]
        self.game_over = self.pyboy.memory[ADDR_GAME_OVER] == 1

        self.current_map = self.pyboy.memory[ADDR_CURRENT_MAP]

        #Score tracking logic
        self.score = bcd_to_dec(
            int.from_bytes(self.pyboy.memory[ADDR_SCORE:ADDR_SCORE + SCORE_BYTE_WIDTH], "little"),
            byte_width=SCORE_BYTE_WIDTH
        ) * 10
        
        self.multiplier = self.pyboy.memory[ADDR_MULTIPLIER]

        #Stage tracking logic
        if self.current_stage != self.pyboy.memory[ADDR_CURRENT_STAGE]:
            self._previous_stage = self.current_stage
            self.current_stage = self.pyboy.memory[ADDR_CURRENT_STAGE]

        #bonus stage tracking logic
        if self.current_stage in AllBonusStageValues and not self._bonus_stage_seen:
            self._bonus_stage_seen = True
            self.bonus_stages_visited += 1
        elif self.current_stage not in AllBonusStageValues:
            self._bonus_stage_seen = False

        #needs testing
        #bonus stage completed tracking logic
        if self._previous_stage in AllBonusStageValues and self.current_stage not in AllBonusStageValues:
            if self.pyboy.memory[ADDR_BONUS_STAGE_WON] == 1:
                self.bonus_stages_completed += 1

                # normally i wouldnt edit game vars in tracking, but setting this to zero is an easy way for me to track it
                # and it will be initialized to 0 at the start of the next bonus stage without being used outside of the bonus stage
                self.pyboy.memory[ADDR_BONUS_STAGE_WON] = 0

        #mostly for debugging TODO track incrementing of this var
        self.ball_size = self.pyboy.memory[ADDR_BALL_SIZE]

        #Ball saver tracking logic
        self.pikachu_saver_charge = self.pyboy.memory[ADDR_PIKACHU_SAVER_CHARGE]
        self.ball_saver_seconds_left = self.pyboy.memory[ADDR_BALL_SAVER_SECONDS_LEFT]

        if self._unlimited_saver:
            self.pyboy.memory[ADDR_BALL_SAVER_SECONDS_LEFT] = 30

        self._update_pokedex()


    def sum_number_on_screen(self, x, y, length, blank_tile_identifier, tile_identifier_offset):
        #TODO remove after done testing
        number = 0
        newx = x
        for i, x in enumerate(self.pyboy.botsupport_manager().tilemap_window()[newx:newx + length, y]):
            if x != blank_tile_identifier:
                number += (x+tile_identifier_offset) * (10**(length - 1 - i))
        return number

    def __repr__(self):
        adjust = 4
        # yapf: disable
        return (
            "PokemonPinball:\n" +
            "Score: " + str(self.score) + "\n" +
            "Actual score: " + str(self.actual_score) + "\n" +
            "Multiplier: " + str(self.multiplier) + "\n" +
            "Balls left: " + str(self.balls_left) + "\n" +
            "Ball type: " + str(BallType(self.ball_type).name) + "\n" +
            "Ball size: " + str(BallSize(self.ball_size).name) + "\n" +
            "Current stage: " + str(Stage(self.current_stage).name) + "\n" +
            "Game over: " + str(self.game_over) + "\n" +
            "Ball saver seconds left: " + str(self.ball_saver_seconds_left) + "\n" +
            "Pokemon caught in session: " + str(self.actual_pokemon_caught_in_session) + "\n"
        )
        # yapf: enable
    
    def _add_hooks(self):
        def completed_evolution(context):
            context.evolution_success_count += 1
        self.pyboy.hook_register(CompleteEvolutionMode_RedField[0], CompleteEvolutionMode_RedField[1], completed_evolution, self)
        self.pyboy.hook_register(CompleteEvolutionMode_BlueField[0], CompleteEvolutionMode_BlueField[1], completed_evolution, self)

        def failed_evolution(context):
            context.evolution_failure_count += 1
        self.pyboy.hook_register(FailEvolutionMode_RedField[0], FailEvolutionMode_RedField[1], failed_evolution, self)
        self.pyboy.hook_register(FailEvolutionMode_BlueField[0], FailEvolutionMode_BlueField[1], failed_evolution, self)

        def pokemon_caught(context):
            context.pokemon_caught_in_session += 1
        self.pyboy.hook_register(AddCaughtPokemonToParty[0], AddCaughtPokemonToParty[1], pokemon_caught, self)

        def pokemon_seen(context):
            context.pokemon_seen_in_session += 1
        self.pyboy.hook_register(SetPokemonSeenFlag[0], SetPokemonSeenFlag[1], pokemon_seen, self)

        def meowth_visited(context):
            context.meowth_stages_visited += 1
        self.pyboy.hook_register(InitMeowthBonusStage[0], InitMeowthBonusStage[1], meowth_visited, self)

        def diglett_visited(context):
            context.diglett_stages_visited += 1
        self.pyboy.hook_register(InitDiglettBonusStage[0], InitDiglettBonusStage[1], diglett_visited, self)

        def gengar_visited(context):
            context.gengar_stages_visited += 1
        self.pyboy.hook_register(InitGengarBonusStage[0], InitGengarBonusStage[1], gengar_visited, self)

        def seel_visited(context):
            context.seel_stages_visited += 1
        self.pyboy.hook_register(InitSeelBonusStage[0], InitSeelBonusStage[1], seel_visited, self)

        def mewtwo_visited(context):
            context.mewtwo_stages_visited += 1
        self.pyboy.hook_register(InitMewtwoBonusStage[0], InitMewtwoBonusStage[1], mewtwo_visited, self)

        def meowth_completed(context):
            context.meowth_stages_completed += 1
        self.pyboy.hook_register(MeowthStageComplete[0], MeowthStageComplete[1], meowth_completed, self)

        def diglett_completed(context):
            context.diglett_stages_completed += 1
        self.pyboy.hook_register(DiglettStageComplete[0], DiglettStageComplete[1], diglett_completed, self)

        def gengar_completed(context):
            context.gengar_stages_completed += 1
        self.pyboy.hook_register(GengarStageComplete[0], GengarStageComplete[1], gengar_completed, self)

        def seel_completed(context):
            context.seel_stages_completed += 1
        self.pyboy.hook_register(SeelStageComplete[0], SeelStageComplete[1], seel_completed, self)

        def mewtwo_completed(context):
            context.mewtwo_stages_completed += 1
        self.pyboy.hook_register(MewtwoStageComplete[0], MewtwoStageComplete[1], mewtwo_completed, self)

        def map_change_attempt(context):
            context.map_change_attempts += 1
        self.pyboy.hook_register(MapChangeAttempt[0], MapChangeAttempt[1], map_change_attempt, self)

        def map_change_success(context):
            context.map_change_successes += 1
        self.pyboy.hook_register(MapChangeSuccess[0], MapChangeSuccess[1], map_change_success, self)

        def pika_saver_increment(context):
            context.pikachu_saver_increments += 1
        self.pyboy.hook_register(PikaSaverIncrement_BlueField[0], PikaSaverIncrement_BlueField[1], pika_saver_increment, self)
        self.pyboy.hook_register(PikaSaverIncrement_RedField[0], PikaSaverIncrement_RedField[1], pika_saver_increment, self)

        def pika_saver_used(context):
            context.pikachu_saver_used += 1
        self.pyboy.hook_register(PikaSaverUsed_BlueField[0], PikaSaverUsed_BlueField[1], pika_saver_used, self)
        self.pyboy.hook_register(PikaSaverUsed_RedField[0], PikaSaverUsed_RedField[1], pika_saver_used, self)

        def ball_upgrade_trigger(context):
            if context.ball_type == BallType.POKEBALL.value:
                context.great_ball_upgrades += 1
            elif context.ball_type == BallType.GREAT_BALL.value:
                context.ultra_ball_upgrades += 1
            elif context.ball_type == BallType.ULTRA_BALL.value:
                context.master_ball_upgrades += 1
        self.pyboy.hook_register(BallUpgradeTrigger_BlueField[0], BallUpgradeTrigger_BlueField[1], ball_upgrade_trigger, self)
        self.pyboy.hook_register(BallUpgradeTrigger_RedField[0], BallUpgradeTrigger_RedField[1], ball_upgrade_trigger, self)



def rom_address_to_bank_and_offset(address):
    """
    Convert a ROM address to a bank and offset

    Args:
        address (int): The ROM address
    Returns:
        tuple: The bank and offset
    """
    if address < 0x4000:
        return 0, address
    else:
        return address // 0x4000, address % 0x4000 + 0x4000

#value starts at 1, increments by 1 for each new ball launch and compares to ADDR_NUM_BALL_LIVES
ADDR_BALLS_LEFT = 0xD49D
#value gets initialized to 3 by red and blue stage initialization code
ADDR_NUM_BALL_LIVES = 0xD49E
ADDR_BALL_TYPE = 0xD47E
ADDR_BALL_SIZE = 0xD4C8
ADDR_EXTRA_BALLS = 0xd49b

ADDR_BALL_SAVER_SECONDS_LEFT = 0xD4A4

ADDR_NUM_MON_HITS = 0xd5c0
ADDR_NUM_CATCH_TILES_FLIPPED = 0xd5b6
ADDR_TILE_ILLUMINATION = 0xd586
TILE_ILLUMINATION_BYTE_WIDTH = 48

ADDR_MESSAGE_BUFFER = 0xc600
MESSAGE_BUFFER_BYTE_WIDTH = 100

ADDR_TIMER_SECONDS = 0xd57a
ADDR_TIMER_MINUTES = 0xd57b
ADDR_TIMER_FRAMES = 0xd57c
ADDR_TIMER_RAN_OUT = 0xd57e # 1 = ran out
ADDR_TIMER_PAUSED = 0xd57f # nz = paused
ADDR_TIMER_ACTIVE = 0xd57d # 1 = active
ADDR_D580 = 0xd580

ADDR_CURRENT_MAP = 0xd54a

ADDR_POKEDEX = 0xd962

#this value caps at 255, and will purposefully overflow to 0 after 255 and grant the player bonus points
#to properly track this value we need to keep track of the overflow
# PROBABLY NOT USED ANYMORE
ADDR_POKEMON_CAUGHT_IN_SESSION = 0xD460

ADDR_STAGE_COLLISION_STATE = 0xD4AF
ADDR_STAGE_COLLISION_STATE_HELPER = 0xD7AD

ADDR_CURRENT_STAGE = 0xD4AC
ADDR_CURRENT_STAGE_BACKUP = 0xD4AD

ADDR_BONUS_STAGE_WON = 0xd49a

ADDR_PIKACHU_SAVER_CHARGE = 0xd517
PIKACHU_SAVER_CHARGE_MAX = 15

ADDR_CURRENT_SLOT_FRAME = 0xD603
#this is the value needed to properly return to main stage after bonus stage
#it sets the current slot frame to 7, which is the frame before getting spit out after bonus stage
CURRENT_SLOT_FRAME_VALUE = 0x7

ADDR_SCORE = 0xD46A
SCORE_BYTE_WIDTH = 6
MAX_SCORE = 999999999999 * 10
"""The maximum score possible, multiplied by ten to add the implied extra 0 the game uses"""

ADDR_GAME_OVER = 0xD616
ADDR_MULTIPLIER = 0xD482

ADDR_TO_NO_OP_STAGE_OVERRIDE = 0x1774 + 37
ADDR_TO_NO_OP_BANK_STAGE_OVERRIDE = 3
NO_OP_BYTE_WIDTH_STAGE_OVERRIDE = 11

ADDR_SPECIAL_MODE = 0xD550
ADDR_SPECIAL_MODE_ACTIVE = 0xD54B
ADDR_SPECIAL_MODE_STATE = 0xD54D # 0 = handleEvolutionMode, 1 = CompleteEvolutionMode, 2 = FailEvolutionMode 
# see here: https://github.com/pret/pokepinball/blob/dcfffa520017ba89108f8be97f51d76c68ea44c9/engine/pinball_game/evolution_mode/evolution_mode_blue_field.asm#L34

ADDR_POKEMON_TO_CATCH = 0xD579
ADDR_RARE_POKEMON_FLAG = 0xd55b

#Evolution hack related addresses
ROM_ADDR_START_EVOLUTION_METHOD = 0x10ab3
ROM_ADDR_PAUSE_BANK = 0xd954
ROM_ADDR_PAUSE_METHOD_CALL = 0xd956
ROM_ADDR_PAUSE_METHOD = 0x86d7

CompleteEvolutionMode_BlueField= (8,0x4d30)
CompleteEvolutionMode_RedField=(8,0x470b)
FailEvolutionMode_BlueField=(8,0x4d7c)
FailEvolutionMode_RedField=(8,0x4757)
AddCaughtPokemonToParty=(4,0x473d)
SetPokemonSeenFlag=(4,0x4753)
InitDiglettBonusStage=(6,0x59f2)
InitMeowthBonusStage=(9,0x4000)
InitGengarBonusStage=(6,0x4099)
InitSeelBonusStage=(9,0x5a7c)
InitMewtwoBonusStage=(6,0x524f)
DiglettStageComplete=(6,0x6bf2)
GengarStageComplete=(6,0x4a14)
MeowthStageComplete=(9,0x444b)
SeelStageComplete=(9,0x5c5a)
MewtwoStageComplete=(6,0x565e)
MapChangeAttempt=(0xc,0x41ec)
MapChangeSuccess=(0xc,0x55d5)
PikaSaverIncrement_BlueField=(0x7,0x4aff)
PikaSaverIncrement_RedField=(0x5,0x4e8a)
PikaSaverUsed_BlueField=(0x7,0x50c9)
PikaSaverUsed_RedField=(0x5,0x6634)
BallUpgradeTrigger_BlueField=(0x7,0x63de)
BallUpgradeTrigger_RedField=(0x5,0x53c0)


RedStageMapWildMons = {
    Maps.PALLET_TOWN: {
        Pokemon.BULBASAUR: 0.0625,
        Pokemon.CHARMANDER: .375,
        Pokemon.PIDGEY: .1875,
        Pokemon.RATTATA: .1875,
        Pokemon.NIDORAN_M: .0625,
        Pokemon.POLIWAG: 0.0625,
        Pokemon.TENTACOOL: 0.0625
    },
    Maps.VIRIDIAN_FOREST: {
        Pokemon.WEEDLE: 0.3125,
        Pokemon.PIDGEY: 0.3125,
        Pokemon.RATTATA: 0.3125,
        Pokemon.PIKACHU: 0.0625
    },
    Maps.PEWTER_CITY: {
        Pokemon.PIDGEY: 0.125,
        Pokemon.SPEAROW: 0.375,
        Pokemon.EKANS: 0.0625,
        Pokemon.JIGGLYPUFF: 0.3125,
        Pokemon.MAGIKARP: 0.125
    },
    Maps.CERULEAN_CITY: {
        Pokemon.WEEDLE: 0.125,
        Pokemon.PIDGEY: 0.0625,
        Pokemon.ODDISH: 0.3125,
        Pokemon.PSYDUCK: 0.0625,
        Pokemon.MANKEY: 0.1875,
        Pokemon.ABRA: 0.125,
        Pokemon.KRABBY: 0.0625,
        Pokemon.GOLDEEN: 0.0625
    },
    Maps.VERMILION_SEASIDE: {
        Pokemon.PIDGEY: 0.0625,
        Pokemon.SPEAROW: 0.0625,
        Pokemon.EKANS: 0.125,
        Pokemon.ODDISH: 0.125,
        Pokemon.MANKEY: 0.125,
        Pokemon.SHELLDER: 0.1875,
        Pokemon.DROWZEE: 0.125,
        Pokemon.KRABBY: 0.1875
    },
    Maps.ROCK_MOUNTAIN: {
        Pokemon.RATTATA: 0.0625,
        Pokemon.SPEAROW: 0.0625,
        Pokemon.EKANS: 0.1875,
        Pokemon.ZUBAT: 0.0625,
        Pokemon.DIGLETT: 0.1875,
        Pokemon.MACHOP: 0.0625,
        Pokemon.GEODUDE: 0.0625,
        Pokemon.SLOWPOKE: 0.0625,
        Pokemon.ONIX: 0.0625,
        Pokemon.VOLTORB: 0.1875
    },
    Maps.LAVENDER_TOWN: {
        Pokemon.PIDGEY: 0.125,
        Pokemon.EKANS: 0.125,
        Pokemon.MANKEY: 0.125,
        Pokemon.GROWLITHE: 0.125,
        Pokemon.MAGNEMITE: 0.125,
        Pokemon.GASTLY: 0.3125,
        Pokemon.CUBONE: 0.0625
    },
    Maps.CYCLING_ROAD: {
        Pokemon.RATTATA: 0.125,
        Pokemon.SPEAROW: 0.125,
        Pokemon.TENTACOOL: 0.125,
        Pokemon.DODUO: 0.1875,
        Pokemon.KRABBY: 0.125,
        Pokemon.LICKITUNG: 0.0625,
        Pokemon.GOLDEEN: 0.125,
        Pokemon.MAGIKARP: 0.125
    },
    Maps.SAFARI_ZONE: {
        Pokemon.NIDORAN_M: 0.25,
        Pokemon.PARAS: 0.25,
        Pokemon.DODUO: 0.25,
        Pokemon.RHYHORN: 0.25
    },
    Maps.SEAFOAM_ISLANDS: {
        Pokemon.ZUBAT: 0.0625,
        Pokemon.PSYDUCK: 0.0625,
        Pokemon.TENTACOOL: 0.0625,
        Pokemon.SLOWPOKE: 0.0625,
        Pokemon.SEEL: 0.0625,
        Pokemon.SHELLDER: 0.0625,
        Pokemon.KRABBY: 0.0625,
        Pokemon.HORSEA: 0.25,
        Pokemon.GOLDEEN: 0.0625,
        Pokemon.STARYU: 0.25
    },
    Maps.CINNABAR_ISLAND: {
        Pokemon.GROWLITHE: 0.25,
        Pokemon.PONYTA: 0.25,
        Pokemon.GRIMER: 0.125,
        Pokemon.KOFFING: 0.25,
        Pokemon.TANGELA: 0.125
    },
    Maps.INDIGO_PLATEAU: {
        Pokemon.SPEAROW: 0.0625,
        Pokemon.EKANS: 0.0625,
        Pokemon.ZUBAT: 0.125,
        Pokemon.MACHOP: 0.1875,
        Pokemon.GEODUDE: 0.1875,
        Pokemon.ONIX: 0.1875,
        Pokemon.DITTO: 0.1875
    }
}

RedStageMapWildMonsRare = {
    Maps.PALLET_TOWN: {
        Pokemon.BULBASAUR: 0.1875,
        Pokemon.CHARMANDER: 0.0625,
        Pokemon.PIDGEY: 0.0625,
        Pokemon.RATTATA: 0.0625,
        Pokemon.NIDORAN_M: 0.1875,
        Pokemon.POLIWAG: 0.25,
        Pokemon.TENTACOOL: 0.1875
    },
    Maps.VIRIDIAN_FOREST: {
        Pokemon.CATERPIE: 0.125,
        Pokemon.WEEDLE: 0.1875,
        Pokemon.PIDGEY: 0.125,
        Pokemon.RATTATA: 0.125,
        Pokemon.PIKACHU: 0.4375
    },
    Maps.PEWTER_CITY: {
        Pokemon.PIDGEY: 0.125,
        Pokemon.SPEAROW: 0.1875,
        Pokemon.EKANS: 0.25,
        Pokemon.JIGGLYPUFF: 0.1875,
        Pokemon.MAGIKARP: 0.25
    },
    Maps.CERULEAN_CITY: {
        Pokemon.CATERPIE: 0.0625,
        Pokemon.NIDORAN_M: 0.0625,
        Pokemon.ODDISH: 0.0625,
        Pokemon.PSYDUCK: 0.125,
        Pokemon.MANKEY: 0.125,
        Pokemon.ABRA: 0.1875,
        Pokemon.KRABBY: 0.0625,
        Pokemon.GOLDEEN: 0.125,
        Pokemon.JYNX: 0.1875
    },
    Maps.VERMILION_SEASIDE: {
        Pokemon.EKANS: 0.25,
        Pokemon.ODDISH: 0.0625,
        Pokemon.MANKEY: 0.0625,
        Pokemon.FARFETCH_D: 0.25,
        Pokemon.SHELLDER: 0.125,
        Pokemon.DROWZEE: 0.125,
        Pokemon.KRABBY: 0.125
    },
    Maps.ROCK_MOUNTAIN: {
        Pokemon.ZUBAT: 0.125,
        Pokemon.DIGLETT: 0.0625,
        Pokemon.MACHOP: 0.125,
        Pokemon.GEODUDE: 0.125,
        Pokemon.SLOWPOKE: 0.125,
        Pokemon.ONIX: 0.125,
        Pokemon.VOLTORB: 0.125,
        Pokemon.MR_MIME: 0.1875
    },
    Maps.LAVENDER_TOWN: {
        Pokemon.EKANS: 0.0625,
        Pokemon.MANKEY: 0.0625,
        Pokemon.GROWLITHE: 0.0625,
        Pokemon.MAGNEMITE: 0.125,
        Pokemon.GASTLY: 0.125,
        Pokemon.CUBONE: 0.1875,
        Pokemon.ELECTABUZZ: 0.1875,
        Pokemon.ZAPDOS: 0.1875
    },
    Maps.CYCLING_ROAD: {
        Pokemon.TENTACOOL: 0.0625,
        Pokemon.DODUO: 0.3125,
        Pokemon.KRABBY: 0.0625,
        Pokemon.LICKITUNG: 0.25,
        Pokemon.GOLDEEN: 0.0625,
        Pokemon.MAGIKARP: 0.0625,
        Pokemon.SNORLAX: 0.1875
    },
    Maps.SAFARI_ZONE: {
        Pokemon.NIDORAN_M: 0.125,
        Pokemon.PARAS: 0.125,
        Pokemon.RHYHORN: 0.125,
        Pokemon.CHANSEY: 0.25,
        Pokemon.SCYTHER: 0.125,
        Pokemon.TAUROS: 0.125,
        Pokemon.DRATINI: 0.125
    },
    Maps.SEAFOAM_ISLANDS: {
        Pokemon.SEEL: 0.3125,
        Pokemon.GOLDEEN: 0.25,
        Pokemon.STARYU: 0.25,
        Pokemon.ARTICUNO: 0.1875
    },
    Maps.CINNABAR_ISLAND: {
        Pokemon.GROWLITHE: 0.125,
        Pokemon.PONYTA: 0.125,
        Pokemon.GRIMER: 0.0625,
        Pokemon.KOFFING: 0.125,
        Pokemon.TANGELA: 0.1875,
        Pokemon.OMANYTE: 0.1875,
        Pokemon.KABUTO: 0.1875
    },
    Maps.INDIGO_PLATEAU: {
        Pokemon.SPEAROW: 0.0625,
        Pokemon.EKANS: 0.0625,
        Pokemon.ZUBAT: 0.0625,
        Pokemon.MACHOP: 0.0625,
        Pokemon.GEODUDE: 0.0625,
        Pokemon.ONIX: 0.0625,
        Pokemon.DITTO: 0.25,
        Pokemon.MOLTRES: 0.1875,
        Pokemon.MEWTWO: 0.1875,
        Pokemon.MEW: 0.0625
    }
}

BlueStageMapWildMons = {
    Maps.VIRIDIAN_CITY: {
        Pokemon.BULBASAUR: 0.0625,
        Pokemon.SQUIRTLE: 0.3125,
        Pokemon.SPEAROW: 0.0625,
        Pokemon.NIDORAN_F: 0.1875,
        Pokemon.NIDORAN_M: 0.1875,
        Pokemon.POLIWAG: 0.0625,
        Pokemon.TENTACOOL: 0.0625,
        Pokemon.GOLDEEN: 0.0625
    },
    Maps.VIRIDIAN_FOREST: {
        Pokemon.CATERPIE: 0.3125,
        Pokemon.PIDGEY: 0.3125,
        Pokemon.RATTATA: 0.3125,
        Pokemon.PIKACHU: 0.0625
    },
    Maps.MT_MOON: {
        Pokemon.RATTATA: 0.0625,
        Pokemon.SPEAROW: 0.125,
        Pokemon.EKANS: 0.125,
        Pokemon.SANDSHREW: 0.125,
        Pokemon.ZUBAT: 0.125,
        Pokemon.PARAS: 0.125,
        Pokemon.PSYDUCK: 0.0625,
        Pokemon.GEODUDE: 0.125,
        Pokemon.KRABBY: 0.0625,
        Pokemon.GOLDEEN: 0.0625
    },
    Maps.CERULEAN_CITY: {
        Pokemon.CATERPIE: 0.125,
        Pokemon.PIDGEY: 0.0625,
        Pokemon.MEOWTH: 0.1875,
        Pokemon.PSYDUCK: 0.0625,
        Pokemon.ABRA: 0.125,
        Pokemon.BELLSPROUT: 0.3125,
        Pokemon.KRABBY: 0.0625,
        Pokemon.GOLDEEN: 0.0625
    },
    Maps.VERMILION_STREETS: {
        Pokemon.PIDGEY: 0.0625,
        Pokemon.SPEAROW: 0.0625,
        Pokemon.SANDSHREW: 0.125,
        Pokemon.MEOWTH: 0.125,
        Pokemon.BELLSPROUT: 0.125,
        Pokemon.SHELLDER: 0.1875,
        Pokemon.DROWZEE: 0.125,
        Pokemon.KRABBY: 0.1875
    },
    Maps.ROCK_MOUNTAIN: {
        Pokemon.RATTATA: 0.0625,
        Pokemon.SPEAROW: 0.0625,
        Pokemon.SANDSHREW: 0.125,
        Pokemon.ZUBAT: 0.0625,
        Pokemon.DIGLETT: 0.25,
        Pokemon.MACHOP: 0.0625,
        Pokemon.GEODUDE: 0.0625,
        Pokemon.SLOWPOKE: 0.0625,
        Pokemon.ONIX: 0.0625,
        Pokemon.VOLTORB: 0.1875
    },
    Maps.CELADON_CITY: {
        Pokemon.PIDGEY: 0.125,
        Pokemon.VULPIX: 0.125,
        Pokemon.ODDISH: 0.125,
        Pokemon.MEOWTH: 0.1875,
        Pokemon.MANKEY: 0.1875,
        Pokemon.GROWLITHE: 0.125,
        Pokemon.BELLSPROUT: 0.125
    },
    Maps.FUCHSIA_CITY: {
        Pokemon.VENONAT: 0.125,
        Pokemon.KRABBY: 0.1875,
        Pokemon.EXEGGCUTE: 0.125,
        Pokemon.KANGASKHAN: 0.125,
        Pokemon.GOLDEEN: 0.1875,
        Pokemon.MAGIKARP: 0.25
    },
    Maps.SAFARI_ZONE: {
        Pokemon.NIDORAN_F: 0.25,
        Pokemon.PARAS: 0.25,
        Pokemon.DODUO: 0.25,
        Pokemon.RHYHORN: 0.25
    },
    Maps.SAFFRON_CITY: {
        Pokemon.PIDGEY: 0.125,
        Pokemon.EKANS: 0.1875,
        Pokemon.SANDSHREW: 0.1875,
        Pokemon.VULPIX: 0.0625,
        Pokemon.ODDISH: 0.125,
        Pokemon.MEOWTH: 0.0625,
        Pokemon.MANKEY: 0.0625,
        Pokemon.GROWLITHE: 0.0625,
        Pokemon.BELLSPROUT: 0.125
    },
    Maps.CINNABAR_ISLAND: {
        Pokemon.VULPIX: 0.1875,
        Pokemon.PONYTA: 0.3125,
        Pokemon.GRIMER: 0.125,
        Pokemon.KOFFING: 0.25,
        Pokemon.TANGELA: 0.125
    },
    Maps.INDIGO_PLATEAU: {
        Pokemon.SPEAROW: 0.0625,
        Pokemon.SANDSHREW: 0.0625,
        Pokemon.ZUBAT: 0.125,
        Pokemon.MACHOP: 0.1875,
        Pokemon.GEODUDE: 0.1875,
        Pokemon.ONIX: 0.1875,
        Pokemon.DITTO: 0.1875
    }
}

BlueStageMapWildMonsRare = {
    Maps.VIRIDIAN_CITY: {
        Pokemon.BULBASAUR: 0.1875,
        Pokemon.SQUIRTLE: 0.0625,
        Pokemon.SPEAROW: 0.125,
        Pokemon.NIDORAN_F: 0.125,
        Pokemon.NIDORAN_M: 0.125,
        Pokemon.POLIWAG: 0.125,
        Pokemon.TENTACOOL: 0.125,
        Pokemon.GOLDEEN: 0.125
    },
    Maps.VIRIDIAN_FOREST: {
        Pokemon.CATERPIE: 0.1875,
        Pokemon.WEEDLE: 0.125,
        Pokemon.PIDGEY: 0.125,
        Pokemon.RATTATA: 0.125,
        Pokemon.PIKACHU: 0.4375
    },
    Maps.MT_MOON: {
        Pokemon.EKANS: 0.125,
        Pokemon.SANDSHREW: 0.125,
        Pokemon.CLEFAIRY: 0.375,
        Pokemon.ZUBAT: 0.125,
        Pokemon.PARAS: 0.125,
        Pokemon.GEODUDE: 0.125
    },
    Maps.CERULEAN_CITY: {
        Pokemon.WEEDLE: 0.0625,
        Pokemon.NIDORAN_M: 0.0625,
        Pokemon.MEOWTH: 0.125,
        Pokemon.PSYDUCK: 0.125,
        Pokemon.ABRA: 0.1875,
        Pokemon.BELLSPROUT: 0.0625,
        Pokemon.KRABBY: 0.0625,
        Pokemon.GOLDEEN: 0.125,
        Pokemon.JYNX: 0.1875
    },
    Maps.VERMILION_STREETS: {
        Pokemon.SANDSHREW: 0.25,
        Pokemon.MEOWTH: 0.0625,
        Pokemon.BELLSPROUT: 0.0625,
        Pokemon.FARFETCH_D: 0.25,
        Pokemon.SHELLDER: 0.125,
        Pokemon.DROWZEE: 0.125,
        Pokemon.KRABBY: 0.125
    },
    Maps.ROCK_MOUNTAIN: {
        Pokemon.ZUBAT: 0.125,
        Pokemon.DIGLETT: 0.0625,
        Pokemon.MACHOP: 0.125,
        Pokemon.GEODUDE: 0.125,
        Pokemon.SLOWPOKE: 0.125,
        Pokemon.ONIX: 0.125,
        Pokemon.VOLTORB: 0.125,
        Pokemon.MR_MIME: 0.1875
    },
    Maps.CELADON_CITY: {
        Pokemon.CLEFAIRY: 0.125,
        Pokemon.ABRA: 0.125,
        Pokemon.SCYTHER: 0.0625,
        Pokemon.PINSIR: 0.0625,
        Pokemon.EEVEE: 0.1875,
        Pokemon.PORYGON: 0.25,
        Pokemon.DRATINI: 0.1875
    },
    Maps.FUCHSIA_CITY: {
        Pokemon.VENONAT: 0.25,
        Pokemon.KRABBY: 0.0625,
        Pokemon.EXEGGCUTE: 0.25,
        Pokemon.KANGASKHAN: 0.25,
        Pokemon.GOLDEEN: 0.0625,
        Pokemon.MAGIKARP: 0.125
    },
    Maps.SAFARI_ZONE: {
        Pokemon.NIDORAN_F: 0.125,
        Pokemon.PARAS: 0.125,
        Pokemon.RHYHORN: 0.125,
        Pokemon.CHANSEY: 0.25,
        Pokemon.PINSIR: 0.125,
        Pokemon.TAUROS: 0.125,
        Pokemon.DRATINI: 0.125
    },
    Maps.SAFFRON_CITY: {
        Pokemon.PIDGEY: 0.0625,
        Pokemon.EKANS: 0.0625,
        Pokemon.SANDSHREW: 0.0625,
        Pokemon.VULPIX: 0.0625,
        Pokemon.MEOWTH: 0.0625,
        Pokemon.MANKEY: 0.0625,
        Pokemon.GROWLITHE: 0.0625,
        Pokemon.HITMONLEE: 0.1875,
        Pokemon.HITMONCHAN: 0.1875,
        Pokemon.LAPRAS: 0.1875
    },
    Maps.CINNABAR_ISLAND: {
        Pokemon.VULPIX: 0.0625,
        Pokemon.PONYTA: 0.125,
        Pokemon.GRIMER: 0.125,
        Pokemon.KOFFING: 0.125,
        Pokemon.TANGELA: 0.1875,
        Pokemon.MAGMAR: 0.1875,
        Pokemon.AERODACTYL: 0.1875
    },
    Maps.INDIGO_PLATEAU: {
        Pokemon.SPEAROW: 0.0625,
        Pokemon.SANDSHREW: 0.0625,
        Pokemon.ZUBAT: 0.0625,
        Pokemon.MACHOP: 0.0625,
        Pokemon.GEODUDE: 0.0625,
        Pokemon.ONIX: 0.0625,
        Pokemon.DITTO: 0.25,
        Pokemon.MOLTRES: 0.1875,
        Pokemon.MEWTWO: 0.1875,
        Pokemon.MEW: 0.0625
    },
}


class SpecialMode(Enum):
    CATCH = 0
    EVOLVE = 1
    STAGE_CHANGE = 2


class BallType(Enum):
    POKEBALL = 0
    GREATBALL = 2
    ULTRABALL = 3
    MASTERBALL = 4


class BallSize(Enum):
    DEFAULT = 0
    MINI = 1
    SUPERMINI = 2

