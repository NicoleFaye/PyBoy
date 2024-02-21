#
# License: See LICENSE.md file
# GitHub: https://github.com/Baekalfen/PyBoy
#
from pyboy.plugins.base_plugin cimport PyBoyGameWrapper


cdef class GameWrapperPokemonPinball(PyBoyGameWrapper):
    cdef public long long score
    cdef long long _previous_score
    cdef int _score_overflow_count
    cdef public long long actual_score
    cdef public int balls_left
    cdef public bint game_over
    cdef public long long fitness
    cdef public bint saver_active
    cdef public int ball_type
    cdef public int multiplier
    cdef public int current_stage
    cdef int ball_size
    cdef public int ball_saver_seconds_left
    cdef public int pokemon_caught_in_session
    cdef int _previous_pokemon_caught_in_session
    cdef int _pokemon_caught_overflow_count
    cdef public int actual_pokemon_caught_in_session
    cdef public int evolution_count
    cdef public list pokedex
    cdef public int bonus_stages_completed
    cdef public int bonus_stages_visited
    cdef bint _bonus_stage_seen
    cdef public int pikachu_saver_charge
    cdef int _previous_stage
    cdef public int current_map
    cdef public int evolution_success_count
    cdef public int evolution_failure_count

    cdef public bint _inEvolutionMode
    cdef public int _prevEvolutionState
    cdef public int _evolutionState 
    cdef public bint _evolutionCounted
    

    cdef bint _unlimited_saver

    cpdef void start_game(self, timer_div=*, stage=*) noexcept
    cpdef void reset_game(self, timer_div=*) noexcept