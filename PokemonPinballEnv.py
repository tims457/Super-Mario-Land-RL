import gymnasium as gym
from gymnasium import spaces
import numpy as np
from pyboy import PyBoy
from enum import Enum


class Actions(Enum):
    IDLE = 0
    LEFT_FLIPPER_PRESS = 1
    RIGHT_FLIPPER_PRESS = 2
    LEFT_FLIPPER_RELEASE = 3
    RIGHT_FLIPPER_RELEASE = 4
    LEFT_TILT = 5
    RIGHT_TILT = 6
    UP_TILT = 7
    LEFT_UP_TILT = 8
    RIGHT_UP_TILT = 9


matrix_shape = (16, 20)
game_area_observation_space = spaces.Box(low=0, high=255, shape=matrix_shape, dtype=np.uint8)


class PokemonPinballEnv(gym.Env):
    def __init__(self, pyboy, debug=False):
        super().__init__()
        self.pyboy = pyboy
        if self.pyboy is None:
            raise ValueError("PyBoy instance is required")
        assert self.pyboy.cartridge_title == "POKEPINBALLVPH"

        self._fitness = 0
        self._previous_fitness = 0

        if not debug:
            self.pyboy.set_emulation_speed(0)

        self.action_space = spaces.Discrete(len(Actions))
        self.observation_space = game_area_observation_space

        self.pyboy.game_wrapper.start_game()

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        # Move the agent
        if action == Actions.IDLE.value:
            pass
        elif action == Actions.LEFT_FLIPPER_PRESS.value:
            self.pyboy.button_press("left")
        elif action == Actions.RIGHT_FLIPPER_PRESS.value:
            self.pyboy.button_press("a")
        elif action == Actions.LEFT_FLIPPER_RELEASE.value:
            self.pyboy.button_release("left")
        elif action == Actions.RIGHT_FLIPPER_RELEASE.value:
            self.pyboy.button_release("a")
        elif action == Actions.LEFT_TILT.value:
            self.pyboy.button("down")
        elif action == Actions.RIGHT_TILT.value:
            self.pyboy.button("b")
        elif action == Actions.UP_TILT.value:
            self.pyboy.button("select")
        elif action == Actions.LEFT_UP_TILT.value:
            self.pyboy.button("select")
            self.pyboy.button("down")
        elif action == Actions.RIGHT_UP_TILT.value:
            self.pyboy.button("select")
            self.pyboy.button("b")

        self.pyboy.tick()

        done = self.pyboy.game_wrapper.game_over

        self._calculate_fitness()
        reward = self._fitness - self._previous_fitness

        observation = self._get_obs()
        info = {}
        truncated = False

        return observation, reward, done, truncated, info

    def reset(self, **kwargs):
        self.pyboy.game_wrapper.reset_game()
        self._fitness = 0
        self._previous_fitness = 0

        observation = self._get_obs()
        info = {}
        return observation, info

    def render(self, mode="human"):
        pass

    def close(self):
        self.pyboy.stop()

    def _get_obs(self):
        return self.pyboy.game_area()

    def _calculate_fitness(self):
        self._previous_fitness = self._fitness
        self._fitness = self.pyboy.game_wrapper.score
