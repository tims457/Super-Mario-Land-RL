from pathlib import Path
import gymnasium as gym
from gymnasium import spaces
from pyboy import PyBoy
import cv2
import numpy as np
from pyboy.utils import WindowEvent

ROM_PATH = str(Path(__file__).parent.parent.joinpath("super_mario_land.gb"))

SCALE_FACTOR = 2

observation_space = spaces.Dict(
    {
        "screen": spaces.Box(
            low=0,
            high=255,
            shape=(126 // SCALE_FACTOR, 160 // SCALE_FACTOR, 1),
            dtype=np.uint8,
        ),
        # "lives_left": spaces.Box(low=0, high=99, dtype=np.int8),
        # "time_left": spaces.Box(low=0, high=999, dtype=np.uint16),
    }
)

NOP = [WindowEvent.PASS]
PRESS_A = [WindowEvent.PRESS_BUTTON_A]
PRESS_B = [WindowEvent.PRESS_BUTTON_B]
UP = [WindowEvent.PRESS_ARROW_UP]
DOWN = [WindowEvent.PRESS_ARROW_DOWN]
LEFT = [WindowEvent.PRESS_ARROW_LEFT]
RIGHT = [WindowEvent.PRESS_ARROW_RIGHT]
SELECT = [WindowEvent.PRESS_BUTTON_SELECT]
LEFT_A = [*LEFT, *PRESS_A]
RIGHT_A = [*RIGHT, *PRESS_A]
LEFT_B = [*LEFT, *PRESS_B]
RIGHT_B = [*RIGHT, *PRESS_B]
LEFT_A_B = [*LEFT, *PRESS_A, *PRESS_B]
RIGHT_A_B = [*RIGHT, *PRESS_A, *PRESS_B]


ACTIONS = [
    LEFT_A,
    RIGHT_A,
    LEFT_B,
    RIGHT_B,
    PRESS_A,
    PRESS_B,
    UP,
    DOWN,
    LEFT,
    RIGHT,
    SELECT,
    LEFT_A,
    RIGHT_A,
    LEFT_B,
    RIGHT_B,
    LEFT_A_B,
    RIGHT_A_B,
]

# All possible buttons "on hardware"
# These two lists are for determining how we want to press and release buttons
BUTTONS = [
    WindowEvent.PRESS_ARROW_UP,
    WindowEvent.PRESS_ARROW_DOWN,
    WindowEvent.PRESS_ARROW_RIGHT,
    WindowEvent.PRESS_ARROW_LEFT,
    WindowEvent.PRESS_BUTTON_A,
    WindowEvent.PRESS_BUTTON_B,
    WindowEvent.PRESS_BUTTON_SELECT,
]

RELEASE_BUTTONS = [
    WindowEvent.RELEASE_ARROW_UP,
    WindowEvent.RELEASE_ARROW_DOWN,
    WindowEvent.RELEASE_ARROW_RIGHT,
    WindowEvent.RELEASE_ARROW_LEFT,
    WindowEvent.RELEASE_BUTTON_A,
    WindowEvent.RELEASE_BUTTON_B,
    WindowEvent.RELEASE_BUTTON_SELECT,
]

# Given a button press, get the "release" version of it
# ex: release_a = RELEASE_BUTTON_LOOKUP[press_a]
RELEASE_BUTTON_LOOKUP = {
    button: r_button for button, r_button in zip(BUTTONS, RELEASE_BUTTONS)
}

# How many frames to advance every action
# 1 = every single frame, a decision is made
# Frame skipping
DEFAULT_NUM_TO_TICK = 8


class SuperMarioLandEnv(gym.Env):
    def __init__(
        self,
        rom_path: str = ROM_PATH,
        render_mode: str = "rgb_array",
        debug: bool = False,
        buf=None,
    ):
        super().__init__()

        win = "SDL2" if render_mode == "human" else "null"
        self.pyboy = PyBoy(rom_path, window=win, debug=debug)
        self.pyboy.game_wrapper.start_game()
        self.is_render_mode_human = True if render_mode == "human" else False

        self.screen = self.pyboy.screen

        self.observation_space = observation_space
        self.action_space = spaces.Discrete(len(ACTIONS))
        # dict of buttons, and True/False for if they're being held or not
        self._currently_held = {button: False for button in BUTTONS}

        self.num_to_tick = DEFAULT_NUM_TO_TICK

        self.progress = 0
        self.old_mem_state = self._get_mem_state_dict()

        self.reset()

    def reset(self, **kwargs):
        self.pyboy.game_wrapper.reset_game()
        self.progress = 0
        info = {}
        return self._get_obs(), info

    def _get_screen_obs(self):
        # remove top bar with lives, score and remove alpha channel
        rgb = self.screen.ndarray[18:, :, :3]

        # h, w = rgb.shape[:2]

        # ?? scale factor? Pufferlib only looks at every SCALE_FACTOR pixels
        smaller = rgb[::SCALE_FACTOR, ::SCALE_FACTOR]

        # convert to grayscale, h,w,c -> h,w
        gray = cv2.cvtColor(smaller, cv2.COLOR_RGB2GRAY)

        # gymnasium expects h,w,c
        gray = np.reshape(gray, gray.shape + (1,))

        return gray

    def _get_mem_state_dict(self) -> dict:
        # values of interest
        lives_left = np.array([self.pyboy.game_wrapper.lives_left], dtype=np.int8)
        score = np.array([self.pyboy.game_wrapper.score], dtype=np.uint16)
        coins = np.array([self.pyboy.game_wrapper.coins], dtype=np.uint16)
        time_left = np.array([self.pyboy.game_wrapper.time_left], dtype=np.float32)
        level_progress = self.pyboy.game_wrapper.level_progress

        if level_progress > self.progress:
            self.progress = level_progress

        return {
            "lives_left": lives_left,
            "score": score,
            "coins": coins,
            "time_left": time_left,
            "level_progress": self.progress,
        }

    def _get_initial_mem_state(self) -> dict:
        time_left = np.array([self.pyboy.game_wrapper.time_left], dtype=np.float32)
        return {
            "lives_left": np.array([2], dtype=np.int8),
            "score": np.array([0], dtype=np.uint16),
            "coins": np.array([0], dtype=np.uint16),
            "time_left": time_left,
            "level_progress": np.array([0], dtype=np.float32),
        }

    def _get_obs(self) -> dict:
        screen = self._get_screen_obs()
        # print(f"screen shape: {screen.shape} mean: {screen.mean()} std: {screen.std()}")

        mem_state = self._get_mem_state_dict()

        dict_obs = {
            "screen": screen,
            # "lives_left": mem_state["lives_left"],
            # "time_left": mem_state["time_left"],
        }

        return dict_obs

    def _calc_reward(self) -> float:
        mem_state = self._get_mem_state_dict()
        # print(f"\nold_mem_state {self.old_mem_state}")
        # print(f"mem_state {mem_state}")

        deltas = dict()
        for k, v in mem_state.items():
            deltas[k] = v - self.old_mem_state[k]

        reward = deltas["score"] / 100
        if deltas["lives_left"] < 0:
            reward -= 10
        # print(f"deltas {deltas}")

        # reward += deltas['level_progress']/100
        # reward += deltas['time_left']/100

        reward += deltas["level_progress"] / 500
        reward += deltas["time_left"] / 100

        self.old_mem_state = mem_state

        return reward

    def do_action_on_emulator(self, action):
        # get buttons currently being held
        holding = [b for b in self._currently_held if self._currently_held[b]]

        # Release buttons we don't need to press anymore
        for held in holding:
            # If the new action doesn't want us to press the button anymore
            if held not in action:
                release = RELEASE_BUTTON_LOOKUP[held]
                self.pyboy.send_input(release)
                self._currently_held[held] = False

        # Press buttons we need to press now
        for button in action:
            # NOP is a list, continaing just the PASS action, which isn't a
            # button we can "press"
            if button in NOP:
                continue
            # Press the button
            self.pyboy.send_input(button)
            # mark it as "pressed"
            self._currently_held[button] = True

    def step(self, action_index) -> tuple:
        action = ACTIONS[action_index]

        self.do_action_on_emulator(action)

        self.pyboy.tick(self.num_to_tick)

        obs = self._get_obs()

        reward = self._calc_reward()

        # PyBoy Cython weirdness makes "game_over()" an int
        done = False if self.pyboy.game_wrapper.game_over() == 0 else True

        if self.pyboy.game_wrapper.time_left == 0:
            done = True
            reward = np.array([-20], dtype=np.float32)

        if done and (self.pyboy.game_wrapper.lives_left == 0):
            reward = np.array([-20], dtype=np.float32)

        if done:
            self.old_mem_state = self._get_initial_mem_state()

        truncated = False
        info = {}
        # print(f"reward: {reward}, done: {done}, truncated: {truncated}")
        return obs, reward, done, truncated, info

    def render(self):
        return self.screen.ndarray

    def close(self):
        self.pyboy.stop()

    def save_screen(self, path="screen.png"):
        cv2.imwrite(path, self.screen.ndarray)
