from pathlib import Path
import gymnasium as gym
from pyboy import PyBoy
from gymnasium import spaces

ROM_PATH = str(Path(__file__).parent.joinpath("super_mario_land.gb"))

observation_space = 



class SuperMarioLandEnv(gym.Env):
    def __init__(self, rom_path=ROM_PATH):
        super().__init__()

        self.pyboy = PyBoy(rom_path)
        self.pyboy.game_wrapper.start_game()
        self.pyboy.tick()
    
    def reset(self):
        self.pyboy.game_wrapper.reset_game()
        return self._get_obs()
        