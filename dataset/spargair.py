import itertools
from os.path import join

import numpy as np
# from higrid.emulate import emulatescene
# from higrid.utils import wavread

from .smir import SMIRDataset


class SPARGAir(SMIRDataset):
    def __init__(self, path) -> None:
        self.path = path
        self.channels = 32

    def emulate_scene(self, snd_path, pos: tuple, room='ii-s05', samples=(0, 96000)):
        pos_dir = self.pos2dir(pos)
        smir_path = join(room, pos_dir)

        return self._emulate_scene(snd_path, smir_path, pos, samples)

    @staticmethod
    def pos2dir(pos):
        return ''.join(map(str, pos))

    @staticmethod
    def calculate_gain(pos: tuple) -> float:
        p0, p1, p2 = pos
        gain = np.sqrt((p0-3.0)**2 + (p1-3.0)** 2 + ((p2-2.0)*0.6)**2)
        return gain

    @staticmethod
    def get_distance(pos: tuple) -> float:
        """Calculate distance from given grid position

        Args:
            pos (tuple): Three dimensional grid position

        Returns:
            float: Distance in meters
        """
        assert len(pos) == 3, "Grid positions must be in three dimensions"
        assert all(isinstance(p, int)
                   for p in pos), "Grid positions must be integer values"

        X, Y, Z = pos
        x = (3 - X) * 0.5
        y = (3 - Y) * 0.5
        z = (2 - Z) * 0.3

        return np.sqrt(sum(p**2 for p in (x, y, z)))

    @staticmethod
    def generate_positions() -> list:
        """Generate all possible grid positions on SPARG AIR Dataset

        Returns:
            list: List of all possible grid positions
        """
        x = list(range(7))
        y = list(range(7))
        z = list(range(5))

        grid = list(itertools.product(x, y, z))

        # Remove the position of the microphone array
        grid.remove((3, 3, 2))
        # Remove the positions right above and right beyond the mic array
        for p in [(3, 3, 0), (3, 3, 1), (3, 3, 3), (3, 3, 4)]:
            grid.remove(p)
        # Remove corrupt positions
        for p in [(0, 0, 1), (0, 1, 2), (0, 1, 4)]:
            grid.remove(p)

        return grid

    @staticmethod
    def generate_perpendicular_positions() -> list:
        return [(0, 3, 2),
                (1, 3, 2),
                (2, 3, 2),
                (3, 0, 2),
                (3, 1, 2),
                (3, 2, 2),
                (3, 4, 2),
                (3, 5, 2),
                (3, 6, 2),
                (4, 3, 2),
                (5, 3, 2),
                (6, 3, 2)]
