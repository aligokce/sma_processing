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
        pos_dir = ''.join(map(str, pos))
        smir_path = join(room, pos_dir)

        sg = self._emulate_scene(snd_path, smir_path, pos, samples)
        # snd = wavread(file_path)
        # snd = snd[0].reshape((snd[0].shape[0]))[samples[0]:samples[1]]
        # gain = self.calculate_gain(pos)

        # sg = emulatescene(snd, gain, join(self.path, room, drtxt))
        return sg

    # def compose_scene(self, file_path_list, positions, room='ii-s05', samples=(0, 96000)):
    #     assert len(file_path_list) == len(positions)

    #     nsamp = samples[1] - samples[0]
    #     sgo = np.zeros((self.channels, nsamp))

    #     for file_path, pos in zip(file_path_list, positions):
    #         sgo += self.emulate_scene(file_path, pos, room, samples)
    #     return sgo

    # def compose_scene(self, file_path_list, positions, room='ii-s05', samples=(0, 96000)):
    #     def _relate_positions(positions):
    #         if positions == 'all':
    #             return self.generate_positions()
    #         elif positions == 'perpendicular':
    #             return self.generate_perpendicular_positions()
    #         else:
    #             return positions

    #     return super().compose_scene(
    #         file_path_list, 
    #         _relate_positions(positions), 
    #         room, 
    #         samples)

    @staticmethod
    def _pos_dir(pos):
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
