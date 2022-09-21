# import itertools
from os.path import join

import numpy as np
# from higrid.emulate import emulatescene
# from higrid.utils import wavread
from .utils import wavread, emulate_scene


class SMIRDataset:
    def __init__(self, path, n_channels) -> None:
        self.path = path
        self.n_channels = n_channels

    def emulate_scene(self, snd_path, pos: tuple, room, samples):
        raise NotImplementedError

    def _emulate_scene(self, snd_path, smir_path, pos: tuple, samples):
        snd = wavread(snd_path)
        snd = snd[0].reshape((snd[0].shape[0]))[samples[0]:samples[1]]
        gain = self.calculate_gain(pos)

        sg = emulate_scene(snd, gain, join(self.path, smir_path))
        return sg

    def compose_scene(self, file_path_list, pos_list, room, samples):
        assert len(file_path_list) == len(pos_list), f"There are {len(file_path_list)} audio(s) but {len(pos_list)} position(s)."

        nsamp = samples[1] - samples[0]
        sgo = np.zeros((self.channels, nsamp))

        for file_path, pos in zip(file_path_list, pos_list):
            sgo += self.emulate_scene(file_path, pos, room, samples)
        return sgo

    @staticmethod
    def pos2dir(pos):
        raise NotImplementedError

    @staticmethod
    def calculate_gain(pos: tuple) -> float:
        raise NotImplementedError

    @staticmethod
    def get_distance(pos: tuple) -> float:
        raise NotImplementedError

    @staticmethod
    def generate_positions() -> list:
        raise NotImplementedError
