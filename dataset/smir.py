from pathlib import Path

import numpy as np


class SMIRDataset:
    def __init__(self, path):
        self.path = Path(path)
        self.files = list(self.path.glob("**/*.wav"))

        self.metadata = self._get_metadata(self.files)
        self.room = self.metadata['room'].unique()[0]

        self.src_coords = self._get_coords(self.room, 'src')
        self.mic_coords = self._get_coords(self.room, 'mic')
        self.n_channels = None

    def get_distance(self, src_pos, mic_pos):
        coord_0 = np.array(self.src_coords.loc[src_pos])
        coord_1 = np.array(self.mic_coords.loc[mic_pos])
        return np.linalg.norm(coord_0 - coord_1)  # type: ignore

    def generate_ir_paths(self, **kwargs):
        raise NotImplementedError

    def calculate_gain(self, src_pos, mic_pos):
        raise NotImplementedError

    @staticmethod
    def parse_file_path(filepath):
        raise NotImplementedError

    @staticmethod
    def _get_metadata(file_list):
        """
        Conventions for column names:
        room, setup, mic, mic_pos, src_pos, src_dir, ch
        """
        raise NotImplementedError

    @staticmethod
    def _get_coords(room, type):
        raise NotImplementedError
