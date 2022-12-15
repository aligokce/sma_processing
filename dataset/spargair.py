from itertools import product
from pathlib import Path

import pandas as pd

from .smir import SMIRDataset


class SPARGAir(SMIRDataset):
    def __init__(self, path):
        self.path = Path(path)
        self.files = list(self.path.glob("**/IR*.wav"))

        self.metadata = self._get_metadata(self.files)
        self.room = 'ii-s05'

        self.src_coords = self._get_coords(self.room, 'src')
        self.mic_coords = self._get_coords(self.room, 'mic')
        self.n_channels = 32

    def generate_ir_paths(self, src_pos, mic='em32', **kwargs):
        # HINT: filepath = "./spargair/em32/000/IR00001.wav"
        ir_paths = [self.path / mic / str(src_pos) / f"IR{ch:05.0f}.wav"
                    for ch in range(1, self.n_channels + 1)]
        return ir_paths

    def calculate_gain(self, src_pos, mic_pos):
        dist = self.get_distance(src_pos, mic_pos)
        return (dist * 2)

    @staticmethod
    def parse_file_path(filepath):
        # HINT: filepath = "./spargair/em32/000/IR00001.wav"
        room = 'ii-s05'

        mic = Path(filepath).parts[-3]
        if mic not in ['em32', 'alctron']:  # Temporary soln for there are two diff structures
            mic = 'em32'

        mic_pos = '332'
        src_pos = Path(filepath).parts[-2]
        ch = int(Path(filepath).stem[2:])

        return dict(
            room=room,
            mic=mic,
            mic_pos=mic_pos,
            src_pos=src_pos,
            ch=ch
        )

    @staticmethod
    def _get_metadata(files):
        df = pd.DataFrame({'fpath': files}) \
            .apply(
                lambda x: x.fpath.parts[-3:-1], 
                axis='columns', result_type='expand') \
            .drop_duplicates() \
            .reset_index(drop=True)
        df.columns = ['mic', 'src_pos']  # type: ignore
        df['mic_pos'] = '332'
        df['room'] = 'ii-s05'
        df['src_dir'] = 'D'
        return df

    @staticmethod
    def _get_coords(room, type):
        def _pos2xyz(pos):
            X, Y, Z = pos
            x = (3 - int(X)) * 0.5
            y = (3 - int(Y)) * 0.5
            z = (2 - int(Z)) * 0.3
            return [x, y, z]

        assert room == 'ii-s05', f"Invalid room: {room}"
        assert type in ['src', 'mic'], f"Invalid type: {type}"

        pos_list = SPARGAir._get_src_positions(
            'all') if type == 'src' else ['332']
        coords = {pos: _pos2xyz(pos) for pos in pos_list}

        return pd.DataFrame.from_dict(coords, orient='index', columns=['x', 'y', 'z'])

    @staticmethod
    def _get_src_positions(kind='all') -> list:
        if kind == 'all':
            x = range(7)
            y = range(7)
            z = range(5)

            grid = list(f"{i}{j}{k}" for i, j, k in product(x, y, z))
            for p in [
                '332',                          # position of the microphone array
                # positions right above and right beyond the mic array
                '330', '331', '333', '334',
                '001', '012', '014'             # corrupt positions
            ]:
                grid.remove(p)

            return grid

        elif kind == 'perpendicular':
            return [
                '032', '132', '232', '302', '312', '322',
                '342', '352', '362', '432', '532', '632']

        else:
            raise NotImplementedError
