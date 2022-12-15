from pathlib import Path

import pandas as pd

from .smir import SMIRDataset


class BBCMaidaValeIR(SMIRDataset):
    def __init__(self, path):
        super().__init__(path)
        self.n_channels = 32

    def generate_ir_paths(self, room, setup, mic, mic_pos, src_pos, src_dir, **kwargs):
        # HINT: filename = "MV4_AS2_Eigen_R_OA-09_S_PA-23_W_Ch_7.wav"
        _base = f"{room}_{setup}_{mic}_R_{mic_pos}_S_{src_pos}_{src_dir}_Ch_"
        ir_paths = [self.path / (_base + str(i) + ".wav")
                    for i in range(1, self.n_channels+1)]  # sorted
        return ir_paths

    def calculate_gain(self, src_pos, mic_pos):
        dist = self.get_distance(src_pos, mic_pos)
        return (dist * 2)

    @staticmethod
    def parse_file_path(filepath):
        # HINT: filename = "MV4_AS2_Eigen_R_OA-09_S_PA-23_W_Ch_7.wav"
        room, setup, mic, _, mic_pos, _, src_pos, src_dir, _, ch = Path(filepath).stem.split('_')
        return dict(
            room = room,
            setup = setup,
            mic = mic,
            mic_pos = mic_pos,
            src_pos = src_pos,
            src_dir = src_dir,
            ch = ch
        )

    @staticmethod
    def _get_metadata(files):
        df = pd.DataFrame({'filename': files}) \
            .apply(lambda x: x.filename.stem, axis=1) \
            .str.split('_', expand=True) \
            .drop([3, 5, 8, 9], axis='columns') \
            .drop_duplicates() \
            .reset_index(drop=True) 
        df.columns = ['room', 'setup', 'mic', 'mic_pos',
                      'src_pos', 'src_dir']  # type: ignore
        return df

    @staticmethod
    def _get_coords(room, type):
        assert room in ['MV4', 'MV5'], f"Invalid room: {room}"
        assert type in ['src', 'mic'], f"Invalid type: {type}"

        # MV4
        if room == 'MV4':
            if type == 'src':
                coords = {
                    "PA-03": [3.0000, -2.7000, 1.5000],
                    "PA-09": [2.0000, -3.7000, 1.2000],
                    "PA-11": [1.0000, -4.7000, 1.2000],
                    "PA-12": [2.0000, -4.7000, 0.0000],
                    "PA-15": [5.0000, -4.7000, 1.5000],
                    "PA-19": [2.0000, -5.7000, 1.2000],
                    "PA-23": [3.0000, -6.7000, 1.5000],
                }
            else:
                coords = {
                    "PA-01": [1.0000, -2.7000, 1.6000],
                    "PA-02": [2.0000, -2.7000, 1.6000],
                    "PA-03": [3.0000, -2.7000, 1.6000],
                    "PA-04": [4.0000, -2.7000, 1.6000],
                    "PA-05": [5.0000, -2.7000, 1.6000],
                    "PA-10": [1.0000, -3.7000, 1.6000],
                    "PA-09": [2.0000, -3.7000, 1.6000],
                    "PA-08": [3.0000, -3.7000, 1.6000],
                    "PA-07": [4.0000, -3.7000, 1.6000],
                    "PA-06": [5.0000, -3.7000, 1.6000],
                    "PA-11": [1.0000, -4.7000, 1.6000],
                    "PA-12": [2.0000, -4.7000, 1.6000],
                    "PA-13": [3.0000, -4.7000, 1.6000],
                    "PA-14": [4.0000, -4.7000, 1.6000],
                    "PA-15": [5.0000, -4.7000, 1.6000],
                    "PA-20": [1.0000, -5.7000, 1.6000],
                    "PA-19": [2.0000, -5.7000, 1.6000],
                    "PA-18": [3.0000, -5.7000, 1.6000],
                    "PA-17": [4.0000, -5.7000, 1.6000],
                    "PA-16": [5.0000, -5.7000, 1.6000],
                    "PA-21": [1.0000, -6.7000, 1.6000],
                    "PA-22": [2.0000, -6.7000, 1.6000],
                    "PA-23": [3.0000, -6.7000, 1.6000],
                    "PA-24": [4.0000, -6.7000, 1.6000],
                    "PA-25": [5.0000, -6.7000, 1.6000],
                    "OA-01": [1.5000, -0.7000, 1.6000],
                    "OA-02": [1.5000, -1.7000, 1.6000],
                    "OA-03": [2.5000, -1.7000, 1.6000],
                    "OA-04": [3.5000, -1.7000, 1.6000],
                    "OA-05": [4.5000, -1.7000, 1.6000],
                    "OA-06": [6.0000, -3.2000, 1.6000],
                    "OA-07": [6.0000, -4.2000, 1.6000],
                    "OA-08": [6.0000, -5.2000, 1.6000],
                    "OA-09": [6.0000, -6.2000, 1.6000],
                    "OA-10": [7.0000, -4.7000, 1.6000],
                    "OA-13": [1.5000, -7.7000, 1.6000],
                    "OA-12": [2.5000, -7.7000, 1.6000],
                    "OA-11": [3.5000, -7.7000, 1.6000],
                    "OA-14": [1.5000, -8.7000, 1.6000],
                    "OA-15": [7.0000, -1.7000, 1.6000],
                    "OA-16": [9.5000, -2.2000, 1.6000],
                    "OA-17": [8.0000, -3.7000, 1.6000],
                    "OA-18": [8.0000, -5.7000, 1.6000],
                    "OA-19": [9.5000, -7.2000, 1.6000],
                    "OA-20": [7.5000, -7.2000, 1.6000],
                }

        # MV5
        elif room == 'MV5':
            if type == 'src':
                coords = {
                    "PA-3A": [1.5000, -1.7500, 1.5000],
                    "PA-3B": [1.5250, -2.0000, 1.0000],
                    "PA-7A": [0.7500, -4.7500, 1.5000],
                    "PA-7B": [1.5000, -5.0000, 1.5000],
                    "PA-7C": [2.2500, -5.2500, 1.5000],
                    "PA-7D": [1.6350, -4.7500, 1.0000],
                }
            else:
                coords = {
                    "PA-01": [0.2500, -1.0000, 1.6000],
                    "PA-02": [2.5000, -1.0000, 1.6000],
                    "PA-04": [0.2500, -3.2000, 1.6000],
                    "PA-05": [2.5000, -3.2000, 1.6000],
                    "PA-06": [5.0000, -3.2000, 1.6000],
                    "PA-7E": [1.2100, -5.5000, 1.6000],
                    "PA-3A": [1.5000, -1.7500, 1.6000],
                    "PA-7B": [1.5000, -5.0000, 1.6000],
                }
            raise NotImplementedError

        else:
            raise AssertionError(f"Invalid room: {room}")

        return pd.DataFrame.from_dict(coords, orient='index', columns=['x', 'y', 'z'])
