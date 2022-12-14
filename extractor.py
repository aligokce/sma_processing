from pathlib import Path

import numpy as np

from dataset import SMIRDataset
from dataset.utils import emulate_scene
from features import rent, shd, srf
from microphone import microphones

from omegaconf import OmegaConf, open_dict


class Extractor:
    def __init__(self, params: OmegaConf, dataset: SMIRDataset) -> None:
        """
        Feature extractor via emulation featuring anechoic sounds and 
        multichannel impulse response dataset.

        Parameters
        ----------
        params : OmegaConf
            Parameters for feature extraction
        dataset : SMIRDataset
            Spherical microphone impulse response dataset
        """
        self.params = self._load_params(params)
        self.dataset = dataset

    def job(self, anechoic_path, scenario, save=True, output_shd=False):
        """
        RENT extraction job

        Parameters
        ----------
        anechoic_path : Union[str, bytes, PathLike]
            Path for anechoic sound file path
        scenario : Dict
            Dictionary describing the current emulation scenario
        save : bool, optional
            Results are saved when true, returned when false, by default True
        output_shd : bool, optional
            Whether to additionaly output intermediate SHD matrix, by default False

        Returns
        -------
        None, NDArray or (NDArray, NDArray)
            Nothing is returned if `save` is enabled.
            RENT matrix is returned if `save` is disabled.
            RENT and SHD matrices are returned if 
            `output_shd` is enabled and `save` is disabled.
        """
        y = self.load_signal(anechoic_path, scenario)

        res = rent_pipe(
            y,
            **self.params,
            mic=microphones[scenario['mic']],
            output_shd=output_shd
        )

        if save:
            return self.save_result(res, anechoic_path, scenario, output_shd)
        return res

    def load_signal(self, sig_path, scenario):
        # NOTE: Should this function belong to the SMIRDataset class?
        # Calculate gain
        mic_pos = scenario['mic_pos']
        src_pos = scenario['src_pos']
        gain = self.dataset.calculate_gain(src_pos, mic_pos)

        # Generate IR paths
        ir_paths = self.dataset.generate_ir_paths(**scenario)

        sig = emulate_scene(sig_path, gain, ir_paths)
        return sig

    def save_result(self, result, anechoic_path, scenario):
        save_folder = ""
        self._save_npy(result, save_folder, anechoic_path, **scenario)

    @staticmethod
    def _load_params(params: OmegaConf):
        p = params.copy()
        with open_dict(p):
            p['fimin'] = int(
                round(params['fl'] / params['fs'] * params['n_fft']))
            p['fimax'] = int(
                round(params['fl'] / params['fs'] * params['n_fft']))
        return p

    @staticmethod
    def _save_npy(npy, save_folder, anechoic_path, mic_pos, src_pos, src_dir, **kwargs):
        # TODO: How to split all variables from here while anechoic filename contains "_"
        # Example output filename: OA-09_PA-19_W__mahler_vl1b_6
        save_name = f"{mic_pos}_{src_pos}_{src_dir}__{anechoic_path.stem}.npy"
        save_path = Path.cwd() / save_folder / save_name

        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(save_path, npy)


def rent_pipe(
    y,
    # shd
    fs,
    n_fft,
    olap,
    n_shd,
    fimin,
    fimax,
    j_nu,
    mic,
    # srf
    n_pix,
    # output_shd=False,
    **kwargs
):
    Anm = shd.extract(y, fs, n_fft, olap, n_shd, fimin, fimax, j_nu, mic)
    S = srf.extract(Anm, n_pix)
    R = rent.extract(S, n_shd, fimin, fimax, j_nu)

    # if output_shd:
    #     return R, Anm
    return R
