# dataset_bcic.py
import numpy as np
from braindecode.datasets.moabb import MOABBDataset
from braindecode.preprocessing.preprocess import Preprocessor, preprocess
from braindecode.preprocessing.preprocess import (
    exponential_moving_standardize, exponential_moving_demean
)

NAME = "bcic_iv_2a"
SUBJECTS = list(range(1, 10))

#BNCI2014_001 (2a): 0–38 / 4–38
FREQ_CONDITIONS = [
    ('0-38', 0.0, 38.0),
    ('4-38', 4.0, 38.0),
]


def load_preprocessed_data(subject_id: int,
                           low_cut_hz: float,
                           high_cut_hz: float,
                           exp_moving_fn: str,
                           only_C: bool,
                           do_car: bool):
    """Exactly your current BCIC path (load -> pick_types -> scale/clip -> resample -> bandpass -> exp. std)."""
    dataset = MOABBDataset(dataset_name="BNCI2014001", subject_ids=[subject_id])

    moving_fn = {
        'standardize': exponential_moving_standardize,
        'demean': exponential_moving_demean
    }[exp_moving_fn]

    factor_new = 1e-3
    init_block_size = 1000

    preprocessors = [
        Preprocessor('load_data'),
        Preprocessor('pick_types', eeg=True, meg=False, stim=False),
        Preprocessor(lambda x: x * 1e6, apply_on_array=True),
        Preprocessor(lambda x: np.clip(x, -800, 800), apply_on_array=True),
    ]

    if do_car:
        preprocessors.append(Preprocessor('set_eeg_reference', ref_channels='average'))

    preprocessors.extend([
        Preprocessor('resample', sfreq=250),
        Preprocessor('filter', l_freq=low_cut_hz, h_freq=high_cut_hz),
        Preprocessor(moving_fn, factor_new=factor_new, init_block_size=init_block_size, apply_on_array=True),
    ])

    preprocess(dataset, preprocessors)
    return dataset
