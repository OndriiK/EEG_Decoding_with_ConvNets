# dataset_hgd.py
import numpy as np
from braindecode.datasets.moabb import MOABBDataset
from braindecode.preprocessing.preprocess import Preprocessor, preprocess
from braindecode.preprocessing.preprocess import (
    exponential_moving_standardize, exponential_moving_demean
)

NAME = "hgd"
SUBJECTS = list(range(1, 15))

# resampled to 250 Hz
FREQ_CONDITIONS = [
    ('0-125', 0.0, None),  #high-pass at 0 Hz
    ('4-125', 4.0, None),  #high-pass at 4 Hz
]

# Motor subset
C_SENSORS = [
    'FC5', 'FC1', 'FC2', 'FC6', 'C3', 'Cz', 'C4', 'CP5', 'CP1', 'CP2', 'CP6',
    'FC3', 'FCz', 'FC4', 'C5', 'C1', 'C2', 'C6', 'CP3', 'CPz', 'CP4',
    'FFC5h', 'FFC3h', 'FFC4h', 'FFC6h', 'FCC5h', 'FCC3h', 'FCC4h', 'FCC6h',
    'CCP5h', 'CCP3h', 'CCP4h', 'CCP6h', 'CPP5h', 'CPP3h', 'CPP4h', 'CPP6h',
    'FFC1h', 'FFC2h', 'FCC1h', 'FCC2h', 'CCP1h', 'CCP2h', 'CPP1h', 'CPP2h'
]


def load_preprocessed_data(subject_id: int,
                           low_cut_hz: float,
                           high_cut_hz: float | None,
                           exp_moving_fn: str,
                           only_C: bool,
                           do_car: bool):
    """Exactly your current HGD preprocessing sequence/order."""
    dataset = MOABBDataset(dataset_name="Schirrmeister2017", subject_ids=[subject_id])

    moving_fn = {
        'standardize': exponential_moving_standardize,
        'demean': exponential_moving_demean
    }[exp_moving_fn]

    factor_new = 1e-3
    init_block_size = 1000

    preprocessors = []
    if only_C:
        preprocessors.append(Preprocessor('pick_channels', ch_names=C_SENSORS, ordered=True))
    preprocessors.append(Preprocessor('load_data'))
    preprocessors.append(Preprocessor(lambda x: x * 1e6, apply_on_array=True))
    preprocessors.append(Preprocessor(lambda x: np.clip(x, -800, 800), apply_on_array=True))

    if do_car:
        preprocessors.append(Preprocessor('set_eeg_reference', ref_channels='average'))

    preprocessors.extend([
        Preprocessor('resample', sfreq=250),
        Preprocessor('filter', l_freq=low_cut_hz, h_freq=high_cut_hz),
        Preprocessor(moving_fn, factor_new=factor_new, init_block_size=init_block_size, apply_on_array=True),
    ])

    preprocess(dataset, preprocessors)
    return dataset
