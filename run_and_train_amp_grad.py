import argparse
import logging
import os.path
import sys
from functools import partial

import numpy as np
import torch
from torch import nn
from braindecode import EEGClassifier
from braindecode.datasets.moabb import MOABBDataset
from braindecode.preprocessing.preprocess import Preprocessor, preprocess
from braindecode.preprocessing.preprocess import exponential_moving_standardize
from braindecode.preprocessing.preprocess import exponential_moving_demean
from braindecode.preprocessing.windowers import create_windows_from_events
from braindecode.models import Deep4Net, EEGResNet
from braindecode.models import ShallowFBCSPNet
from braindecode.models.util import to_dense_prediction_model, get_output_shape
from braindecode.training.losses import CroppedLoss
from braindecode.util import set_random_seeds
from braindecode.visualization.gradients import compute_amplitude_gradients
from skorch.callbacks import LRScheduler
from skorch.helper import predefined_split
from torch.utils.data import Subset
# NEW: reporting imports
import json
from datetime import datetime

log = logging.getLogger(__name__)
# NEW: simple logger to file and stdout
OUTPUT_FILE = 'deepconvnet_hgd_results.log'
def log_line(msg):
    print(msg)
    try:
        with open(OUTPUT_FILE, 'a') as f:
            f.write(str(msg) + '\n')
    except Exception:
        pass

# NEW: robust extractor for validation accuracy from skorch history
def _extract_last_valid_acc(history):
    if not history:
        return None
    # scan from last to first for any valid accuracy-like key
    preferred_keys = [
        'valid_accuracy', 'valid_acc', 'valid_accuracy_best', 'valid_acc_best',
        'valid_balanced_accuracy', 'valid_bal_acc'
    ]
    for row in reversed(history):
        for k in preferred_keys:
            if k in row and row[k] is not None:
                try:
                    return float(row[k])
                except Exception:
                    pass
        # generic search for any key containing both 'valid' and 'acc'
        for k, v in row.items():
            if isinstance(k, str) and ('valid' in k) and ('acc' in k) and v is not None:
                try:
                    return float(v)
                except Exception:
                    continue
    return None

def load_preprocessed_data(subject_id, low_cut_hz, high_cut_hz, exponential_moving_fn,
                           only_C_sensors, do_common_average_reference, set_name):
    log.info("Load dataset...")
    if set_name == 'hgd':
        dataset = MOABBDataset(dataset_name="Schirrmeister2017", subject_ids=[subject_id])
    else:
        assert set_name == "bcic_iv_2a"
        dataset = MOABBDataset(dataset_name="BNCI2014001", subject_ids=[subject_id])

    C_sensors = [
        'FC5', 'FC1', 'FC2', 'FC6', 'C3', 'Cz', 'C4', 'CP5',
        'CP1', 'CP2', 'CP6', 'FC3', 'FCz', 'FC4', 'C5', 'C1', 'C2', 'C6',
        'CP3', 'CPz', 'CP4', 'FFC5h', 'FFC3h', 'FFC4h', 'FFC6h', 'FCC5h',
        'FCC3h', 'FCC4h', 'FCC6h', 'CCP5h', 'CCP3h', 'CCP4h', 'CCP6h', 'CPP5h',
        'CPP3h', 'CPP4h', 'CPP6h', 'FFC1h', 'FFC2h', 'FCC1h', 'FCC2h', 'CCP1h',
        'CCP2h', 'CPP1h', 'CPP2h']
    EEG_sensors = ['Fp1', 'Fp2', 'Fpz', 'F7', 'F3', 'Fz', 'F4', 'F8',
            'FC5', 'FC1', 'FC2', 'FC6', 'M1', 'T7', 'C3', 'Cz', 'C4', 'T8', 'M2',
            'CP5', 'CP1', 'CP2', 'CP6', 'P7', 'P3', 'Pz', 'P4', 'P8', 'POz', 'O1',
            'Oz', 'O2', 'AF7', 'AF3', 'AF4', 'AF8', 'F5', 'F1', 'F2', 'F6', 'FC3',
            'FCz', 'FC4', 'C5', 'C1', 'C2', 'C6', 'CP3', 'CPz', 'CP4', 'P5', 'P1',
            'P2', 'P6', 'PO5', 'PO3', 'PO4', 'PO6', 'FT7', 'FT8', 'TP7', 'TP8',
            'PO7', 'PO8', 'FT9', 'FT10', 'TPP9h', 'TPP10h', 'PO9', 'PO10', 'P9',
            'P10', 'AFF1', 'AFz', 'AFF2', 'FFC5h', 'FFC3h', 'FFC4h', 'FFC6h', 'FCC5h',
            'FCC3h', 'FCC4h', 'FCC6h', 'CCP5h', 'CCP3h', 'CCP4h', 'CCP6h', 'CPP5h',
            'CPP3h', 'CPP4h', 'CPP6h', 'PPO1', 'PPO2', 'I1', 'Iz', 'I2', 'AFp3h',
            'AFp4h', 'AFF5h', 'AFF6h', 'FFT7h', 'FFC1h', 'FFC2h', 'FFT8h', 'FTT9h',
            'FTT7h', 'FCC1h', 'FCC2h', 'FTT8h', 'FTT10h', 'TTP7h', 'CCP1h', 'CCP2h',
            'TTP8h', 'TPP7h', 'CPP1h', 'CPP2h', 'TPP8h', 'PPO9h', 'PPO5h', 'PPO6h',
            'PPO10h', 'POO9h', 'POO3h', 'POO4h', 'POO10h', 'OI1h', 'OI2h']
    if only_C_sensors:
        sensor_names = C_sensors
    else:
        sensor_names = EEG_sensors
    factor_new = 1e-3
    init_block_size = 1000

    log.info("Preprocess dataset...")

    moving_fn = {'standardize': exponential_moving_standardize,
                'demean': exponential_moving_demean}[exponential_moving_fn]
    
    preprocessors = []

    if set_name == "hgd":
        preprocessors.append(Preprocessor(fn='pick_channels', ch_names=sensor_names, ordered=True))
        preprocessors.append(Preprocessor(fn='load_data'))
        # preprocessors.append(Preprocessor(fn='filter', l_freq=low_cut_hz, h_freq=high_cut_hz))
    else:
        assert set_name == 'bcic_iv_2a'
        preprocessors.append(Preprocessor(fn='load_data'))
        preprocessors.append(Preprocessor("pick_types", eeg=True, meg=False, stim=False))
        # preprocessors.append(Preprocessor(fn='filter', l_freq=low_cut_hz, h_freq=high_cut_hz))

    preprocessors.append(Preprocessor(fn=lambda x: x * 1e6, apply_on_array=True))
    preprocessors.append(Preprocessor(fn=lambda x: np.clip(x, -800, 800), apply_on_array=True))

    if do_common_average_reference:
        preprocessors.append(Preprocessor(fn='set_eeg_reference', ref_channels='average'),)
    preprocessors.extend([
        Preprocessor(fn='resample', sfreq=250),
        Preprocessor(fn='filter', l_freq=low_cut_hz, h_freq=high_cut_hz),
        Preprocessor(fn=moving_fn, factor_new=factor_new,
                     init_block_size=init_block_size, apply_on_array=True),
    ])

    preprocess(dataset, preprocessors)
    return dataset


def create_cropped_model(model_name, n_chans, resnet_init_a):
    cuda = torch.cuda.is_available()
    device = 'cuda' if cuda else 'cpu'
    if cuda:
        torch.backends.cudnn.benchmark = True
    seed = 20200220
    set_random_seeds(seed=seed, cuda=cuda)

    n_classes = 4

    if model_name == 'shallow':
        model = ShallowFBCSPNet(
            n_chans,
            n_classes,
            input_window_samples=None,
            final_conv_length=30,
        )
    elif model_name == 'resnet':
        model = EEGResNet(
            n_chans,
            n_classes,
            input_window_samples=None,
            n_first_filters=48,
            final_pool_length=10,
            conv_weight_init_fn=partial(nn.init.kaiming_normal_, a=resnet_init_a))
    else:
        assert model_name == 'deep'
        model = Deep4Net(
            n_chans,
            n_classes,
            input_window_samples=None,
            final_conv_length=2,
        )

    if cuda:
        model.cuda()

    if model_name in ["shallow", "deep"]:
        to_dense_prediction_model(model)
    return model


def cut_windows(dataset, input_window_samples, window_stride_samples):
    trial_start_offset_seconds = -0.5
    sfreq = dataset.datasets[0].raw.info['sfreq']
    assert all([ds.raw.info['sfreq'] == sfreq for ds in dataset.datasets])
    trial_start_offset_samples = int(trial_start_offset_seconds * sfreq)
    windows_dataset = create_windows_from_events(
        dataset,
        trial_start_offset_samples=trial_start_offset_samples,
        trial_stop_offset_samples=0,
        window_size_samples=input_window_samples,
        window_stride_samples=window_stride_samples,
        drop_last_window=False,
        preload=True,
        mapping={'left_hand': 0, 'right_hand': 1, 'feet': 2, 'rest': 3},
    )
    return windows_dataset


def split_into_train_valid(windows_dataset, use_final_eval):
    print("description", windows_dataset.description)
    desc = windows_dataset.description

    if 'session' in desc.columns and len(desc.session.unique()) > 1:
        splitted = windows_dataset.split("session")
        if 'session_T' in splitted and 'session_E' in splitted:
            train_key, test_key = 'session_T', 'session_E'
        elif '0train' in splitted and '1test' in splitted:
            train_key, test_key = '0train', '1test'
        elif 'train' in splitted and 'test' in splitted:
            train_key, test_key = 'train', 'test'
        else:
            keys = list(splitted.keys())
            keys.sort()
            train_key, test_key = keys[0], keys[-1]
    else:
        splitted = windows_dataset.split('run')
        keys = set(splitted.keys())
        if '0train' in keys and '1test' in keys:
            train_key, test_key = '0train', '1test'
        elif 'train' in keys and 'test' in keys:
            train_key, test_key = 'train', 'test'
        elif '0' in keys and '1' in keys:
            train_key, test_key = '0', '1'

    print("splitted", splitted)
    if use_final_eval:
        train_set = splitted[train_key]
        valid_set = splitted[test_key]
    else:
        full_train_set = splitted[train_key]
        n_split = int(np.round(0.8 * len(full_train_set)))
        n_windows_per_trial = 2
        n_split = n_split - (n_split % n_windows_per_trial)
        valid_set = Subset(full_train_set, range(n_split, len(full_train_set)))
        train_set = Subset(full_train_set, range(0, n_split))
    return train_set, valid_set


def run_training(model, model_name, train_set, valid_set, device, n_epochs, resnet_lr,
                 resnet_weight_decay, drop_channel_prob):
    assert model_name in ['deep', 'shallow', 'resnet']
    if model_name == 'shallow':
        lr = 0.0625 * 0.01
        weight_decay = 0
    elif model_name == 'resnet':
        lr = resnet_lr
        weight_decay = resnet_weight_decay
    else:
        assert model_name == 'deep'
        lr = 1 * 0.01
        weight_decay = 0.5 * 0.001

    batch_size = 64
    from braindecode.augmentation import AugmentedDataLoader, ChannelsDropout
    transforms = [ChannelsDropout(1, drop_channel_prob)]

    clf = EEGClassifier(
        model,
        cropped=True,
        criterion=CroppedLoss,
        criterion__loss_function=torch.nn.functional.nll_loss,
        iterator_train=AugmentedDataLoader,
        iterator_train__transforms=transforms,
        optimizer=torch.optim.AdamW,
        train_split=predefined_split(valid_set),
        optimizer__lr=lr,
        optimizer__weight_decay=weight_decay,
        iterator_train__shuffle=True,
        batch_size=batch_size,
        callbacks=[
            "accuracy", ("lr_scheduler", LRScheduler('CosineAnnealingLR', T_max=n_epochs - 1)),
        ],
        device=device,
        classes=["right", "left", "rest", "feet"],
    )
    clf.fit(train_set, y=None, epochs=n_epochs)
    return clf


def compute_and_store_amp_grads(model, train_set, filename):
    amp_grads_per_filter = compute_amplitude_gradients(model, train_set, batch_size=64)
    avg_amp_grads_per_filter = np.mean(amp_grads_per_filter, axis=1)
    np.save(filename, avg_amp_grads_per_filter)


def run_exp(
        seed,
        subject_id,
        low_cut_hz,
        high_cut_hz,
        exponential_moving_fn,
        n_epochs,
        model_name,
        output_dir,
        only_C_sensors,
        do_common_average_reference,
        use_final_eval,
        save_amp_grads,
        save_model,
        resnet_lr,
        resnet_weight_decay,
        resnet_init_a,
        debug,
        set_name,
        drop_channel_prob):
    assert model_name in ['deep', 'shallow', 'resnet']
    set_random_seeds(seed, True)
    log.info(f"Load and preprocess data for subject {subject_id}...")
    dataset = load_preprocessed_data(subject_id, low_cut_hz, high_cut_hz,
                                     exponential_moving_fn=exponential_moving_fn,
                                     only_C_sensors=only_C_sensors,
                                     do_common_average_reference=do_common_average_reference,
                                     set_name=set_name,
                                     )
    n_chans = dataset[0][0].shape[0]
    log.info("Create cropped model...")
    model = create_cropped_model(model_name, n_chans, resnet_init_a)
    log.info("Cut windows from dataset ...")
    input_window_samples = 1000
    n_preds_per_input = get_output_shape(model, n_chans, input_window_samples)[2]
    windows_dataset = cut_windows(
        dataset, input_window_samples, window_stride_samples=n_preds_per_input)
    log.info("Split into train and valid...")
    train_set, valid_set = split_into_train_valid(windows_dataset, use_final_eval=use_final_eval)
    log.info("Run training...")
    clf = run_training(model, model_name, train_set, valid_set, 'cuda', n_epochs, resnet_lr,
                       resnet_weight_decay, drop_channel_prob)
    if save_amp_grads:
        log.info("Compute and store amplitude gradients ...")
        amp_grads_filename = os.path.join(output_dir, f"{subject_id}_avg_amp_grads.npy")
        compute_and_store_amp_grads(model, train_set, filename=amp_grads_filename)
    if (not debug) and (save_model):
        log.info("Save model ...")
        torch.save(model, os.path.join(output_dir, f"model.pth"))
    log.info("... Done.")
    # return valid_set as well for scoring fallback
    return clf, valid_set


# NEW: subject runner that can skip downloads/training
def train_eval_subject(subject, freq_name, low_freq, high_freq, n_epochs):
    log_line(f'  Subject {subject}: {freq_name} ({low_freq}-{high_freq} Hz)')
    seed = 0
    model_name = 'deep'
    output_dir = './results/'
    only_C_sensors = True
    do_common_average_reference = False
    use_final_eval = True
    save_amp_grads = False
    save_model = False
    resnet_lr = 1e-3
    resnet_init_a = 1
    resnet_weight_decay = 1e-5
    debug = False
    set_name = "hgd"
    drop_channel_prob = 0.0
    clf, valid_set = run_exp(
        seed, subject, low_freq, high_freq, "standardize", n_epochs, model_name,
        output_dir, only_C_sensors, do_common_average_reference, use_final_eval,
        save_amp_grads, save_model, resnet_lr, resnet_weight_decay, resnet_init_a,
        debug, set_name, drop_channel_prob
    )
    # try history first
    acc = _extract_last_valid_acc(clf.history)
    # fallback to scoring on valid set
    if acc is None:
        try:
            acc = float(clf.score(valid_set))
        except Exception:
            acc = float('nan')
    # log per-subject result
    log_line(f'  Accuracy: {acc:.3f}' if acc == acc else '  Accuracy: nan')

    return acc


# UPDATED: orchestrator now calls real training
def main():
    with open(OUTPUT_FILE, 'w') as f:
        f.write(f'Deep4Net HIGH-GAMMA (trial mode, Colab-like) - Started at {datetime.now()}\n')
        f.write('='*80 + '\n\n')

    cuda = torch.cuda.is_available()
    device = 'cuda' if cuda else 'cpu'
    SEED = 0
    BATCH_SIZE = 64
    MAX_EPOCHS = 800
    USE_VALIDATION = True

    set_name = 'hgd'

    log_line(f'Using device: {device}')
    if cuda:
        try:
            log_line(f'GPU: {torch.cuda.get_device_name(0)}')
        except Exception:
            pass
    log_line(f'Random seed: {SEED}')
    log_line(f'Batch size: {BATCH_SIZE}')
    log_line(f'Max epochs: {MAX_EPOCHS}')
    log_line(f'Use validation: {USE_VALIDATION}\n')

    

    if set_name == 'hgd':
        subjects = list(range(1, 15))
        freq_conditions = [
            ('0-125', 0.0, None),
            ('4-125', 4.0, None),
        ]
    else:
        subjects = list(range(1, 10))
        freq_conditions = [
            ('0-38', 0.0, 38.0),
            ('4-38', 4.0, 38.0),
        ]

    results = {}

    log_line('='*80)
    if (set_name == 'bcic_iv_2a'):
        log_line('BCIC IV 2a DATASET')
    else:
        assert(set_name == 'hgd')
        log_line('HIGH-GAMMA DATASET')
    log_line('='*80)

    for freq_name, low_freq, high_freq in freq_conditions:
        log_line(f'\n--- Frequency Condition: {freq_name} ({low_freq}-{high_freq} Hz) ---')

        accuracies = []
        for idx, subject in enumerate(subjects):
            log_line(f'\n[Subject {idx+1}/{len(subjects)}]')
            acc = train_eval_subject(subject, freq_name, low_freq, high_freq, n_epochs=MAX_EPOCHS)
            if acc is not None and not np.isnan(acc):
                accuracies.append(acc)

        mean_acc = float(np.mean(accuracies)) if accuracies else float('nan')
        std_acc = float(np.std(accuracies)) if accuracies else float('nan')

        results[freq_name] = {
            'accuracies': accuracies,
            'mean': mean_acc,
            'std': std_acc
        }

        log_line(f'\n{freq_name} Summary:')
        log_line(f'  Mean: {mean_acc:.3f}' if not np.isnan(mean_acc) else '  Mean: nan')
        log_line(f'  Std:  {std_acc:.3f}' if not np.isnan(std_acc) else '  Std: nan')
        log_line(f'  N:    {len(accuracies)}')
        log_line(f'  Results: {[f"{r:.3f}" for r in accuracies]}')

    log_line('\n' + '='*80)
    log_line('FINAL RESULTS SUMMARY')
    log_line('='*80)

    for freq_name, _, _ in freq_conditions:
        r = results[freq_name]
        mean_str = f'{r["mean"]:.3f}' if not np.isnan(r["mean"]) else 'nan'
        std_str = f'{r["std"]:.3f}' if not np.isnan(r["std"]) else 'nan'
        log_line(f'{freq_name}: Mean={mean_str}, Std={std_str}, N={len(r["accuracies"])}')

    json_file = 'deepconvnet_hgd_results.json'
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2)
    log_line(f'\nResults also saved to {json_file}')
    log_line(f'\nCompleted at {datetime.now()}')

if __name__ == '__main__':
    main()
