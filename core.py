# core.py
import logging
import os
import json
from datetime import datetime
from functools import partial

import numpy as np
import torch
from torch.utils.data import Subset

from braindecode import EEGClassifier
from braindecode.models.util import get_output_shape
from braindecode.preprocessing.windowers import create_windows_from_events
from braindecode.training.losses import CroppedLoss
from braindecode.util import set_random_seeds
from braindecode.visualization.gradients import compute_amplitude_gradients
from skorch.callbacks import LRScheduler
from skorch.helper import predefined_split
from braindecode.augmentation import AugmentedDataLoader, ChannelsDropout

import models  # local module with model builders

log = logging.getLogger(__name__)

# Will be set by run_all(ds, ...)
OUTPUT_FILE = "deepconvnet_results.log"


def log_line(msg: str):
    """Print to stdout and append to result log file."""
    print(msg)
    try:
        with open(OUTPUT_FILE, 'a') as f:
            f.write(str(msg) + '\n')
    except Exception:
        pass


def _extract_last_valid_acc(history):
    if not history:
        return None
    keys = [
        'valid_accuracy', 'valid_acc', 'valid_accuracy_best', 'valid_acc_best',
        'valid_balanced_accuracy', 'valid_bal_acc'
    ]
    for row in reversed(history):
        for k in keys:
            if k in row and row[k] is not None:
                try:
                    return float(row[k])
                except Exception:
                    pass
        # generic fallback
        for k, v in row.items():
            if isinstance(k, str) and ('valid' in k) and ('acc' in k) and v is not None:
                try:
                    return float(v)
                except Exception:
                    continue
    return None


def create_cropped_model(model_name: str, n_chans: int, resnet_init_a: float):
    """Exactly the same choices as your current script."""
    cuda = torch.cuda.is_available()
    device = 'cuda' if cuda else 'cpu'
    if cuda:
        torch.backends.cudnn.benchmark = True
    set_random_seeds(seed=20200220, cuda=cuda)

    n_classes = 4
    if model_name == 'shallow':
        model = models.build_shallow(n_chans, n_classes)
    elif model_name == 'resnet':
        model = models.build_resnet(n_chans, n_classes, init_a=resnet_init_a)
    else:
        # default: DeepConvNet
        model = models.build_deep(n_chans, n_classes)

    if cuda:
        model.cuda()

    return model


def cut_windows(dataset, input_window_samples: int, window_stride_samples: int):
    """0.5 s offset, preload=True, 4-class mapping."""
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


def split_into_train_valid(windows_dataset, use_final_eval: bool):
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
            keys = list(splitted.keys()); keys.sort()
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


def run_training(model, model_name, train_set, valid_set, device, n_epochs,
                 resnet_lr, resnet_weight_decay, drop_channel_prob):
    assert model_name in ['deep', 'shallow', 'resnet']
    if model_name == 'shallow':
        lr = 0.0625 * 0.01
        weight_decay = 0
    elif model_name == 'resnet':
        lr = resnet_lr
        weight_decay = resnet_weight_decay
    else:
        # deep
        lr = 1 * 0.01
        weight_decay = 0.5 * 0.001

    batch_size = 64
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
        callbacks=["accuracy",
                   ("lr_scheduler", LRScheduler('CosineAnnealingLR', T_max=n_epochs - 1))],
        device=device,
        classes=["right", "left", "rest", "feet"],
    )
    clf.fit(train_set, y=None, epochs=n_epochs)
    return clf


def run_exp(ds_module,
            subject_id: int,
            low_cut_hz: float,
            high_cut_hz: float | None,
            n_epochs: int,
            model_name: str,
            output_dir: str,
            only_C_sensors: bool,
            do_common_average_reference: bool,
            use_final_eval: bool,
            save_amp_grads: bool,
            save_model: bool,
            resnet_lr: float,
            resnet_weight_decay: float,
            resnet_init_a: float,
            debug: bool,
            drop_channel_prob: float):

    set_random_seeds(0, True)

    # dataset preprocessing
    dataset = ds_module.load_preprocessed_data(
        subject_id,
        low_cut_hz,
        high_cut_hz,
        exp_moving_fn="standardize",
        only_C=only_C_sensors,
        do_car=do_common_average_reference,
    )

    # model
    n_chans = dataset[0][0].shape[0]
    model = create_cropped_model(model_name, n_chans, resnet_init_a)

    #cropped decoding params
    input_window_samples = 1000
    n_preds_per_input = get_output_shape(model, n_chans, input_window_samples)[2]

    # windows and split
    windows_dataset = cut_windows(dataset, input_window_samples, window_stride_samples=n_preds_per_input)
    train_set, valid_set = split_into_train_valid(windows_dataset, use_final_eval=use_final_eval)

    #train
    clf = run_training(model, model_name, train_set, valid_set, 'cuda', n_epochs,
                       resnet_lr, resnet_weight_decay, drop_channel_prob)

    if save_amp_grads:
        amp = compute_amplitude_gradients(model, train_set, batch_size=64)
        np.save(os.path.join(output_dir, f"{subject_id}_avg_amp_grads.npy"), np.mean(amp, axis=1))

    if (not debug) and save_model:
        torch.save(model, os.path.join(output_dir, "model.pth"))

    return clf, valid_set


def train_eval_subject(ds_module, subject, freq_name, low_freq, high_freq, n_epochs):
    log_line(f'  Subject {subject}: {freq_name} ({low_freq}-{high_freq} Hz)')

    clf, valid_set = run_exp(
        ds_module,
        subject_id=subject,
        low_cut_hz=low_freq,
        high_cut_hz=high_freq,
        n_epochs=n_epochs,
        model_name='deep',
        output_dir='./results/',
        only_C_sensors=True,
        do_common_average_reference=False,
        use_final_eval=True,
        save_amp_grads=False,
        save_model=False,
        resnet_lr=1e-3,
        resnet_weight_decay=1e-5,
        resnet_init_a=1.0,
        debug=False,
        drop_channel_prob=0.0,
    )

    # prefer history, fallback to score(valid)
    acc = _extract_last_valid_acc(clf.history)
    if acc is None:
        try:
            acc = float(clf.score(valid_set))
        except Exception:
            acc = float('nan')

    log_line(f'  Accuracy: {acc:.3f}' if acc == acc else '  Accuracy: nan')
    return acc


def run_all(ds_module, max_epochs=800):
    global OUTPUT_FILE
    OUTPUT_FILE = f'deepconvnet_{ds_module.NAME}_results.log'
    with open(OUTPUT_FILE, 'w') as f:
        f.write(f'Deep4Net {ds_module.NAME.upper()} Started at {datetime.now()}\n')
        f.write('=' * 80 + '\n\n')

    cuda = torch.cuda.is_available()
    log_line(f'Using device: {"cuda" if cuda else "cpu"}')
    if cuda:
        try:
            log_line(f'GPU: {torch.cuda.get_device_name(0)}')
        except Exception:
            pass
    log_line(f'Max epochs: {max_epochs}\n')

    results = {}

    log_line('=' * 80)
    log_line(f'{ds_module.NAME.upper()} DATASET')
    log_line('=' * 80)

    for freq_name, low_f, high_f in ds_module.FREQ_CONDITIONS:
        log_line(f'\n--- Frequency Condition: {freq_name} ({low_f}-{high_f} Hz) ---')

        accuracies = []
        for idx, subject in enumerate(ds_module.SUBJECTS):
            log_line(f'\n[Subject {idx + 1}/{len(ds_module.SUBJECTS)}]')
            acc = train_eval_subject(ds_module, subject, freq_name, low_f, high_f, n_epochs=max_epochs)
            if acc is not None and not np.isnan(acc):
                accuracies.append(acc)

        results[freq_name] = {
            'accuracies': accuracies,
            'mean': float(np.mean(accuracies)) if accuracies else float('nan'),
            'std': float(np.std(accuracies)) if accuracies else float('nan'),
        }
        r = results[freq_name]
        log_line(f'\n{freq_name} Summary:')
        log_line(f'  Mean: {r["mean"]:.3f}' if r["mean"] == r["mean"] else '  Mean: nan')
        log_line(f'  Std:  {r["std"]:.3f}' if r["std"] == r["std"] else '  Std: nan')
        log_line(f'  N:    {len(accuracies)}')
        log_line(f'  Results: {[f"{x:.3f}" for x in accuracies]}')

    log_line('\n' + '=' * 80)
    log_line('FINAL RESULTS SUMMARY')
    log_line('=' * 80)
    for freq_name, _, _ in ds_module.FREQ_CONDITIONS:
        r = results[freq_name]
        mean_str = f'{r["mean"]:.3f}' if r["mean"] == r["mean"] else 'nan'
        std_str = f'{r["std"]:.3f}' if r["std"] == r["std"] else 'nan'
        log_line(f'{freq_name}: Mean={mean_str}, Std={std_str}, N={len(r["accuracies"])}')

    json_path = f'deepconvnet_{ds_module.NAME}_results.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    log_line(f'\nResults also saved to {json_path}')
    log_line(f'\nCompleted at {datetime.now()}')
