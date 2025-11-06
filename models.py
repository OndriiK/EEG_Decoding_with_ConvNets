# models.py
from functools import partial
from torch import nn
from braindecode.models import Deep4Net, ShallowFBCSPNet, EEGResNet
from braindecode.models.util import to_dense_prediction_model


def build_deep(n_chans: int, n_classes: int):
    """DeepConvNet as in your current code (final_conv_length=2), dense-converted."""
    m = Deep4Net(
        n_chans, n_classes,
        input_window_samples=None,
        final_conv_length=2,
    )
    to_dense_prediction_model(m)
    return m


def build_shallow(n_chans: int, n_classes: int):
    """ShallowFBCSP as in your current code (final_conv_length=30), dense-converted."""
    m = ShallowFBCSPNet(
        n_chans, n_classes,
        input_window_samples=None,
        final_conv_length=30,
    )
    to_dense_prediction_model(m)
    return m


def build_resnet(n_chans: int, n_classes: int, init_a: float = 1.0):
    """EEGResNet with your init and hyperparams."""
    return EEGResNet(
        n_chans, n_classes,
        input_window_samples=None,
        n_first_filters=48,
        final_pool_length=10,
        conv_weight_init_fn=partial(nn.init.kaiming_normal_, a=init_a),
    )
