import torch.nn as nn
from typing import TypeAlias
import config

from .spectral_cnn import EMGSpectralCNN
from .feature_mlp import EMGFeatureMLP
from .conv_recurrent import EMGConvRecurrent

EMGClassifier: TypeAlias = nn.Module

MODEL_BY_FEATURE: dict[str, type[EMGClassifier]] = {  # type: ignore
    "features": EMGFeatureMLP,  # feature time-domain agregate
    "spectral": EMGSpectralCNN,  # magnitudine |FFT|
    "temporal": EMGConvRecurrent,  # semnal în domeniul timp
}


def get_model(feature, device, n_classes=None, n_channels=None):
    n_classes = config.N_CLASSES if n_classes is None else n_classes
    n_channels = config.N_CHANNELS if n_channels is None else n_channels

    if feature not in MODEL_BY_FEATURE:
        raise ValueError(
            f"feature necunoscut: {feature!r}. Folosește: features, spectral, temporal"
        )

    return MODEL_BY_FEATURE[feature](n_classes, n_channels).to(device)
