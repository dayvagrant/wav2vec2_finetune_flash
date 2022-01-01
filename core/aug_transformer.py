from dataclasses import dataclass

import librosa
import numpy as np
from audiomentations import (AddGaussianNoise, AddShortNoises, Compose,
                             Mp3Compression, PitchShift, Resample, Shift,
                             TimeStretch)
from flash.core.data.io.input_transform import InputTransform


def resampler_rate(waveform, sample_rate, resample_rate=None, mono=True):
    if mono and waveform.ndim > 1:
        waveform = np.mean(waveform, axis=0)
    if resample_rate and resample_rate != sample_rate:
        try:
            waveform = librosa.resample(waveform, sample_rate, resample_rate)
            sample_rate = resample_rate
        except ValueError:
            print([waveform.shape, sample_rate, resample_rate])
    return waveform, sample_rate


class Resampler(object):
    """Apply resample_finc augmentation for waveform."""

    def __init__(self, resample_rate):
        self.resample_rate = resample_rate

    def __call__(self, waveform, sample_rate):
        waveform, sample_rate = resampler_rate(
            waveform, sample_rate, self.resample_rate
        )
        return waveform


class PreEmphasisFilter(object):
    """PreEmphasisFilter - high frequency highlight."""

    def __call__(self, waveform, sample_rate):
        try:
            return librosa.effects.preemphasis(waveform)
        except Exception as ex:
            print("PreEmphasisFilter")
            raise ex


@dataclass
class BaseSpeechInputTransform(InputTransform):
    """Base aug transformer."""

    resample_rate: int = 16000
    augmentation = Compose(
        [
            Resampler(resample_rate),
        ]
    )

    def sample_augmentator(self, sample, resample_rate=resample_rate):
        """Augmentation for audio data [waveform, sample_rate]"""
        sampling_rate = sample["metadata"]["sampling_rate"]
        sample["input"] = self.augmentation(sample["input"], sampling_rate)
        sample["target"] = sample["target"].upper()
        sample["metadata"]["sampling_rate"] = resample_rate
        return sample

    def per_sample_transform(self):
        """Set func for hook catcher."""
        return self.sample_augmentator


@dataclass
class CallsSpeechInputTransform(BaseSpeechInputTransform):
    """Change method of parent class for the waveform like real calls."""

    augmentation = Compose(
        [
            Resampler(resample_rate=BaseSpeechInputTransform.resample_rate),
            PreEmphasisFilter(),
            AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.02, p=0.3),
            Mp3Compression(backend="pydub", p=0.3),
        ]
    )


@dataclass
class PredictCallsSpeechInputTransform(BaseSpeechInputTransform):
    """Change method of parent class for predict real calls."""

    augmentation = Compose(
        [
            Resampler(resample_rate=BaseSpeechInputTransform.resample_rate),
            PreEmphasisFilter(),
        ]
    )


import re


def aug_transfrms_cfg(data_module):
    """Return data_module transform configs."""
    aug_configs = dict()
    transform_selectors = [i for i in data_module.__dict__ if re.findall("_input$", i)]
    for selector in transform_selectors:
        name_config = "".join(["transform", selector])
        try:
            input_atrs = getattr(data_module, selector)
            input_aug_class_name = input_atrs.transform.__class__.__name__
            internal_transforms = input_atrs.transform.__class__.augmentation.transforms
            internal_transforms_params = dict(
                (i.__class__.__name__, i.__dict__) for i in internal_transforms
            )
            input_aug_params = dict(
                input_transform_class_name=input_aug_class_name,
                internal_transforms_params=internal_transforms_params,
            )
            aug_configs.update({name_config: input_aug_params})
        except AttributeError:
            aug_configs.update({name_config: None})
            pass
    return aug_configs
