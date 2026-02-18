"""
Pluggable quantizer classes for GPTQ.

Provides per-channel (per-row) symmetric quantization to different number formats.
Each quantizer implements find_params() to compute scales and quantize_dequantize()
for the round-trip used during GPTQ error compensation.
"""

import torch
from abc import ABC, abstractmethod


class BaseQuantizer(ABC):
    """Abstract base class for weight quantizers."""

    def __init__(self, device="cpu"):
        self.device = device
        self.scale = None
        self.maxq = None

    @abstractmethod
    def find_params(self, x, weight=True):
        """Compute per-channel quantization parameters from weight tensor.

        Args:
            x: Weight tensor [out_features, in_features].
            weight: If True, compute per-channel (per-row) scales.
        """
        pass

    @abstractmethod
    def quantize_dequantize(self, x):
        """Quantize then immediately dequantize a weight column/slice.

        Args:
            x: Weight slice to quantize, shape [out_features] or [out_features, k].

        Returns:
            Dequantized weight in float32, same shape as input.
        """
        pass

    def ready(self):
        """Whether quantization parameters have been computed."""
        return self.scale is not None

    @abstractmethod
    def get_format_name(self):
        """Return string identifier for this format."""
        pass


class FP8E4M3Quantizer(BaseQuantizer):
    """Per-channel symmetric quantization to float8_e4m3fn.

    scale[i] = amax(|W[i,:]|) / 448.0
    The actual cast to float8_e4m3fn ensures the round-trip matches
    what H100 Tensor Cores will see.
    """

    _FP8_MAX = 448.0  # torch.finfo(torch.float8_e4m3fn).max

    def find_params(self, x, weight=True):
        if weight:
            # Per-channel (per-row) scale
            amax = x.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
        else:
            amax = x.abs().amax().clamp(min=1e-12).unsqueeze(0)
        self.scale = amax / self._FP8_MAX
        self.scale = self.scale.to(self.device)

    def quantize_dequantize(self, x):
        scaled = x / self.scale
        clamped = scaled.clamp(-self._FP8_MAX, self._FP8_MAX)
        # Cast to FP8 and back to get true quantization error
        fp8 = clamped.to(torch.float8_e4m3fn)
        dequant = fp8.to(torch.float32) * self.scale
        return dequant

    def get_format_name(self):
        return "fp8_e4m3"


class Int8SymQuantizer(BaseQuantizer):
    """Per-channel symmetric 8-bit integer quantization.

    scale[i] = amax(|W[i,:]|) / 127.0
    """

    _INT8_MAX = 127

    def find_params(self, x, weight=True):
        if weight:
            amax = x.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
        else:
            amax = x.abs().amax().clamp(min=1e-12).unsqueeze(0)
        self.scale = amax / self._INT8_MAX
        self.scale = self.scale.to(self.device)

    def quantize_dequantize(self, x):
        scaled = x / self.scale
        rounded = scaled.round().clamp(-128, 127)
        dequant = rounded * self.scale
        return dequant

    def get_format_name(self):
        return "int8_sym"


QUANTIZER_REGISTRY = {
    "fp8": FP8E4M3Quantizer,
    "int8": Int8SymQuantizer,
}
