"""
GPTQ algorithm implementation.

Adapted from ist-daslab/gptq. Uses second-order (Hessian) information from
calibration data to optimally quantize weight matrices, minimizing output
error via error propagation through the inverse Hessian.

One GPTQ instance per nn.Linear layer.
"""

import torch
import torch.nn as nn
import math


class GPTQ:
    """GPTQ quantizer for a single nn.Linear layer.

    Accumulates the Hessian from calibration inputs, then runs the
    fasterquant algorithm to produce optimally quantized weights.
    """

    def __init__(self, layer):
        """Initialize GPTQ for a linear layer.

        Args:
            layer: nn.Linear module to quantize.
        """
        self.layer = layer
        self.dev = layer.weight.device
        W = layer.weight.data.clone().float()
        self.rows = W.shape[0]  # out_features
        self.columns = W.shape[1]  # in_features
        self.H = torch.zeros((self.columns, self.columns), device=self.dev, dtype=torch.float32)
        self.nsamples = 0
        self.quantizer = None

    def add_batch(self, inp, out):
        """Accumulate Hessian from a calibration batch.

        Args:
            inp: Input activations, shape [batch, seq, in_features] or [batch, in_features].
            out: Output activations (unused, kept for hook compatibility).
        """
        if inp.dim() == 3:
            inp = inp.reshape(-1, inp.shape[-1])
        batch_size = inp.shape[0]

        inp = inp.float().to(self.dev)

        # Incremental Hessian update
        self.H *= self.nsamples / (self.nsamples + batch_size)
        self.nsamples += batch_size
        inp_scaled = math.sqrt(2.0 / self.nsamples) * inp.t()
        self.H += inp_scaled @ inp_scaled.t()

    def fasterquant(self, blocksize=128, percdamp=0.01):
        """Run the GPTQ fasterquant algorithm.

        Quantizes the layer's weights using inverse-Hessian-based error
        compensation, processing columns in blocks for efficiency.

        Args:
            blocksize: Number of columns to process per block.
            percdamp: Damping factor as percentage of mean diagonal.

        Returns:
            Total squared quantization loss.
        """
        W = self.layer.weight.data.clone().float()
        H = self.H

        # Compute per-channel scales
        self.quantizer.find_params(W)

        # Zero out dead columns (no calibration signal)
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        # Damp diagonal for numerical stability
        damp = percdamp * torch.mean(torch.diag(H))
        diag_idx = torch.arange(self.columns, device=self.dev)
        H[diag_idx, diag_idx] += damp

        # Cholesky-based inverse Hessian
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        total_loss = torch.zeros(self.rows, device=self.dev, dtype=torch.float32)

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W_block = W[:, i1:i2].clone()
            Q_block = torch.zeros_like(W_block)
            Err_block = torch.zeros_like(W_block)
            Hinv_block = Hinv[i1:i2, i1:i2]

            for i in range(count):
                w = W_block[:, i]
                d = Hinv_block[i, i]

                q = self.quantizer.quantize_dequantize(w.unsqueeze(1)).squeeze(1)
                Q_block[:, i] = q
                err = (w - q) / d
                Err_block[:, i] = err

                total_loss += err ** 2

                # Propagate error to remaining columns in block
                W_block[:, i:] -= err.unsqueeze(1) * Hinv_block[i, i:].unsqueeze(0)

            # Write quantized block back
            W[:, i1:i2] = Q_block

            # Propagate block error to all remaining columns
            if i2 < self.columns:
                W[:, i2:] -= Err_block @ Hinv[i1:i2, i2:]

        # Write dequantized weights back to layer
        self.layer.weight.data = W.to(self.layer.weight.dtype)

        total_loss = total_loss.sum().item()
        return total_loss

    def free(self):
        """Free Hessian memory."""
        self.H = None
        torch.cuda.empty_cache()
