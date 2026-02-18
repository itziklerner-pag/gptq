"""
Top-level orchestration for GPTQ model quantization.

Coordinates calibration data loading, layer-by-layer Hessian accumulation,
and fasterquant execution. Memory-efficient: only one transformer layer
on GPU at a time.
"""

import torch
import torch.nn as nn

from quant.quantizer import QUANTIZER_REGISTRY
from quant.gptq import GPTQ
from quant.calibration import get_calibration_data, LayerInputCatcher
from quant.model_utils import find_layers, get_model_layers, get_embedding_layers


def gptq_quantize_model(
    model,
    model_name,
    quant_format="fp8",
    dataset="wikitext2",
    nsamples=128,
    seqlen=2048,
    blocksize=128,
    percdamp=0.01,
    seed=0,
    device="cuda",
):
    """Quantize all linear layers in a model using GPTQ.

    Processes transformer layers sequentially (one on GPU at a time) to
    minimize memory usage. For each layer, accumulates the Hessian from
    calibration data, then runs fasterquant to produce optimally quantized
    weights for the target format.

    Args:
        model: HuggingFace causal LM model.
        model_name: Model name string (for tokenizer/dataset loading).
        quant_format: Target format key from QUANTIZER_REGISTRY ("fp8" or "int8").
        dataset: Calibration dataset ("wikitext2" or "c4").
        nsamples: Number of calibration samples.
        seqlen: Sequence length per sample.
        blocksize: GPTQ block size.
        percdamp: GPTQ damping factor.
        seed: Random seed.
        device: Target device for quantization.

    Returns:
        Dict of {layer_dotted_path: quantizer} containing per-channel scales.
    """
    if quant_format not in QUANTIZER_REGISTRY:
        raise ValueError(f"Unknown quant format: {quant_format}. Available: {list(QUANTIZER_REGISTRY.keys())}")

    print(f"[GPTQ] Loading calibration data: {dataset}, {nsamples} samples, seqlen={seqlen}")
    calibration_data = get_calibration_data(model_name, dataset, nsamples, seqlen, seed)

    print(f"[GPTQ] Identifying model structure...")
    layers, arch_type = get_model_layers(model)
    print(f"[GPTQ] Architecture: {arch_type}, {len(layers)} layers")

    # Move embeddings to device for input capture
    embedding_modules = get_embedding_layers(model, arch_type)
    for mod in embedding_modules:
        mod.to(device)

    # Get model dtype and hidden size
    model_dtype = next(model.parameters()).dtype
    hidden_size = model.config.hidden_size

    # Capture first-layer inputs
    print(f"[GPTQ] Capturing first-layer inputs...")
    catcher = LayerInputCatcher(
        layers[0], nsamples, hidden_size, seqlen, model_dtype, device
    )
    layers[0] = catcher

    model.eval()
    with torch.no_grad():
        for i, (input_ids, _) in enumerate(calibration_data):
            try:
                model(input_ids.to(device))
            except ValueError:
                pass  # Expected: LayerInputCatcher aborts forward

    layers[0] = catcher.module
    inps = catcher.inps
    layer_kwargs = catcher.kwargs

    # Move embeddings back to CPU
    for mod in embedding_modules:
        mod.cpu()

    torch.cuda.empty_cache()

    # Track all quantizers for return
    all_quantizers = {}

    # Process layers sequentially
    outs = torch.zeros_like(inps)

    for layer_idx in range(len(layers)):
        layer = layers[layer_idx]
        layer.to(device)

        print(f"[GPTQ] Layer {layer_idx}/{len(layers)-1}: ", end="", flush=True)

        # Find all linear sublayers
        subset = find_layers(layer)

        # Create GPTQ instance + quantizer for each linear
        gptq_instances = {}
        for name, linear in subset.items():
            gptq_inst = GPTQ(linear)
            gptq_inst.quantizer = QUANTIZER_REGISTRY[quant_format](device=device)
            gptq_instances[name] = gptq_inst

        # Register hooks to accumulate Hessian
        handles = []

        def make_hook(gptq_inst):
            def hook(module, inp, out):
                gptq_inst.add_batch(inp[0], out)
            return hook

        for name, gptq_inst in gptq_instances.items():
            handle = subset[name].register_forward_hook(make_hook(gptq_inst))
            handles.append(handle)

        # Run calibration through this layer
        with torch.no_grad():
            for j in range(nsamples):
                outs[j] = layer(inps[j].unsqueeze(0), **layer_kwargs)[0]

        # Remove hooks
        for h in handles:
            h.remove()

        # Run fasterquant on each linear sublayer
        losses = {}
        for name, gptq_inst in gptq_instances.items():
            loss = gptq_inst.fasterquant(blocksize=blocksize, percdamp=percdamp)
            losses[name] = loss
            all_quantizers[f"layer.{layer_idx}.{name}"] = gptq_inst.quantizer
            gptq_inst.free()

        # Print per-sublayer losses
        loss_strs = [f"{n}={l:.2f}" for n, l in losses.items()]
        print(", ".join(loss_strs))

        # Re-run calibration to get updated outputs for next layer
        with torch.no_grad():
            for j in range(nsamples):
                outs[j] = layer(inps[j].unsqueeze(0), **layer_kwargs)[0]

        # Move layer back to CPU, swap inputs/outputs
        layer.cpu()
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    print(f"[GPTQ] Quantization complete. {len(all_quantizers)} sublayers quantized.")
    return all_quantizers
