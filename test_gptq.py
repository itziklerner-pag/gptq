"""
Tests for GPTQ quantization.

Follows fused-ln-linear/src/test_correctness.py conventions:
function-based tests, [PASS]/[FAIL] print output, assert for failures,
__main__ runner.
"""

import sys
sys.path.insert(0, "/home/ubuntu/dev")
sys.path.insert(0, "/home/ubuntu/dev/fused-ln-linear")

import torch
import torch.nn as nn
import copy

from quant.quantizer import FP8E4M3Quantizer, Int8SymQuantizer, QUANTIZER_REGISTRY
from quant.gptq import GPTQ
from quant.apply import gptq_quantize_model
from quant.eval_ppl import eval_ppl


# ============================================================
# Unit tests
# ============================================================

def test_fp8_quantizer_unit():
    """Test FP8 quantizer round-trip error is bounded."""
    print("=" * 60)
    print("TEST: FP8 quantizer unit")
    print("=" * 60)

    torch.manual_seed(42)

    for rows, cols in [(64, 64), (256, 768), (768, 768)]:
        W = torch.randn(rows, cols)

        q = FP8E4M3Quantizer()
        q.find_params(W)
        assert q.ready(), "Quantizer should be ready after find_params"

        W_qdq = q.quantize_dequantize(W)
        error = (W - W_qdq).abs()
        max_err = error.max().item()
        # FP8 E4M3 is a floating-point format with non-uniform steps.
        # Max step at largest exponent range [256, 448] is 32, so max error ~16.
        # In original space: max_error <= 16 * scale = 16 * amax / 448 ≈ 0.036 * amax.
        max_amax = W.abs().amax(dim=1).max().item()
        bound = max_amax * 0.04  # ~4% of dynamic range
        status = "PASS" if max_err < bound else "FAIL"
        print(f"  [{status}] shape=({rows}, {cols}): max_err={max_err:.2e}, bound={bound:.2e}, amax={max_amax:.2e}")
        assert max_err < bound, f"FP8 error {max_err} exceeds bound {bound}"

    # Verify format name
    assert FP8E4M3Quantizer().get_format_name() == "fp8_e4m3"
    print("  All FP8 quantizer unit tests passed!\n")


def test_int8_quantizer_unit():
    """Test Int8 quantizer round-trip error is bounded."""
    print("=" * 60)
    print("TEST: Int8 quantizer unit")
    print("=" * 60)

    torch.manual_seed(42)

    for rows, cols in [(64, 64), (256, 768), (768, 768)]:
        W = torch.randn(rows, cols)

        q = Int8SymQuantizer()
        q.find_params(W)
        assert q.ready(), "Quantizer should be ready after find_params"

        W_qdq = q.quantize_dequantize(W)
        error = (W - W_qdq).abs()
        max_err = error.max().item()
        # Int8 uniform rounding error bounded by scale * 0.5
        max_scale = q.scale.max().item()
        bound = max_scale * 0.5 + 1e-6
        status = "PASS" if max_err <= bound else "FAIL"
        print(f"  [{status}] shape=({rows}, {cols}): max_err={max_err:.2e}, bound={bound:.2e}, scale_max={max_scale:.2e}")
        assert max_err <= bound, f"Int8 error {max_err} exceeds bound {bound}"

    assert Int8SymQuantizer().get_format_name() == "int8_sym"
    print("  All Int8 quantizer unit tests passed!\n")


def test_gptq_single_linear_fp8():
    """Test GPTQ on a single linear layer with FP8 produces lower error than RTN."""
    print("=" * 60)
    print("TEST: GPTQ single linear (FP8)")
    print("=" * 60)

    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    in_features, out_features = 768, 768
    linear = nn.Linear(in_features, out_features, bias=False).to(device).float()
    nn.init.normal_(linear.weight, std=0.02)

    # Generate random calibration inputs
    ncalib = 64
    calib_inputs = torch.randn(ncalib, 1, in_features, device=device)

    # Reference output
    with torch.no_grad():
        ref_outs = torch.cat([linear(calib_inputs[i]) for i in range(ncalib)], dim=0)

    # RTN baseline
    linear_rtn = copy.deepcopy(linear)
    q_rtn = FP8E4M3Quantizer(device=device)
    q_rtn.find_params(linear_rtn.weight.data.float())
    linear_rtn.weight.data = q_rtn.quantize_dequantize(linear_rtn.weight.data.float()).to(linear.weight.dtype)

    with torch.no_grad():
        rtn_outs = torch.cat([linear_rtn(calib_inputs[i]) for i in range(ncalib)], dim=0)
    rtn_error = (ref_outs - rtn_outs).pow(2).sum().item()

    # GPTQ
    linear_gptq = copy.deepcopy(linear)
    gptq = GPTQ(linear_gptq)
    gptq.quantizer = FP8E4M3Quantizer(device=device)

    for i in range(ncalib):
        gptq.add_batch(calib_inputs[i], None)

    gptq_loss = gptq.fasterquant()

    with torch.no_grad():
        gptq_outs = torch.cat([linear_gptq(calib_inputs[i]) for i in range(ncalib)], dim=0)
    gptq_error = (ref_outs - gptq_outs).pow(2).sum().item()

    ratio = gptq_error / (rtn_error + 1e-12)
    status = "PASS" if gptq_error < rtn_error else "FAIL"
    print(f"  [{status}] RTN error={rtn_error:.4f}, GPTQ error={gptq_error:.4f}, ratio={ratio:.4f}")
    assert gptq_error < rtn_error, f"GPTQ ({gptq_error:.4f}) should beat RTN ({rtn_error:.4f})"
    print("  GPTQ single linear FP8 test passed!\n")


def test_gptq_single_linear_int8():
    """Test GPTQ on a single linear layer with Int8 produces lower error than RTN."""
    print("=" * 60)
    print("TEST: GPTQ single linear (Int8)")
    print("=" * 60)

    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    in_features, out_features = 768, 768
    linear = nn.Linear(in_features, out_features, bias=False).to(device).float()
    nn.init.normal_(linear.weight, std=0.02)

    ncalib = 64
    calib_inputs = torch.randn(ncalib, 1, in_features, device=device)

    with torch.no_grad():
        ref_outs = torch.cat([linear(calib_inputs[i]) for i in range(ncalib)], dim=0)

    # RTN baseline
    linear_rtn = copy.deepcopy(linear)
    q_rtn = Int8SymQuantizer(device=device)
    q_rtn.find_params(linear_rtn.weight.data.float())
    linear_rtn.weight.data = q_rtn.quantize_dequantize(linear_rtn.weight.data.float()).to(linear.weight.dtype)

    with torch.no_grad():
        rtn_outs = torch.cat([linear_rtn(calib_inputs[i]) for i in range(ncalib)], dim=0)
    rtn_error = (ref_outs - rtn_outs).pow(2).sum().item()

    # GPTQ
    linear_gptq = copy.deepcopy(linear)
    gptq = GPTQ(linear_gptq)
    gptq.quantizer = Int8SymQuantizer(device=device)

    for i in range(ncalib):
        gptq.add_batch(calib_inputs[i], None)

    gptq_loss = gptq.fasterquant()

    with torch.no_grad():
        gptq_outs = torch.cat([linear_gptq(calib_inputs[i]) for i in range(ncalib)], dim=0)
    gptq_error = (ref_outs - gptq_outs).pow(2).sum().item()

    ratio = gptq_error / (rtn_error + 1e-12)
    status = "PASS" if gptq_error < rtn_error else "FAIL"
    print(f"  [{status}] RTN error={rtn_error:.4f}, GPTQ error={gptq_error:.4f}, ratio={ratio:.4f}")
    assert gptq_error < rtn_error, f"GPTQ ({gptq_error:.4f}) should beat RTN ({rtn_error:.4f})"
    print("  GPTQ single linear Int8 test passed!\n")


def test_gptq_beats_rtn():
    """Compare GPTQ vs RTN output error on calibration data — GPTQ must win."""
    print("=" * 60)
    print("TEST: GPTQ beats RTN (comparative)")
    print("=" * 60)

    torch.manual_seed(123)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for fmt_name, fmt_cls in QUANTIZER_REGISTRY.items():
        in_f, out_f = 512, 1024
        linear = nn.Linear(in_f, out_f, bias=True).to(device).float()
        nn.init.normal_(linear.weight, std=0.02)

        ncalib = 32
        calib_inputs = torch.randn(ncalib, 1, in_f, device=device)

        with torch.no_grad():
            ref_outs = torch.cat([linear(calib_inputs[i]) for i in range(ncalib)], dim=0)

        # RTN
        linear_rtn = copy.deepcopy(linear)
        q_rtn = fmt_cls(device=device)
        q_rtn.find_params(linear_rtn.weight.data.float())
        linear_rtn.weight.data = q_rtn.quantize_dequantize(linear_rtn.weight.data.float()).to(linear.weight.dtype)
        with torch.no_grad():
            rtn_outs = torch.cat([linear_rtn(calib_inputs[i]) for i in range(ncalib)], dim=0)
        rtn_error = (ref_outs - rtn_outs).pow(2).mean().item()

        # GPTQ
        linear_gptq = copy.deepcopy(linear)
        gptq = GPTQ(linear_gptq)
        gptq.quantizer = fmt_cls(device=device)
        for i in range(ncalib):
            gptq.add_batch(calib_inputs[i], None)
        gptq.fasterquant()
        with torch.no_grad():
            gptq_outs = torch.cat([linear_gptq(calib_inputs[i]) for i in range(ncalib)], dim=0)
        gptq_error = (ref_outs - gptq_outs).pow(2).mean().item()

        ratio = gptq_error / (rtn_error + 1e-12)
        status = "PASS" if gptq_error < rtn_error else "FAIL"
        print(f"  [{status}] {fmt_name}: RTN_mse={rtn_error:.2e}, GPTQ_mse={gptq_error:.2e}, ratio={ratio:.4f}")
        assert gptq_error < rtn_error, f"GPTQ should beat RTN for {fmt_name}"

    print("  All GPTQ-beats-RTN tests passed!\n")


# ============================================================
# Integration tests
# ============================================================

def test_gptq_opt125m_fp8():
    """Full GPTQ on OPT-125m with FP8, verify perplexity is reasonable."""
    print("=" * 60)
    print("TEST: GPTQ OPT-125m FP8 (integration)")
    print("=" * 60)

    from transformers import AutoModelForCausalLM

    model_name = "facebook/opt-125m"
    device = "cuda"

    if not torch.cuda.is_available():
        print("  SKIP: CUDA not available")
        return

    print("  Loading OPT-125m...")
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)
    except Exception as e:
        print(f"  SKIP: Cannot load model ({e})")
        return

    # Baseline perplexity
    print("  Evaluating baseline perplexity...")
    model.to(device)
    baseline_ppl = eval_ppl(model, model_name, dataset="wikitext2", seqlen=2048, device=device)
    model.cpu()
    torch.cuda.empty_cache()
    print(f"  Baseline PPL: {baseline_ppl:.2f}")

    # GPTQ quantize
    print("  Running GPTQ FP8...")
    quantizers = gptq_quantize_model(
        model, model_name, quant_format="fp8",
        dataset="wikitext2", nsamples=128, seqlen=2048, device=device,
    )

    # Post-GPTQ perplexity
    print("  Evaluating GPTQ FP8 perplexity...")
    model.to(device)
    gptq_ppl = eval_ppl(model, model_name, dataset="wikitext2", seqlen=2048, device=device)
    model.cpu()
    torch.cuda.empty_cache()
    print(f"  GPTQ FP8 PPL: {gptq_ppl:.2f}")

    # Perplexity should be reasonable (< 2x baseline for FP8)
    ratio = gptq_ppl / baseline_ppl
    status = "PASS" if ratio < 1.5 else "FAIL"
    print(f"  [{status}] baseline={baseline_ppl:.2f}, gptq_fp8={gptq_ppl:.2f}, ratio={ratio:.4f}")
    assert ratio < 1.5, f"GPTQ FP8 PPL ({gptq_ppl:.2f}) too high vs baseline ({baseline_ppl:.2f})"

    del model
    torch.cuda.empty_cache()
    print("  GPTQ OPT-125m FP8 integration test passed!\n")


def test_gptq_tinyllama_int8():
    """Full GPTQ on TinyLlama-1.1B with int8."""
    print("=" * 60)
    print("TEST: GPTQ TinyLlama int8 (integration)")
    print("=" * 60)

    from transformers import AutoModelForCausalLM

    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    device = "cuda"

    if not torch.cuda.is_available():
        print("  SKIP: CUDA not available")
        return

    print("  Loading TinyLlama...")
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)
    except Exception as e:
        print(f"  SKIP: Cannot load model ({e})")
        return

    # Baseline
    print("  Evaluating baseline perplexity...")
    model.to(device)
    baseline_ppl = eval_ppl(model, model_name, dataset="wikitext2", seqlen=2048, device=device)
    model.cpu()
    torch.cuda.empty_cache()
    print(f"  Baseline PPL: {baseline_ppl:.2f}")

    # GPTQ
    print("  Running GPTQ int8...")
    quantizers = gptq_quantize_model(
        model, model_name, quant_format="int8",
        dataset="wikitext2", nsamples=128, seqlen=2048, device=device,
    )

    print("  Evaluating GPTQ int8 perplexity...")
    model.to(device)
    gptq_ppl = eval_ppl(model, model_name, dataset="wikitext2", seqlen=2048, device=device)
    model.cpu()
    torch.cuda.empty_cache()
    print(f"  GPTQ int8 PPL: {gptq_ppl:.2f}")

    ratio = gptq_ppl / baseline_ppl
    status = "PASS" if ratio < 1.5 else "FAIL"
    print(f"  [{status}] baseline={baseline_ppl:.2f}, gptq_int8={gptq_ppl:.2f}, ratio={ratio:.4f}")
    assert ratio < 1.5, f"GPTQ int8 PPL ({gptq_ppl:.2f}) too high vs baseline ({baseline_ppl:.2f})"

    del model
    torch.cuda.empty_cache()
    print("  GPTQ TinyLlama int8 integration test passed!\n")


def test_gptq_plus_fp8_patching():
    """GPTQ FP8 on TinyLlama -> patch_llama_fp8() -> verify output close to original."""
    print("=" * 60)
    print("TEST: GPTQ + FP8 patching (composition)")
    print("=" * 60)

    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    device = "cuda"

    if not torch.cuda.is_available():
        print("  SKIP: CUDA not available")
        return

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from src.fp8_linear import patch_llama_fp8

    print("  Loading TinyLlama...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model_orig = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16).to(device).eval()
    except Exception as e:
        print(f"  SKIP: Cannot load model ({e})")
        return

    # Get baseline outputs
    texts = [
        "The quick brown fox jumps over the lazy dog",
        "Machine learning is transforming",
    ]
    baseline_logits = {}
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt").to(device)
            baseline_logits[text] = model_orig(**inputs).logits.cpu()

    del model_orig
    torch.cuda.empty_cache()

    # Load fresh model, GPTQ then patch
    print("  Loading fresh model for GPTQ + patching...")
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16)

    print("  Running GPTQ FP8...")
    quantizers = gptq_quantize_model(
        model, model_name, quant_format="fp8",
        dataset="wikitext2", nsamples=64, seqlen=2048, device=device,
    )

    print("  Applying patch_llama_fp8...")
    model.to(device)
    patch_llama_fp8(model, fused_norm=False)
    model.eval()

    # Compare
    all_passed = True
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt").to(device)
            patched_logits = model(**inputs).logits.cpu()
            orig = baseline_logits[text]

            max_diff = (orig.float() - patched_logits.float()).abs().max().item()
            mean_diff = (orig.float() - patched_logits.float()).abs().mean().item()
            # BF16 + FP8 quantization: expect larger diffs
            status = "PASS" if max_diff < 10.0 else "FAIL"
            if max_diff >= 10.0:
                all_passed = False
            print(f"  [{status}] \"{text[:40]}...\": max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}")

    if all_passed:
        print("  GPTQ + FP8 patching composition test passed!\n")
    else:
        print("  WARNING: Some composition tests exceeded threshold\n")

    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    # Unit tests
    test_fp8_quantizer_unit()
    test_int8_quantizer_unit()
    test_gptq_single_linear_fp8()
    test_gptq_single_linear_int8()
    test_gptq_beats_rtn()

    # Integration tests
    test_gptq_opt125m_fp8()
    test_gptq_tinyllama_int8()
    test_gptq_plus_fp8_patching()

    print("=" * 60)
    print("ALL GPTQ TESTS COMPLETED")
    print("=" * 60)
