# CUDA Error 802 — H100 SXM Fabric State Stuck "In Progress"

## Summary

Integration tests for GPTQ cannot run because CUDA fails to initialize on this Lambda Labs H100 SXM instance. The NVIDIA driver reports **CUDA error 802: system not yet initialized**, caused by the GPU's fabric state being permanently stuck at **"In Progress"**.

## Environment

| Component | Value |
|-----------|-------|
| Instance | Lambda Labs cloud VM (KVM/QEMU, Q35 chipset) |
| GPU | NVIDIA H100 80GB HBM3 (SXM5), single GPU passthrough |
| Driver | 570.195.03 (open kernel module, Lambda `0lambda0.24.04.1`) |
| CUDA | 12.8 |
| PyTorch | 2.7.0 (system package, compiled with CUDA 12.8) |
| OS | Ubuntu 24.04, kernel 6.11.0-1016-nvidia |
| Fabric Manager | 570.195.03-0lambda0.24.04.1 |
| IMEX Daemon | 570.195.03 (nvidia-imex-570) |

## Root Cause

This is an **H100 SXM5** GPU passed through to a **KVM virtual machine** via VFIO. The H100 SXM5 form factor uses NVLink and requires the NVIDIA Fabric Manager to complete GPU fabric initialization before CUDA can be used. The initialization sequence is:

1. NVIDIA driver loads → sets fabric state to **"In Progress"**
2. Fabric Manager queries NVSwitch devices → configures NVLink topology → sets fabric state to **"Completed"**
3. Only then does `cuInit()` succeed

On this VM, **step 2 fails** because:

- The NVSwitch hardware is **not passed through** to the VM (only the GPU itself is visible via PCIe)
- Fabric Manager queries the NVSwitch driver, gets `NV_WARN_NOTHING_TO_DO`, and exits with error
- The GPU remains in "In Progress" forever, and `cuInit()` returns error 802

This is a **host-side infrastructure issue** — the hypervisor should either:
- Complete fabric initialization on the host before passing the GPU to the VM, or
- Pass through the NVSwitch along with the GPU

## Diagnostic Evidence

### GPU visible but CUDA fails
```
$ nvidia-smi
NVIDIA H100 80GB HBM3 | 570.195.03 | CUDA 12.8 | 0MiB / 81559MiB

$ python3 -c "import torch; print(torch.cuda.is_available())"
CUDA initialization: Unexpected error from cudaGetDeviceCount().
Error 802: system not yet initialized
False
```

### Fabric state stuck
```
$ nvidia-smi -q | grep -A5 "Fabric"
    Fabric
        State                             : In Progress
        Status                            : N/A
```

### Fabric Manager cannot start (no NVSwitch)
```
$ sudo systemctl start nvidia-fabricmanager
nvidia-fabricmanager-start.sh: Detected Pre-NVL5 system
nv-fabricmanager: request to query NVSwitch device information failed:
  WARNING Nothing to do [NV_WARN_NOTHING_TO_DO]
```

### CUDA driver confirms error at lowest level
```python
import ctypes
libcuda = ctypes.CDLL('libcuda.so.1')
print(libcuda.cuInit(0))  # → 802
```

### VM, not bare metal
```
$ systemd-detect-virt
kvm

$ cat /sys/class/dmi/id/product_name
Standard PC (Q35 + ICH9, 2009)

$ lspci | grep NVIDIA
07:00.0 3D controller: NVIDIA Corporation GH100 [H100 SXM5 80GB] (rev a1)
# Note: only 1 NVIDIA device — no NVSwitch visible
```

## What We Tried

| Attempt | Result |
|---------|--------|
| Start Fabric Manager | Fails — no NVSwitch hardware |
| Install + start IMEX daemon (nvidia-imex-570) with single-node config | IMEX runs but doesn't complete fabric init |
| `nvidia-smi -r` (GPU reset) | Fabric state returns to "In Progress" |
| `NVreg_EnableGpuFirmware=0` module parameter | H100 GSP firmware is mandatory, param has no effect |
| GRUB kernel cmdline `nvidia.NVreg_EnableGpuFirmware=0` | Same — GSP cannot be disabled on Hopper |
| Full nvidia module unload + reload with params | Fabric state immediately "In Progress" on load |
| `FM_STAY_RESIDENT_ON_FAILURES=1` in fabricmanager.cfg | FM stays but still can't init without NVSwitch |
| Older PyTorch (2.4.0+cu124) | Same error 802 — driver-level issue |
| Waiting 2+ minutes for async fabric init | State never changes |
| Reboot with updated initramfs | No change |

## Resolution

This **cannot be fixed from inside the VM**. Required action:

1. **Contact Lambda Labs support** — report CUDA error 802 / fabric state "In Progress" on this specific instance. The host needs to have fabric initialized before GPU passthrough.
2. **Request a new instance** — this may be a transient issue with this particular VM's GPU assignment.
3. **Alternative**: request an H100 PCIe instance instead of SXM, as PCIe variants don't require NVSwitch fabric initialization.

## References

- [DigitalOcean: Fix "system not initialized" on multi-GPU Droplets](https://docs.digitalocean.com/support/how-do-i-fix-a-system-not-initialized-error-on-multi-gpu-droplets/)
- [NVIDIA Forums: Error 802 system not yet initialized](https://forums.developer.nvidia.com/t/error-802-system-not-yet-initialized-cuda-11-3/234955)
- [NVIDIA Forums: Fabric Manager initializing CUDA H100](https://forums.developer.nvidia.com/t/nvidia-fabric-manger-initializing-cuda-h100/295716)
- [NVIDIA Forums: CUDA init failure error 802](https://forums.developer.nvidia.com/t/cuda-initialization-failure-with-error-error-802-system-not-yet-initialized/337819)
- [GitHub nvtrust: Pass-through H100 to non-confidential VM](https://github.com/NVIDIA/nvtrust/issues/55)

## Impact on GPTQ Tests

- **Unit tests**: All 5 pass on CPU (quantizer round-trips, GPTQ-vs-RTN comparisons)
- **Integration tests**: 3 tests skip with "CUDA not available" — these require a working GPU:
  - `test_gptq_opt125m_fp8` — OPT-125m end-to-end FP8 quantization + perplexity
  - `test_gptq_tinyllama_int8` — TinyLlama-1.1B end-to-end int8 quantization + perplexity
  - `test_gptq_plus_fp8_patching` — GPTQ FP8 + fused-ln-linear `patch_llama_fp8()` composition
