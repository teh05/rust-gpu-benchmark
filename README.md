# rust-gpu-benchmark (Candle + local CUDA)

Port of `gpu_benchmark_approaches (1).py` / the Colab notebook to **Rust + Candle** on your laptop GPU (RTX 5060 Mobile / sm_120).

Shows why GPU timing without `synchronize` is fake.

## Prerequisites

- NVIDIA driver (`nvidia-smi` works)
- CUDA Toolkit with `nvcc` (tested: 13.1)
- Rust via `rust-toolchain.toml` (**1.94.0+** — Candle needs safe `__cpuid_count`)
- Windows: Visual Studio / MSVC (`cl.exe`)

Check compute capability:

```bat
nvidia-smi --query-gpu=compute_cap --format=csv
```

RTX 50-series / Blackwell → `12.0` (sm_120).

## Run (Windows + VS 2026 + CUDA 13.1)

First build (compiles Candle CUDA kernels — can take several minutes):

```bat
cd rust-gpu-benchmark
scripts\build-release.cmd
```

Then run:

```bat
scripts\run-release.cmd
```

`scripts\nvcc-wrap.cmd` adds `-allow-unsupported-compiler` so CUDA 13.1 accepts VS 2026. Prefer upgrading to **CUDA 13.2+** when you can.

Always use `--release` for meaningful timings.

## What to expect

| Approach | Meaning |
|----------|---------|
| Without Sync | CPU only timed kernel *enqueue* → often far too fast |
| CPU Sync + Timer | Waits for GPU → real wall time |
| Candle synchronize | Same correct method Candle uses in its own benches |

`[WRONG]` should be much smaller than `[CORRECT]`. Absolute ms will differ from Colab (different GPU / 55 W laptop TGP).

## Notes

- Candle does not expose `torch.cuda.Event`-style timing; official practice is `device.synchronize()` around a CPU timer.
- Dependencies track Candle **git main** for better sm_120 support.
