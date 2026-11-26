@echo off
REM Wrapper so Candle kernels build on VS 2026 + CUDA 13.1
REM (MSVC 14.51 wants CUDA 13.2+; nvcc 13.1 only "supports" VS 2019/2022)
set "REAL_NVCC=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\nvcc.exe"
"%REAL_NVCC%" -allow-unsupported-compiler -D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH %*
