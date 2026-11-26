@echo off
setlocal EnableExtensions
cd /d "%~dp0.."

set "VCVARS=C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
if not exist "%VCVARS%" (
  echo Could not find vcvars64.bat at:
  echo   %VCVARS%
  exit /b 1
)

call "%VCVARS%" || exit /b 1

set CUDA_COMPUTE_CAP=120
set "NVCC_CCBIN=C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.51.36231\bin\Hostx64\x64\cl.exe"
set "NVCC=%CD%\scripts\nvcc-wrap.cmd"

cargo run --release %*
exit /b %ERRORLEVEL%
