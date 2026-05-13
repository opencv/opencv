# MLAS (Microsoft Linear Algebra Subprograms)

MLAS is a compute library containing processor-optimized GEMM kernels and
platform-specific threading code. It is the default math kernel library used
internally by ONNX Runtime.

## Provenance

- **Upstream**: https://github.com/microsoft/onnxruntime
- **Source path**: `onnxruntime/core/mlas/`
- **Imported**: 2026-05-04
- **Upstream commit**: TODO — fill in the exact ORT commit SHA on next re-vendor
- **License**: MIT (see [LICENSE](LICENSE))

## What is vendored

The SGEMM (single-precision GEMM) subset of MLAS plus `MlasFlashAttention`
(fused multi-head attention). The rest of MLAS (quantized GEMM, conv,
FP16-dispatch SoftMax, etc.) is excluded so the OpenCV DNN module gets the
fast SGEMM and FlashAttention paths without dragging in the full library.

Source files imported verbatim from upstream:

- `lib/sgemm.cpp` — SGEMM dispatch and host-side glue.
- `lib/compute.cpp` — softmax / exp / row-max / sum-exp kernels. Only the
  portable C++ fallbacks for `MlasReduceMaximumF32Kernel` and
  `MlasComputeSumExpF32Kernel` are exercised; no per-arch `.S` softmax
  kernels are vendored. The file is imported whole (FP16 / GQA template
  specializations compile but never run — `SoftmaxDispatch` stays nullptr).
- `lib/flashattn.cpp` — the `MlasFlashAttention` / `MlasFlashAttentionThreaded`
  entry points. Depends on `MlasSgemmOperation` (in `sgemm.cpp`) and the two
  portable kernels above.
- `lib/softmax.h` — header included by `compute.cpp`; pure FP16-dispatch
  typedefs, harmless under FP32-only builds.
- Per-arch SGEMM kernels under `lib/<arch>/`.

Top-level layout:

- `inc/` — public MLAS headers (kept verbatim from upstream).
- `lib/` — implementation, kept verbatim from upstream except for the local
  patches listed below. Per-architecture kernels live in subdirectories
  (`x86_64/`, `aarch64/`, `arm/`, `power/`, `riscv64/`, `loongarch64/`,
  `s390x/`, `sve/`, `kleidiai/`).
- `CMakeLists.txt` — OpenCV-side build glue. Builds an OBJECT library
  (`opencv_dnn_mlas`) whose objects are linked directly into `opencv_dnn`.
- `threading_opencv.cpp` — OpenCV-side replacement for `lib/threading.cpp`
  (see "Local patches" below). Carries the OpenCV license header.

## Copyright

Most files are © Microsoft Corporation and licensed MIT. Some upstream
contributions in `lib/` carry additional MIT-licensed copyrights — they are
preserved verbatim in the file headers:

- `lib/kleidiai/mlasi_kleidiai.h` — © Arm Limited 2025.
- `lib/erf_neon_fp16.{h,cpp}`, `lib/gelu_neon_fp16.{h,cpp}` — © FUJITSU
  LIMITED 2025 (jointly with Microsoft).

The OpenCV-authored files in this directory (`CMakeLists.txt`,
`threading_opencv.cpp`, this `README.md`) are licensed under OpenCV's
top-level license (Apache 2.0).

## Local patches against upstream

These deviate from a clean upstream import and must be re-applied on every
re-vendor:

1. `lib/threading.cpp` is dropped. Its three threading entry points
   (`MlasExecuteThreaded`, `MlasTrySimpleParallel`, `MlasTryBatchParallel`)
   are reimplemented in `threading_opencv.cpp` on top of `cv::parallel_for_`.
2. `lib/mlasi.h` — `MlasGetMaximumThreadCount()` returns
   `cv::getNumThreads()` when `MLAS_OPENCV_THREADING` is defined. Look for
   the `MLAS_OPENCV_THREADING` block.
3. `lib/platform.cpp` — non-SGEMM dispatch is wrapped in `#if 0` blocks
   (search for `MLAS_GEMM_ONLY`). Allows building the SGEMM-only subset
   without pulling in the rest of the MLAS sources. The top-of-file
   `erf_neon_fp16.h` / `gelu_neon_fp16.h` includes are also gated with
   `!defined(MLAS_GEMM_ONLY)` because those headers transitively pull in
   non-vendored FP16 sources (`fp16_common.h`, `softmax_kernel_neon.h`).
   The `MLAS_GEMM_ONLY` ctor also assigns `ReduceMaximumF32Kernel` and
   `ComputeSumExpF32Kernel` to the portable `compute.cpp` fallbacks so
   `MlasFlashAttention` works without per-arch softmax kernels.
4. `lib/core/common/{narrow,common}.h` — minimal shims for ORT internals
   that MLAS calls; not present upstream as MLAS sources, only as ORT
   includes.

## Build flags

- `HAVE_MLAS` is set by this directory's `CMakeLists.txt` when the host
  arch/OS is wired up.
- `BUILD_MLAS_NO_ONNXRUNTIME=1`, `MLAS_OPENCV_THREADING=1`,
  `MLAS_GEMM_ONLY=1` are set as private compile definitions on the OBJECT
  library.

## Caller in OpenCV

The thin wrapper that dispatches OpenCV GEMMs to MLAS lives at
[modules/dnn/src/layers/cpu_kernels/mlas_gemm.{hpp,cpp}](../../modules/dnn/src/layers/cpu_kernels/).
It only includes `mlas.h` (the public header) and falls back to the existing
fast_gemm path when MLAS is unavailable or the requested shape is unsupported.

## Upstream unit tests

Unit tests for the SGEMM kernels live in upstream ONNX Runtime under
`onnxruntime/test/mlas`. They are not vendored here; OpenCV's own DNN tests
exercise the integration.
