# OpenCV Hardware Acceleration Layer (HAL) {#api_hal}

## Detailed Description

OpenCV ships a single source tree that must run efficiently on wildly diverse hardware architectures, including x86, Arm, and RISC-V. Because each architecture utilizes wildly different SIMD widths and instruction sets (such as SSE2, AVX-512, NEON, SVE, and RVV), writing separate hand-optimized kernels for every algorithm is an unmaintainable approach.

To solve this, the performance architecture relies on the **two pillars of OpenCV**:

- **Universal Intrinsics:** A framework allowing developers to write vectorized code (a vector loop) just once using portable wrappers. The compiler then emits the highly optimized native intrinsics for the target architecture. (For an in-depth guide on writing vectorized code, visit the @ref tutorial_univ_intrin "Universal Intrinsics Documentation").
- **The Hardware Acceleration Layer (HAL):** A replaceable, stable C-interface that allows hardware vendors to bypass generic implementations entirely and inject their own highly tuned, silicon-specific libraries into OpenCV's core operations.

This document focuses entirely on the second pillar: HAL.

## What is HAL?

The Hardware Acceleration Layer (HAL) acts as the thin, interceptable layer between OpenCV's high-level algorithms and the hardware metal. While Universal Intrinsics make generic code fast, hardware vendors often possess proprietary, hand-tuned libraries that utilize specific cache blocking strategies, custom ISAs, or hardware accelerators that outpace generic implementations.

HAL allows these vendors to reach OpenCV's enormous user base without needing to fork the repository or maintain complex patch sets.

## Architecture & The Call Chain

When an OpenCV application calls a standard algorithm, the execution follows a strict fallback chain to guarantee maximum performance and correctness:

1. **The OpenCV Algorithm** (e.g., `cv::resize`, `cv::cvtColor`, `cv::gemm`) is invoked.
2. **HAL Dispatcher:** OpenCV queries the `cv_hal_*` interface to check if a registered vendor backend has an optimized implementation for this specific operation and data type.
3. **Vendor Backend Execution:** If implemented, the vendor library (e.g., IPP, KleidiCV) executes the operation and returns the result.
4. **Graceful Fallback:** If the backend is absent, or returns a `NOT_IMPLEMENTED` status, OpenCV transparently falls back to its own dispatched Universal Intrinsics. If no SIMD path exists, it falls back to portable C++ scalar code.

## Scope of Interceptable Operations

HAL exposes approximately 280 interceptable operations across the library. A vendor does not need to implement all of them; covering just the ~20 hotspots that dominate real computer vision workloads (color conversions, resizes, warps, filters, and GEMM) captures most of the runtime.

| Module | `cv_hal_*` Entry Points | Examples |
|---|---|---|
| core | ~199 | add/sub/mul/div (all depths), gemm, SVD, LU, norm, split/merge, convertScale |
| imgproc | ~73 | resize, warpAffine/Perspective, remap, filter, sepFilter, morph, cvtColor, threshold, canny |
| geometry | ~4 | geometric transforms |
| features | ~4 | FAST corners, descriptor distance |
| video | ~3 | optical-flow primitives |

## Design Principles of the Replaceable Interface

The HAL interface is engineered to be as friction-less as possible for both OpenCV maintainers and hardware vendors:

- **Plain C API:** Avoids complex C++ ABI headaches, making compiled vendor libraries easy to ship and link.
- **Subset Implementation:** Vendors are encouraged to implement only the operations where their silicon has a distinct advantage. Partial coverage is a first-class feature.
- **Stable Signatures:** HAL function signatures are immutable. Backends built for older versions will survive and function across future OpenCV releases.
- **Low-Overhead Dispatch (New in OpenCV 5.0):** To reduce API bloat and eliminate calling overhead inside inner loops, modern OpenCV architecture uses a highly efficient function-pointer style dispatcher (`cv_hal_get_*_func()`) rather than rigid, type-specific C entry points. Through this new interface, rewritten geometric operations show performance gains ranging from 10% to over 300%.

## Plugging a Backend In

Vendor backends are integrated at build time via CMake. OpenCV provides built-in shortcut flags (like `WITH_IPP` or `WITH_KLEIDICV`) that automatically register known backends.

Alternatively, custom HALs can be pointed to directly:

```bash
cmake -DOpenCV_HAL="myvendor_hal;ipp"
```

**Runtime Coexistence:** Multiple HALs can coexist in a single build. At configure time, OpenCV registers every enabled HAL into an ordered list. At runtime, they are tried in registration order. This allows a user to combine a highly specialized custom HAL with a general-purpose HAL (like IPP) to achieve best-available acceleration automatically.

## Implementing a Custom HAL

If you are a hardware vendor or an advanced developer looking to build your own custom acceleration layer, OpenCV provides templates to get you started quickly.

### 1. Start with the Templates

Do not start from scratch. Navigate to the OpenCV source tree and look at the provided samples:

- `samples/hal/c_hal`: A barebones skeleton template demonstrating the required structure.
- `samples/hal/slow_hal`: A fully working example that demonstrates how to properly wire up the CMake configuration and intercept calls.

### 2. Implement the Interface

Write your optimized C code targeting the `cv_hal_<op>` signatures (or the new `cv_hal_get_<op>_func` signatures for OpenCV 5.0).

- If your library successfully computes the result, return `CV_HAL_OK` (or the equivalent success macro).
- If your library does not support the specific array size, data type, or operation, simply return `NOT_IMPLEMENTED`. OpenCV will gracefully handle the failure and compute the result using its own generic SIMD code.

### 3. CMake Configuration Requirements

When OpenCV searches for your custom HAL at build time, it runs `find_package(<your_hal_name>)`. Your CMake configuration must export the following variables so OpenCV can link against it:

- `<hal_name>_FOUND`
- `<hal_name>_LIBRARIES`
- `<hal_name>_HEADERS`
- `<hal_name>_INCLUDE_DIRS`

### 4. Build OpenCV with Your HAL

Once your custom library is compiled and your CMake configuration is ready, build OpenCV by explicitly pointing it to your build directory:

```bash
cmake -DOpenCV_HAL="myvendor_hal" -DOpenCV_HAL_DIR=<path-to-your-build> ..
```

Verify the integration by checking the CMake configure summary for the `OpenCV_USED_HAL` status line.

## The HAL Ecosystem

OpenCV supports multiple production-ready vendor backends natively, featuring deeply integrated support for modern architectures. These can be enabled at build time using the following CMake shortcuts:

| Backend | Vendor | Target | Scope | CMake Flag |
|---|---|---|---|---|
| IPP/IPPICV | Intel | x86/x86-64 | stats, transforms, warp | `WITH_IPP` |
| KleidiCV | Arm | AArch64 (NEON/SVE2/SME2) | color, filter, resize, warp | `WITH_KLEIDICV` |
| Carotene | NVIDIA | Arm NEON (v7/v8) | 80+ imgproc/arith ops | `WITH_CAROTENE` |
| FastCV | Qualcomm | Snapdragon (Android/Linux Arm) | core + imgproc | `WITH_FASTCV` |
| RVV HAL | OpenCV China, SpacemiT | RISC-V RVV 1.0 | ~119 core+imgproc ops | `WITH_HAL_RVV` |
| NDSRVP | Andes | RISC-V P-extension (DSP) | filter, morph, warp, remap | `WITH_NDSRVP` |
| ARMPL | Arm | AArch64 | BLAS/LAPACK linear algebra | `WITH_ARMPL` |
