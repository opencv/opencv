Metal UMat Backend {#metal_umat_backend}
==================

@ingroup core_metal

The Metal UMat backend accelerates selected Transparent API (`cv::UMat`) paths
on Apple platforms with Apple Metal. User code keeps using the regular OpenCV
functions and passes `cv::UMat` inputs and outputs, as with the OpenCL UMat
backend.

## Build and Runtime Selection

Metal support is controlled by the `WITH_METAL` CMake option. The option is
visible on Apple platforms and defaults to `ON` when OpenCV optimizations are
enabled.

```text
cmake -S . -B build -DWITH_METAL=ON
```

When Metal is available on Apple platforms, OpenCV disables the OpenCL UMat
backend during configuration. This keeps `cv::UMat` backed by a single active
device runtime on Apple systems:

- Apple platforms with Metal: `cv::UMat` uses the Metal allocator and Metal
  kernels for supported operations.
- Other platforms, or Apple builds without Metal: `cv::UMat` can use the
  existing OpenCL backend when OpenCL is enabled and available.
- Unsupported Metal operations fall back to the existing CPU path.

At runtime, `cv::metal::haveMetal()` reports whether the configured build can
create a default Metal device and command queue.

## Supported UMat Operations

The initial Metal backend focuses on memory correctness, fallback behavior, and
a small set of common 2D operations.

| Area | Supported path | Current limits |
| ---- | -------------- | -------------- |
| Allocation | `cv::UMat` allocation through the Metal allocator | Apple Metal device required |
| Host transfer | `Mat -> UMat`, `UMat -> Mat`, map/unmap | Managed storage is not implemented |
| Device copy | `UMat -> UMat`, including ROI copies | Non-continuous copies may use staging |
| `copyTo` | Unmasked blit copy and masked 2D copy | Masked path supports current Metal kernel limits |
| `setTo` | Scalar fill with optional mask | 2D supported types/channels only |
| Arithmetic | `add`, `subtract`, `multiply` | 2D `CV_8U` and `CV_32F`, 1/3/4 channels |
| Bitwise | `bitwise_and`, `bitwise_or`, `bitwise_xor`, `bitwise_not` | No mask acceleration yet |
| Compare | `compare` | Matrix-vs-matrix cases only |
| Convert | `UMat::convertTo` | `CV_8U` and `CV_32F`, 1/3/4 channels |
| Threshold | Simple `threshold` types | No masks, OTSU, TRIANGLE, or dry-run acceleration |

## Testing

Focused Metal backend tests can be run with:

```text
opencv_test_core --gtest_filter=Core_Metal_UMat.*
opencv_test_imgproc --gtest_filter=Imgproc_Metal_Threshold.*
```

Regression testing should also cover a build where Metal is disabled and an
OpenCL-enabled build on systems with OpenCL runtime support.

## Sample

The `samples/tapi/umat_backend.cpp` sample demonstrates the intended user model:
write regular OpenCV code with `cv::UMat`, then let the configured UMat backend
choose Metal or OpenCL where available and fall back to CPU where needed.
