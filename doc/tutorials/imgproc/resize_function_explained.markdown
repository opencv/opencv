Understanding Image Resize in OpenCV {#tutorial_resize_function_explained}
====================================

@tableofcontents

@prev_tutorial{tutorial_distance_transform}
@next_tutorial{tutorial_warp_affine}

|    |    |
| -: | :- |
| Original author | OpenCV Documentation |
| Compatibility | OpenCV >= 3.0 |

## Goals

In this tutorial you will learn:
-   How OpenCV's `cv::resize()` function works internally
-   The different interpolation methods available
-   The algorithm architecture and optimization strategies
-   How to choose the right interpolation method for your use case

## Note: imresize() vs resize()

If you're coming from MATLAB, you may be looking for `imresize()`. In OpenCV, the equivalent function is called `cv::resize()` (C++) or `cv2.resize()` (Python). This tutorial explains how this function works.

## Function Signature

The `resize()` function is declared in `opencv2/imgproc.hpp`:

@code{.cpp}
void cv::resize(InputArray src, OutputArray dst,
                Size dsize, double fx = 0, double fy = 0,
                int interpolation = INTER_LINEAR);
@endcode

### Parameters

-   **src** – Input image
-   **dst** – Output image (will be created/resized automatically)
-   **dsize** – Output image size. If it equals zero, it is computed as:
    @f[
    \texttt{dsize = Size(round(fx*src.cols), round(fy*src.rows))}
    @f]
    Either `dsize` or both `fx` and `fy` must be non-zero.
-   **fx** – Scale factor along the horizontal axis. When it equals 0, it is computed as:
    @f[
    \texttt{fx = (double)dsize.width/src.cols}
    @f]
-   **fy** – Scale factor along the vertical axis. When it equals 0, it is computed as:
    @f[
    \texttt{fy = (double)dsize.height/src.rows}
    @f]
-   **interpolation** – Interpolation method (see the list below)

## Interpolation Methods

OpenCV supports several interpolation methods for resizing images. Each has different trade-offs between quality and speed:

### INTER_NEAREST
Nearest neighbor interpolation. This is the fastest method but produces the lowest quality results, especially when enlarging images. It simply picks the nearest pixel value without any blending.

**Best for:** Speed-critical applications, pixel art, images with discrete values/labels

### INTER_LINEAR (Default)
Bilinear interpolation. Computes the output pixel value as a weighted average of the 4 nearest input pixels. This is the default method and provides a good balance between speed and quality.

**Best for:** General-purpose resizing, real-time applications

### INTER_AREA
Resampling using pixel area relation. This method produces moiré-free results when downsampling. It averages all pixels that contribute to an output pixel. When upsampling, it behaves like `INTER_NEAREST`.

**Best for:** Shrinking images, downsampling

### INTER_CUBIC
Bicubic interpolation over 4×4 pixel neighborhood. Produces smoother results than bilinear interpolation but is slower. Uses cubic convolution.

**Best for:** High-quality upsampling, image enlargement

### INTER_LANCZOS4
Lanczos interpolation over 8×8 pixel neighborhood. Produces the highest quality results but is the slowest method. Uses a Lanczos filter with a=4.

**Best for:** Maximum quality upsampling, professional image processing

### INTER_LINEAR_EXACT
Bit-exact bilinear interpolation. Provides reproducible results across different platforms and OpenCV versions.

**Best for:** When reproducibility is required

### INTER_NEAREST_EXACT
Bit-exact nearest neighbor interpolation. Provides reproducible results across different platforms and OpenCV versions.

**Best for:** When reproducibility is required with nearest neighbor

## Algorithm Architecture

The `resize()` function uses an efficient **separable approach** to achieve high performance:

### 1. Separable Processing

Instead of computing each output pixel directly from the entire input image, resize operates in two passes:

1. **Horizontal resize**: Resizes each row independently to the target width
2. **Vertical resize**: Resizes each column of the intermediate result to the target height

This reduces complexity from O(M×N×m×n) to O(M×N×(m+n)) where:
- M×N is the output size
- m×n is the interpolation kernel size

### 2. Pre-computation of Coefficients

Before processing pixels, the algorithm pre-computes:
- **Offset tables**: Which input pixels contribute to each output pixel
- **Alpha coefficients**: Interpolation weights for horizontal resizing
- **Beta coefficients**: Interpolation weights for vertical resizing

This allows the main loop to focus on arithmetic operations without repeatedly calculating positions and weights.

### 3. Parallel Execution

The implementation uses `cv::parallel_for_` to process multiple rows simultaneously on multi-core CPUs, significantly improving performance.

### 4. SIMD Optimizations

Platform-specific implementations exist for:
- **AVX2** (Advanced Vector Extensions 2)
- **SSE4.1** (Streaming SIMD Extensions 4.1)
- **NEON** (ARM SIMD)
- **LASX** (LoongArch SIMD)
- **OpenCL** (GPU acceleration)

These vectorized implementations process multiple pixels simultaneously.

### 5. Fixed-Point Arithmetic

For integer image types, the algorithm uses fixed-point arithmetic instead of floating-point to maintain precision while improving performance. The scale factor `INTER_RESIZE_COEF_SCALE = 2048` is used to convert floating-point weights to integers.

## Implementation Details

### Key Components

The implementation in `modules/imgproc/src/resize.cpp` includes several specialized classes:

-   **resizeGeneric_Invoker**: Generic parallel loop body for resize operations
-   **resizeNNInvoker**: Optimized nearest neighbor interpolation
-   **ResizeArea_Invoker**: Area-based averaging with decimation tables
-   **HResize/VResize templates**: Generic horizontal and vertical resizers

### Fast Paths

Special optimizations exist for common cases:
-   **2×2 area resizing**: Ultra-fast integer downsampling by factor of 2
-   **Nearest neighbor with SIMD**: Vectorized nearest neighbor for integer scales
-   **Linear interpolation with SIMD**: Vectorized bilinear interpolation

## Example Usage

### Resize by Specifying Output Size

@code{.cpp}
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

int main() {
    cv::Mat src = cv::imread("input.jpg");
    cv::Mat dst;
    
    // Resize to 640x480
    cv::resize(src, dst, cv::Size(640, 480), 0, 0, cv::INTER_LINEAR);
    
    cv::imwrite("output.jpg", dst);
    return 0;
}
@endcode

### Resize by Specifying Scale Factors

@code{.cpp}
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

int main() {
    cv::Mat src = cv::imread("input.jpg");
    cv::Mat dst;
    
    // Scale to 2x the original size
    cv::resize(src, dst, cv::Size(), 2.0, 2.0, cv::INTER_CUBIC);
    
    cv::imwrite("output.jpg", dst);
    return 0;
}
@endcode

### Python Example

@code{.py}
import cv2 as cv

# Read image
img = cv.imread('input.jpg')

# Method 1: Specify exact dimensions
resized = cv.resize(img, (640, 480), interpolation=cv.INTER_LINEAR)

# Method 2: Specify scale factors
resized = cv.resize(img, None, fx=2, fy=2, interpolation=cv.INTER_CUBIC)

# Method 3: Shrink image (use INTER_AREA for best quality)
resized = cv.resize(img, None, fx=0.5, fy=0.5, interpolation=cv.INTER_AREA)

cv.imwrite('output.jpg', resized)
@endcode

## Choosing the Right Interpolation Method

| Use Case | Recommended Method | Reason |
|----------|-------------------|--------|
| Shrinking images | INTER_AREA | Produces moiré-free results |
| Enlarging images (quality) | INTER_CUBIC or INTER_LANCZOS4 | Smooth, high-quality results |
| Enlarging images (speed) | INTER_LINEAR | Good balance of quality and speed |
| Real-time processing | INTER_NEAREST or INTER_LINEAR | Fastest methods |
| Reproducibility required | INTER_LINEAR_EXACT | Platform-independent results |
| Pixel art / Label images | INTER_NEAREST | Preserves discrete values |

## Performance Considerations

1. **Separable processing**: O(M×N×(m+n)) complexity instead of O(M×N×m×n)
2. **Pre-computed coefficients**: Reduces runtime calculations
3. **Parallel execution**: Scales with CPU cores
4. **SIMD acceleration**: 4-8x speedup on supported platforms
5. **Fixed-point arithmetic**: Faster than floating-point for integers

## Mathematical Background

### Bilinear Interpolation (INTER_LINEAR)

For a point (x, y) in the output image, bilinear interpolation computes the value as:

@f[
f(x,y) = (1-\alpha)(1-\beta)I(x_0,y_0) + \alpha(1-\beta)I(x_1,y_0) + (1-\alpha)\beta I(x_0,y_1) + \alpha\beta I(x_1,y_1)
@f]

where:
- @f$(x_0, y_0)@f$ is the top-left pixel
- @f$\alpha = x - x_0@f$ is the horizontal fraction
- @f$\beta = y - y_0@f$ is the vertical fraction

### Bicubic Interpolation (INTER_CUBIC)

Uses a cubic kernel function to interpolate over a 4×4 neighborhood. The kernel is typically defined as:

@f[
W(x) = \begin{cases}
(a+2)|x|^3 - (a+3)|x|^2 + 1 & \text{for } |x| \leq 1 \\
a|x|^3 - 5a|x|^2 + 8a|x| - 4a & \text{for } 1 < |x| < 2 \\
0 & \text{otherwise}
\end{cases}
@f]

where a = -0.5 (common choice).

### Lanczos Interpolation (INTER_LANCZOS4)

Uses a windowed sinc function:

@f[
L(x) = \begin{cases}
\frac{\sin(\pi x) \sin(\pi x / a)}{\pi^2 x^2 / a} & \text{for } |x| < a \\
0 & \text{otherwise}
\end{cases}
@f]

where a = 4 for INTER_LANCZOS4.

## Source Code References

The implementation can be found in:
- **Declaration**: `modules/imgproc/include/opencv2/imgproc.hpp`
- **Implementation**: `modules/imgproc/src/resize.cpp`
- **SIMD optimizations**:
  - `modules/imgproc/src/resize.avx2.cpp` (AVX2)
  - `modules/imgproc/src/resize.sse4_1.cpp` (SSE4.1)
  - `modules/imgproc/src/resize.lasx.cpp` (LASX)
- **OpenCL**: `modules/imgproc/src/opencl/resize.cl`
- **Tests**: `modules/imgproc/test/test_resize_bitexact.cpp`

## Additional Resources

-   [OpenCV Image Processing Documentation](https://docs.opencv.org/master/d7/da8/tutorial_table_of_content_imgproc.html)
-   [Geometric Transformations Tutorial](@ref tutorial_py_geometric_transformations)
-   "Digital Image Processing" by Rafael C. Gonzalez and Richard E. Woods
-   "Computer Vision: Algorithms and Applications" by Richard Szeliski

## Summary

The `cv::resize()` function in OpenCV is a highly optimized implementation that:
- Uses separable filtering for efficiency
- Supports multiple interpolation methods
- Leverages parallel execution and SIMD instructions
- Pre-computes coefficients to minimize runtime calculations
- Provides both quality and performance options

Understanding these internals helps you choose the right parameters and understand the performance characteristics of your image processing pipeline.
