# Understanding the imresize() / resize() Function in OpenCV

## Quick Summary

**Important Note:** There is **no `imresize()` function** in OpenCV. 

- `imresize()` is a **MATLAB** function
- OpenCV's equivalent is **`cv::resize()`** (C++) or **`cv2.resize()`** (Python)

## What I've Created

This investigation has produced comprehensive documentation about how image resizing works in OpenCV:

### 1. Documentation (`doc/tutorials/imgproc/resize_function_explained.markdown`)

A complete tutorial explaining:
- Function signature and parameters
- All 7 interpolation methods (INTER_NEAREST, INTER_LINEAR, INTER_AREA, INTER_CUBIC, INTER_LANCZOS4, etc.)
- Internal algorithm architecture (separable processing, parallel execution, SIMD optimization)
- Mathematical background for each interpolation method
- Performance considerations
- How to choose the right interpolation method
- Source code references

### 2. C++ Demo Program (`samples/cpp/tutorial_code/ImgProc/resize_demo.cpp`)

Demonstrates:
- Resizing by specifying output size
- Resizing by scale factors
- Comparison of all interpolation methods for upscaling
- Comparison of interpolation methods for downscaling
- Non-uniform scaling (changing aspect ratio)

### 3. Python Demo Program (`samples/python/tutorial_code/imgProc/resize_demo.py`)

Python equivalent of the C++ demo, plus:
- Performance timing comparison of different interpolation methods
- Shows speed vs. quality trade-offs

## Key Findings About How resize() Works

### Algorithm Architecture

1. **Separable Processing**: Instead of computing each output pixel from the entire input image in one step, resize operates in two passes:
   - Horizontal resize: Resizes each row to target width
   - Vertical resize: Resizes columns of intermediate result to target height
   - This reduces complexity from O(M×N×m×n) to O(M×N×(m+n))

2. **Pre-computed Coefficients**: Before processing pixels:
   - Calculates which input pixels contribute to each output pixel
   - Pre-computes interpolation weights (alpha/beta coefficients)
   - Main loop focuses on arithmetic without repeated calculations

3. **Parallel Execution**: Uses `cv::parallel_for_` to process multiple rows simultaneously on multi-core CPUs

4. **SIMD Optimizations**: Platform-specific implementations for AVX2, SSE4.1, NEON, LASX, and OpenCL

5. **Fixed-Point Arithmetic**: For integer images, uses fixed-point math with scale factor 2048 for precision and performance

### Interpolation Methods Summary

| Method | Speed | Quality | Best For |
|--------|-------|---------|----------|
| INTER_NEAREST | Fastest | Lowest | Pixel art, labels, speed-critical |
| INTER_LINEAR | Fast | Good | General purpose (default) |
| INTER_AREA | Fast | Best for shrinking | Downsampling images |
| INTER_CUBIC | Moderate | Better | High-quality upsampling |
| INTER_LANCZOS4 | Slowest | Best | Maximum quality upsampling |
| INTER_LINEAR_EXACT | Fast | Good | Reproducible results |
| INTER_NEAREST_EXACT | Fastest | Lowest | Reproducible nearest neighbor |

### Performance Features

- **Separable filtering**: More efficient computation
- **Pre-computation**: Offset tables and coefficients calculated once
- **Parallelization**: Scales with CPU cores
- **SIMD acceleration**: 4-8x speedup on supported platforms
- **Special fast paths**: Optimized 2×2 area resizing, SIMD nearest neighbor

## Source Code Locations

- **Main implementation**: `modules/imgproc/src/resize.cpp`
- **Function declaration**: `modules/imgproc/include/opencv2/imgproc.hpp`
- **SIMD optimizations**:
  - `modules/imgproc/src/resize.avx2.cpp`
  - `modules/imgproc/src/resize.sse4_1.cpp`
  - `modules/imgproc/src/resize.lasx.cpp`
- **OpenCL**: `modules/imgproc/src/opencl/resize.cl`
- **Tests**: `modules/imgproc/test/test_resize_bitexact.cpp`

## How to Use

### C++
```cpp
#include <opencv2/imgproc.hpp>

cv::Mat src = cv::imread("input.jpg");
cv::Mat dst;

// Method 1: Specify exact size
cv::resize(src, dst, cv::Size(640, 480), 0, 0, cv::INTER_LINEAR);

// Method 2: Use scale factors
cv::resize(src, dst, cv::Size(), 2.0, 2.0, cv::INTER_CUBIC);
```

### Python
```python
import cv2 as cv

img = cv.imread('input.jpg')

# Method 1: Specify exact size
resized = cv.resize(img, (640, 480), interpolation=cv.INTER_LINEAR)

# Method 2: Use scale factors
resized = cv.resize(img, None, fx=2, fy=2, interpolation=cv.INTER_CUBIC)
```

## Running the Demo Programs

### C++
```bash
cd build
cmake ..
make
./bin/resize_demo path/to/image.jpg
```

### Python
```bash
python samples/python/tutorial_code/imgProc/resize_demo.py --input path/to/image.jpg
```

## Conclusion

The `cv::resize()` function is a highly optimized implementation that balances quality and performance through:
- Intelligent algorithm design (separable processing)
- Multi-core parallelization
- SIMD vectorization
- Multiple interpolation methods for different use cases
- Pre-computation of coefficients

Understanding these internals helps you choose the right parameters for your specific image processing needs.
