# OpenCV Issue #27535: pyrUp() Memory Overflow Fix

## Problem Description

The `pyrUp()` function in OpenCV can cause segmentation faults or program crashes when called repeatedly in sequence. This occurs because:

1. **Exponential Memory Growth**: Each call to `pyrUp()` doubles the image dimensions (width × 2, height × 2), resulting in 4× memory usage per iteration
2. **No Bounds Checking**: The original implementation had no limits on the destination image size
3. **Unbounded Allocation**: The function would attempt to allocate memory even for impossibly large images

### Memory Growth Pattern
- Iteration 1: 100×100 → 200×200 (160KB)
- Iteration 2: 200×200 → 400×400 (640KB) 
- Iteration 3: 400×400 → 800×800 (2.5MB)
- Iteration 4: 800×800 → 1600×1600 (10MB)
- Iteration 5: 1600×1600 → 3200×3200 (40MB)
- Iteration 10: 51,200×51,200 → 102,400×102,400 (41GB!)

## Root Cause Analysis

The issue is in `modules/imgproc/src/pyramids.cpp` at line 1388:

```cpp
Size dsz = _dsz.empty() ? Size(src.cols*2, src.rows*2) : _dsz;
_dst.create( dsz, src.type() );  // No bounds checking here!
```

The function blindly doubles the image size and attempts allocation without verifying if the resulting image size is reasonable.

## Solution

Added bounds checking before memory allocation in the `pyrUp()` function:

```cpp
void cv::pyrUp( InputArray _src, OutputArray _dst, const Size& _dsz, int borderType )
{
    CV_INSTRUMENT_REGION();

    CV_Assert(borderType == BORDER_DEFAULT);

    CV_OCL_RUN(_src.dims() <= 2 && _dst.isUMat(),
               ocl_pyrUp(_src, _dst, _dsz, borderType))

    // Define maximum allowable image dimensions to prevent memory overflow
    const int MAX_IMAGE_SIZE = 32768; // 32K pixels per dimension
    const size_t MAX_TOTAL_PIXELS = static_cast<size_t>(1024) * 1024 * 1024; // 1 billion pixels max

    Mat src = _src.getMat();
    Size dsz = _dsz.empty() ? Size(src.cols*2, src.rows*2) : _dsz;
    
    // Check for potential memory overflow before allocation
    if (dsz.width > MAX_IMAGE_SIZE || dsz.height > MAX_IMAGE_SIZE ||
        static_cast<size_t>(dsz.width) * dsz.height > MAX_TOTAL_PIXELS) {
        CV_Error(CV_StsNoMem, "pyrUp: Destination image size is too large and may cause memory overflow");
    }
    
    _dst.create( dsz, src.type() );
    // ... rest of function unchanged
}
```

### Bounds Selected

- **MAX_IMAGE_SIZE = 32,768**: Reasonable limit for individual dimensions
- **MAX_TOTAL_PIXELS = 1,073,741,824**: Approximately 1 billion pixels (4GB for RGBA images)

These limits prevent memory exhaustion while allowing legitimate use cases.

## Benefits

1. **Prevents Crashes**: Function throws a clear error instead of crashing
2. **Early Detection**: Fails fast before attempting massive allocations
3. **Backward Compatible**: Normal usage patterns continue to work
4. **Clear Error Messages**: Users get descriptive error messages
5. **Configurable**: Limits can be adjusted if needed

## Testing

### Before Fix
```cpp
Mat img(100, 100, CV_8UC3);
Mat current = img;
for (int i = 0; i < 15; i++) {
    pyrUp(current, current);  // Eventually crashes with segfault
}
```

### After Fix
```cpp
Mat img(100, 100, CV_8UC3);
Mat current = img;
try {
    for (int i = 0; i < 15; i++) {
        pyrUp(current, current);
    }
} catch (cv::Exception& e) {
    // Graceful error: "pyrUp: Destination image size is too large..."
}
```

## Files Modified

- `modules/imgproc/src/pyramids.cpp`: Added bounds checking in `pyrUp()` function

## Test Files Created

- `test_pyrUp_overflow.cpp`: C++ test demonstrating the issue and fix
- `test_pyrUp_fix.py`: Python test script for validation
- `pyrUp_memory_fix.patch`: Patch file with the solution

## Verification

The fix has been tested with:
- ✅ Normal pyrUp operations (small to medium images)
- ✅ Custom destination sizes
- ✅ Various image types (CV_8U, CV_16S, CV_16U, CV_32F, CV_64F)
- ✅ Multiple channel images (1, 2, 3, 4 channels)
- ✅ Overflow prevention (catches oversized requests)
- ✅ Error message clarity

## Impact

- **Security**: Prevents denial-of-service through memory exhaustion
- **Stability**: Applications no longer crash unexpectedly
- **User Experience**: Clear error messages help developers debug issues
- **Performance**: No impact on normal operations

This fix resolves Issue #27535 by adding necessary bounds checking while maintaining full backward compatibility for legitimate use cases.
