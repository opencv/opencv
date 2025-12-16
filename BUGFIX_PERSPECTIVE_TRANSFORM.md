# OpenCV perspectiveTransform Bug Fix - Issue #26947

## Issue Summary
After upgrading from OpenCV 4.11.0 to 4.12.0, the `perspectiveTransform` function returns zeros for some inputs when using Android/Java bindings. The calculation fails for specific coordinate transformations.

## Root Cause
The bug is located in [modules/core/src/matmul.simd.hpp](modules/core/src/matmul.simd.hpp#L1885-L1910) in the generic fallback path of the `perspectiveTransform_` template function.

### The Problem
In perspective transformation, points are transformed using homogeneous coordinates:
```
(x', y') = (ax + by + c, dx + ey + f) / (gx + hy + i)
```

The code correctly computes the denominator `w = gx + hy + i`, but in the generic path (used when channels don't match optimized cases), it **fails to invert `w`** before multiplying.

### Affected Code Path
The bug occurs in the `else` branch starting at line 1885, which handles generic channel combinations that don't match the optimized paths (2D→2D, 3D→3D, 3D→2D).

**Before Fix (line 1892-1901):**
```cpp
if( fabs(w) > eps )
{
    _m = m;
    for( j = 0; j < dcn; j++, _m += scn + 1 )
    {
        double s = _m[scn];
        for( k = 0; k < scn; k++ )
            s += _m[k]*src[k];
        dst[j] = (T)(s*w);  // BUG: Should divide by w, not multiply
    }
}
```

The code multiplies `s * w` when it should divide `s / w`. The optimized paths correctly handle this by computing `w = 1./w` first (line 1839, 1854, 1871), but the generic path omits this crucial step.

## The Fix
Added the missing inversion step at line 1894:

```cpp
if( fabs(w) > eps )
{
    w = 1./w;  // FIX: Invert w so multiplication becomes division
    _m = m;
    for( j = 0; j < dcn; j++, _m += scn + 1 )
    {
        double s = _m[scn];
        for( k = 0; k < scn; k++ )
            s += _m[k]*src[k];
        dst[j] = (T)(s*w);  // Now correctly divides
    }
}
```

## Why This Wasn't Caught Earlier
- The optimized code paths (2D→2D, 3D→3D, 3D→2D) work correctly because they include `w = 1./w`
- The bug only manifests in the generic fallback path used for other channel combinations
- The user's Android code likely triggers this generic path based on how Mat objects are constructed in Java bindings
- Existing tests may primarily exercise the optimized paths

## Impact
This bug causes incorrect perspective transformations for:
- Any channel combinations not covered by optimized paths
- Potentially affects coordinate mapping applications, image warping, and geometric transformations
- Results in coordinates becoming zeros or incorrect values

## Testing
Created test case: [test_perspective_fix.cpp](test_perspective_fix.cpp)

To test:
```bash
# Build OpenCV with the fix
cmake -DBUILD_TESTS=ON ..
make

# Run the test
./test_perspective_fix

# Run official tests
make test
# or
ctest -R Core_PerspectiveTransform
```

## Files Modified
- [modules/core/src/matmul.simd.hpp](modules/core/src/matmul.simd.hpp#L1894) - Added `w = 1./w;`

## References
- Original issue: User report for Android app using OpenCV 4.12.0
- Related code: 
  - Optimized 2D path: line 1839
  - Optimized 3D→3D path: line 1854  
  - Optimized 3D→2D path: line 1871
  - Generic path (fixed): line 1894
