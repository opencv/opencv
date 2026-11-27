libjpeg-turbo SIMD Extensions
=============================

This directory contains the libjpeg-turbo SIMD extensions, hand-coded assembly
or compiler intrinsics modules that plug into the modular interfaces of the
libjpeg API library and use various SIMD instruction sets to accelerate most of
the 8-bit-per-sample lossy JPEG compression and decompression algorithms.

(Note that, since the TurboJPEG API library wraps the libjpeg API library, it
uses the libjpeg-turbo SIMD extensions implicitly.)

Algorithm Coverage
------------------

The following 8-bit-per-sample lossy JPEG compression algorithms are currently
implemented as SIMD modules:

- Color Conversion (see [jccolor.c](../src/jccolor.c))
  * RGB-to-YCbCr Color Conversion
  * RGB-to-Grayscale Color Conversion
- Downsampling (see [jcsample.c](../src/jcsample.c))
  * H2V1 (4:2:2) Downsampling
  * H2V2 (4:2:0) Downsampling
- Sample Conversion (see [jcdctmgr.c](../src/jcdctmgr.c))
  * Integer Sample Conversion
  * Floating Point Sample Conversion (legacy feature)
- Forward DCT (see [jcdctmgr.c](../src/jcdctmgr.c))
  * Accurate Integer Forward DCT
  * Fast Integer Forward DCT (legacy feature)
  * Floating Point Forward DCT (legacy feature)
- Quantization (see [jcdctmgr.c](../src/jcdctmgr.c))
  * Integer Quantization
  * Floating Point Quantization (legacy feature)
- Entropy Encoding
  * Huffman Encoding (see [jchuff.c](../src/jchuff.c))
  * Progressive Huffman Encoding (see [jcphuff.c](../src/jcphuff.c))

The following 8-bit-per-sample lossy JPEG decompression algorithms are
currently implemented as SIMD modules:

- Inverse DCT (see [jddctmgr.c](../src/jddctmgr.c))
  * Accurate Integer Inverse DCT
  * Fast Integer Inverse DCT (legacy feature)
  * Floating Point Inverse DCT (legacy feature)
  * 2x2 (1/4 Scaling) Integer Inverse DCT (infrequently used)
  * 4x4 (1/2 Scaling) Integer Inverse DCT (infrequently used)
- Upsampling (see [jdsample.c](../src/jdsample.c))
  * H2V1 (4:2:2) Fancy (Smooth) Upsampling
  * H2V2 (4:2:0) Fancy (Smooth) Upsampling
  * H1V2 (4:4:0) Fancy (Smooth) Upsampling
  * H2V1 (4:2:2) Plain Upsampling (infrequently used)
  * H2V2 (4:2:0) Plain Upsampling (infrequently used)
- Merged Upsampling/Color Conversion (see [jdmerge.c](../src/jdmerge.c))
  * H2V1 (4:2:2) Merged Upsampling/Color Conversion
  * H2V2 (4:2:0) Merged Upsampling/Color Conversion
- Color Deconversion (see [jdcolor.c](../src/jdcolor.c))
  * YCbCr-to-RGB Color Conversion
  * YCbCr-to-RGB565 Color Conversion (infrequently used)

Refer to <https://libjpeg-turbo.org/About/SIMDCoverage> for a list of SIMD
modules that are implemented for the algorithms above using specific SIMD
instruction sets.

Legacy features are features that were designed to work around hardware
performance limitations that no longer exist.  They generally have little or no
utility on modern hardware and are retained only for backward compatibility
with libjpeg.

Infrequently used features may be useful for specific applications but are not
the "common case" for JPEG compression and decompression.

SIMD Dispatcher Operation
-------------------------

When initializing a particular compression or decompression module (which
occurs during `jpeg_start_compress()`, `jpeg_start_decompress()`,
`tj3Compress*()`, or `tj3Decompress*()`), the libjpeg API library calls a SIMD
dispatcher function (`jsimd_set_*()`) for each algorithm listed above.  The
SIMD dispatcher functions are defined in [jsimd.h](jsimd.h),
[jsimddct.h](jsimddct.h), and [jsimd.c](jsimd.c).  Each function

- determines which SIMD instruction sets are available for the current
  architecture (if such has not already been determined for the current API
  instance),
- determines whether a compatible SIMD module exists for the specified
  algorithm, and
- (if so) plugs the best available SIMD module for that algorithm into the
  appropriate compression or decompression module, thus causing a SIMD
  implementation of the algorithm to be used instead of the scalar/C
  implementation.

You can use environment variables to override the dispatchers' choice of SIMD
modules:

- `JSIMD_FORCENONE=1` disables all SIMD modules.
- `JSIMD_NOHUFFENC=1` disables only the Huffman encoding SIMD modules.
- `JSIMD_FORCESSE2=1` (x86) enables only the SSE2 and SSE SIMD modules, even if
  the CPU supports newer instruction sets.
- `JSIMD_FORCESSE=1` (i386) enables only the SSE and MMX SIMD modules, even if
  the CPU supports newer instruction sets.
- `JSIMD_FORCEMMX=1` (i386) enables only the MMX SIMD modules, even if the CPU
  supports newer instruction sets.
- `JSIMD_FORCENEON=1` (AArch32) force-enables the Neon SIMD modules, bypassing
  **/proc/cpuinfo** feature detection (which may be unreliable in QEMU and
  other emulation/virtualization environments.)
- `JSIMD_FORCEMMI=1` (Loongson) force-enables the MMI SIMD modules, bypassing
  **/proc/cpuinfo** feature detection (which may be unreliable in QEMU and
  other emulation/virtualization environments.)

The **simdcoverage** program reports which SIMD modules will be used, taking
into account the current architecture, detected CPU features, and
aforementioned overrides.

SIMD Module Performance Profiling
---------------------------------

When built with the `WITH_PROFILE` CMake variable enabled, the libjpeg API
library measures the average throughput of each lossy JPEG algorithm as an
image is compressed or decompressed, and it prints the results to the command
line when `jpeg_destroy_compress()`, `jpeg_destroy_decompress()`, or
`tj3Destroy()` is called.  This allows developers to easily study the
performance of each SIMD module in isolation and compare it to the performance
of the corresponding scalar/C module (or a previous implementation of the same
SIMD module.)

The most effective way to use the profiling feature is with the TJBench application
and a suitably large image, such as one of the 8-bit RGB images from
[imagecompression.info](http://imagecompression.info/test_images).  Use one of
the following command lines to obtain the performance of specific algorithms.
(Adjust the warmup and benchmark times to suit your needs.)

- `tjbench {image}.ppm 95 -rgb -quiet -nowrite -benchtime 10 -warmup 10 -subsamp 422`
  * "Color conversion" reports the performance of the RGB-to-YCbCr Color
    Conversion algorithm.
  * "Downsampling" reports the performance of the H2V1 (4:2:2) Downsampling
    algorithm.
  * "Sample conversion" reports the performance of the Integer Sample
    Conversion algorithm.
  * "Forward DCT" reports the performance of the Accurate Integer Forward DCT
    algorithm.
  * "Quantization" reports the performance of the Integer Quantization
    algorithm.
  * "Entropy encoding" reports the performance of the Huffman Encoding
    algorithm.
  * "Inverse DCT" reports the performance of the Accurate Integer Inverse DCT
    algorithm.
  * "Upsampling" reports the performance of the H2V1 (4:2:2) Fancy (Smooth)
    Upsampling algorithm.
  * "Color deconversion" reports the performance of the YCbCr-to-RGB Color
    Conversion algorithm.

- `tjbench {image}.ppm 95 -rgb -quiet -nowrite -benchtime 10 -warmup 10 -subsamp gray`
  * "Color conversion" reports the performance of the RGB-to-Grayscale Color
    Conversion algorithm.

- `tjbench {image}.ppm 95 -rgb -quiet -nowrite -benchtime 10 -warmup 10 -subsamp 420`
  * "Downsampling" reports the performance of the H2V2 (4:2:0) Downsampling
    algorithm.
  * "Upsampling" reports the performance of the H2V2 (4:2:0) Fancy (Smooth)
    Upsampling algorithm.

- `tjbench {image}.ppm 95 -rgb -quiet -nowrite -benchtime 10 -warmup 10 -subsamp 440`
  * "Upsampling" reports the performance of the H1V2 (4:4:0) Fancy (Smooth)
    Upsampling algorithm.

- `tjbench {image}.ppm 95 -cmyk -quiet -nowrite -benchtime 10 -warmup 10 -subsamp 422 -nosmooth`
  * "Upsampling" reports the performance of the H2V1 (4:2:2) Plain Upsampling
    algorithm.

- `tjbench {image}.ppm 95 -cmyk -quiet -nowrite -benchtime 10 -warmup 10 -subsamp 420 -nosmooth`
  * "Upsampling" reports the performance of the H2V2 (4:2:0) Plain Upsampling
    algorithm.

- `tjbench {image}.ppm 95 -rgb -quiet -nowrite -benchtime 10 -warmup 10 -subsamp 422 -nosmooth`
  * "Merged upsampling" reports the performance of the H2V1 (4:2:2) Merged
    Upsampling/Color Conversion algorithm.

- `tjbench {image}.ppm 95 -rgb -quiet -nowrite -benchtime 10 -warmup 10 -subsamp 420 -nosmooth`
  * "Merged upsampling" reports the performance of the H2V2 (4:2:0) Merged
    Upsampling/Color Conversion algorithm.

- `tjbench {image}.ppm 95 -rgb -quiet -nowrite -benchtime 10 -warmup 10 -subsamp 422 -progressive`
  * "Entropy encoding" reports the performance of the Progressive Huffman
    Encoding algorithm.

- `tjbench {image}.ppm 95 -rgb -quiet -nowrite -benchtime 10 -warmup 10 -subsamp 422 -scale 1/4`
  * "Inverse DCT" reports the performance of the 2x2 (1/4 Scaling) Integer
    Inverse DCT algorithm.

- `tjbench {image}.ppm 95 -rgb -quiet -nowrite -benchtime 10 -warmup 10 -subsamp 422 -scale 1/2`
  * "Inverse DCT" reports the performance of the 4x4 (1/2 Scaling) Integer
    Inverse DCT algorithm.

- `tjbench {image}.ppm 95 -rgb -quiet -nowrite -benchtime 10 -warmup 10 -subsamp 422 -dct fast`
  * "Forward DCT" reports the performance of the Fast Integer Forward DCT
    algorithm.
  * "Inverse DCT" reports the performance of the Fast Integer Inverse DCT
    algorithm.
