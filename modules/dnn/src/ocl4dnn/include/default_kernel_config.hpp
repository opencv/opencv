#ifndef _OPENCV_OCL4DNN_DEFAULT_KERNEL_CONFIG_HPP_
#define _OPENCV_OCL4DNN_DEFAULT_KERNEL_CONFIG_HPP_
const char *default_kernel_config_intel[] = {
// Below is the information for OpenCL based on which these configurations tuned
/*******************************************************************************
Number of platforms                               1
  Platform Name                                   Intel(R) OpenCL
  Platform Vendor                                 Intel(R) Corporation
  Platform Version                                OpenCL 2.0
  Platform Profile                                FULL_PROFILE
  Platform Extensions                             cl_intel_accelerator cl_intel_advanced_motion_estimation cl_intel_device_side_avc_motion_estimation cl_intel_driver_diagnostics cl_intel_media_block_io cl_intel_motion_estimation cl_intel_planar_yuv cl_intel_packed_yuv cl_intel_required_subgroup_size cl_intel_subgroups cl_intel_subgroups_short cl_intel_va_api_media_sharing cl_khr_3d_image_writes cl_khr_byte_addressable_store cl_khr_depth_images cl_khr_fp16 cl_khr_fp64 cl_khr_global_int32_base_atomics cl_khr_global_int32_extended_atomics cl_khr_icd cl_khr_image2d_from_buffer cl_khr_local_int32_base_atomics cl_khr_local_int32_extended_atomics cl_khr_mipmap_image cl_khr_mipmap_image_writes cl_khr_spir cl_khr_subgroups
  Platform Extensions function suffix             INTEL

  Platform Name                                   Intel(R) OpenCL
Number of devices                                 1
  Device Name                                     Intel(R) HD Graphics
  Device Vendor                                   Intel(R) Corporation
  Device Vendor ID                                0x8086
  Device Version                                  OpenCL 2.0
  Driver Version                                  r4.1.61547
  Device OpenCL C Version                         OpenCL C 2.0
  Device Type                                     GPU
  Device Profile                                  FULL_PROFILE
  Max compute units                               72
  Max clock frequency                             950MHz
  Device Partition                                (core)
    Max number of sub-devices                     0
    Supported partition types                     by <unknown> (0x7FE000000000)
  Max work item dimensions                        3
  Max work item sizes                             256x256x256
  Max work group size                             256
  Preferred work group size multiple              32
  Preferred / native vector sizes
    char                                                16 / 16
    short                                                8 / 8
    int                                                  4 / 4
    long                                                 1 / 1
    half                                                 8 / 8        (cl_khr_fp16)
    float                                                1 / 1
    double                                               1 / 1        (cl_khr_fp64)
  Half-precision Floating-point support           (cl_khr_fp16)
    Denormals                                     Yes
    Infinity and NANs                             Yes
    Round to nearest                              Yes
    Round to zero                                 Yes
    Round to infinity                             Yes
    IEEE754-2008 fused multiply-add               Yes
    Support is emulated in software               No
    Correctly-rounded divide and sqrt operations  No
  Single-precision Floating-point support         (core)
    Denormals                                     Yes
    Infinity and NANs                             Yes
    Round to nearest                              Yes
    Round to zero                                 Yes
    Round to infinity                             Yes
    IEEE754-2008 fused multiply-add               Yes
    Support is emulated in software               No
    Correctly-rounded divide and sqrt operations  Yes
  Double-precision Floating-point support         (cl_khr_fp64)
    Denormals                                     Yes
    Infinity and NANs                             Yes
    Round to nearest                              Yes
    Round to zero                                 Yes
    Round to infinity                             Yes
    IEEE754-2008 fused multiply-add               Yes
    Support is emulated in software               No
    Correctly-rounded divide and sqrt operations  No
  Address bits                                    64, Little-Endian
  Global memory size                              26887677543 (25.04GiB)
  Error Correction support                        No
  Max memory allocation                           4294959103 (4GiB)
  Unified memory for Host and Device              Yes
  Shared Virtual Memory (SVM) capabilities        (core)
    Coarse-grained buffer sharing                 Yes
    Fine-grained buffer sharing                   No
    Fine-grained system sharing                   No
    Atomics                                       No
  Minimum alignment for any data type             128 bytes
  Alignment of base address                       1024 bits (128 bytes)
  Preferred alignment for atomics
    SVM                                           64 bytes
    Global                                        64 bytes
    Local                                         64 bytes
  Max size for global variable                    65536 (64KiB)
  Preferred total size of global vars             4294959103 (4GiB)
  Global Memory cache type                        Read/Write
  Global Memory cache size                        1572864
  Global Memory cache line                        64 bytes
  Image support                                   Yes
    Max number of samplers per kernel             16
    Max size for 1D images from buffer            268434943 pixels
    Max 1D or 2D image array size                 2048 images
    Base address alignment for 2D image buffers   4 bytes
    Pitch alignment for 2D image buffers          4 bytes
    Max 2D image size                             16384x16384 pixels
    Max 3D image size                             16384x16384x2048 pixels
    Max number of read image args                 128
    Max number of write image args                128
    Max number of read/write image args           128
  Max number of pipe args                         16
  Max active pipe reservations                    1
  Max pipe packet size                            1024
  Local memory type                               Local
  Local memory size                               65536 (64KiB)
  Max constant buffer size                        4294959103 (4GiB)
  Max number of constant args                     8
  Max size of kernel argument                     1024
  Queue properties (on host)
    Out-of-order execution                        Yes
    Profiling                                     Yes
  Queue properties (on device)
    Out-of-order execution                        Yes
    Profiling                                     Yes
    Preferred size                                131072 (128KiB)
    Max size                                      67108864 (64MiB)
  Max queues on device                            1
  Max events on device                            1024
  Prefer user sync for interop                    Yes
  Profiling timer resolution                      83ns
  Execution capabilities
    Run OpenCL kernels                            Yes
    Run native kernels                            No
    SPIR versions                                 1.2
  printf() buffer size                            4194304 (4MiB)
  Built-in kernels                                block_motion_estimate_intel;block_advanced_motion_estimate_check_intel;block_advanced_motion_estimate_bidirectional_check_intel
  Motion Estimation accelerator version	(Intel)   2
  Device Available                                Yes
  Compiler Available                              Yes
  Linker Available                                Yes
  Device Extensions                               cl_intel_accelerator cl_intel_advanced_motion_estimation cl_intel_device_side_avc_motion_estimation cl_intel_driver_diagnostics cl_intel_media_block_io cl_intel_motion_estimation cl_intel_planar_yuv cl_intel_packed_yuv cl_intel_required_subgroup_size cl_intel_subgroups cl_intel_subgroups_short cl_intel_va_api_media_sharing cl_khr_3d_image_writes cl_khr_byte_addressable_store cl_khr_depth_images cl_khr_fp16 cl_khr_fp64 cl_khr_global_int32_base_atomics cl_khr_global_int32_extended_atomics cl_khr_icd cl_khr_image2d_from_buffer cl_khr_local_int32_base_atomics cl_khr_local_int32_extended_atomics cl_khr_mipmap_image cl_khr_mipmap_image_writes cl_khr_spir cl_khr_subgroups

NULL platform behavior
  clGetPlatformInfo(NULL, CL_PLATFORM_NAME, ...)  No platform
  clGetDeviceIDs(NULL, CL_DEVICE_TYPE_ALL, ...)   No platform
  clCreateContext(NULL, ...) [default]            No platform
  clCreateContext(NULL, ...) [other]              Success [INTEL]
  clCreateContextFromType(NULL, CL_DEVICE_TYPE_CPU)  No platform
  clCreateContextFromType(NULL, CL_DEVICE_TYPE_GPU)  No platform
  clCreateContextFromType(NULL, CL_DEVICE_TYPE_ACCELERATOR)  No platform
  clCreateContextFromType(NULL, CL_DEVICE_TYPE_CUSTOM)  No platform
  clCreateContextFromType(NULL, CL_DEVICE_TYPE_ALL)  No platform
********************************************************************************/
"EU72_k1x1_cn832_g1_s1x1_d1x1_b1_in16x16_p0x0_num2_M128","4 6 8 2 1 1 8 1 0 ",
"EU72_k1x1_cn256_g1_s1x1_d1x1_b1_in32x32_p0x0_num1_M32","12 2 8 2 1 1 8 1 0 ",
"EU72_k7x7_cn3_g1_s2x2_d1x1_b1_in224x224_p3x3_num1_M64","1 8 32 5 1 8 1 1 0 ",
"EU72_k5x5_cn48_g1_s1x1_d1x1_b1_in16x16_p2x2_num1_M128","4 2 16 2 1 1 16 1 0 ",
"EU72_k1x1_cn128_g1_s1x1_d1x1_b0_in64x64_p0x0_num1_M64","1 8 32 5 1 8 1 1 0 ",
"EU72_k3x3_cn64_g1_s1x1_d1x1_b1_in64x64_p1x1_num2_M192","2 7 16 2 1 1 16 1 0 ",
"EU72_k5x5_cn16_g1_s1x1_d1x1_b1_in16x16_p2x2_num1_M48","4 3 16 2 1 1 16 1 0 ",
"EU72_k1x1_cn832_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M32","4 6 8 2 1 1 8 1 0 ",
"EU72_k1x1_cn192_g1_s1x1_d1x1_b1_in32x32_p0x0_num1_M96","1 8 32 5 1 8 1 1 0 ",
"EU72_k11x7_cn3_g1_s3x4_d1x1_b1_in64x64_p3x2_num1_M64","4 1 16 2 1 1 16 1 0 ",
"EU72_k1x1_cn512_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M64","8 3 8 2 1 1 8 1 0 ",
"EU72_k1x1_cn832_g1_s1x1_d1x1_b1_in16x16_p0x0_num2_M32","4 6 8 2 1 1 8 1 0 ",
"EU72_k3x3_cn4_g1_s1x1_d1x1_b1_in256x256_p1x1_num1_M4","14 1 16 2 1 1 16 1 0 ",
"EU72_k1x1_cn3_g1_s1x1_d1x1_b1_in16x16_p0x0_num2_M4","4 4 8 2 1 1 8 1 0 ",
"EU72_k5x5_cn32_g1_s1x1_d1x1_b1_in16x16_p2x2_num2_M128","4 2 16 2 1 1 16 1 0 ",
"EU72_k1x1_cn480_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M192","1 8 32 5 1 8 1 1 0 ",
"EU72_k1x1_cn480_g1_s1x1_d1x1_b1_in16x16_p0x0_num2_M192","1 8 32 5 1 8 1 1 0 ",
"EU72_k3x3_cn96_g1_s1x1_d1x1_b1_in16x16_p1x1_num1_M208","2 6 16 2 1 1 16 1 0 ",
"EU72_k1x1_cn512_g1_s1x1_d1x1_b1_in16x16_p0x0_num2_M32","8 3 8 2 1 1 8 1 0 ",
"EU72_k1x1_cn832_g1_s1x1_d1x1_b1_in16x16_p0x0_num2_M384","1 8 32 5 1 8 1 1 0 ",
"EU72_k3x3_cn160_g1_s1x1_d1x1_b1_in16x16_p1x1_num1_M320","2 5 16 2 1 1 16 1 0 ",
"EU72_k1x1_cn528_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M160","8 3 8 2 1 1 8 1 0 ",
"EU72_k1x1_cn528_g1_s1x1_d1x1_b1_in16x16_p0x0_num2_M256","1 8 32 5 1 8 1 1 0 ",
"EU72_k5x1_cn32_g1_s1x1_d1x1_b0_in64x64_p2x0_num1_M32","4 6 16 2 1 1 16 1 0 ",
"EU72_k1x1_cn16_g1_s1x1_d1x1_b0_in256x256_p0x0_num1_M4","12 2 16 2 1 1 16 1 0 ",
"EU72_k1x1_cn64_g1_s1x1_d1x1_b1_in64x64_p0x0_num1_M64","2 8 32 5 1 8 1 1 0 ",
"EU72_k1x1_cn192_g1_s1x1_d1x1_b1_in32x32_p0x0_num2_M16","8 3 8 2 1 1 8 1 0 ",
"EU72_k1x1_cn32_g1_s1x1_d1x1_b0_in64x64_p0x0_num1_M128","1 16 32 5 1 16 1 1 0 ",
"EU72_k3x3_cn32_g1_s1x1_d2x2_b1_in64x64_p2x2_num1_M32","3 6 16 2 1 1 16 1 0 ",
"EU72_k3x3_cn32_g1_s1x1_d16x16_b1_in64x64_p16x16_num1_M32","1 16 32 5 1 16 1 1 0 ",
"EU72_k1x1_cn128_g1_s1x1_d1x1_b0_in32x32_p0x0_num1_M512","2 8 32 5 1 8 1 1 0 ",
"EU72_k3x3_cn192_g1_s1x1_d1x1_b1_in16x16_p1x1_num2_M384","2 7 16 2 1 1 16 1 0 ",
"EU72_k5x4_cn6_g3_s3x2_d1x1_b1_in128x80_p1x0_num2_M4","1 1 1 4 1 1 1 0 1 ",
"EU72_k5x5_cn32_g1_s1x1_d1x1_b1_in32x32_p2x2_num2_M96","4 5 16 2 1 1 16 1 0 ",
"EU72_k3x3_cn64_g1_s1x1_d1x1_b1_in64x64_p1x1_num1_M192","10 2 16 2 1 1 16 1 0 ",
"EU72_k3x3_cn128_g1_s1x1_d1x1_b1_in32x32_p1x1_num1_M192","6 4 16 2 1 1 16 1 0 ",
"EU72_k1x1_cn4_g1_s1x1_d1x1_b0_in256x256_p0x0_num1_M16","2 8 32 5 1 8 1 1 0 ",
"EU72_k1x1_cn480_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M96","8 3 8 2 1 1 8 1 0 ",
"EU72_k5x5_cn16_g1_s1x1_d1x1_b1_in32x32_p2x2_num1_M32","8 1 16 2 1 1 16 1 0 ",
"EU72_k1x1_cn512_g1_s1x1_d1x1_b1_in16x16_p0x0_num2_M64","1 8 32 5 1 8 1 1 0 ",
"EU72_k1x1_cn832_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M384","4 7 8 2 1 1 8 1 0 ",
"EU72_k3x3_cn128_g1_s1x1_d1x1_b1_in16x16_p1x1_num1_M256","2 6 16 2 1 1 16 1 0 ",
"EU72_k3x3_cn96_g1_s1x1_d1x1_b1_in32x32_p1x1_num1_M128","6 4 16 2 1 1 16 1 0 ",
"EU72_k5x5_cn24_g1_s1x1_d1x1_b1_in16x16_p2x2_num2_M64","4 4 16 2 1 1 16 1 0 ",
"EU72_k5x5_cn16_g1_s1x1_d1x1_b1_in16x16_p2x2_num2_M48","4 3 16 2 1 1 16 1 0 ",
"EU72_k3x3_cn3_g1_s1x1_d1x1_b1_in16x16_p0x0_num2_M5","2 3 16 2 1 1 16 1 0 ",
"EU72_k1x1_cn512_g1_s1x1_d1x1_b1_in16x16_p0x0_num2_M24","8 2 8 2 1 1 8 1 0 ",
"EU72_k3x3_cn128_g1_s1x1_d1x1_b0_in32x32_p1x1_num1_M128","1 8 32 5 1 8 1 1 0 ",
"EU72_k3x3_cn96_g1_s1x1_d1x1_b1_in32x32_p1x1_num2_M128","2 7 16 2 1 1 16 1 0 ",
"EU72_k1x1_cn128_g1_s1x1_d1x1_b0_in64x64_p0x0_num1_M32","1 16 32 5 1 16 1 1 0 ",
"EU72_k1x1_cn512_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M112","8 2 8 2 1 1 8 1 0 ",
"EU72_k1x1_cn192_g1_s1x1_d1x1_b1_in32x32_p0x0_num2_M64","1 8 32 5 1 8 1 1 0 ",
"EU72_k1x1_cn528_g1_s1x1_d1x1_b1_in16x16_p0x0_num2_M160","1 8 32 5 1 8 1 1 0 ",
"EU72_k5x5_cn32_g1_s1x1_d1x1_b1_in16x16_p2x2_num1_M128","4 3 16 2 1 1 16 1 0 ",
"EU72_k1x1_cn64_g1_s1x1_d1x1_b1_in64x64_p0x0_num2_M64","1 16 32 5 1 16 1 1 0 ",
"EU72_k1x1_cn64_g1_s1x1_d1x1_b0_in128x128_p0x0_num1_M16","2 8 32 5 1 8 1 1 0 ",
"EU72_k1x1_cn512_g1_s1x1_d1x1_b1_in16x16_p0x0_num2_M144","1 8 32 5 1 8 1 1 0 ",
"EU72_k1x1_cn512_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M128","8 2 8 2 1 1 8 1 0 ",
"EU72_k1x1_cn16_g1_s1x1_d1x1_b0_in128x128_p0x0_num1_M64","1 16 32 5 1 16 1 1 0 ",
"EU72_k3x3_cn112_g1_s1x1_d1x1_b1_in16x16_p1x1_num2_M224","2 7 16 2 1 1 16 1 0 ",
"EU72_k1x1_cn832_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M256","4 6 8 2 1 1 8 1 0 ",
"EU72_k5x5_cn32_g1_s1x1_d1x1_b1_in32x32_p2x2_num1_M96","4 3 16 2 1 1 16 1 0 ",
"EU72_k1x1_cn512_g1_s2x2_d1x1_b0_in32x32_p0x0_num1_M256","1 8 32 5 1 8 1 1 0 ",
"EU72_k3x3_cn128_g1_s1x1_d1x1_b1_in32x32_p1x1_num2_M192","10 2 16 2 1 1 16 1 0 ",
"EU72_k1x1_cn480_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M64","12 2 8 2 1 1 8 1 0 ",
"EU72_k3x3_cn384_g2_s1x1_d1x1_b1_in16x16_p1x1_num1_M128","2 5 16 2 1 1 16 1 0 ",
"EU72_k1x1_cn832_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M48","4 6 8 2 1 1 8 1 0 ",
"EU72_k1x1_cn832_g1_s1x1_d1x1_b1_in16x16_p0x0_num2_M48","1 8 32 5 1 8 1 1 0 ",
"EU72_k1x1_cn528_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M256","8 3 8 2 1 1 8 1 0 ",
"EU72_k1x1_cn256_g1_s1x1_d1x1_b0_in64x64_p0x0_num1_M64","2 8 32 5 1 8 1 1 0 ",
"EU72_k3x3_cn144_g1_s1x1_d1x1_b1_in16x16_p1x1_num1_M288","2 5 16 2 1 1 16 1 0 ",
"EU72_k1x1_cn528_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M128","1 8 32 5 1 8 1 1 0 ",
"EU72_k1x1_cn512_g1_s1x1_d1x1_b1_in16x16_p0x0_num2_M128","1 8 32 5 1 8 1 1 0 ",
"EU72_k1x1_cn1024_g1_s1x1_d1x1_b0_in16x16_p0x0_num1_M256","1 8 32 5 1 8 1 1 0 ",
"EU72_k1x1_cn480_g1_s1x1_d1x1_b1_in16x16_p0x0_num2_M96","1 8 32 5 1 8 1 1 0 ",
"EU72_k1x1_cn512_g1_s2x2_d1x1_b0_in32x32_p0x0_num1_M1024","1 16 32 5 1 16 1 1 0 ",
"EU72_k1x1_cn2048_g1_s1x1_d1x1_b0_in16x16_p0x0_num1_M512","4 6 8 2 1 1 8 1 0 ",
"EU72_k3x3_cn512_g1_s1x1_d1x1_b0_in16x16_p1x1_num1_M512","2 5 16 2 1 1 16 1 0 ",
"EU72_k1x1_cn256_g1_s1x1_d1x1_b1_in32x32_p0x0_num1_M128","1 8 32 5 1 8 1 1 0 ",
"EU72_k1x1_cn192_g1_s1x1_d1x1_b1_in32x32_p0x0_num1_M16","8 2 8 2 1 1 8 1 0 ",
"EU72_k5x5_cn32_g1_s1x1_d1x1_b1_in16x16_p2x2_num1_M64","4 2 16 2 1 1 16 1 0 ",
"EU72_k1x1_cn528_g1_s1x1_d1x1_b1_in16x16_p0x0_num2_M128","8 3 8 2 1 1 8 1 0 ",
"EU72_k3x3_cn144_g1_s1x1_d1x1_b1_in16x16_p1x1_num2_M288","2 7 16 2 1 1 16 1 0 ",
"EU72_k3x3_cn16_g1_s1x1_d1x1_b1_in128x128_p1x1_num1_M16","2 5 16 2 1 1 16 1 0 ",
"EU72_k3x3_cn32_g1_s1x1_d8x8_b1_in64x64_p8x8_num1_M32","1 8 32 5 1 8 1 1 0 ",
"EU72_k1x1_cn64_g1_s1x1_d1x1_b0_in128x128_p0x0_num1_M4","8 3 8 2 1 1 8 1 0 ",
"EU72_k3x3_cn128_g1_s1x1_d1x1_b1_in16x16_p1x1_num2_M256","2 7 16 2 1 1 16 1 0 ",
"EU72_k3x3_cn256_g1_s1x1_d1x1_b0_in16x16_p1x1_num1_M256","2 5 16 2 1 1 16 1 0 ",
"EU72_k3x3_cn112_g1_s1x1_d1x1_b1_in16x16_p1x1_num1_M224","2 5 16 2 1 1 16 1 0 ",
"EU72_k1x1_cn832_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M128","1 8 32 5 1 8 1 1 0 ",
"EU72_k2x2_cn16_g1_s2x2_d1x1_b0_in256x256_p0x0_num1_M16","6 4 16 2 1 1 16 1 0 ",
"EU72_k1x1_cn832_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M192","4 6 8 2 1 1 8 1 0 ",
"EU72_k1x1_cn1024_g1_s2x2_d1x1_b0_in16x16_p0x0_num1_M512","1 8 32 5 1 8 1 1 0 ",
"EU72_k1x1_cn512_g1_s1x1_d1x1_b1_in16x16_p0x0_num2_M160","1 8 32 5 1 8 1 1 0 ",
"EU72_k3x3_cn384_g2_s1x1_d1x1_b1_in16x16_p1x1_num1_M192","2 5 16 2 1 1 16 1 0 ",
"EU72_k5x5_cn96_g2_s1x1_d1x1_b1_in32x32_p2x2_num1_M128","4 3 16 2 1 1 16 1 0 ",
"EU72_k1x1_cn512_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M32","8 2 8 2 1 1 8 1 0 ",
"EU72_k2x2_cn64_g1_s2x2_d1x1_b0_in128x128_p0x0_num1_M32","8 3 16 2 1 1 16 1 0 ",
"EU72_k1x1_cn64_g1_s1x1_d1x1_b0_in64x64_p0x0_num1_M256","1 16 32 5 1 16 1 1 0 ",
"EU72_k1x1_cn192_g1_s1x1_d1x1_b1_in32x32_p0x0_num1_M32","12 2 8 2 1 1 8 1 0 ",
"EU72_k5x5_cn16_g1_s1x1_d1x1_b1_in32x32_p2x2_num2_M32","4 2 16 2 1 1 16 1 0 ",
"EU72_k1x1_cn480_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M16","12 1 8 2 1 1 8 1 0 ",
"EU72_k11x11_cn3_g1_s4x4_d1x1_b1_in224x224_p0x0_num1_M96","1 8 32 5 1 8 1 1 0 ",
"EU72_k1x1_cn256_g1_s1x1_d1x1_b1_in32x32_p0x0_num2_M64","1 8 32 5 1 8 1 1 0 ",
"EU72_k1x1_cn832_g1_s1x1_d1x1_b1_in16x16_p0x0_num2_M256","4 7 8 2 1 1 8 1 0 ",
"EU72_k3x3_cn192_g1_s1x1_d1x1_b1_in16x16_p1x1_num1_M384","2 5 16 2 1 1 16 1 0 ",
"EU72_k1x1_cn480_g1_s1x1_d1x1_b1_in16x16_p0x0_num2_M16","12 1 8 2 1 1 8 1 0 ",
"EU72_k1x1_cn512_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M160","8 3 8 2 1 1 8 1 0 ",
"EU72_k1x1_cn256_g1_s2x2_d1x1_b0_in64x64_p0x0_num1_M512","1 16 32 5 1 16 1 1 0 ",
"EU72_k1x1_cn128_g1_s1x1_d1x1_b0_in64x64_p0x0_num1_M16","2 8 32 5 1 8 1 1 0 ",
"EU72_k1x1_cn832_g1_s1x1_d1x1_b1_in16x16_p0x0_num2_M192","4 6 8 2 1 1 8 1 0 ",
"EU72_k1x1_cn192_g1_s1x1_d1x1_b1_in32x32_p0x0_num2_M96","1 16 32 5 1 16 1 1 0 ",
"EU72_k1x1_cn528_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M32","12 1 8 2 1 1 8 1 0 ",
"EU72_k1x1_cn480_g1_s1x1_d1x1_b1_in16x16_p0x0_num2_M64","12 2 8 2 1 1 8 1 0 ",
"EU72_k3x3_cn256_g1_s1x1_d1x1_b1_in16x16_p1x1_num1_M384","2 7 16 2 1 1 16 1 0 ",
"EU72_k5x5_cn24_g1_s1x1_d1x1_b1_in16x16_p2x2_num1_M64","4 2 16 2 1 1 16 1 0 ",
"EU72_k1x1_cn512_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M24","12 1 8 2 1 1 8 1 0 ",
"EU72_k5x5_cn48_g1_s1x1_d1x1_b1_in16x16_p2x2_num2_M128","4 2 16 2 1 1 16 1 0 ",
"EU72_k1x1_cn832_g1_s1x1_d1x1_b1_in16x16_p0x0_num2_M160","4 6 8 2 1 1 8 1 0 ",
"EU72_k1x1_cn512_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M144","1 8 32 5 1 8 1 1 0 ",
"EU72_k3x3_cn32_g1_s1x1_d4x4_b1_in64x64_p4x4_num1_M32","1 8 32 5 1 8 1 1 0 ",
"EU72_k3x3_cn3_g1_s2x2_d1x1_b1_in256x256_p1x1_num1_M13","1 1 1 4 1 1 1 0 1 ",
"EU72_k3x3_cn32_g1_s1x1_d1x1_b1_in64x64_p1x1_num1_M32","6 4 16 2 1 1 16 1 0 ",
"EU72_k1x1_cn256_g1_s1x1_d1x1_b1_in32x32_p0x0_num2_M32","1 8 32 5 1 8 1 1 0 ",
"EU72_k3x3_cn64_g1_s1x1_d1x1_b0_in64x64_p1x1_num1_M64","2 7 16 2 1 1 16 1 0 ",
"EU72_k1x1_cn256_g1_s1x1_d1x1_b0_in16x16_p0x0_num1_M1024","2 8 32 5 1 8 1 1 0 ",
"EU72_k3x3_cn160_g1_s1x1_d1x1_b1_in16x16_p1x1_num2_M320","1 8 32 5 1 8 1 1 0 ",
"EU72_k1x5_cn32_g1_s1x1_d1x1_b1_in64x64_p0x2_num1_M32","4 6 16 2 1 1 16 1 0 ",
"EU72_k1x1_cn64_g1_s1x1_d1x1_b0_in64x64_p0x0_num1_M64","1 16 32 5 1 16 1 1 0 ",
"EU72_k1x1_cn832_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M160","4 6 8 2 1 1 8 1 0 ",
"EU72_k1x1_cn512_g1_s1x1_d1x1_b0_in32x32_p0x0_num1_M128","1 8 32 5 1 8 1 1 0 ",
"EU72_k1x1_cn192_g1_s1x1_d1x1_b1_in32x32_p0x0_num1_M64","8 3 8 2 1 1 8 1 0 ",
"EU72_k1x1_cn256_g1_s1x1_d1x1_b1_in32x32_p0x0_num1_M64","12 2 8 2 1 1 8 1 0 ",
"EU72_k1x1_cn256_g1_s1x1_d1x1_b1_in32x32_p0x0_num2_M128","2 8 32 5 1 8 1 1 0 ",
"EU72_k1x1_cn528_g1_s1x1_d1x1_b1_in16x16_p0x0_num2_M32","8 3 8 2 1 1 8 1 0 ",
"EU72_k1x1_cn192_g1_s1x1_d1x1_b1_in32x32_p0x0_num2_M32","1 8 32 5 1 8 1 1 0 ",
"EU72_k1x1_cn512_g1_s1x1_d1x1_b1_in16x16_p0x0_num2_M112","1 8 32 5 1 8 1 1 0 ",
"EU72_k4x4_cn3_g1_s1x1_d1x1_b1_in16x16_p0x0_num2_M2","1 3 16 2 1 1 16 1 0 ",
"EU72_k1x1_cn1024_g1_s2x2_d1x1_b0_in16x16_p0x0_num1_M2048","1 8 32 5 1 8 1 1 0 ",
"EU72_k1x1_cn256_g1_s2x2_d1x1_b0_in64x64_p0x0_num1_M128","1 8 32 5 1 8 1 1 0 ",
"EU72_k7x7_cn3_g1_s2x2_d1x1_b1_in224x224_p3x3_num2_M64","1 8 32 5 1 8 1 1 0 ",
"EU72_k1x1_cn512_g1_s1x1_d1x1_b0_in16x16_p0x0_num1_M2048","1 8 32 5 1 8 1 1 0 ",
"EU72_k5x5_cn32_g1_s1x1_d1x1_b1_in16x16_p2x2_num2_M64","8 1 16 2 1 1 16 1 0 ",
"EU72_k3x3_cn96_g1_s1x1_d1x1_b1_in16x16_p1x1_num2_M208","2 7 16 2 1 1 16 1 0 ",
// Below is the information for OpenCL based on which these configurations tuned
/*******************************************************************************
Number of platforms                               1
  Platform Name                                   Intel(R) OpenCL
  Platform Vendor                                 Intel(R) Corporation
  Platform Version                                OpenCL 2.0
  Platform Profile                                FULL_PROFILE
  Platform Extensions                             cl_intel_accelerator cl_intel_advanced_motion_estimation cl_intel_driver_diagnostics cl_intel_motion_estimation cl_intel_packed_yuv cl_intel_required_subgroup_size cl_intel_subgroups cl_intel_subgroups_short cl_intel_va_api_media_sharing cl_khr_3d_image_writes cl_khr_byte_addressable_store cl_khr_depth_images cl_khr_fp16 cl_khr_fp64 cl_khr_global_int32_base_atomics cl_khr_global_int32_extended_atomics cl_khr_icd cl_khr_image2d_from_buffer cl_khr_local_int32_base_atomics cl_khr_local_int32_extended_atomics cl_khr_mipmap_image cl_khr_mipmap_image_writes cl_khr_spir cl_khr_subgroups
  Platform Extensions function suffix             INTEL

  Platform Name                                   Intel(R) OpenCL
Number of devices                                 1
  Device Name                                     Intel(R) HD Graphics
  Device Vendor                                   Intel(R) Corporation
  Device Vendor ID                                0x8086
  Device Version                                  OpenCL 2.0
  Driver Version                                  16.5.56875
  Device OpenCL C Version                         OpenCL C 2.0 ( using IGC )
  Device Type                                     GPU
  Device Profile                                  FULL_PROFILE
  Max compute units                               48
  Max clock frequency                             950MHz
  Device Partition                                (core)
    Max number of sub-devices                     0
    Supported partition types                     by <unknown> (0x7F4B00000000)
  Max work item dimensions                        3
  Max work item sizes                             256x256x256
  Max work group size                             256
  Preferred work group size multiple              32
  Preferred / native vector sizes
    char                                                16 / 16
    short                                                8 / 8
    int                                                  4 / 4
    long                                                 1 / 1
    half                                                 8 / 8        (cl_khr_fp16)
    float                                                1 / 1
    double                                               1 / 1        (cl_khr_fp64)
  Half-precision Floating-point support           (cl_khr_fp16)
    Denormals                                     Yes
    Infinity and NANs                             Yes
    Round to nearest                              Yes
    Round to zero                                 Yes
    Round to infinity                             Yes
    IEEE754-2008 fused multiply-add               Yes
    Support is emulated in software               No
    Correctly-rounded divide and sqrt operations  No
  Single-precision Floating-point support         (core)
    Denormals                                     Yes
    Infinity and NANs                             Yes
    Round to nearest                              Yes
    Round to zero                                 Yes
    Round to infinity                             Yes
    IEEE754-2008 fused multiply-add               Yes
    Support is emulated in software               No
    Correctly-rounded divide and sqrt operations  Yes
  Double-precision Floating-point support         (cl_khr_fp64)
    Denormals                                     Yes
    Infinity and NANs                             Yes
    Round to nearest                              Yes
    Round to zero                                 Yes
    Round to infinity                             Yes
    IEEE754-2008 fused multiply-add               Yes
    Support is emulated in software               No
    Correctly-rounded divide and sqrt operations  No
  Address bits                                    64, Little-Endian
  Global memory size                              13361912218 (12.44GiB)
  Error Correction support                        No
  Max memory allocation                           4294959103 (4GiB)
  Unified memory for Host and Device              Yes
  Shared Virtual Memory (SVM) capabilities        (core)
    Coarse-grained buffer sharing                 Yes
    Fine-grained buffer sharing                   No
    Fine-grained system sharing                   No
    Atomics                                       No
  Minimum alignment for any data type             128 bytes
  Alignment of base address                       1024 bits (128 bytes)
  Preferred alignment for atomics
    SVM                                           64 bytes
    Global                                        64 bytes
    Local                                         64 bytes
  Max size for global variable                    65536 (64KiB)
  Preferred total size of global vars             4294959103 (4GiB)
  Global Memory cache type                        Read/Write
  Global Memory cache size                        1048576
  Global Memory cache line                        64 bytes
  Image support                                   Yes
    Max number of samplers per kernel             16
    Max size for 1D images from buffer            268434943 pixels
    Max 1D or 2D image array size                 2048 images
    Base address alignment for 2D image buffers   4 bytes
    Pitch alignment for 2D image buffers          4 bytes
    Max 2D image size                             16384x16384 pixels
    Max 3D image size                             16384x16384x2048 pixels
    Max number of read image args                 128
    Max number of write image args                128
    Max number of read/write image args           128
  Max number of pipe args                         16
  Max active pipe reservations                    1
  Max pipe packet size                            1024
  Local memory type                               Local
  Local memory size                               65536 (64KiB)
  Max constant buffer size                        4294959103 (4GiB)
  Max number of constant args                     8
  Max size of kernel argument                     1024
  Queue properties (on host)
    Out-of-order execution                        Yes
    Profiling                                     Yes
  Queue properties (on device)
    Out-of-order execution                        Yes
    Profiling                                     Yes
    Preferred size                                131072 (128KiB)
    Max size                                      67108864 (64MiB)
  Max queues on device                            1
  Max events on device                            1024
  Prefer user sync for interop                    Yes
  Profiling timer resolution                      83ns
  Execution capabilities
    Run OpenCL kernels                            Yes
    Run native kernels                            No
    SPIR versions                                 1.2
  printf() buffer size                            4194304 (4MiB)
  Built-in kernels                                block_motion_estimate_intel;block_advanced_motion_estimate_check_intel;block_advanced_motion_estimate_bidirectional_check_intel
  Motion Estimation accelerator version	(Intel)   2
  Device Available                                Yes
  Compiler Available                              Yes
  Linker Available                                Yes
  Device Extensions                               cl_intel_accelerator cl_intel_advanced_motion_estimation cl_intel_driver_diagnostics cl_intel_motion_estimation cl_intel_packed_yuv cl_intel_required_subgroup_size cl_intel_subgroups cl_intel_subgroups_short cl_intel_va_api_media_sharing cl_khr_3d_image_writes cl_khr_byte_addressable_store cl_khr_depth_images cl_khr_fp16 cl_khr_fp64 cl_khr_global_int32_base_atomics cl_khr_global_int32_extended_atomics cl_khr_icd cl_khr_image2d_from_buffer cl_khr_local_int32_base_atomics cl_khr_local_int32_extended_atomics cl_khr_mipmap_image cl_khr_mipmap_image_writes cl_khr_spir cl_khr_subgroups

NULL platform behavior
  clGetPlatformInfo(NULL, CL_PLATFORM_NAME, ...)  No platform
  clGetDeviceIDs(NULL, CL_DEVICE_TYPE_ALL, ...)   No platform
  clCreateContext(NULL, ...) [default]            No platform
  clCreateContext(NULL, ...) [other]              Success [INTEL]
  clCreateContextFromType(NULL, CL_DEVICE_TYPE_CPU)  No platform
  clCreateContextFromType(NULL, CL_DEVICE_TYPE_GPU)  No platform
  clCreateContextFromType(NULL, CL_DEVICE_TYPE_ACCELERATOR)  No platform
  clCreateContextFromType(NULL, CL_DEVICE_TYPE_CUSTOM)  No platform
  clCreateContextFromType(NULL, CL_DEVICE_TYPE_ALL)  No platform
********************************************************************************/
"EU48_k1x1_cn512_g1_s1x1_d1x1_b1_in16x16_p0x0_num2_M32","8 3 8 2 1 1 8 1 0 ",
"EU48_k1x1_cn480_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M64","8 2 8 2 1 1 8 1 0 ",
"EU48_k1x1_cn32_g1_s1x1_d1x1_b0_in64x64_p0x0_num1_M128","1 16 32 5 1 16 1 1 0 ",
"EU48_k5x5_cn16_g1_s1x1_d1x1_b1_in32x32_p2x2_num1_M32","8 1 16 2 1 1 16 1 0 ",
"EU48_k1x1_cn512_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M144","1 8 32 5 1 8 1 1 0 ",
"EU48_k1x1_cn192_g1_s1x1_d1x1_b1_in32x32_p0x0_num1_M96","1 16 32 5 1 16 1 1 0 ",
"EU48_k3x3_cn128_g1_s1x1_d1x1_b0_in32x32_p1x1_num1_M128","6 4 16 2 1 1 16 1 0 ",
"EU48_k1x1_cn256_g1_s1x1_d1x1_b1_in32x32_p0x0_num1_M128","2 8 32 5 1 8 1 1 0 ",
"EU48_k5x5_cn32_g1_s1x1_d1x1_b1_in16x16_p2x2_num1_M64","8 1 16 2 1 1 16 1 0 ",
"EU48_k2x2_cn16_g1_s2x2_d1x1_b0_in256x256_p0x0_num1_M16","2 7 16 2 1 1 16 1 0 ",
"EU48_k3x3_cn4_g1_s1x1_d1x1_b1_in256x256_p1x1_num1_M4","6 4 8 2 1 1 8 1 0 ",
"EU48_k1x1_cn128_g1_s1x1_d1x1_b0_in32x32_p0x0_num1_M512","2 8 32 5 1 8 1 1 0 ",
"EU48_k1x1_cn512_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M112","8 3 8 2 1 1 8 1 0 ",
"EU48_k3x3_cn512_g1_s1x1_d1x1_b0_in16x16_p1x1_num1_M512","2 7 8 2 1 1 8 1 0 ",
"EU48_k1x1_cn192_g1_s1x1_d1x1_b1_in32x32_p0x0_num2_M64","2 8 32 5 1 8 1 1 0 ",
"EU48_k1x1_cn832_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M384","4 6 8 2 1 1 8 1 0 ",
"EU48_k1x1_cn480_g1_s1x1_d1x1_b1_in16x16_p0x0_num2_M16","8 2 8 2 1 1 8 1 0 ",
"EU48_k1x1_cn480_g1_s1x1_d1x1_b1_in16x16_p0x0_num2_M96","1 8 32 5 1 8 1 1 0 ",
"EU48_k1x1_cn256_g1_s1x1_d1x1_b0_in16x16_p0x0_num1_M1024","1 8 32 5 1 8 1 1 0 ",
"EU48_k1x1_cn832_g1_s1x1_d1x1_b1_in16x16_p0x0_num2_M192","4 7 8 2 1 1 8 1 0 ",
"EU48_k1x1_cn128_g1_s1x1_d1x1_b0_in64x64_p0x0_num1_M64","1 8 32 5 1 8 1 1 0 ",
"EU48_k3x3_cn160_g1_s1x1_d1x1_b1_in16x16_p1x1_num2_M320","2 7 16 2 1 1 16 1 0 ",
"EU48_k7x7_cn3_g1_s2x2_d1x1_b1_in224x224_p3x3_num1_M64","1 8 32 5 1 8 1 1 0 ",
"EU48_k5x5_cn16_g1_s1x1_d1x1_b1_in16x16_p2x2_num1_M48","4 2 16 2 1 1 16 1 0 ",
"EU48_k1x1_cn528_g1_s1x1_d1x1_b1_in16x16_p0x0_num2_M256","2 8 32 5 1 8 1 1 0 ",
"EU48_k3x3_cn64_g1_s1x1_d1x1_b1_in64x64_p1x1_num1_M192","2 8 16 2 1 1 16 1 0 ",
"EU48_k11x11_cn3_g1_s4x4_d1x1_b1_in224x224_p0x0_num1_M96","1 8 32 5 1 8 1 1 0 ",
"EU48_k1x1_cn512_g1_s1x1_d1x1_b1_in16x16_p0x0_num2_M112","1 8 32 5 1 8 1 1 0 ",
"EU48_k1x1_cn512_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M32","12 1 8 2 1 1 8 1 0 ",
"EU48_k1x1_cn512_g1_s2x2_d1x1_b0_in32x32_p0x0_num1_M256","1 8 32 5 1 8 1 1 0 ",
"EU48_k1x1_cn512_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M128","12 2 8 2 1 1 8 1 0 ",
"EU48_k1x1_cn512_g1_s1x1_d1x1_b1_in16x16_p0x0_num2_M64","8 2 8 2 1 1 8 1 0 ",
"EU48_k3x3_cn128_g1_s1x1_d1x1_b1_in32x32_p1x1_num2_M192","2 7 16 2 1 1 16 1 0 ",
"EU48_k3x3_cn128_g1_s1x1_d1x1_b1_in16x16_p1x1_num1_M256","2 5 16 2 1 1 16 1 0 ",
"EU48_k1x1_cn16_g1_s1x1_d1x1_b0_in256x256_p0x0_num1_M4","8 3 8 2 1 1 8 1 0 ",
"EU48_k1x5_cn32_g1_s1x1_d1x1_b1_in64x64_p0x2_num1_M32","4 7 16 2 1 1 16 1 0 ",
"EU48_k1x1_cn832_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M256","4 7 8 2 1 1 8 1 0 ",
"EU48_k3x3_cn3_g1_s2x2_d1x1_b1_in256x256_p1x1_num1_M13","1 1 1 4 1 1 1 0 1 ",
"EU48_k11x7_cn3_g1_s3x4_d1x1_b1_in64x64_p3x2_num1_M64","4 1 16 2 1 1 16 1 0 ",
"EU48_k1x1_cn480_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M96","8 3 8 2 1 1 8 1 0 ",
"EU48_k1x1_cn128_g1_s1x1_d1x1_b0_in64x64_p0x0_num1_M16","1 8 32 5 1 8 1 1 0 ",
"EU48_k3x3_cn32_g1_s1x1_d2x2_b1_in64x64_p2x2_num1_M32","3 3 16 2 1 1 16 1 0 ",
"EU48_k3x3_cn32_g1_s1x1_d8x8_b1_in64x64_p8x8_num1_M32","1 8 32 5 1 8 1 1 0 ",
"EU48_k1x1_cn192_g1_s1x1_d1x1_b1_in32x32_p0x0_num2_M96","2 8 32 5 1 8 1 1 0 ",
"EU48_k2x2_cn64_g1_s2x2_d1x1_b0_in128x128_p0x0_num1_M32","4 4 16 2 1 1 16 1 0 ",
"EU48_k1x1_cn528_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M128","4 3 8 2 1 1 8 1 0 ",
"EU48_k1x1_cn512_g1_s1x1_d1x1_b0_in32x32_p0x0_num1_M128","2 8 32 5 1 8 1 1 0 ",
"EU48_k3x3_cn16_g1_s1x1_d1x1_b1_in128x128_p1x1_num1_M16","2 7 16 2 1 1 16 1 0 ",
"EU48_k1x1_cn4_g1_s1x1_d1x1_b0_in256x256_p0x0_num1_M16","2 8 32 5 1 8 1 1 0 ",
"EU48_k3x3_cn384_g2_s1x1_d1x1_b1_in16x16_p1x1_num1_M128","6 2 8 2 1 1 8 1 0 ",
"EU48_k1x1_cn3_g1_s1x1_d1x1_b1_in16x16_p0x0_num2_M4","4 2 8 2 1 1 8 1 0 ",
"EU48_k1x1_cn512_g1_s1x1_d1x1_b1_in16x16_p0x0_num2_M144","1 8 32 5 1 8 1 1 0 ",
"EU48_k1x1_cn512_g1_s1x1_d1x1_b1_in16x16_p0x0_num2_M160","1 8 32 5 1 8 1 1 0 ",
"EU48_k1x1_cn832_g1_s1x1_d1x1_b1_in16x16_p0x0_num2_M384","1 8 32 5 1 8 1 1 0 ",
"EU48_k1x1_cn256_g1_s2x2_d1x1_b0_in64x64_p0x0_num1_M128","1 16 32 5 1 16 1 1 0 ",
"EU48_k1x1_cn480_g1_s1x1_d1x1_b1_in16x16_p0x0_num2_M192","1 8 32 5 1 8 1 1 0 ",
"EU48_k1x1_cn1024_g1_s2x2_d1x1_b0_in16x16_p0x0_num1_M2048","1 16 32 5 1 16 1 1 0 ",
"EU48_k3x3_cn192_g1_s1x1_d1x1_b1_in16x16_p1x1_num2_M384","1 8 32 5 1 8 1 1 0 ",
"EU48_k1x1_cn256_g1_s1x1_d1x1_b1_in32x32_p0x0_num2_M128","1 8 32 5 1 8 1 1 0 ",
"EU48_k1x1_cn16_g1_s1x1_d1x1_b0_in128x128_p0x0_num1_M64","1 16 32 5 1 16 1 1 0 ",
"EU48_k1x1_cn832_g1_s1x1_d1x1_b1_in16x16_p0x0_num2_M32","4 7 8 2 1 1 8 1 0 ",
"EU48_k3x3_cn384_g2_s1x1_d1x1_b1_in16x16_p1x1_num1_M192","2 5 16 2 1 1 16 1 0 ",
"EU48_k3x3_cn96_g1_s1x1_d1x1_b1_in32x32_p1x1_num1_M128","6 4 16 2 1 1 16 1 0 ",
"EU48_k1x1_cn528_g1_s1x1_d1x1_b1_in16x16_p0x0_num2_M32","8 3 8 2 1 1 8 1 0 ",
"EU48_k1x1_cn480_g1_s1x1_d1x1_b1_in16x16_p0x0_num2_M64","12 2 8 2 1 1 8 1 0 ",
"EU48_k1x1_cn256_g1_s1x1_d1x1_b1_in32x32_p0x0_num2_M64","2 8 32 5 1 8 1 1 0 ",
"EU48_k1x1_cn2048_g1_s1x1_d1x1_b0_in16x16_p0x0_num1_M512","4 7 8 2 1 1 8 1 0 ",
"EU48_k1x1_cn256_g1_s1x1_d1x1_b1_in32x32_p0x0_num1_M64","12 2 8 2 1 1 8 1 0 ",
"EU48_k3x3_cn112_g1_s1x1_d1x1_b1_in16x16_p1x1_num1_M224","1 8 32 5 1 8 1 1 0 ",
"EU48_k3x3_cn256_g1_s1x1_d1x1_b1_in16x16_p1x1_num1_M384","2 7 16 2 1 1 16 1 0 ",
"EU48_k3x3_cn32_g1_s1x1_d4x4_b1_in64x64_p4x4_num1_M32","1 8 32 5 1 8 1 1 0 ",
"EU48_k1x1_cn528_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M256","1 8 32 5 1 8 1 1 0 ",
"EU48_k3x3_cn192_g1_s1x1_d1x1_b1_in16x16_p1x1_num1_M384","2 4 16 2 1 1 16 1 0 ",
"EU48_k3x3_cn144_g1_s1x1_d1x1_b1_in16x16_p1x1_num2_M288","2 4 16 2 1 1 16 1 0 ",
"EU48_k1x1_cn832_g1_s1x1_d1x1_b1_in16x16_p0x0_num2_M48","4 6 8 2 1 1 8 1 0 ",
"EU48_k5x5_cn32_g1_s1x1_d1x1_b1_in16x16_p2x2_num2_M64","8 1 16 2 1 1 16 1 0 ",
"EU48_k1x1_cn528_g1_s1x1_d1x1_b1_in16x16_p0x0_num2_M160","1 8 32 5 1 8 1 1 0 ",
"EU48_k1x1_cn528_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M160","12 2 8 2 1 1 8 1 0 ",
"EU48_k1x1_cn256_g1_s1x1_d1x1_b0_in64x64_p0x0_num1_M64","2 8 32 5 1 8 1 1 0 ",
"EU48_k1x1_cn832_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M192","4 6 8 2 1 1 8 1 0 ",
"EU48_k5x5_cn96_g2_s1x1_d1x1_b1_in32x32_p2x2_num1_M128","4 5 16 2 1 1 16 1 0 ",
"EU48_k3x3_cn256_g1_s1x1_d1x1_b0_in16x16_p1x1_num1_M256","2 6 16 2 1 1 16 1 0 ",
"EU48_k1x1_cn528_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M32","8 3 8 2 1 1 8 1 0 ",
"EU48_k5x5_cn16_g1_s1x1_d1x1_b1_in16x16_p2x2_num2_M48","4 2 16 2 1 1 16 1 0 ",
"EU48_k3x3_cn64_g1_s1x1_d1x1_b0_in64x64_p1x1_num1_M64","10 2 16 2 1 1 16 1 0 ",
"EU48_k1x1_cn832_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M160","4 6 8 2 1 1 8 1 0 ",
"EU48_k1x1_cn832_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M32","4 5 8 2 1 1 8 1 0 ",
"EU48_k3x3_cn96_g1_s1x1_d1x1_b1_in16x16_p1x1_num1_M208","2 5 16 2 1 1 16 1 0 ",
"EU48_k1x1_cn832_g1_s1x1_d1x1_b1_in16x16_p0x0_num2_M256","4 6 8 2 1 1 8 1 0 ",
"EU48_k1x1_cn512_g1_s1x1_d1x1_b0_in16x16_p0x0_num1_M2048","2 8 32 5 1 8 1 1 0 ",
"EU48_k1x1_cn832_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M48","4 6 8 2 1 1 8 1 0 ",
"EU48_k1x1_cn64_g1_s1x1_d1x1_b0_in64x64_p0x0_num1_M256","1 16 32 5 1 16 1 1 0 ",
"EU48_k3x3_cn112_g1_s1x1_d1x1_b1_in16x16_p1x1_num2_M224","2 7 16 2 1 1 16 1 0 ",
"EU48_k1x1_cn192_g1_s1x1_d1x1_b1_in32x32_p0x0_num2_M32","1 8 32 5 1 8 1 1 0 ",
"EU48_k5x1_cn32_g1_s1x1_d1x1_b0_in64x64_p2x0_num1_M32","2 8 32 5 1 8 1 1 0 ",
"EU48_k1x1_cn64_g1_s1x1_d1x1_b0_in64x64_p0x0_num1_M64","1 8 32 5 1 8 1 1 0 ",
"EU48_k3x3_cn144_g1_s1x1_d1x1_b1_in16x16_p1x1_num1_M288","2 7 16 2 1 1 16 1 0 ",
"EU48_k3x3_cn128_g1_s1x1_d1x1_b1_in32x32_p1x1_num1_M192","2 7 16 2 1 1 16 1 0 ",
"EU48_k5x5_cn16_g1_s1x1_d1x1_b1_in32x32_p2x2_num2_M32","4 3 16 2 1 1 16 1 0 ",
"EU48_k5x5_cn32_g1_s1x1_d1x1_b1_in32x32_p2x2_num2_M96","4 2 16 2 1 1 16 1 0 ",
"EU48_k3x3_cn96_g1_s1x1_d1x1_b1_in16x16_p1x1_num2_M208","2 5 16 2 1 1 16 1 0 ",
"EU48_k5x5_cn32_g1_s1x1_d1x1_b1_in32x32_p2x2_num1_M96","4 2 16 2 1 1 16 1 0 ",
"EU48_k1x1_cn512_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M24","12 1 8 2 1 1 8 1 0 ",
"EU48_k1x1_cn64_g1_s1x1_d1x1_b0_in128x128_p0x0_num1_M16","4 7 16 2 1 1 16 1 0 ",
"EU48_k1x1_cn256_g1_s2x2_d1x1_b0_in64x64_p0x0_num1_M512","2 8 32 5 1 8 1 1 0 ",
"EU48_k1x1_cn1024_g1_s1x1_d1x1_b0_in16x16_p0x0_num1_M256","1 8 32 5 1 8 1 1 0 ",
"EU48_k1x1_cn512_g1_s1x1_d1x1_b1_in16x16_p0x0_num2_M128","1 8 32 5 1 8 1 1 0 ",
"EU48_k1x1_cn256_g1_s1x1_d1x1_b1_in32x32_p0x0_num2_M32","1 8 32 5 1 8 1 1 0 ",
"EU48_k3x3_cn160_g1_s1x1_d1x1_b1_in16x16_p1x1_num1_M320","2 8 16 2 1 1 16 1 0 ",
"EU48_k3x3_cn64_g1_s1x1_d1x1_b1_in64x64_p1x1_num2_M192","6 4 16 2 1 1 16 1 0 ",
"EU48_k5x5_cn32_g1_s1x1_d1x1_b1_in16x16_p2x2_num2_M128","4 3 16 2 1 1 16 1 0 ",
"EU48_k1x1_cn832_g1_s1x1_d1x1_b1_in16x16_p0x0_num2_M160","4 6 8 2 1 1 8 1 0 ",
"EU48_k1x1_cn528_g1_s1x1_d1x1_b1_in16x16_p0x0_num2_M128","1 8 32 5 1 8 1 1 0 ",
"EU48_k3x3_cn3_g1_s1x1_d1x1_b1_in16x16_p0x0_num2_M5","2 3 8 2 1 1 8 1 0 ",
"EU48_k1x1_cn192_g1_s1x1_d1x1_b1_in32x32_p0x0_num1_M64","1 8 32 5 1 8 1 1 0 ",
"EU48_k1x1_cn480_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M192","1 8 32 5 1 8 1 1 0 ",
"EU48_k1x1_cn192_g1_s1x1_d1x1_b1_in32x32_p0x0_num1_M32","8 3 8 2 1 1 8 1 0 ",
"EU48_k1x1_cn64_g1_s1x1_d1x1_b1_in64x64_p0x0_num2_M64","1 16 32 5 1 16 1 1 0 ",
"EU48_k5x5_cn32_g1_s1x1_d1x1_b1_in16x16_p2x2_num1_M128","8 2 16 2 1 1 16 1 0 ",
"EU48_k7x7_cn3_g1_s2x2_d1x1_b1_in224x224_p3x3_num2_M64","1 8 32 5 1 8 1 1 0 ",
"EU48_k1x1_cn832_g1_s1x1_d1x1_b1_in16x16_p0x0_num2_M128","4 6 8 2 1 1 8 1 0 ",
"EU48_k1x1_cn128_g1_s1x1_d1x1_b0_in64x64_p0x0_num1_M32","1 16 32 5 1 16 1 1 0 ",
"EU48_k1x1_cn1024_g1_s2x2_d1x1_b0_in16x16_p0x0_num1_M512","1 8 32 5 1 8 1 1 0 ",
"EU48_k5x5_cn24_g1_s1x1_d1x1_b1_in16x16_p2x2_num2_M64","4 4 16 2 1 1 16 1 0 ",
"EU48_k1x1_cn256_g1_s1x1_d1x1_b1_in32x32_p0x0_num1_M32","8 3 8 2 1 1 8 1 0 ",
"EU48_k1x1_cn512_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M160","12 2 8 2 1 1 8 1 0 ",
"EU48_k1x1_cn512_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M64","8 3 8 2 1 1 8 1 0 ",
"EU48_k1x1_cn512_g1_s2x2_d1x1_b0_in32x32_p0x0_num1_M1024","1 8 32 5 1 8 1 1 0 ",
"EU48_k5x4_cn6_g3_s3x2_d1x1_b1_in128x80_p1x0_num2_M4","1 1 1 4 1 1 1 0 1 ",
"EU48_k3x3_cn128_g1_s1x1_d1x1_b1_in16x16_p1x1_num2_M256","2 7 16 2 1 1 16 1 0 ",
"EU48_k1x1_cn512_g1_s1x1_d1x1_b1_in16x16_p0x0_num2_M24","8 2 8 2 1 1 8 1 0 ",
"EU48_k1x1_cn480_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M16","12 1 8 2 1 1 8 1 0 ",
"EU48_k3x3_cn96_g1_s1x1_d1x1_b1_in32x32_p1x1_num2_M128","10 2 16 2 1 1 16 1 0 ",
"EU48_k1x1_cn64_g1_s1x1_d1x1_b1_in64x64_p0x0_num1_M64","1 16 32 5 1 16 1 1 0 ",
"EU48_k3x3_cn32_g1_s1x1_d16x16_b1_in64x64_p16x16_num1_M32","1 16 32 5 1 16 1 1 0 ",
"EU48_k1x1_cn832_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M128","4 7 8 2 1 1 8 1 0 ",
"EU48_k1x1_cn192_g1_s1x1_d1x1_b1_in32x32_p0x0_num1_M16","12 2 8 2 1 1 8 1 0 ",
"EU48_k4x4_cn3_g1_s1x1_d1x1_b1_in16x16_p0x0_num2_M2","1 4 16 2 1 1 16 1 0 ",
"EU48_k1x1_cn64_g1_s1x1_d1x1_b0_in128x128_p0x0_num1_M4","8 2 8 2 1 1 8 1 0 ",
"EU48_k5x5_cn24_g1_s1x1_d1x1_b1_in16x16_p2x2_num1_M64","4 2 16 2 1 1 16 1 0 ",
"EU48_k1x1_cn192_g1_s1x1_d1x1_b1_in32x32_p0x0_num2_M16","1 8 32 5 1 8 1 1 0 ",
"EU48_k5x5_cn48_g1_s1x1_d1x1_b1_in16x16_p2x2_num2_M128","4 2 16 2 1 1 16 1 0 ",
"EU48_k3x3_cn32_g1_s1x1_d1x1_b1_in64x64_p1x1_num1_M32","2 8 16 2 1 1 16 1 0 ",
"EU48_k5x5_cn48_g1_s1x1_d1x1_b1_in16x16_p2x2_num1_M128","4 2 16 2 1 1 16 1 0 ",
// Below is the information for OpenCL based on which these configurations tuned
/*******************************************************************************
Number of platforms                               1
  Platform Name                                   Intel(R) OpenCL
  Platform Vendor                                 Intel(R) Corporation
  Platform Version                                OpenCL 2.0
  Platform Profile                                FULL_PROFILE
  Platform Extensions                             cl_intel_accelerator cl_intel_advanced_motion_estimation cl_intel_device_side_avc_motion_estimation cl_intel_driver_diagnostics cl_intel_media_block_io cl_intel_motion_estimation cl_intel_planar_yuv cl_intel_packed_yuv cl_intel_required_subgroup_size cl_intel_subgroups cl_intel_subgroups_short cl_intel_va_api_media_sharing cl_khr_3d_image_writes cl_khr_byte_addressable_store cl_khr_depth_images cl_khr_fp16 cl_khr_fp64 cl_khr_global_int32_base_atomics cl_khr_global_int32_extended_atomics cl_khr_icd cl_khr_image2d_from_buffer cl_khr_local_int32_base_atomics cl_khr_local_int32_extended_atomics cl_khr_mipmap_image cl_khr_mipmap_image_writes cl_khr_spir cl_khr_subgroups
  Platform Extensions function suffix             INTEL

  Platform Name                                   Intel(R) OpenCL
Number of devices                                 1
  Device Name                                     Intel(R) HD Graphics
  Device Vendor                                   Intel(R) Corporation
  Device Vendor ID                                0x8086
  Device Version                                  OpenCL 2.0
  Driver Version                                  16.5.59288
  Device OpenCL C Version                         OpenCL C 2.0
  Device Type                                     GPU
  Device Profile                                  FULL_PROFILE
  Max compute units                               24
  Max clock frequency                             1050MHz
  Device Partition                                (core)
    Max number of sub-devices                     0
    Supported partition types                     by <unknown> (0x7F5100000000)
  Max work item dimensions                        3
  Max work item sizes                             256x256x256
  Max work group size                             256
  Preferred work group size multiple              32
  Preferred / native vector sizes
    char                                                16 / 16
    short                                                8 / 8
    int                                                  4 / 4
    long                                                 1 / 1
    half                                                 8 / 8        (cl_khr_fp16)
    float                                                1 / 1
    double                                               1 / 1        (cl_khr_fp64)
  Half-precision Floating-point support           (cl_khr_fp16)
    Denormals                                     Yes
    Infinity and NANs                             Yes
    Round to nearest                              Yes
    Round to zero                                 Yes
    Round to infinity                             Yes
    IEEE754-2008 fused multiply-add               Yes
    Support is emulated in software               No
    Correctly-rounded divide and sqrt operations  No
  Single-precision Floating-point support         (core)
    Denormals                                     Yes
    Infinity and NANs                             Yes
    Round to nearest                              Yes
    Round to zero                                 Yes
    Round to infinity                             Yes
    IEEE754-2008 fused multiply-add               Yes
    Support is emulated in software               No
    Correctly-rounded divide and sqrt operations  Yes
  Double-precision Floating-point support         (cl_khr_fp64)
    Denormals                                     Yes
    Infinity and NANs                             Yes
    Round to nearest                              Yes
    Round to zero                                 Yes
    Round to infinity                             Yes
    IEEE754-2008 fused multiply-add               Yes
    Support is emulated in software               No
    Correctly-rounded divide and sqrt operations  No
  Address bits                                    64, Little-Endian
  Global memory size                              6588802663 (6.136GiB)
  Error Correction support                        No
  Max memory allocation                           3294401331 (3.068GiB)
  Unified memory for Host and Device              Yes
  Shared Virtual Memory (SVM) capabilities        (core)
    Coarse-grained buffer sharing                 Yes
    Fine-grained buffer sharing                   No
    Fine-grained system sharing                   No
    Atomics                                       No
  Minimum alignment for any data type             128 bytes
  Alignment of base address                       1024 bits (128 bytes)
  Preferred alignment for atomics
    SVM                                           64 bytes
    Global                                        64 bytes
    Local                                         64 bytes
  Max size for global variable                    65536 (64KiB)
  Preferred total size of global vars             3294401331 (3.068GiB)
  Global Memory cache type                        Read/Write
  Global Memory cache size                        524288
  Global Memory cache line                        64 bytes
  Image support                                   Yes
    Max number of samplers per kernel             16
    Max size for 1D images from buffer            205900083 pixels
    Max 1D or 2D image array size                 2048 images
    Base address alignment for 2D image buffers   4 bytes
    Pitch alignment for 2D image buffers          4 bytes
    Max 2D image size                             16384x16384 pixels
    Max 3D image size                             16384x16384x2048 pixels
    Max number of read image args                 128
    Max number of write image args                128
    Max number of read/write image args           128
  Max number of pipe args                         16
  Max active pipe reservations                    1
  Max pipe packet size                            1024
  Local memory type                               Local
  Local memory size                               65536 (64KiB)
  Max constant buffer size                        3294401331 (3.068GiB)
  Max number of constant args                     8
  Max size of kernel argument                     1024
  Queue properties (on host)
    Out-of-order execution                        Yes
    Profiling                                     Yes
  Queue properties (on device)
    Out-of-order execution                        Yes
    Profiling                                     Yes
    Preferred size                                131072 (128KiB)
    Max size                                      67108864 (64MiB)
  Max queues on device                            1
  Max events on device                            1024
  Prefer user sync for interop                    Yes
  Profiling timer resolution                      83ns
  Execution capabilities
    Run OpenCL kernels                            Yes
    Run native kernels                            No
    SPIR versions                                 1.2
  printf() buffer size                            4194304 (4MiB)
  Built-in kernels                                block_motion_estimate_intel;block_advanced_motion_estimate_check_intel;block_advanced_motion_estimate_bidirectional_check_intel
  Motion Estimation accelerator version	(Intel)   2
  Device Available                                Yes
  Compiler Available                              Yes
  Linker Available                                Yes
  Device Extensions                               cl_intel_accelerator cl_intel_advanced_motion_estimation cl_intel_device_side_avc_motion_estimation cl_intel_driver_diagnostics cl_intel_media_block_io cl_intel_motion_estimation cl_intel_planar_yuv cl_intel_packed_yuv cl_intel_required_subgroup_size cl_intel_subgroups cl_intel_subgroups_short cl_intel_va_api_media_sharing cl_khr_3d_image_writes cl_khr_byte_addressable_store cl_khr_depth_images cl_khr_fp16 cl_khr_fp64 cl_khr_global_int32_base_atomics cl_khr_global_int32_extended_atomics cl_khr_icd cl_khr_image2d_from_buffer cl_khr_local_int32_base_atomics cl_khr_local_int32_extended_atomics cl_khr_mipmap_image cl_khr_mipmap_image_writes cl_khr_spir cl_khr_subgroups

NULL platform behavior
  clGetPlatformInfo(NULL, CL_PLATFORM_NAME, ...)  No platform
  clGetDeviceIDs(NULL, CL_DEVICE_TYPE_ALL, ...)   No platform
  clCreateContext(NULL, ...) [default]            No platform
  clCreateContext(NULL, ...) [other]              Success [INTEL]
  clCreateContextFromType(NULL, CL_DEVICE_TYPE_CPU)  No platform
  clCreateContextFromType(NULL, CL_DEVICE_TYPE_GPU)  No platform
  clCreateContextFromType(NULL, CL_DEVICE_TYPE_ACCELERATOR)  No platform
  clCreateContextFromType(NULL, CL_DEVICE_TYPE_CUSTOM)  No platform
  clCreateContextFromType(NULL, CL_DEVICE_TYPE_ALL)  No platform
********************************************************************************/
"EU24_k1x1_cn192_g1_s1x1_d1x1_b1_in32x32_p0x0_num1_M32","1 8 32 5 1 8 1 1 0 ",
"EU24_k1x1_cn256_g1_s1x1_d1x1_b1_in32x32_p0x0_num1_M64","2 8 32 5 1 8 1 1 0 ",
"EU24_k5x1_cn32_g1_s1x1_d1x1_b0_in64x64_p2x0_num1_M32","4 6 16 2 1 1 16 1 0 ",
"EU24_k1x1_cn256_g1_s1x1_d1x1_b1_in32x32_p0x0_num1_M128","1 8 32 5 1 8 1 1 0 ",
"EU24_k5x5_cn48_g1_s1x1_d1x1_b1_in16x16_p2x2_num1_M128","4 2 16 2 1 1 16 1 0 ",
"EU24_k3x3_cn112_g1_s1x1_d1x1_b1_in16x16_p1x1_num1_M224","2 5 16 2 1 1 16 1 0 ",
"EU24_k1x1_cn256_g1_s1x1_d1x1_b1_in32x32_p0x0_num1_M32","1 8 32 5 1 8 1 1 0 ",
"EU24_k2x2_cn16_g1_s2x2_d1x1_b0_in256x256_p0x0_num1_M16","1 8 32 5 1 8 1 1 0 ",
"EU24_k5x5_cn32_g1_s1x1_d1x1_b1_in16x16_p2x2_num2_M128","4 3 16 2 1 1 16 1 0 ",
"EU24_k1x1_cn256_g1_s1x1_d1x1_b0_in64x64_p0x0_num1_M64","1 8 32 5 1 8 1 1 0 ",
"EU24_k1x1_cn528_g1_s1x1_d1x1_b1_in16x16_p0x0_num2_M256","1 8 32 5 1 8 1 1 0 ",
"EU24_k3x3_cn192_g1_s1x1_d1x1_b1_in16x16_p1x1_num2_M384","1 8 32 5 1 8 1 1 0 ",
"EU24_k3x3_cn256_g1_s1x1_d1x1_b1_in16x16_p1x1_num1_M384","2 7 16 2 1 1 16 1 0 ",
"EU24_k1x1_cn2048_g1_s1x1_d1x1_b0_in16x16_p0x0_num1_M512","1 8 32 5 1 8 1 1 0 ",
"EU24_k1x1_cn192_g1_s1x1_d1x1_b1_in32x32_p0x0_num2_M16","2 8 32 5 1 8 1 1 0 ",
"EU24_k3x3_cn384_g2_s1x1_d1x1_b1_in16x16_p1x1_num1_M128","2 7 16 2 1 1 16 1 0 ",
"EU24_k1x1_cn832_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M192","1 8 32 5 1 8 1 1 0 ",
"EU24_k3x3_cn112_g1_s1x1_d1x1_b1_in16x16_p1x1_num2_M224","2 7 16 2 1 1 16 1 0 ",
"EU24_k3x3_cn32_g1_s1x1_d8x8_b1_in64x64_p8x8_num1_M32","1 8 32 5 1 8 1 1 0 ",
"EU24_k3x3_cn96_g1_s1x1_d1x1_b1_in16x16_p1x1_num1_M208","2 7 16 2 1 1 16 1 0 ",
"EU24_k11x11_cn3_g1_s4x4_d1x1_b1_in224x224_p0x0_num1_M96","1 8 32 5 1 8 1 1 0 ",
"EU24_k1x1_cn512_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M160","1 8 32 5 1 8 1 1 0 ",
"EU24_k7x7_cn3_g1_s2x2_d1x1_b1_in224x224_p3x3_num2_M64","1 8 32 5 1 8 1 1 0 ",
"EU24_k3x3_cn32_g1_s1x1_d2x2_b1_in64x64_p2x2_num1_M32","3 3 16 2 1 1 16 1 0 ",
"EU24_k1x1_cn128_g1_s1x1_d1x1_b0_in64x64_p0x0_num1_M64","1 8 32 5 1 8 1 1 0 ",
"EU24_k1x1_cn512_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M24","8 3 8 2 1 1 8 1 0 ",
"EU24_k3x3_cn128_g1_s1x1_d1x1_b0_in32x32_p1x1_num1_M128","6 4 16 2 1 1 16 1 0 ",
"EU24_k1x1_cn512_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M144","1 8 32 5 1 8 1 1 0 ",
"EU24_k1x1_cn1024_g1_s1x1_d1x1_b0_in16x16_p0x0_num1_M256","2 8 32 5 1 8 1 1 0 ",
"EU24_k3x3_cn96_g1_s1x1_d1x1_b1_in16x16_p1x1_num2_M208","2 7 16 2 1 1 16 1 0 ",
"EU24_k5x5_cn32_g1_s1x1_d1x1_b1_in16x16_p2x2_num1_M128","4 3 16 2 1 1 16 1 0 ",
"EU24_k5x5_cn16_g1_s1x1_d1x1_b1_in16x16_p2x2_num2_M48","4 2 16 2 1 1 16 1 0 ",
"EU24_k1x1_cn512_g1_s1x1_d1x1_b0_in16x16_p0x0_num1_M2048","4 7 16 2 1 1 16 1 0 ",
"EU24_k1x1_cn128_g1_s1x1_d1x1_b0_in64x64_p0x0_num1_M16","2 8 32 5 1 8 1 1 0 ",
"EU24_k3x3_cn64_g1_s1x1_d1x1_b1_in64x64_p1x1_num1_M192","6 4 16 2 1 1 16 1 0 ",
"EU24_k1x1_cn256_g1_s1x1_d1x1_b1_in32x32_p0x0_num2_M128","1 8 32 5 1 8 1 1 0 ",
"EU24_k1x1_cn256_g1_s1x1_d1x1_b0_in16x16_p0x0_num1_M1024","1 8 32 5 1 8 1 1 0 ",
"EU24_k1x1_cn32_g1_s1x1_d1x1_b0_in64x64_p0x0_num1_M128","1 16 32 5 1 16 1 1 0 ",
"EU24_k1x1_cn4_g1_s1x1_d1x1_b0_in256x256_p0x0_num1_M16","1 8 32 5 1 8 1 1 0 ",
"EU24_k3x3_cn192_g1_s1x1_d1x1_b1_in16x16_p1x1_num1_M384","2 7 16 2 1 1 16 1 0 ",
"EU24_k1x1_cn832_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M256","1 8 32 5 1 8 1 1 0 ",
"EU24_k1x1_cn128_g1_s1x1_d1x1_b0_in64x64_p0x0_num1_M32","1 8 32 5 1 8 1 1 0 ",
"EU24_k1x1_cn832_g1_s1x1_d1x1_b1_in16x16_p0x0_num2_M32","4 6 8 2 1 1 8 1 0 ",
"EU24_k5x5_cn48_g1_s1x1_d1x1_b1_in16x16_p2x2_num2_M128","4 4 16 2 1 1 16 1 0 ",
"EU24_k1x1_cn512_g1_s1x1_d1x1_b1_in16x16_p0x0_num2_M128","2 8 32 5 1 8 1 1 0 ",
"EU24_k1x1_cn512_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M32","8 2 8 2 1 1 8 1 0 ",
"EU24_k1x1_cn512_g1_s1x1_d1x1_b1_in16x16_p0x0_num2_M64","1 8 32 5 1 8 1 1 0 ",
"EU24_k1x1_cn1024_g1_s2x2_d1x1_b0_in16x16_p0x0_num1_M2048","1 16 32 5 1 16 1 1 0 ",
"EU24_k5x5_cn32_g1_s1x1_d1x1_b1_in16x16_p2x2_num2_M64","4 3 16 2 1 1 16 1 0 ",
"EU24_k1x1_cn512_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M128","1 8 32 5 1 8 1 1 0 ",
"EU24_k1x1_cn832_g1_s1x1_d1x1_b1_in16x16_p0x0_num2_M384","1 8 32 5 1 8 1 1 0 ",
"EU24_k5x4_cn6_g3_s3x2_d1x1_b1_in128x80_p1x0_num2_M4","1 1 1 4 1 1 1 0 1 ",
"EU24_k3x3_cn128_g1_s1x1_d1x1_b1_in32x32_p1x1_num2_M192","6 4 16 2 1 1 16 1 0 ",
"EU24_k3x3_cn256_g1_s1x1_d1x1_b0_in16x16_p1x1_num1_M256","2 7 16 2 1 1 16 1 0 ",
"EU24_k3x3_cn160_g1_s1x1_d1x1_b1_in16x16_p1x1_num1_M320","2 8 16 2 1 1 16 1 0 ",
"EU24_k1x1_cn512_g1_s1x1_d1x1_b1_in16x16_p0x0_num2_M160","1 8 32 5 1 8 1 1 0 ",
"EU24_k1x1_cn256_g1_s1x1_d1x1_b1_in32x32_p0x0_num2_M64","1 8 32 5 1 8 1 1 0 ",
"EU24_k1x1_cn512_g1_s2x2_d1x1_b0_in32x32_p0x0_num1_M256","2 8 32 5 1 8 1 1 0 ",
"EU24_k1x1_cn64_g1_s1x1_d1x1_b1_in64x64_p0x0_num2_M64","1 8 32 5 1 8 1 1 0 ",
"EU24_k1x1_cn480_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M192","1 8 32 5 1 8 1 1 0 ",
"EU24_k1x1_cn528_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M256","2 8 32 5 1 8 1 1 0 ",
"EU24_k3x3_cn128_g1_s1x1_d1x1_b1_in16x16_p1x1_num1_M256","2 5 16 2 1 1 16 1 0 ",
"EU24_k5x5_cn24_g1_s1x1_d1x1_b1_in16x16_p2x2_num2_M64","4 3 16 2 1 1 16 1 0 ",
"EU24_k1x1_cn480_g1_s1x1_d1x1_b1_in16x16_p0x0_num2_M16","8 3 8 2 1 1 8 1 0 ",
"EU24_k1x1_cn528_g1_s1x1_d1x1_b1_in16x16_p0x0_num2_M128","2 8 32 5 1 8 1 1 0 ",
"EU24_k1x1_cn512_g1_s1x1_d1x1_b1_in16x16_p0x0_num2_M112","2 8 32 5 1 8 1 1 0 ",
"EU24_k1x1_cn64_g1_s1x1_d1x1_b0_in128x128_p0x0_num1_M16","2 8 32 5 1 8 1 1 0 ",
"EU24_k1x1_cn480_g1_s1x1_d1x1_b1_in16x16_p0x0_num2_M96","1 8 32 5 1 8 1 1 0 ",
"EU24_k1x1_cn64_g1_s1x1_d1x1_b0_in64x64_p0x0_num1_M256","2 8 32 5 1 8 1 1 0 ",
"EU24_k1x1_cn832_g1_s1x1_d1x1_b1_in16x16_p0x0_num2_M128","1 8 32 5 1 8 1 1 0 ",
"EU24_k5x5_cn16_g1_s1x1_d1x1_b1_in32x32_p2x2_num2_M32","4 2 16 2 1 1 16 1 0 ",
"EU24_k1x1_cn480_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M96","8 3 8 2 1 1 8 1 0 ",
"EU24_k3x3_cn16_g1_s1x1_d1x1_b1_in128x128_p1x1_num1_M16","6 3 16 2 1 1 16 1 0 ",
"EU24_k1x1_cn512_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M112","1 8 32 5 1 8 1 1 0 ",
"EU24_k5x5_cn32_g1_s1x1_d1x1_b1_in32x32_p2x2_num2_M96","4 3 16 2 1 1 16 1 0 ",
"EU24_k1x1_cn528_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M32","8 2 8 2 1 1 8 1 0 ",
"EU24_k1x1_cn832_g1_s1x1_d1x1_b1_in16x16_p0x0_num2_M192","1 8 32 5 1 8 1 1 0 ",
"EU24_k1x1_cn832_g1_s1x1_d1x1_b1_in16x16_p0x0_num2_M160","1 8 32 5 1 8 1 1 0 ",
"EU24_k3x3_cn144_g1_s1x1_d1x1_b1_in16x16_p1x1_num2_M288","2 8 16 2 1 1 16 1 0 ",
"EU24_k3x3_cn144_g1_s1x1_d1x1_b1_in16x16_p1x1_num1_M288","2 7 16 2 1 1 16 1 0 ",
"EU24_k7x7_cn3_g1_s2x2_d1x1_b1_in224x224_p3x3_num1_M64","1 8 32 5 1 8 1 1 0 ",
"EU24_k1x1_cn512_g1_s1x1_d1x1_b0_in32x32_p0x0_num1_M128","1 8 32 5 1 8 1 1 0 ",
"EU24_k3x3_cn4_g1_s1x1_d1x1_b1_in256x256_p1x1_num1_M4","10 2 8 2 1 1 8 1 0 ",
"EU24_k3x3_cn32_g1_s1x1_d16x16_b1_in64x64_p16x16_num1_M32","1 8 32 5 1 8 1 1 0 ",
"EU24_k1x1_cn480_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M16","8 2 8 2 1 1 8 1 0 ",
"EU24_k1x1_cn192_g1_s1x1_d1x1_b1_in32x32_p0x0_num1_M64","1 16 32 5 1 16 1 1 0 ",
"EU24_k1x5_cn32_g1_s1x1_d1x1_b1_in64x64_p0x2_num1_M32","4 7 16 2 1 1 16 1 0 ",
"EU24_k1x1_cn528_g1_s1x1_d1x1_b1_in16x16_p0x0_num2_M32","8 3 8 2 1 1 8 1 0 ",
"EU24_k3x3_cn384_g2_s1x1_d1x1_b1_in16x16_p1x1_num1_M192","2 7 16 2 1 1 16 1 0 ",
"EU24_k1x1_cn832_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M32","4 6 8 2 1 1 8 1 0 ",
"EU24_k1x1_cn832_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M128","4 6 8 2 1 1 8 1 0 ",
"EU24_k3x3_cn32_g1_s1x1_d4x4_b1_in64x64_p4x4_num1_M32","1 8 32 5 1 8 1 1 0 ",
"EU24_k2x2_cn64_g1_s2x2_d1x1_b0_in128x128_p0x0_num1_M32","2 8 32 5 1 8 1 1 0 ",
"EU24_k5x5_cn96_g2_s1x1_d1x1_b1_in32x32_p2x2_num1_M128","4 3 16 2 1 1 16 1 0 ",
"EU24_k5x5_cn16_g1_s1x1_d1x1_b1_in16x16_p2x2_num1_M48","8 1 16 2 1 1 16 1 0 ",
"EU24_k1x1_cn16_g1_s1x1_d1x1_b0_in256x256_p0x0_num1_M4","8 3 8 2 1 1 8 1 0 ",
"EU24_k1x1_cn832_g1_s1x1_d1x1_b1_in16x16_p0x0_num2_M256","1 8 32 5 1 8 1 1 0 ",
"EU24_k1x1_cn512_g1_s1x1_d1x1_b1_in16x16_p0x0_num2_M144","2 8 32 5 1 8 1 1 0 ",
"EU24_k3x3_cn96_g1_s1x1_d1x1_b1_in32x32_p1x1_num1_M128","6 4 16 2 1 1 16 1 0 ",
"EU24_k1x1_cn256_g1_s1x1_d1x1_b1_in32x32_p0x0_num2_M32","2 8 32 5 1 8 1 1 0 ",
"EU24_k3x3_cn128_g1_s1x1_d1x1_b1_in32x32_p1x1_num1_M192","2 7 16 2 1 1 16 1 0 ",
"EU24_k5x5_cn32_g1_s1x1_d1x1_b1_in16x16_p2x2_num1_M64","4 2 16 2 1 1 16 1 0 ",
"EU24_k1x1_cn528_g1_s1x1_d1x1_b1_in16x16_p0x0_num2_M160","1 8 32 5 1 8 1 1 0 ",
"EU24_k5x5_cn32_g1_s1x1_d1x1_b1_in32x32_p2x2_num1_M96","4 4 16 2 1 1 16 1 0 ",
"EU24_k1x1_cn832_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M160","4 6 8 2 1 1 8 1 0 ",
"EU24_k1x1_cn192_g1_s1x1_d1x1_b1_in32x32_p0x0_num2_M32","2 8 32 5 1 8 1 1 0 ",
"EU24_k3x3_cn32_g1_s1x1_d1x1_b1_in64x64_p1x1_num1_M32","2 8 16 2 1 1 16 1 0 ",
"EU24_k3x3_cn96_g1_s1x1_d1x1_b1_in32x32_p1x1_num2_M128","10 2 16 2 1 1 16 1 0 ",
"EU24_k3x3_cn160_g1_s1x1_d1x1_b1_in16x16_p1x1_num2_M320","2 7 16 2 1 1 16 1 0 ",
"EU24_k1x1_cn512_g1_s1x1_d1x1_b1_in16x16_p0x0_num2_M32","8 3 8 2 1 1 8 1 0 ",
"EU24_k3x3_cn64_g1_s1x1_d1x1_b0_in64x64_p1x1_num1_M64","2 8 16 2 1 1 16 1 0 ",
"EU24_k3x3_cn3_g1_s1x1_d1x1_b1_in16x16_p0x0_num2_M5","2 3 8 2 1 1 8 1 0 ",
"EU24_k1x1_cn16_g1_s1x1_d1x1_b0_in128x128_p0x0_num1_M64","1 16 32 5 1 16 1 1 0 ",
"EU24_k1x1_cn832_g1_s1x1_d1x1_b1_in16x16_p0x0_num2_M48","4 6 8 2 1 1 8 1 0 ",
"EU24_k5x5_cn24_g1_s1x1_d1x1_b1_in16x16_p2x2_num1_M64","4 2 16 2 1 1 16 1 0 ",
"EU24_k1x1_cn64_g1_s1x1_d1x1_b0_in128x128_p0x0_num1_M4","8 2 8 2 1 1 8 1 0 ",
"EU24_k1x1_cn512_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M64","8 2 8 2 1 1 8 1 0 ",
"EU24_k1x1_cn192_g1_s1x1_d1x1_b1_in32x32_p0x0_num2_M96","1 8 32 5 1 8 1 1 0 ",
"EU24_k1x1_cn64_g1_s1x1_d1x1_b0_in64x64_p0x0_num1_M64","1 8 32 5 1 8 1 1 0 ",
"EU24_k1x1_cn480_g1_s1x1_d1x1_b1_in16x16_p0x0_num2_M192","1 8 32 5 1 8 1 1 0 ",
"EU24_k1x1_cn832_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M48","4 6 8 2 1 1 8 1 0 ",
"EU24_k3x3_cn128_g1_s1x1_d1x1_b1_in16x16_p1x1_num2_M256","2 7 16 2 1 1 16 1 0 ",
"EU24_k1x1_cn3_g1_s1x1_d1x1_b1_in16x16_p0x0_num2_M4","4 4 16 2 1 1 16 1 0 ",
"EU24_k4x4_cn3_g1_s1x1_d1x1_b1_in16x16_p0x0_num2_M2","1 3 16 2 1 1 16 1 0 ",
"EU24_k1x1_cn192_g1_s1x1_d1x1_b1_in32x32_p0x0_num1_M96","1 8 32 5 1 8 1 1 0 ",
"EU24_k3x3_cn512_g1_s1x1_d1x1_b0_in16x16_p1x1_num1_M512","2 7 16 2 1 1 16 1 0 ",
"EU24_k1x1_cn256_g1_s2x2_d1x1_b0_in64x64_p0x0_num1_M128","1 8 32 5 1 8 1 1 0 ",
"EU24_k1x1_cn512_g1_s2x2_d1x1_b0_in32x32_p0x0_num1_M1024","1 8 32 5 1 8 1 1 0 ",
"EU24_k1x1_cn192_g1_s1x1_d1x1_b1_in32x32_p0x0_num2_M64","1 8 32 5 1 8 1 1 0 ",
"EU24_k11x7_cn3_g1_s3x4_d1x1_b1_in64x64_p3x2_num1_M64","4 1 16 2 1 1 16 1 0 ",
"EU24_k3x3_cn64_g1_s1x1_d1x1_b1_in64x64_p1x1_num2_M192","6 4 16 2 1 1 16 1 0 ",
"EU24_k1x1_cn480_g1_s1x1_d1x1_b1_in16x16_p0x0_num2_M64","1 8 32 5 1 8 1 1 0 ",
"EU24_k1x1_cn64_g1_s1x1_d1x1_b1_in64x64_p0x0_num1_M64","1 16 32 5 1 16 1 1 0 ",
"EU24_k1x1_cn192_g1_s1x1_d1x1_b1_in32x32_p0x0_num1_M16","8 3 8 2 1 1 8 1 0 ",
"EU24_k1x1_cn128_g1_s1x1_d1x1_b0_in32x32_p0x0_num1_M512","1 8 32 5 1 8 1 1 0 ",
"EU24_k1x1_cn1024_g1_s2x2_d1x1_b0_in16x16_p0x0_num1_M512","1 8 32 5 1 8 1 1 0 ",
"EU24_k1x1_cn528_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M128","1 8 32 5 1 8 1 1 0 ",
"EU24_k1x1_cn832_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M384","4 7 8 2 1 1 8 1 0 ",
"EU24_k1x1_cn528_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M160","1 8 32 5 1 8 1 1 0 ",
"EU24_k1x1_cn480_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M64","8 3 8 2 1 1 8 1 0 ",
"EU24_k3x3_cn3_g1_s2x2_d1x1_b1_in256x256_p1x1_num1_M13","1 1 1 4 1 1 1 0 1 ",
"EU24_k1x1_cn256_g1_s2x2_d1x1_b0_in64x64_p0x0_num1_M512","2 8 32 5 1 8 1 1 0 ",
"EU24_k1x1_cn512_g1_s1x1_d1x1_b1_in16x16_p0x0_num2_M24","8 3 8 2 1 1 8 1 0 ",
"EU24_k5x5_cn16_g1_s1x1_d1x1_b1_in32x32_p2x2_num1_M32","4 3 16 2 1 1 16 1 0 ",
};
#endif
