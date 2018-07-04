#ifndef _OPENCV_OCL4DNN_DEFAULT_KERNEL_CONFIG_HPP_
#define _OPENCV_OCL4DNN_DEFAULT_KERNEL_CONFIG_HPP_
const char *default_kernel_config_intel_fp32[] = {
// Below is the information for OpenCL based on which these configurations tuned
/*******************************************************************************
Number of platforms                               1
  Platform Name                                   Intel(R) OpenCL HD Graphics
  Platform Vendor                                 Intel(R) Corporation
  Platform Version                                OpenCL 2.1
  Platform Profile                                FULL_PROFILE
  Platform Extensions                             cl_khr_3d_image_writes cl_khr_byte_addressable_store cl_khr_fp16 cl_khr_depth_images cl_khr_global_int32_base_atomics cl_khr_global_int32_extended_atomics cl_khr_icd cl_khr_image2d_from_buffer cl_khr_local_int32_base_atomics cl_khr_local_int32_extended_atomics cl_intel_subgroups cl_intel_required_subgroup_size cl_intel_subgroups_short cl_khr_spir cl_intel_accelerator cl_intel_media_block_io cl_intel_driver_diagnostics cl_intel_device_side_avc_motion_estimation cl_khr_priority_hints cl_khr_throttle_hints cl_khr_create_command_queue cl_khr_subgroups cl_khr_il_program cl_khr_mipmap_image cl_khr_mipmap_image_writes cl_khr_fp64 cl_intel_planar_yuv cl_intel_packed_yuv cl_intel_motion_estimation cl_intel_advanced_motion_estimation
  Platform Host timer resolution                  1ns
  Platform Extensions function suffix             INTEL

  Platform Name                                   Intel(R) OpenCL HD Graphics
Number of devices                                 1
  Device Name                                     Intel(R) Gen9 HD Graphics NEO
  Device Vendor                                   Intel(R) Corporation
  Device Vendor ID                                0x8086
  Device Version                                  OpenCL 2.1 NEO
  Driver Version                                  2018ww15-010713
  Device OpenCL C Version                         OpenCL C 2.0
  Device Type                                     GPU
  Device Profile                                  FULL_PROFILE
  Max compute units                               72
  Max clock frequency                             950MHz
  Device Partition                                (core)
    Max number of sub-devices                     0
    Supported partition types                     None
  Max work item dimensions                        3
  Max work item sizes                             256x256x256
  Max work group size                             256
  Preferred work group size multiple              32
  Max sub-groups per work group                   32
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
  Global memory size                              26892222464 (25.05GiB)
  Error Correction support                        No
  Max memory allocation                           4294959104 (4GiB)
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
  Preferred total size of global vars             4294959104 (4GiB)
  Global Memory cache type                        Read/Write
  Global Memory cache size                        1572864
  Global Memory cache line                        64 bytes
  Image support                                   Yes
    Max number of samplers per kernel             16
    Max size for 1D images from buffer            268434944 pixels
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
  Max constant buffer size                        4294959104 (4GiB)
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
    Sub-group independent forward progress        Yes
    IL version                                    SPIR-V_1.0
    SPIR versions                                 1.2
  printf() buffer size                            4194304 (4MiB)
  Built-in kernels                                block_motion_estimate_intel;block_advanced_motion_estimate_check_intel;block_advanced_motion_estimate_bidirectional_check_intel;
  Motion Estimation accelerator version	(Intel)   2
  Device Available                                Yes
  Compiler Available                              Yes
  Linker Available                                Yes
  Device Extensions                               cl_khr_3d_image_writes cl_khr_byte_addressable_store cl_khr_fp16 cl_khr_depth_images cl_khr_global_int32_base_atomics cl_khr_global_int32_extended_atomics cl_khr_icd cl_khr_image2d_from_buffer cl_khr_local_int32_base_atomics cl_khr_local_int32_extended_atomics cl_intel_subgroups cl_intel_required_subgroup_size cl_intel_subgroups_short cl_khr_spir cl_intel_accelerator cl_intel_media_block_io cl_intel_driver_diagnostics cl_intel_device_side_avc_motion_estimation cl_khr_priority_hints cl_khr_throttle_hints cl_khr_create_command_queue cl_khr_subgroups cl_khr_il_program cl_khr_mipmap_image cl_khr_mipmap_image_writes cl_khr_fp64 cl_intel_planar_yuv cl_intel_packed_yuv cl_intel_motion_estimation cl_intel_advanced_motion_estimation

NULL platform behavior
  clGetPlatformInfo(NULL, CL_PLATFORM_NAME, ...)  Intel(R) OpenCL HD Graphics
  clGetDeviceIDs(NULL, CL_DEVICE_TYPE_ALL, ...)   Success [INTEL]
  clCreateContext(NULL, ...) [default]            Success [INTEL]
  clCreateContextFromType(NULL, CL_DEVICE_TYPE_CPU)  No devices found in platform
  clCreateContextFromType(NULL, CL_DEVICE_TYPE_GPU)  Success (1)
    Platform Name                                 Intel(R) OpenCL HD Graphics
    Device Name                                   Intel(R) Gen9 HD Graphics NEO
  clCreateContextFromType(NULL, CL_DEVICE_TYPE_ACCELERATOR)  No devices found in platform
  clCreateContextFromType(NULL, CL_DEVICE_TYPE_CUSTOM)  No devices found in platform
  clCreateContextFromType(NULL, CL_DEVICE_TYPE_ALL)  Success (1)
    Platform Name                                 Intel(R) OpenCL HD Graphics
    Device Name                                   Intel(R) Gen9 HD Graphics NEO

ICD loader properties
  ICD loader Name                                 OpenCL ICD Loader
  ICD loader Vendor                               OCL Icd free software
  ICD loader Version                              2.2.8
  ICD loader Profile                              OpenCL 1.2
      NOTE:	your OpenCL library declares to support OpenCL 1.2,
                but it seems to support up to OpenCL 2.1 too.
********************************************************************************/

"EU72_k1x1_cn1024_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M1024_activ1_eltwise0_FP32", "1 8 32 5 1 8 1 1 0",
"EU72_k1x1_cn1024_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M1024_activ5_eltwise0_FP32", "1 8 32 5 1 8 1 1 0",
"EU72_k1x1_cn1024_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M126_activ0_eltwise0_FP32", "6 1 16 2 1 1 16 1 0",
"EU72_k1x1_cn1024_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M24_activ0_eltwise0_FP32", "2 1 8 2 1 1 8 1 0",
"EU72_k1x1_cn1024_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M256_activ1_eltwise0_FP32", "8 1 16 2 1 1 16 1 0",
"EU72_k1x1_cn1024_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M256_activ5_eltwise0_FP32", "5 1 16 2 1 1 16 1 0",
"EU72_k1x1_cn1024_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M546_activ0_eltwise0_FP32", "10 1 16 2 1 1 16 1 0",
"EU72_k1x1_cn128_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M126_activ0_eltwise0_FP32", "1 1 16 2 1 1 16 1 0",
"EU72_k1x1_cn128_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M24_activ0_eltwise0_FP32", "1 1 16 2 1 1 16 1 0",
"EU72_k1x1_cn128_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M546_activ0_eltwise0_FP32", "1 1 16 2 1 1 16 1 0",
"EU72_k1x1_cn128_g1_s1x1_d1x1_b1_in48x48_p0x0_num1_M256_activ1_eltwise0_FP32", "1 8 32 5 1 8 1 1 0",
"EU72_k1x1_cn128_g1_s1x1_d1x1_b1_in48x48_p0x0_num1_M256_activ5_eltwise0_FP32", "12 2 16 2 1 1 16 1 0",
"EU72_k1x1_cn128_g1_s1x1_d1x1_b1_in80x80_p0x0_num1_M128_activ1_eltwise0_FP32", "2 8 32 5 1 8 1 1 0",
"EU72_k1x1_cn128_g1_s1x1_d1x1_b1_in80x80_p0x0_num1_M128_activ5_eltwise0_FP32", "8 1 16 2 1 1 16 1 0",
"EU72_k1x1_cn192_g1_s1x1_d1x1_b1_in32x32_p0x0_num1_M16_activ1_eltwise0_FP32", "8 1 16 2 1 1 16 1 0",
"EU72_k1x1_cn192_g1_s1x1_d1x1_b1_in32x32_p0x0_num1_M32_activ1_eltwise0_FP32", "8 1 8 2 1 1 8 1 0",
"EU72_k1x1_cn192_g1_s1x1_d1x1_b1_in32x32_p0x0_num1_M64_activ1_eltwise0_FP32", "8 1 16 2 1 1 16 1 0",
"EU72_k1x1_cn192_g1_s1x1_d1x1_b1_in32x32_p0x0_num1_M96_activ1_eltwise0_FP32", "14 1 16 2 1 1 16 1 0",
"EU72_k1x1_cn256_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M126_activ0_eltwise0_FP32", "1 3 8 2 1 1 8 1 0",
"EU72_k1x1_cn256_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M128_activ1_eltwise0_FP32", "2 1 8 2 1 1 8 1 0",
"EU72_k1x1_cn256_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M128_activ5_eltwise0_FP32", "2 1 16 2 1 1 16 1 0",
"EU72_k1x1_cn256_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M24_activ0_eltwise0_FP32", "1 1 8 2 1 1 8 1 0",
"EU72_k1x1_cn256_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M546_activ0_eltwise0_FP32", "3 1 16 2 1 1 16 1 0",
"EU72_k1x1_cn256_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M64_activ1_eltwise0_FP32", "1 1 8 2 1 1 8 1 0",
"EU72_k1x1_cn256_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M64_activ5_eltwise0_FP32", "2 1 8 2 1 1 8 1 0",
"EU72_k1x1_cn256_g1_s1x1_d1x1_b1_in32x32_p0x0_num1_M128_activ1_eltwise0_FP32", "1 8 32 5 1 8 1 1 0",
"EU72_k1x1_cn256_g1_s1x1_d1x1_b1_in32x32_p0x0_num1_M32_activ1_eltwise0_FP32", "8 1 8 2 1 1 8 1 0",
"EU72_k1x1_cn256_g1_s1x1_d1x1_b1_in32x32_p0x0_num1_M512_activ1_eltwise0_FP32", "1 8 32 5 1 8 1 1 0",
"EU72_k1x1_cn256_g1_s1x1_d1x1_b1_in32x32_p0x0_num1_M512_activ5_eltwise0_FP32", "10 1 16 2 1 1 16 1 0",
"EU72_k1x1_cn256_g1_s1x1_d1x1_b1_in32x32_p0x0_num1_M64_activ1_eltwise0_FP32", "8 1 8 2 1 1 8 1 0",
"EU72_k1x1_cn256_g1_s1x1_d1x1_b1_in48x48_p0x0_num1_M256_activ1_eltwise0_FP32", "1 8 32 5 1 8 1 1 0",
"EU72_k1x1_cn256_g1_s1x1_d1x1_b1_in48x48_p0x0_num1_M256_activ5_eltwise0_FP32", "8 1 16 2 1 1 16 1 0",
"EU72_k1x1_cn32_g1_s1x1_d1x1_b1_in160x160_p0x0_num1_M64_activ1_eltwise0_FP32", "1 8 32 5 1 8 1 1 0",
"EU72_k1x1_cn32_g1_s1x1_d1x1_b1_in160x160_p0x0_num1_M64_activ5_eltwise0_FP32", "1 16 32 5 1 16 1 1 0",
"EU72_k1x1_cn480_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M16_activ1_eltwise0_FP32", "5 1 8 2 1 1 8 1 0",
"EU72_k1x1_cn480_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M192_activ1_eltwise0_FP32", "8 1 16 2 1 1 16 1 0",
"EU72_k1x1_cn480_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M64_activ1_eltwise0_FP32", "7 1 8 2 1 1 8 1 0",
"EU72_k1x1_cn480_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M96_activ1_eltwise0_FP32", "7 2 8 2 1 1 8 1 0",
"EU72_k1x1_cn512_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M1024_activ1_eltwise0_FP32", "10 1 16 2 1 1 16 1 0",
"EU72_k1x1_cn512_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M1024_activ5_eltwise0_FP32", "1 8 32 5 1 8 1 1 0",
"EU72_k1x1_cn512_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M112_activ1_eltwise0_FP32", "8 1 16 2 1 1 16 1 0",
"EU72_k1x1_cn512_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M126_activ0_eltwise0_FP32", "5 1 8 2 1 1 8 1 0",
"EU72_k1x1_cn512_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M128_activ1_eltwise0_FP32", "8 2 8 2 1 1 8 1 0",
"EU72_k1x1_cn512_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M128_activ5_eltwise0_FP32", "5 1 8 2 1 1 8 1 0",
"EU72_k1x1_cn512_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M144_activ1_eltwise0_FP32", "7 1 16 2 1 1 16 1 0",
"EU72_k1x1_cn512_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M160_activ1_eltwise0_FP32", "8 2 8 2 1 1 8 1 0",
"EU72_k1x1_cn512_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M24_activ0_eltwise0_FP32", "3 1 8 2 1 1 8 1 0",
"EU72_k1x1_cn512_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M24_activ1_eltwise0_FP32", "4 1 16 2 1 1 16 1 0",
"EU72_k1x1_cn512_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M32_activ1_eltwise0_FP32", "7 1 8 2 1 1 8 1 0",
"EU72_k1x1_cn512_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M546_activ0_eltwise0_FP32", "3 1 16 2 1 1 16 1 0",
"EU72_k1x1_cn512_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M64_activ1_eltwise0_FP32", "8 1 8 2 1 1 8 1 0",
"EU72_k1x1_cn512_g1_s1x1_d1x1_b1_in32x32_p0x0_num1_M12_activ0_eltwise0_FP32", "6 1 16 2 1 1 16 1 0",
"EU72_k1x1_cn512_g1_s1x1_d1x1_b1_in32x32_p0x0_num1_M273_activ0_eltwise0_FP32", "10 1 16 2 1 1 16 1 0",
"EU72_k1x1_cn512_g1_s1x1_d1x1_b1_in32x32_p0x0_num1_M512_activ1_eltwise0_FP32", "10 1 16 2 1 1 16 1 0",
"EU72_k1x1_cn512_g1_s1x1_d1x1_b1_in32x32_p0x0_num1_M512_activ5_eltwise0_FP32", "10 1 16 2 1 1 16 1 0",
"EU72_k1x1_cn512_g1_s1x1_d1x1_b1_in32x32_p0x0_num1_M63_activ0_eltwise0_FP32", "7 1 16 2 1 1 16 1 0",
"EU72_k1x1_cn528_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M128_activ1_eltwise0_FP32", "8 1 16 2 1 1 16 1 0",
"EU72_k1x1_cn528_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M160_activ1_eltwise0_FP32", "8 2 8 2 1 1 8 1 0",
"EU72_k1x1_cn528_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M256_activ1_eltwise0_FP32", "14 1 16 2 1 1 16 1 0",
"EU72_k1x1_cn528_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M32_activ1_eltwise0_FP32", "8 1 8 2 1 1 8 1 0",
"EU72_k1x1_cn64_g1_s1x1_d1x1_b1_in64x64_p0x0_num1_M64_activ1_eltwise0_FP32", "1 8 32 5 1 8 1 1 0",
"EU72_k1x1_cn64_g1_s1x1_d1x1_b1_in80x80_p0x0_num1_M128_activ1_eltwise0_FP32", "1 8 32 5 1 8 1 1 0",
"EU72_k1x1_cn64_g1_s1x1_d1x1_b1_in80x80_p0x0_num1_M128_activ5_eltwise0_FP32", "2 8 32 5 1 8 1 1 0",
"EU72_k1x1_cn832_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M128_activ1_eltwise0_FP32", "7 1 8 2 1 1 8 1 0",
"EU72_k1x1_cn832_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M160_activ1_eltwise0_FP32", "7 1 8 2 1 1 8 1 0",
"EU72_k1x1_cn832_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M192_activ1_eltwise0_FP32", "7 1 8 2 1 1 8 1 0",
"EU72_k1x1_cn832_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M256_activ1_eltwise0_FP32", "7 1 8 2 1 1 8 1 0",
"EU72_k1x1_cn832_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M32_activ1_eltwise0_FP32", "5 1 8 2 1 1 8 1 0",
"EU72_k1x1_cn832_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M384_activ1_eltwise0_FP32", "7 2 8 2 1 1 8 1 0",
"EU72_k1x1_cn832_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M48_activ1_eltwise0_FP32", "5 1 8 2 1 1 8 1 0",
"EU72_k3x3_cn1024_g1024_s1x1_d1x1_b1_in16x16_p1x1_num1_M1_activ1_eltwise0_FP32", "1 1 1 6 1 1 1 0 0",
"EU72_k3x3_cn1024_g1024_s1x1_d1x1_b1_in16x16_p1x1_num1_M1_activ5_eltwise0_FP32", "1 1 1 6 1 1 1 0 0",
"EU72_k3x3_cn112_g1_s1x1_d1x1_b1_in16x16_p1x1_num1_M224_activ1_eltwise0_FP32", "14 1 16 2 1 1 16 1 0",
"EU72_k3x3_cn128_g128_s1x1_d1x1_b1_in80x80_p1x1_num1_M1_activ1_eltwise0_FP32", "1 1 1 6 1 1 1 0 0",
"EU72_k3x3_cn128_g128_s1x1_d1x1_b1_in80x80_p1x1_num1_M1_activ5_eltwise0_FP32", "1 1 1 6 1 1 1 0 0",
"EU72_k3x3_cn128_g128_s2x2_d1x1_b1_in80x80_p1x1_num1_M1_activ1_eltwise0_FP32", "1 1 1 6 1 1 1 0 0",
"EU72_k3x3_cn128_g128_s2x2_d1x1_b1_in80x80_p1x1_num1_M1_activ5_eltwise0_FP32", "1 1 1 6 1 1 1 0 0",
"EU72_k3x3_cn128_g1_s1x1_d1x1_b1_in16x16_p1x1_num1_M256_activ1_eltwise0_FP32", "14 2 16 2 1 1 16 1 0",
"EU72_k3x3_cn128_g1_s1x1_d1x1_b1_in32x32_p1x1_num1_M192_activ1_eltwise0_FP32", "14 2 16 2 1 1 16 1 0",
"EU72_k3x3_cn128_g1_s2x2_d1x1_b1_in16x16_p0x0_num1_M256_activ5_eltwise0_FP32", "3 1 8 2 1 1 8 1 0",
"EU72_k3x3_cn128_g1_s2x2_d1x1_b1_in16x16_p1x1_num1_M256_activ1_eltwise0_FP32", "1 1 16 2 1 1 16 1 0",
"EU72_k3x3_cn144_g1_s1x1_d1x1_b1_in16x16_p1x1_num1_M288_activ1_eltwise0_FP32", "14 2 16 2 1 1 16 1 0",
"EU72_k3x3_cn160_g1_s1x1_d1x1_b1_in16x16_p1x1_num1_M320_activ1_eltwise0_FP32", "14 1 16 2 1 1 16 1 0",
"EU72_k3x3_cn192_g1_s1x1_d1x1_b1_in16x16_p1x1_num1_M384_activ1_eltwise0_FP32", "7 1 16 2 1 1 16 1 0",
"EU72_k3x3_cn256_g1_s2x2_d1x1_b1_in16x16_p0x0_num1_M512_activ5_eltwise0_FP32", "3 1 16 2 1 1 16 1 0",
"EU72_k3x3_cn256_g1_s2x2_d1x1_b1_in16x16_p1x1_num1_M512_activ1_eltwise0_FP32", "5 1 16 2 1 1 16 1 0",
"EU72_k3x3_cn256_g256_s1x1_d1x1_b1_in48x48_p1x1_num1_M1_activ1_eltwise0_FP32", "1 1 1 6 1 1 1 0 0",
"EU72_k3x3_cn256_g256_s1x1_d1x1_b1_in48x48_p1x1_num1_M1_activ5_eltwise0_FP32", "1 1 1 6 1 1 1 0 0",
"EU72_k3x3_cn256_g256_s2x2_d1x1_b1_in48x48_p0x0_num1_M1_activ5_eltwise0_FP32", "1 1 1 6 1 1 1 0 0",
"EU72_k3x3_cn256_g256_s2x2_d1x1_b1_in48x48_p1x1_num1_M1_activ1_eltwise0_FP32", "1 1 1 6 1 1 1 0 0",
"EU72_k3x3_cn32_g32_s1x1_d1x1_b1_in160x160_p1x1_num1_M1_activ1_eltwise0_FP32", "1 1 1 6 1 1 1 0 0",
"EU72_k3x3_cn32_g32_s1x1_d1x1_b1_in160x160_p1x1_num1_M1_activ5_eltwise0_FP32", "1 1 1 6 1 1 1 0 0",
"EU72_k3x3_cn3_g1_s2x2_d1x1_b1_in256x256_p0x0_num1_M32_activ5_eltwise0_FP32", "1 8 32 5 1 8 1 1 0",
"EU72_k3x3_cn3_g1_s2x2_d1x1_b1_in256x256_p1x1_num1_M32_activ1_eltwise0_FP32", "1 8 32 5 1 8 1 1 0",
"EU72_k3x3_cn512_g512_s1x1_d1x1_b1_in32x32_p1x1_num1_M1_activ1_eltwise0_FP32", "1 1 1 6 1 1 1 0 0",
"EU72_k3x3_cn512_g512_s1x1_d1x1_b1_in32x32_p1x1_num1_M1_activ5_eltwise0_FP32", "1 1 1 6 1 1 1 0 0",
"EU72_k3x3_cn512_g512_s2x2_d1x1_b1_in32x32_p1x1_num1_M1_activ1_eltwise0_FP32", "1 1 1 6 1 1 1 0 0",
"EU72_k3x3_cn512_g512_s2x2_d1x1_b1_in32x32_p1x1_num1_M1_activ5_eltwise0_FP32", "1 1 1 6 1 1 1 0 0",
"EU72_k3x3_cn64_g1_s1x1_d1x1_b1_in64x64_p1x1_num1_M192_activ1_eltwise0_FP32", "14 2 16 2 1 1 16 1 0",
"EU72_k3x3_cn64_g1_s2x2_d1x1_b1_in16x16_p0x0_num1_M128_activ5_eltwise0_FP32", "1 1 8 2 1 1 8 1 0",
"EU72_k3x3_cn64_g1_s2x2_d1x1_b1_in16x16_p1x1_num1_M128_activ1_eltwise0_FP32", "1 1 16 2 1 1 16 1 0",
"EU72_k3x3_cn64_g64_s2x2_d1x1_b1_in160x160_p0x0_num1_M1_activ5_eltwise0_FP32", "1 1 1 6 1 1 1 0 0",
"EU72_k3x3_cn64_g64_s2x2_d1x1_b1_in160x160_p1x1_num1_M1_activ1_eltwise0_FP32", "1 1 1 6 1 1 1 0 0",
"EU72_k3x3_cn96_g1_s1x1_d1x1_b1_in16x16_p1x1_num1_M208_activ1_eltwise0_FP32", "7 2 16 2 1 1 16 1 0",
"EU72_k3x3_cn96_g1_s1x1_d1x1_b1_in32x32_p1x1_num1_M128_activ1_eltwise0_FP32", "7 2 16 2 1 1 16 1 0",
"EU72_k5x5_cn16_g1_s1x1_d1x1_b1_in16x16_p2x2_num1_M48_activ1_eltwise0_FP32", "4 2 8 2 1 1 8 1 0",
"EU72_k5x5_cn16_g1_s1x1_d1x1_b1_in32x32_p2x2_num1_M32_activ1_eltwise0_FP32", "6 1 16 2 1 1 16 1 0",
"EU72_k5x5_cn24_g1_s1x1_d1x1_b1_in16x16_p2x2_num1_M64_activ1_eltwise0_FP32", "5 1 16 2 1 1 16 1 0",
"EU72_k5x5_cn32_g1_s1x1_d1x1_b1_in16x16_p2x2_num1_M128_activ1_eltwise0_FP32", "5 1 16 2 1 1 16 1 0",
"EU72_k5x5_cn32_g1_s1x1_d1x1_b1_in16x16_p2x2_num1_M64_activ1_eltwise0_FP32", "6 1 16 2 1 1 16 1 0",
"EU72_k5x5_cn32_g1_s1x1_d1x1_b1_in32x32_p2x2_num1_M96_activ1_eltwise0_FP32", "7 2 16 2 1 1 16 1 0",
"EU72_k5x5_cn48_g1_s1x1_d1x1_b1_in16x16_p2x2_num1_M128_activ1_eltwise0_FP32", "4 1 16 2 1 1 16 1 0",
"EU72_k7x7_cn3_g1_s2x2_d1x1_b1_in224x224_p3x3_num1_M64_activ1_eltwise0_FP32", "4 3 16 2 1 1 16 1 0",
// Below is the information for OpenCL based on which these configurations tuned
/*******************************************************************************
Number of platforms                               1
  Platform Name                                   Intel(R) OpenCL HD Graphics
  Platform Vendor                                 Intel(R) Corporation
  Platform Version                                OpenCL 2.1
  Platform Profile                                FULL_PROFILE
  Platform Extensions                             cl_khr_3d_image_writes cl_khr_byte_addressable_store cl_khr_fp16 cl_khr_depth_images cl_khr_global_int32_base_atomics cl_khr_global_int32_extended_atomics cl_khr_icd cl_khr_image2d_from_buffer cl_khr_local_int32_base_atomics cl_khr_local_int32_extended_atomics cl_intel_subgroups cl_intel_required_subgroup_size cl_intel_subgroups_short cl_khr_spir cl_intel_accelerator cl_intel_media_block_io cl_intel_driver_diagnostics cl_intel_device_side_avc_motion_estimation cl_khr_priority_hints cl_khr_throttle_hints cl_khr_create_command_queue cl_khr_subgroups cl_khr_il_program cl_khr_mipmap_image cl_khr_mipmap_image_writes cl_khr_fp64 cl_intel_planar_yuv cl_intel_packed_yuv cl_intel_motion_estimation cl_intel_advanced_motion_estimation
  Platform Host timer resolution                  1ns
  Platform Extensions function suffix             INTEL

  Platform Name                                   Intel(R) OpenCL HD Graphics
Number of devices                                 1
  Device Name                                     Intel(R) Gen9 HD Graphics NEO
  Device Vendor                                   Intel(R) Corporation
  Device Vendor ID                                0x8086
  Device Version                                  OpenCL 2.1 NEO
  Driver Version                                  18.21.10858
  Device OpenCL C Version                         OpenCL C 2.0
  Device Type                                     GPU
  Device Profile                                  FULL_PROFILE
  Max compute units                               48
  Max clock frequency                             950MHz
  Device Partition                                (core)
    Max number of sub-devices                     0
    Supported partition types                     None
  Max work item dimensions                        3
  Max work item sizes                             256x256x256
  Max work group size                             256
  Preferred work group size multiple              32
  Max sub-groups per work group                   32
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
  Global memory size                              13364170752 (12.45GiB)
  Error Correction support                        No
  Max memory allocation                           4294959104 (4GiB)
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
  Preferred total size of global vars             4294959104 (4GiB)
  Global Memory cache type                        Read/Write
  Global Memory cache size                        1048576
  Global Memory cache line                        64 bytes
  Image support                                   Yes
    Max number of samplers per kernel             16
    Max size for 1D images from buffer            268434944 pixels
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
  Max constant buffer size                        4294959104 (4GiB)
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
    Sub-group independent forward progress        Yes
    IL version                                    SPIR-V_1.0
    SPIR versions                                 1.2
  printf() buffer size                            4194304 (4MiB)
  Built-in kernels                                block_motion_estimate_intel;block_advanced_motion_estimate_check_intel;block_advanced_motion_estimate_bidirectional_check_intel;
  Motion Estimation accelerator version	(Intel)   2
  Device Available                                Yes
  Compiler Available                              Yes
  Linker Available                                Yes
  Device Extensions                               cl_khr_3d_image_writes cl_khr_byte_addressable_store cl_khr_fp16 cl_khr_depth_images cl_khr_global_int32_base_atomics cl_khr_global_int32_extended_atomics cl_khr_icd cl_khr_image2d_from_buffer cl_khr_local_int32_base_atomics cl_khr_local_int32_extended_atomics cl_intel_subgroups cl_intel_required_subgroup_size cl_intel_subgroups_short cl_khr_spir cl_intel_accelerator cl_intel_media_block_io cl_intel_driver_diagnostics cl_intel_device_side_avc_motion_estimation cl_khr_priority_hints cl_khr_throttle_hints cl_khr_create_command_queue cl_khr_subgroups cl_khr_il_program cl_khr_mipmap_image cl_khr_mipmap_image_writes cl_khr_fp64 cl_intel_planar_yuv cl_intel_packed_yuv cl_intel_motion_estimation cl_intel_advanced_motion_estimation

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
"EU48_k1x1_cn1024_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M1024_activ1_eltwise0_FP32", "2 8 32 5 1 8 1 1 0",
"EU48_k1x1_cn1024_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M1024_activ5_eltwise0_FP32", "2 8 32 5 1 8 1 1 0",
"EU48_k1x1_cn1024_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M126_activ0_eltwise0_FP32", "10 1 16 2 1 1 16 1 0",
"EU48_k1x1_cn1024_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M24_activ0_eltwise0_FP32", "5 1 8 2 1 1 8 1 0",
"EU48_k1x1_cn1024_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M256_activ1_eltwise0_FP32", "10 1 16 2 1 1 16 1 0",
"EU48_k1x1_cn1024_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M256_activ5_eltwise0_FP32", "10 1 16 2 1 1 16 1 0",
"EU48_k1x1_cn1024_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M546_activ0_eltwise0_FP32", "10 1 16 2 1 1 16 1 0",
"EU48_k1x1_cn128_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M126_activ0_eltwise0_FP32", "1 1 8 2 1 1 8 1 0",
"EU48_k1x1_cn128_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M24_activ0_eltwise0_FP32", "1 1 8 2 1 1 8 1 0",
"EU48_k1x1_cn128_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M546_activ0_eltwise0_FP32", "1 1 16 2 1 1 16 1 0",
"EU48_k1x1_cn128_g1_s1x1_d1x1_b1_in48x48_p0x0_num1_M256_activ1_eltwise0_FP32", "2 8 32 5 1 8 1 1 0",
"EU48_k1x1_cn128_g1_s1x1_d1x1_b1_in48x48_p0x0_num1_M256_activ5_eltwise0_FP32", "1 8 32 5 1 8 1 1 0",
"EU48_k1x1_cn128_g1_s1x1_d1x1_b1_in80x80_p0x0_num1_M128_activ1_eltwise0_FP32", "1 8 32 5 1 8 1 1 0",
"EU48_k1x1_cn128_g1_s1x1_d1x1_b1_in80x80_p0x0_num1_M128_activ5_eltwise0_FP32", "1 8 32 5 1 8 1 1 0",
"EU48_k1x1_cn192_g1_s1x1_d1x1_b1_in32x32_p0x0_num1_M16_activ1_eltwise0_FP32", "14 1 16 2 1 1 16 1 0",
"EU48_k1x1_cn192_g1_s1x1_d1x1_b1_in32x32_p0x0_num1_M32_activ1_eltwise0_FP32", "14 1 16 2 1 1 16 1 0",
"EU48_k1x1_cn192_g1_s1x1_d1x1_b1_in32x32_p0x0_num1_M64_activ1_eltwise0_FP32", "14 1 16 2 1 1 16 1 0",
"EU48_k1x1_cn192_g1_s1x1_d1x1_b1_in32x32_p0x0_num1_M96_activ1_eltwise0_FP32", "14 1 16 2 1 1 16 1 0",
"EU48_k1x1_cn256_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M126_activ0_eltwise0_FP32", "3 1 8 2 1 1 8 1 0",
"EU48_k1x1_cn256_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M128_activ1_eltwise0_FP32", "3 1 8 2 1 1 8 1 0",
"EU48_k1x1_cn256_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M128_activ5_eltwise0_FP32", "3 1 8 2 1 1 8 1 0",
"EU48_k1x1_cn256_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M24_activ0_eltwise0_FP32", "1 1 8 2 1 1 8 1 0",
"EU48_k1x1_cn256_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M546_activ0_eltwise0_FP32", "3 1 16 2 1 1 16 1 0",
"EU48_k1x1_cn256_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M64_activ1_eltwise0_FP32", "2 1 8 2 1 1 8 1 0",
"EU48_k1x1_cn256_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M64_activ5_eltwise0_FP32", "1 1 8 2 1 1 8 1 0",
"EU48_k1x1_cn256_g1_s1x1_d1x1_b1_in32x32_p0x0_num1_M128_activ1_eltwise0_FP32", "1 8 32 5 1 8 1 1 0",
"EU48_k1x1_cn256_g1_s1x1_d1x1_b1_in32x32_p0x0_num1_M32_activ1_eltwise0_FP32", "1 8 32 5 1 8 1 1 0",
"EU48_k1x1_cn256_g1_s1x1_d1x1_b1_in32x32_p0x0_num1_M512_activ1_eltwise0_FP32", "2 8 32 5 1 8 1 1 0",
"EU48_k1x1_cn256_g1_s1x1_d1x1_b1_in32x32_p0x0_num1_M512_activ5_eltwise0_FP32", "1 8 32 5 1 8 1 1 0",
"EU48_k1x1_cn256_g1_s1x1_d1x1_b1_in32x32_p0x0_num1_M64_activ1_eltwise0_FP32", "1 8 32 5 1 8 1 1 0",
"EU48_k1x1_cn256_g1_s1x1_d1x1_b1_in48x48_p0x0_num1_M256_activ1_eltwise0_FP32", "1 8 32 5 1 8 1 1 0",
"EU48_k1x1_cn256_g1_s1x1_d1x1_b1_in48x48_p0x0_num1_M256_activ5_eltwise0_FP32", "2 8 32 5 1 8 1 1 0",
"EU48_k1x1_cn32_g1_s1x1_d1x1_b1_in160x160_p0x0_num1_M64_activ1_eltwise0_FP32", "1 16 32 5 1 16 1 1 0",
"EU48_k1x1_cn32_g1_s1x1_d1x1_b1_in160x160_p0x0_num1_M64_activ5_eltwise0_FP32", "1 16 32 5 1 16 1 1 0",
"EU48_k1x1_cn480_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M16_activ1_eltwise0_FP32", "8 1 8 2 1 1 8 1 0",
"EU48_k1x1_cn480_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M192_activ1_eltwise0_FP32", "14 1 16 2 1 1 16 1 0",
"EU48_k1x1_cn480_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M64_activ1_eltwise0_FP32", "14 1 16 2 1 1 16 1 0",
"EU48_k1x1_cn480_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M96_activ1_eltwise0_FP32", "14 1 16 2 1 1 16 1 0",
"EU48_k1x1_cn512_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M1024_activ1_eltwise0_FP32", "10 1 16 2 1 1 16 1 0",
"EU48_k1x1_cn512_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M1024_activ5_eltwise0_FP32", "2 8 32 5 1 8 1 1 0",
"EU48_k1x1_cn512_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M112_activ1_eltwise0_FP32", "1 8 32 5 1 8 1 1 0",
"EU48_k1x1_cn512_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M126_activ0_eltwise0_FP32", "5 1 8 2 1 1 8 1 0",
"EU48_k1x1_cn512_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M128_activ1_eltwise0_FP32", "14 1 16 2 1 1 16 1 0",
"EU48_k1x1_cn512_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M128_activ5_eltwise0_FP32", "5 1 8 2 1 1 8 1 0",
"EU48_k1x1_cn512_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M144_activ1_eltwise0_FP32", "1 8 32 5 1 8 1 1 0",
"EU48_k1x1_cn512_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M160_activ1_eltwise0_FP32", "14 1 16 2 1 1 16 1 0",
"EU48_k1x1_cn512_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M24_activ0_eltwise0_FP32", "4 1 8 2 1 1 8 1 0",
"EU48_k1x1_cn512_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M24_activ1_eltwise0_FP32", "8 1 8 2 1 1 8 1 0",
"EU48_k1x1_cn512_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M32_activ1_eltwise0_FP32", "8 1 8 2 1 1 8 1 0",
"EU48_k1x1_cn512_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M546_activ0_eltwise0_FP32", "5 1 16 2 1 1 16 1 0",
"EU48_k1x1_cn512_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M64_activ1_eltwise0_FP32", "14 1 16 2 1 1 16 1 0",
"EU48_k1x1_cn512_g1_s1x1_d1x1_b1_in32x32_p0x0_num1_M12_activ0_eltwise0_FP32", "9 1 16 2 1 1 16 1 0",
"EU48_k1x1_cn512_g1_s1x1_d1x1_b1_in32x32_p0x0_num1_M273_activ0_eltwise0_FP32", "10 1 16 2 1 1 16 1 0",
"EU48_k1x1_cn512_g1_s1x1_d1x1_b1_in32x32_p0x0_num1_M512_activ1_eltwise0_FP32", "1 8 32 5 1 8 1 1 0",
"EU48_k1x1_cn512_g1_s1x1_d1x1_b1_in32x32_p0x0_num1_M512_activ5_eltwise0_FP32", "1 8 32 5 1 8 1 1 0",
"EU48_k1x1_cn512_g1_s1x1_d1x1_b1_in32x32_p0x0_num1_M63_activ0_eltwise0_FP32", "10 1 16 2 1 1 16 1 0",
"EU48_k1x1_cn528_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M128_activ1_eltwise0_FP32", "14 1 16 2 1 1 16 1 0",
"EU48_k1x1_cn528_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M160_activ1_eltwise0_FP32", "14 1 16 2 1 1 16 1 0",
"EU48_k1x1_cn528_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M256_activ1_eltwise0_FP32", "1 8 32 5 1 8 1 1 0",
"EU48_k1x1_cn528_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M32_activ1_eltwise0_FP32", "7 1 16 2 1 1 16 1 0",
"EU48_k1x1_cn64_g1_s1x1_d1x1_b1_in64x64_p0x0_num1_M64_activ1_eltwise0_FP32", "1 8 32 5 1 8 1 1 0",
"EU48_k1x1_cn64_g1_s1x1_d1x1_b1_in80x80_p0x0_num1_M128_activ1_eltwise0_FP32", "1 8 32 5 1 8 1 1 0",
"EU48_k1x1_cn64_g1_s1x1_d1x1_b1_in80x80_p0x0_num1_M128_activ5_eltwise0_FP32", "2 8 32 5 1 8 1 1 0",
"EU48_k1x1_cn832_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M128_activ1_eltwise0_FP32", "7 2 8 2 1 1 8 1 0",
"EU48_k1x1_cn832_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M160_activ1_eltwise0_FP32", "7 1 16 2 1 1 16 1 0",
"EU48_k1x1_cn832_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M192_activ1_eltwise0_FP32", "7 1 16 2 1 1 16 1 0",
"EU48_k1x1_cn832_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M256_activ1_eltwise0_FP32", "7 1 16 2 1 1 16 1 0",
"EU48_k1x1_cn832_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M32_activ1_eltwise0_FP32", "4 1 8 2 1 1 8 1 0",
"EU48_k1x1_cn832_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M384_activ1_eltwise0_FP32", "7 1 16 2 1 1 16 1 0",
"EU48_k1x1_cn832_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M48_activ1_eltwise0_FP32", "4 1 8 2 1 1 8 1 0",
"EU48_k3x3_cn1024_g1024_s1x1_d1x1_b1_in16x16_p1x1_num1_M1_activ1_eltwise0_FP32", "1 1 1 6 1 1 1 0 0",
"EU48_k3x3_cn1024_g1024_s1x1_d1x1_b1_in16x16_p1x1_num1_M1_activ5_eltwise0_FP32", "1 1 1 6 1 1 1 0 0",
"EU48_k3x3_cn112_g1_s1x1_d1x1_b1_in16x16_p1x1_num1_M224_activ1_eltwise0_FP32", "14 1 16 2 1 1 16 1 0",
"EU48_k3x3_cn128_g128_s1x1_d1x1_b1_in80x80_p1x1_num1_M1_activ1_eltwise0_FP32", "1 1 1 6 1 1 1 0 0",
"EU48_k3x3_cn128_g128_s1x1_d1x1_b1_in80x80_p1x1_num1_M1_activ5_eltwise0_FP32", "1 1 1 6 1 1 1 0 0",
"EU48_k3x3_cn128_g128_s2x2_d1x1_b1_in80x80_p1x1_num1_M1_activ1_eltwise0_FP32", "1 1 1 6 1 1 1 0 0",
"EU48_k3x3_cn128_g128_s2x2_d1x1_b1_in80x80_p1x1_num1_M1_activ5_eltwise0_FP32", "1 1 1 6 1 1 1 0 0",
"EU48_k3x3_cn128_g1_s1x1_d1x1_b1_in16x16_p1x1_num1_M256_activ1_eltwise0_FP32", "14 1 16 2 1 1 16 1 0",
"EU48_k3x3_cn128_g1_s1x1_d1x1_b1_in32x32_p1x1_num1_M192_activ1_eltwise0_FP32", "14 2 16 2 1 1 16 1 0",
"EU48_k3x3_cn128_g1_s2x2_d1x1_b1_in16x16_p0x0_num1_M256_activ5_eltwise0_FP32", "2 1 16 2 1 1 16 1 0",
"EU48_k3x3_cn128_g1_s2x2_d1x1_b1_in16x16_p1x1_num1_M256_activ1_eltwise0_FP32", "3 1 8 2 1 1 8 1 0",
"EU48_k3x3_cn144_g1_s1x1_d1x1_b1_in16x16_p1x1_num1_M288_activ1_eltwise0_FP32", "14 1 16 2 1 1 16 1 0",
"EU48_k3x3_cn160_g1_s1x1_d1x1_b1_in16x16_p1x1_num1_M320_activ1_eltwise0_FP32", "14 1 16 2 1 1 16 1 0",
"EU48_k3x3_cn192_g1_s1x1_d1x1_b1_in16x16_p1x1_num1_M384_activ1_eltwise0_FP32", "7 1 16 2 1 1 16 1 0",
"EU48_k3x3_cn256_g1_s2x2_d1x1_b1_in16x16_p0x0_num1_M512_activ5_eltwise0_FP32", "5 1 16 2 1 1 16 1 0",
"EU48_k3x3_cn256_g1_s2x2_d1x1_b1_in16x16_p1x1_num1_M512_activ1_eltwise0_FP32", "5 1 16 2 1 1 16 1 0",
"EU48_k3x3_cn256_g256_s1x1_d1x1_b1_in48x48_p1x1_num1_M1_activ1_eltwise0_FP32", "1 1 1 6 1 1 1 0 0",
"EU48_k3x3_cn256_g256_s1x1_d1x1_b1_in48x48_p1x1_num1_M1_activ5_eltwise0_FP32", "1 1 1 6 1 1 1 0 0",
"EU48_k3x3_cn256_g256_s2x2_d1x1_b1_in48x48_p0x0_num1_M1_activ5_eltwise0_FP32", "1 1 1 6 1 1 1 0 0",
"EU48_k3x3_cn256_g256_s2x2_d1x1_b1_in48x48_p1x1_num1_M1_activ1_eltwise0_FP32", "1 1 1 6 1 1 1 0 0",
"EU48_k3x3_cn32_g32_s1x1_d1x1_b1_in160x160_p1x1_num1_M1_activ1_eltwise0_FP32", "1 1 1 6 1 1 1 0 0",
"EU48_k3x3_cn32_g32_s1x1_d1x1_b1_in160x160_p1x1_num1_M1_activ5_eltwise0_FP32", "1 1 1 6 1 1 1 0 0",
"EU48_k3x3_cn3_g1_s2x2_d1x1_b1_in256x256_p0x0_num1_M32_activ5_eltwise0_FP32", "1 8 32 5 1 8 1 1 0",
"EU48_k3x3_cn3_g1_s2x2_d1x1_b1_in256x256_p1x1_num1_M32_activ1_eltwise0_FP32", "1 8 32 5 1 8 1 1 0",
"EU48_k3x3_cn512_g512_s1x1_d1x1_b1_in32x32_p1x1_num1_M1_activ1_eltwise0_FP32", "1 1 1 6 1 1 1 0 0",
"EU48_k3x3_cn512_g512_s1x1_d1x1_b1_in32x32_p1x1_num1_M1_activ5_eltwise0_FP32", "1 1 1 6 1 1 1 0 0",
"EU48_k3x3_cn512_g512_s2x2_d1x1_b1_in32x32_p1x1_num1_M1_activ1_eltwise0_FP32", "1 1 1 6 1 1 1 0 0",
"EU48_k3x3_cn512_g512_s2x2_d1x1_b1_in32x32_p1x1_num1_M1_activ5_eltwise0_FP32", "1 1 1 6 1 1 1 0 0",
"EU48_k3x3_cn64_g1_s1x1_d1x1_b1_in64x64_p1x1_num1_M192_activ1_eltwise0_FP32", "14 2 16 2 1 1 16 1 0",
"EU48_k3x3_cn64_g1_s2x2_d1x1_b1_in16x16_p0x0_num1_M128_activ5_eltwise0_FP32", "1 1 8 2 1 1 8 1 0",
"EU48_k3x3_cn64_g1_s2x2_d1x1_b1_in16x16_p1x1_num1_M128_activ1_eltwise0_FP32", "1 1 8 2 1 1 8 1 0",
"EU48_k3x3_cn64_g64_s2x2_d1x1_b1_in160x160_p0x0_num1_M1_activ5_eltwise0_FP32", "1 1 1 6 1 1 1 0 0",
"EU48_k3x3_cn64_g64_s2x2_d1x1_b1_in160x160_p1x1_num1_M1_activ1_eltwise0_FP32", "1 1 1 6 1 1 1 0 0",
"EU48_k3x3_cn96_g1_s1x1_d1x1_b1_in16x16_p1x1_num1_M208_activ1_eltwise0_FP32", "14 1 16 2 1 1 16 1 0",
"EU48_k3x3_cn96_g1_s1x1_d1x1_b1_in32x32_p1x1_num1_M128_activ1_eltwise0_FP32", "14 2 16 2 1 1 16 1 0",
"EU48_k5x5_cn16_g1_s1x1_d1x1_b1_in16x16_p2x2_num1_M48_activ1_eltwise0_FP32", "4 2 8 2 1 1 8 1 0",
"EU48_k5x5_cn16_g1_s1x1_d1x1_b1_in32x32_p2x2_num1_M32_activ1_eltwise0_FP32", "4 3 8 2 1 1 8 1 0",
"EU48_k5x5_cn24_g1_s1x1_d1x1_b1_in16x16_p2x2_num1_M64_activ1_eltwise0_FP32", "4 1 16 2 1 1 16 1 0",
"EU48_k5x5_cn32_g1_s1x1_d1x1_b1_in16x16_p2x2_num1_M128_activ1_eltwise0_FP32", "7 1 16 2 1 1 16 1 0",
"EU48_k5x5_cn32_g1_s1x1_d1x1_b1_in16x16_p2x2_num1_M64_activ1_eltwise0_FP32", "4 2 8 2 1 1 8 1 0",
"EU48_k5x5_cn32_g1_s1x1_d1x1_b1_in32x32_p2x2_num1_M96_activ1_eltwise0_FP32", "4 7 8 2 1 1 8 1 0",
"EU48_k5x5_cn48_g1_s1x1_d1x1_b1_in16x16_p2x2_num1_M128_activ1_eltwise0_FP32", "4 1 16 2 1 1 16 1 0",
"EU48_k7x7_cn3_g1_s2x2_d1x1_b1_in224x224_p3x3_num1_M64_activ1_eltwise0_FP32", "5 4 16 2 1 1 16 1 0",
"EU48_k11x11_cn3_g1_s4x4_d1x1_b1_in240x240_p0x0_num1_M96_activ1_eltwise0_FP32", "1 8 32 5 1 8 1 1 0",
"EU48_k3x3_cn256_g1_s1x1_d1x1_b1_in16x16_p1x1_num1_M384_activ1_eltwise0_FP32", "13 1 16 2 1 1 16 1 0",
"EU48_k3x3_cn384_g2_s1x1_d1x1_b1_in16x16_p1x1_num1_M128_activ1_eltwise0_FP32", "7 1 16 2 1 1 16 1 0",
"EU48_k3x3_cn384_g2_s1x1_d1x1_b1_in16x16_p1x1_num1_M192_activ1_eltwise0_FP32", "7 1 16 2 1 1 16 1 0",
"EU48_k5x5_cn96_g2_s1x1_d1x1_b1_in32x32_p2x2_num1_M128_activ1_eltwise0_FP32", "3 9 8 2 1 1 8 1 0",
// Below is the information for OpenCL based on which these configurations tuned
/*******************************************************************************
Number of platforms                               1
  Platform Name                                   Intel(R) OpenCL HD Graphics
  Platform Vendor                                 Intel(R) Corporation
  Platform Version                                OpenCL 2.1
  Platform Profile                                FULL_PROFILE
  Platform Extensions                             cl_khr_3d_image_writes cl_khr_byte_addressable_store cl_khr_fp16 cl_khr_depth_images cl_khr_global_int32_base_atomics cl_khr_global_int32_extended_atomics cl_khr_icd cl_khr_image2d_from_buffer cl_khr_local_int32_base_atomics cl_khr_local_int32_extended_atomics cl_intel_subgroups cl_intel_required_subgroup_size cl_intel_subgroups_short cl_khr_spir cl_intel_accelerator cl_intel_media_block_io cl_intel_driver_diagnostics cl_intel_device_side_avc_motion_estimation cl_khr_priority_hints cl_khr_throttle_hints cl_khr_create_command_queue cl_khr_fp64 cl_khr_subgroups cl_khr_il_program cl_khr_mipmap_image cl_khr_mipmap_image_writes cl_intel_planar_yuv cl_intel_packed_yuv cl_intel_motion_estimation cl_intel_advanced_motion_estimation cl_intel_va_api_media_sharing
  Platform Host timer resolution                  1ns
  Platform Extensions function suffix             INTEL

  Platform Name                                   Intel(R) OpenCL HD Graphics
Number of devices                                 1
  Device Name                                     Intel(R) Gen9 HD Graphics NEO
  Device Vendor                                   Intel(R) Corporation
  Device Vendor ID                                0x8086
  Device Version                                  OpenCL 2.1 NEO
  Driver Version                                  18.23.10915
  Device OpenCL C Version                         OpenCL C 2.0
  Device Type                                     GPU
  Device Profile                                  FULL_PROFILE
  Max compute units                               24
  Max clock frequency                             1150MHz
  Device Partition                                (core)
    Max number of sub-devices                     0
    Supported partition types                     None
  Max work item dimensions                        3
  Max work item sizes                             256x256x256
  Max work group size                             256
  Preferred work group size multiple              32
  Max sub-groups per work group                   32
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
  Global memory size                              6575288320 (6.124GiB)
  Error Correction support                        No
  Max memory allocation                           3287644160 (3.062GiB)
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
  Preferred total size of global vars             3287644160 (3.062GiB)
  Global Memory cache type                        Read/Write
  Global Memory cache size                        524288
  Global Memory cache line                        64 bytes
  Image support                                   Yes
    Max number of samplers per kernel             16
    Max size for 1D images from buffer            205477760 pixels
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
  Max constant buffer size                        3287644160 (3.062GiB)
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
    Sub-group independent forward progress        Yes
    IL version                                    SPIR-V_1.0
    SPIR versions                                 1.2
  printf() buffer size                            4194304 (4MiB)
  Built-in kernels                                block_motion_estimate_intel;block_advanced_motion_estimate_check_intel;block_advanced_motion_estimate_bidirectional_check_intel;
  Motion Estimation accelerator version	(Intel)   2
  Device Available                                Yes
  Compiler Available                              Yes
  Linker Available                                Yes
  Device Extensions                               cl_khr_3d_image_writes cl_khr_byte_addressable_store cl_khr_fp16 cl_khr_depth_images cl_khr_global_int32_base_atomics cl_khr_global_int32_extended_atomics cl_khr_icd cl_khr_image2d_from_buffer cl_khr_local_int32_base_atomics cl_khr_local_int32_extended_atomics cl_intel_subgroups cl_intel_required_subgroup_size cl_intel_subgroups_short cl_khr_spir cl_intel_accelerator cl_intel_media_block_io cl_intel_driver_diagnostics cl_intel_device_side_avc_motion_estimation cl_khr_priority_hints cl_khr_throttle_hints cl_khr_create_command_queue cl_khr_fp64 cl_khr_subgroups cl_khr_il_program cl_khr_mipmap_image cl_khr_mipmap_image_writes cl_intel_planar_yuv cl_intel_packed_yuv cl_intel_motion_estimation cl_intel_advanced_motion_estimation cl_intel_va_api_media_sharing

NULL platform behavior
  clGetPlatformInfo(NULL, CL_PLATFORM_NAME, ...)  Intel(R) OpenCL HD Graphics
  clGetDeviceIDs(NULL, CL_DEVICE_TYPE_ALL, ...)   Success [INTEL]
  clCreateContext(NULL, ...) [default]            Success [INTEL]
  clCreateContextFromType(NULL, CL_DEVICE_TYPE_CPU)  No devices found in platform
  clCreateContextFromType(NULL, CL_DEVICE_TYPE_GPU)  Success (1)
    Platform Name                                 Intel(R) OpenCL HD Graphics
    Device Name                                   Intel(R) Gen9 HD Graphics NEO
  clCreateContextFromType(NULL, CL_DEVICE_TYPE_ACCELERATOR)  No devices found in platform
  clCreateContextFromType(NULL, CL_DEVICE_TYPE_CUSTOM)  No devices found in platform
  clCreateContextFromType(NULL, CL_DEVICE_TYPE_ALL)  Success (1)
    Platform Name                                 Intel(R) OpenCL HD Graphics
    Device Name                                   Intel(R) Gen9 HD Graphics NEO

ICD loader properties
  ICD loader Name                                 OpenCL ICD Loader
  ICD loader Vendor                               OCL Icd free software
  ICD loader Version                              2.2.8
  ICD loader Profile                              OpenCL 1.2
        NOTE:	your OpenCL library declares to support OpenCL 1.2,
                but it seems to support up to OpenCL 2.1 too.
********************************************************************************/
"EU24_k11x11_cn3_g1_s4x4_d1x1_b1_in240x240_p0x0_num1_M96_activ1_eltwise0_FP32", "2 5 16 2 1 1 16 1 0",
"EU24_k1x1_cn1024_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M256_activ1_eltwise0_FP32", "14 1 16 2 1 1 16 1 0",
"EU24_k1x1_cn1024_g1_s2x2_d1x1_b1_in16x16_p0x0_num1_M2048_activ0_eltwise0_FP32", "7 4 16 2 1 1 16 1 0",
"EU24_k1x1_cn1024_g1_s2x2_d1x1_b1_in16x16_p0x0_num1_M512_activ1_eltwise0_FP32", "7 2 16 2 1 1 16 1 0",
"EU24_k1x1_cn128_g1_s1x1_d1x1_b1_in32x32_p0x0_num1_M512_activ1_eltwise1_FP32", "2 8 32 5 1 8 1 1 0",
"EU24_k1x1_cn192_g1_s1x1_d1x1_b1_in32x32_p0x0_num1_M16_activ1_eltwise0_FP32", "10 1 16 2 1 1 16 1 0",
"EU24_k1x1_cn192_g1_s1x1_d1x1_b1_in32x32_p0x0_num1_M32_activ1_eltwise0_FP32", "14 1 16 2 1 1 16 1 0",
"EU24_k1x1_cn192_g1_s1x1_d1x1_b1_in32x32_p0x0_num1_M64_activ1_eltwise0_FP32", "10 2 16 2 1 1 16 1 0",
"EU24_k1x1_cn192_g1_s1x1_d1x1_b1_in32x32_p0x0_num1_M96_activ1_eltwise0_FP32", "14 1 16 2 1 1 16 1 0",
"EU24_k1x1_cn2048_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M512_activ1_eltwise0_FP32", "7 1 16 2 1 1 16 1 0",
"EU24_k1x1_cn256_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M1024_activ1_eltwise1_FP32", "14 1 16 2 1 1 16 1 0",
"EU24_k1x1_cn256_g1_s1x1_d1x1_b1_in32x32_p0x0_num1_M128_activ1_eltwise0_FP32", "14 2 16 2 1 1 16 1 0",
"EU24_k1x1_cn256_g1_s1x1_d1x1_b1_in32x32_p0x0_num1_M32_activ1_eltwise0_FP32", "10 1 16 2 1 1 16 1 0",
"EU24_k1x1_cn256_g1_s1x1_d1x1_b1_in32x32_p0x0_num1_M64_activ1_eltwise0_FP32", "14 1 16 2 1 1 16 1 0",
"EU24_k1x1_cn256_g1_s1x1_d1x1_b1_in64x64_p0x0_num1_M64_activ1_eltwise0_FP32", "14 1 16 2 1 1 16 1 0",
"EU24_k1x1_cn256_g1_s2x2_d1x1_b1_in64x64_p0x0_num1_M128_activ1_eltwise0_FP32", "1 8 32 5 1 8 1 1 0",
"EU24_k1x1_cn256_g1_s2x2_d1x1_b1_in64x64_p0x0_num1_M512_activ0_eltwise0_FP32", "2 8 32 5 1 8 1 1 0",
"EU24_k1x1_cn480_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M16_activ1_eltwise0_FP32", "7 1 16 2 1 1 16 1 0",
"EU24_k1x1_cn480_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M192_activ1_eltwise0_FP32", "14 1 16 2 1 1 16 1 0",
"EU24_k1x1_cn480_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M64_activ1_eltwise0_FP32", "8 1 16 2 1 1 16 1 0",
"EU24_k1x1_cn480_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M96_activ1_eltwise0_FP32", "14 1 16 2 1 1 16 1 0",
"EU24_k1x1_cn512_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M112_activ1_eltwise0_FP32", "14 1 16 2 1 1 16 1 0",
"EU24_k1x1_cn512_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M128_activ1_eltwise0_FP32", "8 1 16 2 1 1 16 1 0",
"EU24_k1x1_cn512_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M144_activ1_eltwise0_FP32", "14 1 16 2 1 1 16 1 0",
"EU24_k1x1_cn512_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M160_activ1_eltwise0_FP32", "14 1 16 2 1 1 16 1 0",
"EU24_k1x1_cn512_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M2048_activ1_eltwise1_FP32", "7 1 16 2 1 1 16 1 0",
"EU24_k1x1_cn512_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M24_activ1_eltwise0_FP32", "7 1 16 2 1 1 16 1 0",
"EU24_k1x1_cn512_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M32_activ1_eltwise0_FP32", "7 1 16 2 1 1 16 1 0",
"EU24_k1x1_cn512_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M64_activ1_eltwise0_FP32", "7 1 16 2 1 1 16 1 0",
"EU24_k1x1_cn512_g1_s1x1_d1x1_b1_in32x32_p0x0_num1_M128_activ1_eltwise0_FP32", "14 1 16 2 1 1 16 1 0",
"EU24_k1x1_cn512_g1_s2x2_d1x1_b1_in32x32_p0x0_num1_M1024_activ0_eltwise0_FP32", "2 8 32 5 1 8 1 1 0",
"EU24_k1x1_cn512_g1_s2x2_d1x1_b1_in32x32_p0x0_num1_M256_activ1_eltwise0_FP32", "7 3 16 2 1 1 16 1 0",
"EU24_k1x1_cn528_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M128_activ1_eltwise0_FP32", "14 1 16 2 1 1 16 1 0",
"EU24_k1x1_cn528_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M160_activ1_eltwise0_FP32", "14 1 16 2 1 1 16 1 0",
"EU24_k1x1_cn528_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M256_activ1_eltwise0_FP32", "14 1 16 2 1 1 16 1 0",
"EU24_k1x1_cn528_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M32_activ1_eltwise0_FP32", "7 1 16 2 1 1 16 1 0",
"EU24_k1x1_cn64_g1_s1x1_d1x1_b1_in64x64_p0x0_num1_M256_activ0_eltwise0_FP32", "2 8 32 5 1 8 1 1 0",
"EU24_k1x1_cn64_g1_s1x1_d1x1_b1_in64x64_p0x0_num1_M256_activ1_eltwise1_FP32", "1 16 32 5 1 16 1 1 0",
"EU24_k1x1_cn64_g1_s1x1_d1x1_b1_in64x64_p0x0_num1_M64_activ1_eltwise0_FP32", "2 8 32 5 1 8 1 1 0",
"EU24_k1x1_cn832_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M128_activ1_eltwise0_FP32", "7 1 16 2 1 1 16 1 0",
"EU24_k1x1_cn832_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M160_activ1_eltwise0_FP32", "7 1 16 2 1 1 16 1 0",
"EU24_k1x1_cn832_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M192_activ1_eltwise0_FP32", "7 2 16 2 1 1 16 1 0",
"EU24_k1x1_cn832_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M256_activ1_eltwise0_FP32", "7 1 16 2 1 1 16 1 0",
"EU24_k1x1_cn832_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M32_activ1_eltwise0_FP32", "7 1 8 2 1 1 8 1 0",
"EU24_k1x1_cn832_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M384_activ1_eltwise0_FP32", "7 1 16 2 1 1 16 1 0",
"EU24_k1x1_cn832_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M48_activ1_eltwise0_FP32", "4 1 16 2 1 1 16 1 0",
"EU24_k3x3_cn112_g1_s1x1_d1x1_b1_in16x16_p1x1_num1_M224_activ1_eltwise0_FP32", "14 1 16 2 1 1 16 1 0",
"EU24_k3x3_cn128_g1_s1x1_d1x1_b1_in16x16_p1x1_num1_M256_activ1_eltwise0_FP32", "14 1 16 2 1 1 16 1 0",
"EU24_k3x3_cn128_g1_s1x1_d1x1_b1_in32x32_p1x1_num1_M128_activ1_eltwise0_FP32", "14 1 16 2 1 1 16 1 0",
"EU24_k3x3_cn128_g1_s1x1_d1x1_b1_in32x32_p1x1_num1_M192_activ1_eltwise0_FP32", "14 2 16 2 1 1 16 1 0",
"EU24_k3x3_cn144_g1_s1x1_d1x1_b1_in16x16_p1x1_num1_M288_activ1_eltwise0_FP32", "14 1 16 2 1 1 16 1 0",
"EU24_k3x3_cn160_g1_s1x1_d1x1_b1_in16x16_p1x1_num1_M320_activ1_eltwise0_FP32", "7 1 16 2 1 1 16 1 0",
"EU24_k3x3_cn192_g1_s1x1_d1x1_b1_in16x16_p1x1_num1_M384_activ1_eltwise0_FP32", "7 1 16 2 1 1 16 1 0",
"EU24_k3x3_cn256_g1_s1x1_d1x1_b1_in16x16_p1x1_num1_M256_activ1_eltwise0_FP32", "14 1 16 2 1 1 16 1 0",
"EU24_k3x3_cn256_g1_s1x1_d1x1_b1_in16x16_p1x1_num1_M384_activ1_eltwise0_FP32", "13 1 16 2 1 1 16 1 0",
"EU24_k3x3_cn384_g2_s1x1_d1x1_b1_in16x16_p1x1_num1_M128_activ1_eltwise0_FP32", "13 1 16 2 1 1 16 1 0",
"EU24_k3x3_cn384_g2_s1x1_d1x1_b1_in16x16_p1x1_num1_M192_activ1_eltwise0_FP32", "13 1 16 2 1 1 16 1 0",
"EU24_k3x3_cn512_g1_s1x1_d1x1_b1_in16x16_p1x1_num1_M512_activ1_eltwise0_FP32", "7 1 16 2 1 1 16 1 0",
"EU24_k3x3_cn64_g1_s1x1_d1x1_b1_in64x64_p1x1_num1_M192_activ1_eltwise0_FP32", "14 2 16 2 1 1 16 1 0",
"EU24_k3x3_cn64_g1_s1x1_d1x1_b1_in64x64_p1x1_num1_M64_activ1_eltwise0_FP32", "14 2 16 2 1 1 16 1 0",
"EU24_k3x3_cn96_g1_s1x1_d1x1_b1_in16x16_p1x1_num1_M208_activ1_eltwise0_FP32", "14 2 16 2 1 1 16 1 0",
"EU24_k3x3_cn96_g1_s1x1_d1x1_b1_in32x32_p1x1_num1_M128_activ1_eltwise0_FP32", "14 1 16 2 1 1 16 1 0",
"EU24_k5x5_cn16_g1_s1x1_d1x1_b1_in16x16_p2x2_num1_M48_activ1_eltwise0_FP32", "5 1 16 2 1 1 16 1 0",
"EU24_k5x5_cn16_g1_s1x1_d1x1_b1_in32x32_p2x2_num1_M32_activ1_eltwise0_FP32", "4 3 8 2 1 1 8 1 0",
"EU24_k5x5_cn24_g1_s1x1_d1x1_b1_in16x16_p2x2_num1_M64_activ1_eltwise0_FP32", "5 1 16 2 1 1 16 1 0",
"EU24_k5x5_cn32_g1_s1x1_d1x1_b1_in16x16_p2x2_num1_M128_activ1_eltwise0_FP32", "4 1 16 2 1 1 16 1 0",
"EU24_k5x5_cn32_g1_s1x1_d1x1_b1_in16x16_p2x2_num1_M64_activ1_eltwise0_FP32", "7 1 16 2 1 1 16 1 0",
"EU24_k5x5_cn32_g1_s1x1_d1x1_b1_in32x32_p2x2_num1_M96_activ1_eltwise0_FP32", "7 2 16 2 1 1 16 1 0",
"EU24_k5x5_cn48_g1_s1x1_d1x1_b1_in16x16_p2x2_num1_M128_activ1_eltwise0_FP32", "3 7 8 2 1 1 8 1 0",
"EU24_k5x5_cn96_g2_s1x1_d1x1_b1_in32x32_p2x2_num1_M128_activ1_eltwise0_FP32", "9 3 16 2 1 1 16 1 0",
"EU24_k7x7_cn3_g1_s2x2_d1x1_b1_in224x224_p3x3_num1_M64_activ1_eltwise0_FP32", "4 4 16 2 1 1 16 1 0",
};

const char *default_kernel_config_intel_fp16[] = {
// Below is the information for OpenCL based on which these configurations tuned
/*******************************************************************************
Number of platforms                               1
  Platform Name                                   Intel(R) OpenCL HD Graphics
  Platform Vendor                                 Intel(R) Corporation
  Platform Version                                OpenCL 2.1
  Platform Profile                                FULL_PROFILE
  Platform Extensions                             cl_khr_3d_image_writes cl_khr_byte_addressable_store cl_khr_fp16 cl_khr_depth_images cl_khr_global_int32_base_atomics cl_khr_global_int32_extended_atomics cl_khr_icd cl_khr_image2d_from_buffer cl_khr_local_int32_base_atomics cl_khr_local_int32_extended_atomics cl_intel_subgroups cl_intel_required_subgroup_size cl_intel_subgroups_short cl_khr_spir cl_intel_accelerator cl_intel_media_block_io cl_intel_driver_diagnostics cl_intel_device_side_avc_motion_estimation cl_khr_priority_hints cl_khr_throttle_hints cl_khr_create_command_queue cl_khr_subgroups cl_khr_il_program cl_khr_mipmap_image cl_khr_mipmap_image_writes cl_khr_fp64 cl_intel_planar_yuv cl_intel_packed_yuv cl_intel_motion_estimation cl_intel_advanced_motion_estimation
  Platform Host timer resolution                  1ns
  Platform Extensions function suffix             INTEL

  Platform Name                                   Intel(R) OpenCL HD Graphics
Number of devices                                 1
  Device Name                                     Intel(R) Gen9 HD Graphics NEO
  Device Vendor                                   Intel(R) Corporation
  Device Vendor ID                                0x8086
  Device Version                                  OpenCL 2.1 NEO
  Driver Version                                  18.21.10858
  Device OpenCL C Version                         OpenCL C 2.0
  Device Type                                     GPU
  Device Profile                                  FULL_PROFILE
  Max compute units                               48
  Max clock frequency                             950MHz
  Device Partition                                (core)
    Max number of sub-devices                     0
    Supported partition types                     None
  Max work item dimensions                        3
  Max work item sizes                             256x256x256
  Max work group size                             256
  Preferred work group size multiple              32
  Max sub-groups per work group                   32
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
  Global memory size                              13364170752 (12.45GiB)
  Error Correction support                        No
  Max memory allocation                           4294959104 (4GiB)
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
  Preferred total size of global vars             4294959104 (4GiB)
  Global Memory cache type                        Read/Write
  Global Memory cache size                        1048576
  Global Memory cache line                        64 bytes
  Image support                                   Yes
    Max number of samplers per kernel             16
    Max size for 1D images from buffer            268434944 pixels
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
  Max constant buffer size                        4294959104 (4GiB)
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
    Sub-group independent forward progress        Yes
    IL version                                    SPIR-V_1.0
    SPIR versions                                 1.2
  printf() buffer size                            4194304 (4MiB)
  Built-in kernels                                block_motion_estimate_intel;block_advanced_motion_estimate_check_intel;block_advanced_motion_estimate_bidirectional_check_intel;
  Motion Estimation accelerator version	(Intel)   2
  Device Available                                Yes
  Compiler Available                              Yes
  Linker Available                                Yes
  Device Extensions                               cl_khr_3d_image_writes cl_khr_byte_addressable_store cl_khr_fp16 cl_khr_depth_images cl_khr_global_int32_base_atomics cl_khr_global_int32_extended_atomics cl_khr_icd cl_khr_image2d_from_buffer cl_khr_local_int32_base_atomics cl_khr_local_int32_extended_atomics cl_intel_subgroups cl_intel_required_subgroup_size cl_intel_subgroups_short cl_khr_spir cl_intel_accelerator cl_intel_media_block_io cl_intel_driver_diagnostics cl_intel_device_side_avc_motion_estimation cl_khr_priority_hints cl_khr_throttle_hints cl_khr_create_command_queue cl_khr_subgroups cl_khr_il_program cl_khr_mipmap_image cl_khr_mipmap_image_writes cl_khr_fp64 cl_intel_planar_yuv cl_intel_packed_yuv cl_intel_motion_estimation cl_intel_advanced_motion_estimation

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
"EU48_k1x1_cn192_g1_s1x1_d1x1_b1_in32x32_p0x0_num1_M16_activ1_eltwise0_FP16", "11 1 16 2 1 1 16 1 0",
"EU48_k1x1_cn192_g1_s1x1_d1x1_b1_in32x32_p0x0_num1_M32_activ1_eltwise0_FP16", "14 1 16 2 1 1 16 1 0",
"EU48_k1x1_cn192_g1_s1x1_d1x1_b1_in32x32_p0x0_num1_M64_activ1_eltwise0_FP16", "14 1 16 2 1 1 16 1 0",
"EU48_k1x1_cn192_g1_s1x1_d1x1_b1_in32x32_p0x0_num1_M96_activ1_eltwise0_FP16", "14 1 16 2 1 1 16 1 0",
"EU48_k1x1_cn256_g1_s1x1_d1x1_b1_in32x32_p0x0_num1_M128_activ1_eltwise0_FP16", "14 1 16 2 1 1 16 1 0",
"EU48_k1x1_cn256_g1_s1x1_d1x1_b1_in32x32_p0x0_num1_M32_activ1_eltwise0_FP16", "14 1 16 2 1 1 16 1 0",
"EU48_k1x1_cn256_g1_s1x1_d1x1_b1_in32x32_p0x0_num1_M64_activ1_eltwise0_FP16", "14 1 16 2 1 1 16 1 0",
"EU48_k1x1_cn480_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M16_activ1_eltwise0_FP16", "8 1 8 2 1 1 8 1 0",
"EU48_k1x1_cn480_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M192_activ1_eltwise0_FP16", "14 1 16 2 1 1 16 1 0",
"EU48_k1x1_cn480_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M64_activ1_eltwise0_FP16", "7 1 16 2 1 1 16 1 0",
"EU48_k1x1_cn480_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M96_activ1_eltwise0_FP16", "14 1 16 2 1 1 16 1 0",
"EU48_k1x1_cn512_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M112_activ1_eltwise0_FP16", "14 1 16 2 1 1 16 1 0",
"EU48_k1x1_cn512_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M128_activ1_eltwise0_FP16", "14 1 16 2 1 1 16 1 0",
"EU48_k1x1_cn512_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M144_activ1_eltwise0_FP16", "14 1 16 2 1 1 16 1 0",
"EU48_k1x1_cn512_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M160_activ1_eltwise0_FP16", "14 1 16 2 1 1 16 1 0",
"EU48_k1x1_cn512_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M24_activ1_eltwise0_FP16", "7 1 8 2 1 1 8 1 0",
"EU48_k1x1_cn512_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M32_activ1_eltwise0_FP16", "8 1 16 2 1 1 16 1 0",
"EU48_k1x1_cn512_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M64_activ1_eltwise0_FP16", "7 1 16 2 1 1 16 1 0",
"EU48_k1x1_cn528_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M128_activ1_eltwise0_FP16", "14 1 16 2 1 1 16 1 0",
"EU48_k1x1_cn528_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M160_activ1_eltwise0_FP16", "14 1 16 2 1 1 16 1 0",
"EU48_k1x1_cn528_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M256_activ1_eltwise0_FP16", "14 1 16 2 1 1 16 1 0",
"EU48_k1x1_cn528_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M32_activ1_eltwise0_FP16", "6 1 16 2 1 1 16 1 0",
"EU48_k1x1_cn64_g1_s1x1_d1x1_b1_in64x64_p0x0_num1_M64_activ1_eltwise0_FP16", "1 16 32 5 1 16 1 1 0",
"EU48_k1x1_cn832_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M128_activ1_eltwise0_FP16", "7 1 16 2 1 1 16 1 0",
"EU48_k1x1_cn832_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M160_activ1_eltwise0_FP16", "7 1 16 2 1 1 16 1 0",
"EU48_k1x1_cn832_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M192_activ1_eltwise0_FP16", "7 1 16 2 1 1 16 1 0",
"EU48_k1x1_cn832_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M256_activ1_eltwise0_FP16", "7 1 16 2 1 1 16 1 0",
"EU48_k1x1_cn832_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M32_activ1_eltwise0_FP16", "4 1 8 2 1 1 8 1 0",
"EU48_k1x1_cn832_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M384_activ1_eltwise0_FP16", "7 1 16 2 1 1 16 1 0",
"EU48_k1x1_cn832_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M48_activ1_eltwise0_FP16", "7 1 8 2 1 1 8 1 0",
"EU48_k3x3_cn112_g1_s1x1_d1x1_b1_in16x16_p1x1_num1_M224_activ1_eltwise0_FP16", "14 1 16 2 1 1 16 1 0",
"EU48_k3x3_cn128_g1_s1x1_d1x1_b1_in16x16_p1x1_num1_M256_activ1_eltwise0_FP16", "14 1 16 2 1 1 16 1 0",
"EU48_k3x3_cn128_g1_s1x1_d1x1_b1_in32x32_p1x1_num1_M192_activ1_eltwise0_FP16", "14 2 16 2 1 1 16 1 0",
"EU48_k3x3_cn144_g1_s1x1_d1x1_b1_in16x16_p1x1_num1_M288_activ1_eltwise0_FP16", "14 1 16 2 1 1 16 1 0",
"EU48_k3x3_cn160_g1_s1x1_d1x1_b1_in16x16_p1x1_num1_M320_activ1_eltwise0_FP16", "14 1 16 2 1 1 16 1 0",
"EU48_k3x3_cn192_g1_s1x1_d1x1_b1_in16x16_p1x1_num1_M384_activ1_eltwise0_FP16", "7 1 16 2 1 1 16 1 0",
"EU48_k3x3_cn64_g1_s1x1_d1x1_b1_in64x64_p1x1_num1_M192_activ1_eltwise0_FP16", "14 2 16 2 1 1 16 1 0",
"EU48_k3x3_cn96_g1_s1x1_d1x1_b1_in16x16_p1x1_num1_M208_activ1_eltwise0_FP16", "14 1 16 2 1 1 16 1 0",
"EU48_k3x3_cn96_g1_s1x1_d1x1_b1_in32x32_p1x1_num1_M128_activ1_eltwise0_FP16", "14 2 16 2 1 1 16 1 0",
"EU48_k5x5_cn16_g1_s1x1_d1x1_b1_in16x16_p2x2_num1_M48_activ1_eltwise0_FP16", "5 1 16 2 1 1 16 1 0",
"EU48_k5x5_cn16_g1_s1x1_d1x1_b1_in32x32_p2x2_num1_M32_activ1_eltwise0_FP16", "7 1 16 2 1 1 16 1 0",
"EU48_k5x5_cn24_g1_s1x1_d1x1_b1_in16x16_p2x2_num1_M64_activ1_eltwise0_FP16", "5 1 16 2 1 1 16 1 0",
"EU48_k5x5_cn32_g1_s1x1_d1x1_b1_in16x16_p2x2_num1_M128_activ1_eltwise0_FP16", "8 1 16 2 1 1 16 1 0",
"EU48_k5x5_cn32_g1_s1x1_d1x1_b1_in16x16_p2x2_num1_M64_activ1_eltwise0_FP16", "7 1 16 2 1 1 16 1 0",
"EU48_k5x5_cn32_g1_s1x1_d1x1_b1_in32x32_p2x2_num1_M96_activ1_eltwise0_FP16", "10 2 16 2 1 1 16 1 0",
"EU48_k5x5_cn48_g1_s1x1_d1x1_b1_in16x16_p2x2_num1_M128_activ1_eltwise0_FP16", "5 1 16 2 1 1 16 1 0",
"EU48_k7x7_cn3_g1_s2x2_d1x1_b1_in224x224_p3x3_num1_M64_activ1_eltwise0_FP16", "5 6 16 2 1 1 16 1 0",
"EU48_k11x11_cn3_g1_s4x4_d1x1_b1_in240x240_p0x0_num1_M96_activ1_eltwise0_FP16", "2 8 16 2 1 1 16 1 0",
"EU48_k3x3_cn256_g1_s1x1_d1x1_b1_in16x16_p1x1_num1_M384_activ1_eltwise0_FP16", "13 1 16 2 1 1 16 1 0",
"EU48_k3x3_cn384_g2_s1x1_d1x1_b1_in16x16_p1x1_num1_M128_activ1_eltwise0_FP16", "13 1 16 2 1 1 16 1 0",
"EU48_k3x3_cn384_g2_s1x1_d1x1_b1_in16x16_p1x1_num1_M192_activ1_eltwise0_FP16", "13 1 16 2 1 1 16 1 0",
"EU48_k5x5_cn96_g2_s1x1_d1x1_b1_in32x32_p2x2_num1_M128_activ1_eltwise0_FP16", "9 2 16 2 1 1 16 1 0",
// Below is the information for OpenCL based on which these configurations tuned
/*******************************************************************************
Number of platforms                               1
  Platform Name                                   Intel(R) OpenCL HD Graphics
  Platform Vendor                                 Intel(R) Corporation
  Platform Version                                OpenCL 2.1
  Platform Profile                                FULL_PROFILE
  Platform Extensions                             cl_khr_3d_image_writes cl_khr_byte_addressable_store cl_khr_fp16 cl_khr_depth_images cl_khr_global_int32_base_atomics cl_khr_global_int32_extended_atomics cl_khr_icd cl_khr_image2d_from_buffer cl_khr_local_int32_base_atomics cl_khr_local_int32_extended_atomics cl_intel_subgroups cl_intel_required_subgroup_size cl_intel_subgroups_short cl_khr_spir cl_intel_accelerator cl_intel_media_block_io cl_intel_driver_diagnostics cl_intel_device_side_avc_motion_estimation cl_khr_priority_hints cl_khr_throttle_hints cl_khr_create_command_queue cl_khr_fp64 cl_khr_subgroups cl_khr_il_program cl_khr_mipmap_image cl_khr_mipmap_image_writes cl_intel_planar_yuv cl_intel_packed_yuv cl_intel_motion_estimation cl_intel_advanced_motion_estimation cl_intel_va_api_media_sharing
  Platform Host timer resolution                  1ns
  Platform Extensions function suffix             INTEL

  Platform Name                                   Intel(R) OpenCL HD Graphics
Number of devices                                 1
  Device Name                                     Intel(R) Gen9 HD Graphics NEO
  Device Vendor                                   Intel(R) Corporation
  Device Vendor ID                                0x8086
  Device Version                                  OpenCL 2.1 NEO
  Driver Version                                  18.23.10915
  Device OpenCL C Version                         OpenCL C 2.0
  Device Type                                     GPU
  Device Profile                                  FULL_PROFILE
  Max compute units                               24
  Max clock frequency                             1150MHz
  Device Partition                                (core)
    Max number of sub-devices                     0
    Supported partition types                     None
  Max work item dimensions                        3
  Max work item sizes                             256x256x256
  Max work group size                             256
  Preferred work group size multiple              32
  Max sub-groups per work group                   32
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
  Global memory size                              6575288320 (6.124GiB)
  Error Correction support                        No
  Max memory allocation                           3287644160 (3.062GiB)
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
  Preferred total size of global vars             3287644160 (3.062GiB)
  Global Memory cache type                        Read/Write
  Global Memory cache size                        524288
  Global Memory cache line                        64 bytes
  Image support                                   Yes
    Max number of samplers per kernel             16
    Max size for 1D images from buffer            205477760 pixels
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
  Max constant buffer size                        3287644160 (3.062GiB)
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
    Sub-group independent forward progress        Yes
    IL version                                    SPIR-V_1.0
    SPIR versions                                 1.2
  printf() buffer size                            4194304 (4MiB)
  Built-in kernels                                block_motion_estimate_intel;block_advanced_motion_estimate_check_intel;block_advanced_motion_estimate_bidirectional_check_intel;
  Motion Estimation accelerator version	(Intel)   2
  Device Available                                Yes
  Compiler Available                              Yes
  Linker Available                                Yes
  Device Extensions                               cl_khr_3d_image_writes cl_khr_byte_addressable_store cl_khr_fp16 cl_khr_depth_images cl_khr_global_int32_base_atomics cl_khr_global_int32_extended_atomics cl_khr_icd cl_khr_image2d_from_buffer cl_khr_local_int32_base_atomics cl_khr_local_int32_extended_atomics cl_intel_subgroups cl_intel_required_subgroup_size cl_intel_subgroups_short cl_khr_spir cl_intel_accelerator cl_intel_media_block_io cl_intel_driver_diagnostics cl_intel_device_side_avc_motion_estimation cl_khr_priority_hints cl_khr_throttle_hints cl_khr_create_command_queue cl_khr_fp64 cl_khr_subgroups cl_khr_il_program cl_khr_mipmap_image cl_khr_mipmap_image_writes cl_intel_planar_yuv cl_intel_packed_yuv cl_intel_motion_estimation cl_intel_advanced_motion_estimation cl_intel_va_api_media_sharing

NULL platform behavior
  clGetPlatformInfo(NULL, CL_PLATFORM_NAME, ...)  Intel(R) OpenCL HD Graphics
  clGetDeviceIDs(NULL, CL_DEVICE_TYPE_ALL, ...)   Success [INTEL]
  clCreateContext(NULL, ...) [default]            Success [INTEL]
  clCreateContextFromType(NULL, CL_DEVICE_TYPE_CPU)  No devices found in platform
  clCreateContextFromType(NULL, CL_DEVICE_TYPE_GPU)  Success (1)
    Platform Name                                 Intel(R) OpenCL HD Graphics
    Device Name                                   Intel(R) Gen9 HD Graphics NEO
  clCreateContextFromType(NULL, CL_DEVICE_TYPE_ACCELERATOR)  No devices found in platform
  clCreateContextFromType(NULL, CL_DEVICE_TYPE_CUSTOM)  No devices found in platform
  clCreateContextFromType(NULL, CL_DEVICE_TYPE_ALL)  Success (1)
    Platform Name                                 Intel(R) OpenCL HD Graphics
    Device Name                                   Intel(R) Gen9 HD Graphics NEO

ICD loader properties
  ICD loader Name                                 OpenCL ICD Loader
  ICD loader Vendor                               OCL Icd free software
  ICD loader Version                              2.2.8
  ICD loader Profile                              OpenCL 1.2
        NOTE:	your OpenCL library declares to support OpenCL 1.2,
                but it seems to support up to OpenCL 2.1 too.
********************************************************************************/
"EU24_k11x11_cn3_g1_s4x4_d1x1_b1_in240x240_p0x0_num1_M96_activ1_eltwise0_FP16", "2 7 16 2 1 1 16 1 0",
"EU24_k1x1_cn1024_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M256_activ1_eltwise0_FP16", "14 2 16 2 1 1 16 1 0",
"EU24_k1x1_cn1024_g1_s2x2_d1x1_b1_in16x16_p0x0_num1_M2048_activ0_eltwise0_FP16", "7 4 16 2 1 1 16 1 0",
"EU24_k1x1_cn1024_g1_s2x2_d1x1_b1_in16x16_p0x0_num1_M512_activ1_eltwise0_FP16", "7 1 16 2 1 1 16 1 0",
"EU24_k1x1_cn128_g1_s1x1_d1x1_b1_in32x32_p0x0_num1_M512_activ1_eltwise1_FP16", "1 16 32 5 1 16 1 1 0",
"EU24_k1x1_cn192_g1_s1x1_d1x1_b1_in32x32_p0x0_num1_M16_activ1_eltwise0_FP16", "14 1 16 2 1 1 16 1 0",
"EU24_k1x1_cn192_g1_s1x1_d1x1_b1_in32x32_p0x0_num1_M32_activ1_eltwise0_FP16", "14 1 16 2 1 1 16 1 0",
"EU24_k1x1_cn192_g1_s1x1_d1x1_b1_in32x32_p0x0_num1_M64_activ1_eltwise0_FP16", "10 3 16 2 1 1 16 1 0",
"EU24_k1x1_cn192_g1_s1x1_d1x1_b1_in32x32_p0x0_num1_M96_activ1_eltwise0_FP16", "14 2 16 2 1 1 16 1 0",
"EU24_k1x1_cn2048_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M512_activ1_eltwise0_FP16", "7 1 16 2 1 1 16 1 0",
"EU24_k1x1_cn256_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M1024_activ1_eltwise1_FP16", "14 2 16 2 1 1 16 1 0",
"EU24_k1x1_cn256_g1_s1x1_d1x1_b1_in32x32_p0x0_num1_M128_activ1_eltwise0_FP16", "14 1 16 2 1 1 16 1 0",
"EU24_k1x1_cn256_g1_s1x1_d1x1_b1_in32x32_p0x0_num1_M32_activ1_eltwise0_FP16", "10 2 16 2 1 1 16 1 0",
"EU24_k1x1_cn256_g1_s1x1_d1x1_b1_in32x32_p0x0_num1_M64_activ1_eltwise0_FP16", "14 2 16 2 1 1 16 1 0",
"EU24_k1x1_cn256_g1_s1x1_d1x1_b1_in64x64_p0x0_num1_M64_activ1_eltwise0_FP16", "14 2 16 2 1 1 16 1 0",
"EU24_k1x1_cn256_g1_s2x2_d1x1_b1_in64x64_p0x0_num1_M128_activ1_eltwise0_FP16", "8 4 16 2 1 1 16 1 0",
"EU24_k1x1_cn256_g1_s2x2_d1x1_b1_in64x64_p0x0_num1_M512_activ0_eltwise0_FP16", "1 16 32 5 1 16 1 1 0",
"EU24_k1x1_cn480_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M16_activ1_eltwise0_FP16", "7 1 16 2 1 1 16 1 0",
"EU24_k1x1_cn480_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M192_activ1_eltwise0_FP16", "14 1 16 2 1 1 16 1 0",
"EU24_k1x1_cn480_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M64_activ1_eltwise0_FP16", "14 1 16 2 1 1 16 1 0",
"EU24_k1x1_cn480_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M96_activ1_eltwise0_FP16", "7 2 16 2 1 1 16 1 0",
"EU24_k1x1_cn512_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M112_activ1_eltwise0_FP16", "14 1 16 2 1 1 16 1 0",
"EU24_k1x1_cn512_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M128_activ1_eltwise0_FP16", "14 1 16 2 1 1 16 1 0",
"EU24_k1x1_cn512_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M144_activ1_eltwise0_FP16", "14 1 16 2 1 1 16 1 0",
"EU24_k1x1_cn512_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M160_activ1_eltwise0_FP16", "14 1 16 2 1 1 16 1 0",
"EU24_k1x1_cn512_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M2048_activ1_eltwise1_FP16", "7 1 16 2 1 1 16 1 0",
"EU24_k1x1_cn512_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M24_activ1_eltwise0_FP16", "10 1 16 2 1 1 16 1 0",
"EU24_k1x1_cn512_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M32_activ1_eltwise0_FP16", "9 1 16 2 1 1 16 1 0",
"EU24_k1x1_cn512_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M64_activ1_eltwise0_FP16", "7 1 16 2 1 1 16 1 0",
"EU24_k1x1_cn512_g1_s1x1_d1x1_b1_in32x32_p0x0_num1_M128_activ1_eltwise0_FP16", "14 2 16 2 1 1 16 1 0",
"EU24_k1x1_cn512_g1_s2x2_d1x1_b1_in32x32_p0x0_num1_M1024_activ0_eltwise0_FP16", "8 3 16 2 1 1 16 1 0",
"EU24_k1x1_cn512_g1_s2x2_d1x1_b1_in32x32_p0x0_num1_M256_activ1_eltwise0_FP16", "7 3 16 2 1 1 16 1 0",
"EU24_k1x1_cn528_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M128_activ1_eltwise0_FP16", "14 1 16 2 1 1 16 1 0",
"EU24_k1x1_cn528_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M160_activ1_eltwise0_FP16", "14 2 16 2 1 1 16 1 0",
"EU24_k1x1_cn528_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M256_activ1_eltwise0_FP16", "14 2 16 2 1 1 16 1 0",
"EU24_k1x1_cn528_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M32_activ1_eltwise0_FP16", "8 1 16 2 1 1 16 1 0",
"EU24_k1x1_cn64_g1_s1x1_d1x1_b1_in64x64_p0x0_num1_M256_activ0_eltwise0_FP16", "1 16 32 5 1 16 1 1 0",
"EU24_k1x1_cn64_g1_s1x1_d1x1_b1_in64x64_p0x0_num1_M256_activ1_eltwise1_FP16", "1 16 32 5 1 16 1 1 0",
"EU24_k1x1_cn64_g1_s1x1_d1x1_b1_in64x64_p0x0_num1_M64_activ1_eltwise0_FP16", "1 16 32 5 1 16 1 1 0",
"EU24_k1x1_cn832_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M128_activ1_eltwise0_FP16", "7 1 16 2 1 1 16 1 0",
"EU24_k1x1_cn832_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M160_activ1_eltwise0_FP16", "7 1 16 2 1 1 16 1 0",
"EU24_k1x1_cn832_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M192_activ1_eltwise0_FP16", "7 1 16 2 1 1 16 1 0",
"EU24_k1x1_cn832_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M256_activ1_eltwise0_FP16", "7 1 16 2 1 1 16 1 0",
"EU24_k1x1_cn832_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M32_activ1_eltwise0_FP16", "6 1 16 2 1 1 16 1 0",
"EU24_k1x1_cn832_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M384_activ1_eltwise0_FP16", "7 1 16 2 1 1 16 1 0",
"EU24_k1x1_cn832_g1_s1x1_d1x1_b1_in16x16_p0x0_num1_M48_activ1_eltwise0_FP16", "7 1 16 2 1 1 16 1 0",
"EU24_k3x3_cn112_g1_s1x1_d1x1_b1_in16x16_p1x1_num1_M224_activ1_eltwise0_FP16", "14 2 16 2 1 1 16 1 0",
"EU24_k3x3_cn128_g1_s1x1_d1x1_b1_in16x16_p1x1_num1_M256_activ1_eltwise0_FP16", "14 2 16 2 1 1 16 1 0",
"EU24_k3x3_cn128_g1_s1x1_d1x1_b1_in32x32_p1x1_num1_M128_activ1_eltwise0_FP16", "10 2 16 2 1 1 16 1 0",
"EU24_k3x3_cn128_g1_s1x1_d1x1_b1_in32x32_p1x1_num1_M192_activ1_eltwise0_FP16", "14 2 16 2 1 1 16 1 0",
"EU24_k3x3_cn144_g1_s1x1_d1x1_b1_in16x16_p1x1_num1_M288_activ1_eltwise0_FP16", "14 1 16 2 1 1 16 1 0",
"EU24_k3x3_cn160_g1_s1x1_d1x1_b1_in16x16_p1x1_num1_M320_activ1_eltwise0_FP16", "14 2 16 2 1 1 16 1 0",
"EU24_k3x3_cn192_g1_s1x1_d1x1_b1_in16x16_p1x1_num1_M384_activ1_eltwise0_FP16", "7 1 16 2 1 1 16 1 0",
"EU24_k3x3_cn256_g1_s1x1_d1x1_b1_in16x16_p1x1_num1_M256_activ1_eltwise0_FP16", "14 2 16 2 1 1 16 1 0",
"EU24_k3x3_cn256_g1_s1x1_d1x1_b1_in16x16_p1x1_num1_M384_activ1_eltwise0_FP16", "13 2 16 2 1 1 16 1 0",
"EU24_k3x3_cn384_g2_s1x1_d1x1_b1_in16x16_p1x1_num1_M128_activ1_eltwise0_FP16", "13 1 16 2 1 1 16 1 0",
"EU24_k3x3_cn384_g2_s1x1_d1x1_b1_in16x16_p1x1_num1_M192_activ1_eltwise0_FP16", "13 1 16 2 1 1 16 1 0",
"EU24_k3x3_cn512_g1_s1x1_d1x1_b1_in16x16_p1x1_num1_M512_activ1_eltwise0_FP16", "7 2 16 2 1 1 16 1 0",
"EU24_k3x3_cn64_g1_s1x1_d1x1_b1_in64x64_p1x1_num1_M192_activ1_eltwise0_FP16", "14 2 16 2 1 1 16 1 0",
"EU24_k3x3_cn64_g1_s1x1_d1x1_b1_in64x64_p1x1_num1_M64_activ1_eltwise0_FP16", "14 2 16 2 1 1 16 1 0",
"EU24_k3x3_cn96_g1_s1x1_d1x1_b1_in16x16_p1x1_num1_M208_activ1_eltwise0_FP16", "14 2 16 2 1 1 16 1 0",
"EU24_k3x3_cn96_g1_s1x1_d1x1_b1_in32x32_p1x1_num1_M128_activ1_eltwise0_FP16", "14 1 16 2 1 1 16 1 0",
"EU24_k5x5_cn16_g1_s1x1_d1x1_b1_in16x16_p2x2_num1_M48_activ1_eltwise0_FP16", "7 1 16 2 1 1 16 1 0",
"EU24_k5x5_cn16_g1_s1x1_d1x1_b1_in32x32_p2x2_num1_M32_activ1_eltwise0_FP16", "7 2 16 2 1 1 16 1 0",
"EU24_k5x5_cn24_g1_s1x1_d1x1_b1_in16x16_p2x2_num1_M64_activ1_eltwise0_FP16", "7 1 16 2 1 1 16 1 0",
"EU24_k5x5_cn32_g1_s1x1_d1x1_b1_in16x16_p2x2_num1_M128_activ1_eltwise0_FP16", "7 2 16 2 1 1 16 1 0",
"EU24_k5x5_cn32_g1_s1x1_d1x1_b1_in16x16_p2x2_num1_M64_activ1_eltwise0_FP16", "7 1 16 2 1 1 16 1 0",
"EU24_k5x5_cn32_g1_s1x1_d1x1_b1_in32x32_p2x2_num1_M96_activ1_eltwise0_FP16", "7 2 16 2 1 1 16 1 0",
"EU24_k5x5_cn48_g1_s1x1_d1x1_b1_in16x16_p2x2_num1_M128_activ1_eltwise0_FP16", "4 1 16 2 1 1 16 1 0",
"EU24_k5x5_cn96_g2_s1x1_d1x1_b1_in32x32_p2x2_num1_M128_activ1_eltwise0_FP16", "7 3 16 2 1 1 16 1 0",
"EU24_k7x7_cn3_g1_s2x2_d1x1_b1_in224x224_p3x3_num1_M64_activ1_eltwise0_FP16", "4 7 16 2 1 1 16 1 0",
};
#endif
