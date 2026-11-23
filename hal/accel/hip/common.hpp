// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef OPENCV_HAL_ACCEL_HIP_COMMON_HPP
#define OPENCV_HAL_ACCEL_HIP_COMMON_HPP

#include <hip/hip_runtime.h>

#include <limits>
#include <type_traits>

namespace cv { namespace accel { namespace hip_impl { namespace kernels {

// Alignment decides the copy width: an unaligned Bytes<16> is 16 byte-moves, an aligned one
// is a single dwordx4. Callers must check the pointers and steps before using the aligned form.
template<int N> struct Bytes { unsigned char b[N]; };
template<> struct __align__(2)  Bytes<2>  { unsigned char b[2];  };
template<> struct __align__(4)  Bytes<4>  { unsigned char b[4];  };
template<> struct __align__(8)  Bytes<8>  { unsigned char b[8];  };
template<> struct __align__(16) Bytes<16> { unsigned char b[16]; };

template<typename T>
__host__ __device__ __forceinline__ bool hipAligned(const void* p, size_t step)
{
    return ((reinterpret_cast<uintptr_t>(p) | step) & (alignof(T) - 1)) == 0;
}

// Copy N bytes without forming a Bytes<N> temporary: an array member of a struct is placed in
// scratch (VRAM) rather than registers, which costs far more than the copy itself.
template<int N>
__device__ __forceinline__ void hipCopyN(unsigned char* d, const unsigned char* s)
{
#pragma unroll
    for (int i = 0; i < N; i++)
        d[i] = s[i];
}

#define CV_ACCEL_HIP_COPY_WORDS(N_, WORD_) \
    template<> __device__ __forceinline__ void hipCopyN<N_>(unsigned char* d, const unsigned char* s) \
    { \
        if ((((uintptr_t)d | (uintptr_t)s) & (sizeof(WORD_) - 1)) == 0) \
            *(WORD_*)d = *(const WORD_*)s; \
        else \
        { \
            _Pragma("unroll") \
            for (int i = 0; i < N_; i++) d[i] = s[i]; \
        } \
    }
CV_ACCEL_HIP_COPY_WORDS(2,  ushort)
CV_ACCEL_HIP_COPY_WORDS(4,  uint)
CV_ACCEL_HIP_COPY_WORDS(8,  uint2)
CV_ACCEL_HIP_COPY_WORDS(16, uint4)
#undef CV_ACCEL_HIP_COPY_WORDS

template<typename T, typename D> struct HipWorkType
{
    static const bool wide = std::is_same<T, int>::value || std::is_same<T, double>::value ||
                             std::is_same<D, int>::value || std::is_same<D, double>::value;
    typedef typename std::conditional<wide, double, float>::type type;
};

const int BLOCK_X = 32, BLOCK_Y = 8;

// gridDim.y is limited: taller images are covered by a loop over y
const unsigned MAX_GRID_Y = 65535;

inline dim3 grid(int rows, int cols)
{
    const unsigned gy = (unsigned)((rows + BLOCK_Y - 1) / BLOCK_Y);
    return dim3((unsigned)((cols + BLOCK_X - 1) / BLOCK_X), gy < MAX_GRID_Y ? gy : MAX_GRID_Y);
}

#define CV_ACCEL_HIP_BY_SIZE(fn, ...) \
    switch (esz) \
    { \
    case 1: return fn<1>(__VA_ARGS__); \
    case 2: return fn<2>(__VA_ARGS__); \
    case 3: return fn<3>(__VA_ARGS__); \
    case 4: return fn<4>(__VA_ARGS__); \
    case 6: return fn<6>(__VA_ARGS__); \
    case 8: return fn<8>(__VA_ARGS__); \
    case 12: return fn<12>(__VA_ARGS__); \
    case 16: return fn<16>(__VA_ARGS__); \
    case 24: return fn<24>(__VA_ARGS__); \
    case 32: return fn<32>(__VA_ARGS__); \
    default: return hipErrorInvalidValue; \
    }

template<typename D, typename S>
__host__ __device__ __forceinline__ D hip_saturate_cast(S val) {
#if defined(__HIP_DEVICE_COMPILE__)
    // Narrow integer destinations from a float source stay in 32-bit float: a double round
    // trip is many times slower on RDNA, and float holds every value these types can take.
    if constexpr (std::is_same<S, float>::value && std::is_integral<D>::value && sizeof(D) <= 2)
    {
        const float lo = static_cast<float>(std::numeric_limits<D>::lowest());
        const float hi = static_cast<float>(std::numeric_limits<D>::max());
        return static_cast<D>(__float2int_rn(fminf(fmaxf(val, lo), hi)));
    }
#endif
    const double d = static_cast<double>(val);
    if (d < static_cast<double>(std::numeric_limits<D>::lowest())) return std::numeric_limits<D>::lowest();
    if (d > static_cast<double>(std::numeric_limits<D>::max()))    return std::numeric_limits<D>::max();
    // Integer destinations round to nearest (matches saturate_cast/cvRound).
    return static_cast<D>(std::numeric_limits<D>::is_integer ? rint(d) : d);
}

#define HIP_SAT_FSPEC(D, S) \
    template<> __host__ __device__ __forceinline__ D hip_saturate_cast<D,S>(S v) { return static_cast<D>(v); }
HIP_SAT_FSPEC(float,  unsigned char)   HIP_SAT_FSPEC(float,  signed char)
HIP_SAT_FSPEC(float,  unsigned short)  HIP_SAT_FSPEC(float,  short)
HIP_SAT_FSPEC(float,  int)             HIP_SAT_FSPEC(float,  float)
HIP_SAT_FSPEC(float,  double)
HIP_SAT_FSPEC(double, unsigned char)   HIP_SAT_FSPEC(double, signed char)
HIP_SAT_FSPEC(double, unsigned short)  HIP_SAT_FSPEC(double, short)
HIP_SAT_FSPEC(double, int)             HIP_SAT_FSPEC(double, float)
HIP_SAT_FSPEC(double, double)
#undef HIP_SAT_FSPEC

template<typename T> struct HipVecTraits;

#define DEFINE_VT1(T_, ET_) \
    template<> struct HipVecTraits<T_> { \
        typedef ET_ elem_type; \
        static __host__ __device__ __forceinline__ T_ make(const double* v) { \
            return hip_saturate_cast<T_>(v[0]); \
        } \
    }
#define DEFINE_VT2(T_, ET_, mk_) \
    template<> struct HipVecTraits<T_> { \
        typedef ET_ elem_type; \
        static __host__ __device__ __forceinline__ T_ make(const double* v) { \
            return mk_(hip_saturate_cast<ET_>(v[0]), hip_saturate_cast<ET_>(v[1])); \
        } \
    }
#define DEFINE_VT3(T_, ET_, mk_) \
    template<> struct HipVecTraits<T_> { \
        typedef ET_ elem_type; \
        static __host__ __device__ __forceinline__ T_ make(const double* v) { \
            return mk_(hip_saturate_cast<ET_>(v[0]), hip_saturate_cast<ET_>(v[1]), \
                       hip_saturate_cast<ET_>(v[2])); \
        } \
    }
#define DEFINE_VT4(T_, ET_, mk_) \
    template<> struct HipVecTraits<T_> { \
        typedef ET_ elem_type; \
        static __host__ __device__ __forceinline__ T_ make(const double* v) { \
            return mk_(hip_saturate_cast<ET_>(v[0]), hip_saturate_cast<ET_>(v[1]), \
                       hip_saturate_cast<ET_>(v[2]), hip_saturate_cast<ET_>(v[3])); \
        } \
    }

DEFINE_VT1(unsigned char,  unsigned char);
DEFINE_VT2(uchar2,  unsigned char,  make_uchar2);
DEFINE_VT3(uchar3,  unsigned char,  make_uchar3);
DEFINE_VT4(uchar4,  unsigned char,  make_uchar4);
DEFINE_VT1(signed char,    signed char);
DEFINE_VT2(char2,   signed char,    make_char2);
DEFINE_VT3(char3,   signed char,    make_char3);
DEFINE_VT4(char4,   signed char,    make_char4);
DEFINE_VT1(unsigned short, unsigned short);
DEFINE_VT2(ushort2, unsigned short, make_ushort2);
DEFINE_VT3(ushort3, unsigned short, make_ushort3);
DEFINE_VT4(ushort4, unsigned short, make_ushort4);
DEFINE_VT1(short,   short);
DEFINE_VT2(short2,  short,  make_short2);
DEFINE_VT3(short3,  short,  make_short3);
DEFINE_VT4(short4,  short,  make_short4);
DEFINE_VT1(int,     int);
DEFINE_VT2(int2,    int,    make_int2);
DEFINE_VT3(int3,    int,    make_int3);
DEFINE_VT4(int4,    int,    make_int4);
DEFINE_VT1(float,   float);
DEFINE_VT2(float2,  float,  make_float2);
DEFINE_VT3(float3,  float,  make_float3);
DEFINE_VT4(float4,  float,  make_float4);
DEFINE_VT1(double,  double);
DEFINE_VT2(double2, double, make_double2);
DEFINE_VT3(double3, double, make_double3);
DEFINE_VT4(double4, double, make_double4);

#undef DEFINE_VT1
#undef DEFINE_VT2
#undef DEFINE_VT3
#undef DEFINE_VT4

}}}} // namespace cv::accel::hip_impl::kernels

#endif // OPENCV_HAL_ACCEL_HIP_COMMON_HPP
