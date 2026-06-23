// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "opencv2/core/hip.hpp"
#include "opencv2/core/hipdev.hpp"
#include "cvconfig.h"

#include <limits>

using namespace cv;

// ThrustAllocator is needed only in standalone mode; combined mode uses gpu_mat.cu's.
#ifdef HAVE_HIP_STANDALONE

namespace cv { namespace cuda { namespace device {
struct CV_EXPORTS ThrustAllocator {
    virtual __device__ __host__ uchar* allocate(size_t numBytes) = 0;
    virtual __device__ __host__ void deallocate(uchar* ptr, size_t numBytes) = 0;
    virtual ~ThrustAllocator();
    CV_EXPORTS static ThrustAllocator& getAllocator();
    CV_EXPORTS static void setAllocator(ThrustAllocator* allocator);
};
}}}

cv::cuda::device::ThrustAllocator::~ThrustAllocator() {}

namespace {
class DefaultThrustAllocator : public cv::cuda::device::ThrustAllocator {
public:
    __device__ __host__ uchar* allocate(size_t n) CV_OVERRIDE {
#ifndef __HIP_DEVICE_COMPILE__
        uchar* p; CV_HIP_SAFE_CALL(hipMalloc(&p, n)); return p;
#else
        return NULL;
#endif
    }
    __device__ __host__ void deallocate(uchar* p, size_t) CV_OVERRIDE {
#ifndef __HIP_DEVICE_COMPILE__
        CV_HIP_SAFE_CALL(hipFree(p));
#endif
    }
};
DefaultThrustAllocator defaultThrustAllocator;
cv::cuda::device::ThrustAllocator* g_thrustAllocator = &defaultThrustAllocator;
}

cv::cuda::device::ThrustAllocator& cv::cuda::device::ThrustAllocator::getAllocator() { return *g_thrustAllocator; }
void cv::cuda::device::ThrustAllocator::setAllocator(cv::cuda::device::ThrustAllocator* a) {
    g_thrustAllocator = a ? a : &defaultThrustAllocator;
}

#endif // HAVE_HIP_STANDALONE

// hip_saturate_cast: integer dests clamp via double; float/double use the static_cast specialisations below.

template<typename D, typename S>
__host__ __device__ __forceinline__ D hip_saturate_cast(S val) {
    const double d = static_cast<double>(val);
    if (d < static_cast<double>(std::numeric_limits<D>::lowest())) return std::numeric_limits<D>::lowest();
    if (d > static_cast<double>(std::numeric_limits<D>::max()))    return std::numeric_limits<D>::max();
    // Integer destinations round to nearest (matches saturate_cast/cvRound).
    return static_cast<D>(std::numeric_limits<D>::is_integer ? rint(d) : d);
}

#define HIP_SAT_FSPEC(D, S) \
    template<> __host__ __device__ __forceinline__ D hip_saturate_cast<D,S>(S v) { return static_cast<D>(v); }
HIP_SAT_FSPEC(float,  uchar)  HIP_SAT_FSPEC(float,  schar)  HIP_SAT_FSPEC(float,  ushort)
HIP_SAT_FSPEC(float,  short)  HIP_SAT_FSPEC(float,  int)    HIP_SAT_FSPEC(float,  float)
HIP_SAT_FSPEC(float,  double)
HIP_SAT_FSPEC(double, uchar)  HIP_SAT_FSPEC(double, schar)  HIP_SAT_FSPEC(double, ushort)
HIP_SAT_FSPEC(double, short)  HIP_SAT_FSPEC(double, int)    HIP_SAT_FSPEC(double, float)
HIP_SAT_FSPEC(double, double)
#undef HIP_SAT_FSPEC

// HipVecTraits

template<typename T> struct HipVecTraits;

#define DEFINE_VT1(T_, ET_) \
    template<> struct HipVecTraits<T_> { \
        typedef ET_ elem_type; \
        static __host__ __device__ __forceinline__ T_ make(const ET_* v) { \
            return static_cast<T_>(v[0]); \
        } \
    }
#define DEFINE_VT2(T_, ET_, mk_) \
    template<> struct HipVecTraits<T_> { \
        typedef ET_ elem_type; \
        static __host__ __device__ __forceinline__ T_ make(const ET_* v) { \
            return mk_(v[0], v[1]); \
        } \
    }
#define DEFINE_VT3(T_, ET_, mk_) \
    template<> struct HipVecTraits<T_> { \
        typedef ET_ elem_type; \
        static __host__ __device__ __forceinline__ T_ make(const ET_* v) { \
            return mk_(v[0], v[1], v[2]); \
        } \
    }
#define DEFINE_VT4(T_, ET_, mk_) \
    template<> struct HipVecTraits<T_> { \
        typedef ET_ elem_type; \
        static __host__ __device__ __forceinline__ T_ make(const ET_* v) { \
            return mk_(v[0], v[1], v[2], v[3]); \
        } \
    }

DEFINE_VT1(uchar,   uchar);
DEFINE_VT2(uchar2,  uchar,  make_uchar2);
DEFINE_VT3(uchar3,  uchar,  make_uchar3);
DEFINE_VT4(uchar4,  uchar,  make_uchar4);
DEFINE_VT1(schar,   schar);
DEFINE_VT2(char2,   schar,  make_char2);
DEFINE_VT3(char3,   schar,  make_char3);
DEFINE_VT4(char4,   schar,  make_char4);
DEFINE_VT1(ushort,  ushort);
DEFINE_VT2(ushort2, ushort, make_ushort2);
DEFINE_VT3(ushort3, ushort, make_ushort3);
DEFINE_VT4(ushort4, ushort, make_ushort4);
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

// HipLargerType
template<typename A, typename B> struct HipLargerType          { typedef float  type; };
template<>                       struct HipLargerType<double, float> { typedef double type; };

// Grid helpers

static dim3 hipBlock() { return dim3(32, 32); }
static dim3 hipGrid(int rows, int cols) { return dim3((cols + 31) / 32, (rows + 31) / 32); }

// setToWithoutMask

template<typename T>
__global__ void setToKernel(uchar* data, size_t step, int rows, int cols, T val)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < cols && y < rows)
        *reinterpret_cast<T*>(data + y * step + x * sizeof(T)) = val;
}

namespace {
template<typename T>
void setToImpl(uchar* data, size_t step, int rows, int cols, const Scalar& s, hipStream_t stream)
{
    typedef typename HipVecTraits<T>::elem_type E;
    Scalar_<E> sc = s;
    T val = HipVecTraits<T>::make(sc.val);
    hipLaunchKernelGGL(setToKernel<T>, hipGrid(rows, cols), hipBlock(), 0, stream,
                       data, step, rows, cols, val);
}
}

void cv::hip::device::setToWithoutMask(void* data_, size_t step, int rows, int cols, int type,
                                       Scalar val)
{
    uchar* data = static_cast<uchar*>(data_);
    const int depth = CV_MAT_DEPTH(type), cn = CV_MAT_CN(type);
    const size_t elemSize = CV_ELEM_SIZE(type);
    const hipStream_t s = 0;  // UMat T-API has no stream concept; use the default stream
    CV_DbgAssert(data && depth <= CV_64F && cn <= 4);

    if (val[0] == 0.0 && val[1] == 0.0 && val[2] == 0.0 && val[3] == 0.0) {
        CV_HIP_SAFE_CALL(hipMemset2D(data, step, 0, cols * elemSize, rows));
        return;
    }
    if (depth == CV_8U) {
        if (cn == 1 || (cn == 2 && val[0]==val[1]) ||
            (cn == 3 && val[0]==val[1] && val[0]==val[2]) ||
            (cn == 4 && val[0]==val[1] && val[0]==val[2] && val[0]==val[3])) {
            CV_HIP_SAFE_CALL(hipMemset2D(data, step, hip_saturate_cast<uchar>(val[0]),
                             cols * elemSize, rows));
            return;
        }
    }

    typedef void (*func_t)(uchar*, size_t, int, int, const Scalar&, hipStream_t);
    static const func_t funcs[7][4] = {
        {setToImpl<uchar>,  setToImpl<uchar2>,  setToImpl<uchar3>,  setToImpl<uchar4>},
        {setToImpl<schar>,  setToImpl<char2>,   setToImpl<char3>,   setToImpl<char4>},
        {setToImpl<ushort>, setToImpl<ushort2>, setToImpl<ushort3>, setToImpl<ushort4>},
        {setToImpl<short>,  setToImpl<short2>,  setToImpl<short3>,  setToImpl<short4>},
        {setToImpl<int>,    setToImpl<int2>,    setToImpl<int3>,    setToImpl<int4>},
        {setToImpl<float>,  setToImpl<float2>,  setToImpl<float3>,  setToImpl<float4>},
        {setToImpl<double>, setToImpl<double2>, setToImpl<double3>, setToImpl<double4>}
    };
    funcs[depth][cn - 1](data, step, rows, cols, val, s);
    CV_HIP_SAFE_CALL(hipGetLastError());
    CV_HIP_SAFE_CALL(hipStreamSynchronize(s));
}

// setToWithMask

template<typename T>
__global__ void setToMaskedKernel(uchar* data, size_t step,
                                   const uchar* mask, size_t maskStep,
                                   int rows, int cols, T val)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < cols && y < rows && mask[y * maskStep + x])
        *reinterpret_cast<T*>(data + y * step + x * sizeof(T)) = val;
}

namespace {
template<typename T>
void setToMaskedImpl(uchar* data, size_t step,
                     const uchar* mask, size_t maskStep,
                     int rows, int cols, const Scalar& s, hipStream_t stream)
{
    typedef typename HipVecTraits<T>::elem_type E;
    Scalar_<E> sc = s;
    T val = HipVecTraits<T>::make(sc.val);
    hipLaunchKernelGGL(setToMaskedKernel<T>, hipGrid(rows, cols), hipBlock(), 0, stream,
                       data, step, mask, maskStep, rows, cols, val);
}
}

void cv::hip::device::setToWithMask(void* data_, size_t step, int rows, int cols, int type,
                                    const void* mask_, size_t maskStep,
                                    Scalar val)
{
    uchar* data = static_cast<uchar*>(data_);
    const uchar* mask = static_cast<const uchar*>(mask_);
    const int depth = CV_MAT_DEPTH(type), cn = CV_MAT_CN(type);
    const hipStream_t s = 0;  // UMat T-API has no stream concept; use the default stream
    CV_DbgAssert(data && mask && depth <= CV_64F && cn <= 4);

    typedef void (*func_t)(uchar*, size_t, const uchar*, size_t, int, int, const Scalar&, hipStream_t);
    static const func_t funcs[7][4] = {
        {setToMaskedImpl<uchar>,  setToMaskedImpl<uchar2>,  setToMaskedImpl<uchar3>,  setToMaskedImpl<uchar4>},
        {setToMaskedImpl<schar>,  setToMaskedImpl<char2>,   setToMaskedImpl<char3>,   setToMaskedImpl<char4>},
        {setToMaskedImpl<ushort>, setToMaskedImpl<ushort2>, setToMaskedImpl<ushort3>, setToMaskedImpl<ushort4>},
        {setToMaskedImpl<short>,  setToMaskedImpl<short2>,  setToMaskedImpl<short3>,  setToMaskedImpl<short4>},
        {setToMaskedImpl<int>,    setToMaskedImpl<int2>,    setToMaskedImpl<int3>,    setToMaskedImpl<int4>},
        {setToMaskedImpl<float>,  setToMaskedImpl<float2>,  setToMaskedImpl<float3>,  setToMaskedImpl<float4>},
        {setToMaskedImpl<double>, setToMaskedImpl<double2>, setToMaskedImpl<double3>, setToMaskedImpl<double4>}
    };
    funcs[depth][cn - 1](data, step, mask, maskStep, rows, cols, val, s);
    CV_HIP_SAFE_CALL(hipGetLastError());
    CV_HIP_SAFE_CALL(hipStreamSynchronize(s));
}

// copyToWithMask

template<typename T>
__global__ void copyMaskedKernel(const uchar* src, size_t srcStep,
                                  uchar* dst, size_t dstStep,
                                  const uchar* mask, size_t maskStep,
                                  int rows, int cols)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < cols && y < rows && mask[y * maskStep + x])
        *reinterpret_cast<T*>(dst + y * dstStep + x * sizeof(T)) =
            *reinterpret_cast<const T*>(src + y * srcStep + x * sizeof(T));
}

namespace {
template<typename T>
void copyMaskedImpl(const uchar* src, size_t srcStep,
                    uchar* dst, size_t dstStep,
                    const uchar* mask, size_t maskStep,
                    int rows, int cols, hipStream_t stream)
{
    hipLaunchKernelGGL(copyMaskedKernel<T>, hipGrid(rows, cols), hipBlock(), 0, stream,
                       src, srcStep, dst, dstStep, mask, maskStep, rows, cols);
}
}

void cv::hip::device::copyToWithMask(const void* src_, size_t srcStep,
                                     void* dst_, size_t dstStep,
                                     const void* mask_, size_t maskStep,
                                     int rows, int cols, int type, int maskCn)
{
    const uchar* src  = static_cast<const uchar*>(src_);
    uchar*       dst  = static_cast<uchar*>(dst_);
    const uchar* mask = static_cast<const uchar*>(mask_);
    const int    cn   = CV_MAT_CN(type);
    const size_t esz1 = CV_ELEM_SIZE1(type);
    const hipStream_t s = 0;  // UMat T-API has no stream concept; use the default stream
    CV_DbgAssert(src && CV_MAT_DEPTH(type) <= CV_64F && cn <= 4 &&
                 (maskCn == 1 || maskCn == cn));

    // funcs indexed by [byte-width-of-one-channel][cn-1]; copy ignores signedness.
    typedef void (*func_t)(const uchar*, size_t, uchar*, size_t,
                           const uchar*, size_t, int, int, hipStream_t);
    static const func_t funcs[9][4] = {
        {0,0,0,0},
        {copyMaskedImpl<uchar>,  copyMaskedImpl<uchar2>,  copyMaskedImpl<uchar3>,  copyMaskedImpl<uchar4>},
        {copyMaskedImpl<ushort>, copyMaskedImpl<ushort2>, copyMaskedImpl<ushort3>, copyMaskedImpl<ushort4>},
        {0,0,0,0},
        {copyMaskedImpl<int>,    copyMaskedImpl<int2>,    copyMaskedImpl<int3>,    copyMaskedImpl<int4>},
        {0,0,0,0},
        {0,0,0,0},
        {0,0,0,0},
        {copyMaskedImpl<double>, copyMaskedImpl<double2>, copyMaskedImpl<double3>, copyMaskedImpl<double4>}
    };

    if (maskCn == cn) {
        // Per-channel mask: flatten channels into the width and copy single elements.
        const func_t func = funcs[esz1][0];
        CV_DbgAssert(func);
        func(src, srcStep, dst, dstStep, mask, maskStep, rows, cols * cn, s);
    } else {
        // Single-channel mask gates whole pixels: use the cn-wide vector copy.
        const func_t func = funcs[esz1][cn - 1];
        CV_DbgAssert(func);
        func(src, srcStep, dst, dstStep, mask, maskStep, rows, cols, s);
    }
    CV_HIP_SAFE_CALL(hipGetLastError());
    CV_HIP_SAFE_CALL(hipStreamSynchronize(s));
}

// convertToScale

template<typename T, typename D, typename S>
__global__ void convertScaleKernel(const uchar* src, size_t srcStep,
                                    uchar* dst, size_t dstStep,
                                    int rows, int cols, S alpha, S beta)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < cols && y < rows)
        *reinterpret_cast<D*>(dst + y * dstStep + x * sizeof(D)) =
            hip_saturate_cast<D>(
                alpha * static_cast<S>(*reinterpret_cast<const T*>(src + y * srcStep + x * sizeof(T)))
                + beta);
}

namespace {
template<typename T, typename D>
void convertScaleImpl(const uchar* src, size_t srcStep,
                       uchar* dst, size_t dstStep,
                       int rows, int cols,
                       double alpha, double beta, hipStream_t stream)
{
    typedef typename HipVecTraits<T>::elem_type src_e;
    typedef typename HipLargerType<src_e, float>::type scalar_t;
    hipLaunchKernelGGL((convertScaleKernel<T,D,scalar_t>),
                       hipGrid(rows, cols), hipBlock(), 0, stream,
                       src, srcStep, dst, dstStep, rows, cols,
                       hip_saturate_cast<scalar_t>(alpha), hip_saturate_cast<scalar_t>(beta));
}
}

void cv::hip::device::convertToScale(const void* src_, size_t srcStep, int stype,
                                     void* dst_, size_t dstStep, int dtype,
                                     int rows, int cols, double alpha, double beta)
{
    const uchar* src = static_cast<const uchar*>(src_);
    uchar*       dst = static_cast<uchar*>(dst_);
    const hipStream_t s = 0;  // UMat T-API has no stream concept; use the default stream
    const int sd = CV_MAT_DEPTH(stype), dd = CV_MAT_DEPTH(dtype);
    CV_Assert(sd <= CV_64F && dd <= CV_64F);

    // convertTo preserves channel count; flatten channels into the width.
    const int cols1 = cols * CV_MAT_CN(stype);

    typedef void (*func_t)(const uchar*, size_t, uchar*, size_t,
                           int, int, double, double, hipStream_t);
    static const func_t funcs[7][7] = {
        {convertScaleImpl<uchar,uchar>, convertScaleImpl<uchar,schar>, convertScaleImpl<uchar,ushort>, convertScaleImpl<uchar,short>, convertScaleImpl<uchar,int>, convertScaleImpl<uchar,float>, convertScaleImpl<uchar,double>},
        {convertScaleImpl<schar,uchar>, convertScaleImpl<schar,schar>, convertScaleImpl<schar,ushort>, convertScaleImpl<schar,short>, convertScaleImpl<schar,int>, convertScaleImpl<schar,float>, convertScaleImpl<schar,double>},
        {convertScaleImpl<ushort,uchar>, convertScaleImpl<ushort,schar>, convertScaleImpl<ushort,ushort>, convertScaleImpl<ushort,short>, convertScaleImpl<ushort,int>, convertScaleImpl<ushort,float>, convertScaleImpl<ushort,double>},
        {convertScaleImpl<short,uchar>, convertScaleImpl<short,schar>, convertScaleImpl<short,ushort>, convertScaleImpl<short,short>, convertScaleImpl<short,int>, convertScaleImpl<short,float>, convertScaleImpl<short,double>},
        {convertScaleImpl<int,uchar>, convertScaleImpl<int,schar>, convertScaleImpl<int,ushort>, convertScaleImpl<int,short>, convertScaleImpl<int,int>, convertScaleImpl<int,float>, convertScaleImpl<int,double>},
        {convertScaleImpl<float,uchar>, convertScaleImpl<float,schar>, convertScaleImpl<float,ushort>, convertScaleImpl<float,short>, convertScaleImpl<float,int>, convertScaleImpl<float,float>, convertScaleImpl<float,double>},
        {convertScaleImpl<double,uchar>, convertScaleImpl<double,schar>, convertScaleImpl<double,ushort>, convertScaleImpl<double,short>, convertScaleImpl<double,int>, convertScaleImpl<double,float>, convertScaleImpl<double,double>},
    };

    const func_t func = funcs[sd][dd];
    CV_Assert(func);
    func(src, srcStep, dst, dstStep, rows, cols1, alpha, beta, s);
    CV_HIP_SAFE_CALL(hipGetLastError());
    CV_HIP_SAFE_CALL(hipStreamSynchronize(s));
}

