/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "precomp.hpp"

using namespace cv;
using namespace cv::gpu;


namespace 
{
    // Compares value to set using the given comparator. Returns true if
    // there is at least one element x in the set satisfying to: x cmp value
    // predicate.
    template <typename Comparer>
    bool compareToSet(const std::string& set_as_str, int value, Comparer cmp)
    {
        if (set_as_str.find_first_not_of(" ") == string::npos)
            return false;

        std::stringstream stream(set_as_str);
        int cur_value;

        while (!stream.eof())
        {
            stream >> cur_value;
            if (cmp(cur_value, value))
                return true;
        }

        return false;
    }
}


bool cv::gpu::TargetArchs::builtWith(cv::gpu::FeatureSet feature_set)
{
#if defined (HAVE_CUDA)
    return ::compareToSet(CUDA_ARCH_FEATURES, feature_set, std::greater_equal<int>());
#else
	(void)feature_set;
	return false;
#endif
}


bool cv::gpu::TargetArchs::has(int major, int minor)
{
    return hasPtx(major, minor) || hasBin(major, minor);
}


bool cv::gpu::TargetArchs::hasPtx(int major, int minor)
{
#if defined (HAVE_CUDA)
    return ::compareToSet(CUDA_ARCH_PTX, major * 10 + minor, std::equal_to<int>());
#else
	(void)major;
	(void)minor;
	return false;
#endif
}


bool cv::gpu::TargetArchs::hasBin(int major, int minor)
{
#if defined (HAVE_CUDA)
    return ::compareToSet(CUDA_ARCH_BIN, major * 10 + minor, std::equal_to<int>());
#else
	(void)major;
	(void)minor;
	return false;
#endif
}


bool cv::gpu::TargetArchs::hasEqualOrLessPtx(int major, int minor)
{
#if defined (HAVE_CUDA)
    return ::compareToSet(CUDA_ARCH_PTX, major * 10 + minor, 
                     std::less_equal<int>());
#else
	(void)major;
	(void)minor;
	return false;
#endif
}


bool cv::gpu::TargetArchs::hasEqualOrGreater(int major, int minor)
{
    return hasEqualOrGreaterPtx(major, minor) ||
           hasEqualOrGreaterBin(major, minor);
}


bool cv::gpu::TargetArchs::hasEqualOrGreaterPtx(int major, int minor)
{
#if defined (HAVE_CUDA)
    return ::compareToSet(CUDA_ARCH_PTX, major * 10 + minor, 
                     std::greater_equal<int>());
#else
	(void)major;
	(void)minor;
	return false;
#endif
}


bool cv::gpu::TargetArchs::hasEqualOrGreaterBin(int major, int minor)
{
#if defined (HAVE_CUDA)
    return ::compareToSet(CUDA_ARCH_BIN, major * 10 + minor, 
                     std::greater_equal<int>());
#else
	(void)major;
	(void)minor;
	return false;
#endif
}


#if !defined (HAVE_CUDA)

int cv::gpu::getCudaEnabledDeviceCount() { return 0; }
void cv::gpu::setDevice(int) { throw_nogpu(); } 
int cv::gpu::getDevice() { throw_nogpu(); return 0; }
void cv::gpu::resetDevice() { throw_nogpu(); }
size_t cv::gpu::DeviceInfo::freeMemory() const { throw_nogpu(); return 0; }
size_t cv::gpu::DeviceInfo::totalMemory() const { throw_nogpu(); return 0; }
bool cv::gpu::DeviceInfo::supports(cv::gpu::FeatureSet) const { throw_nogpu(); return false; }
bool cv::gpu::DeviceInfo::isCompatible() const { throw_nogpu(); return false; }
void cv::gpu::DeviceInfo::query() { throw_nogpu(); }
void cv::gpu::DeviceInfo::queryMemory(size_t&, size_t&) const { throw_nogpu(); }

#else /* !defined (HAVE_CUDA) */

int cv::gpu::getCudaEnabledDeviceCount()
{
    int count;
    cudaError_t error = cudaGetDeviceCount( &count );

    if (error == cudaErrorInsufficientDriver)
        return -1;

    if (error == cudaErrorNoDevice)
        return 0;

    cudaSafeCall(error);
    return count;  
}


void cv::gpu::setDevice(int device)
{
    cudaSafeCall( cudaSetDevice( device ) );
}


int cv::gpu::getDevice()
{
    int device;    
    cudaSafeCall( cudaGetDevice( &device ) );
    return device;
}


void cv::gpu::resetDevice()
{
    cudaSafeCall( cudaDeviceReset() );
}


size_t cv::gpu::DeviceInfo::freeMemory() const
{
    size_t free_memory, total_memory;
    queryMemory(free_memory, total_memory);
    return free_memory;
}


size_t cv::gpu::DeviceInfo::totalMemory() const
{
    size_t free_memory, total_memory;
    queryMemory(free_memory, total_memory);
    return total_memory;
}


bool cv::gpu::DeviceInfo::supports(cv::gpu::FeatureSet feature_set) const
{
    int version = majorVersion() * 10 + minorVersion();
    return version >= feature_set;
}


bool cv::gpu::DeviceInfo::isCompatible() const
{
    // Check PTX compatibility
    if (TargetArchs::hasEqualOrLessPtx(majorVersion(), minorVersion()))
        return true;

    // Check BIN compatibility
    for (int i = minorVersion(); i >= 0; --i)
        if (TargetArchs::hasBin(majorVersion(), i))
            return true;

    return false;
}


void cv::gpu::DeviceInfo::query()
{
    cudaDeviceProp prop;
    cudaSafeCall(cudaGetDeviceProperties(&prop, device_id_));
    name_ = prop.name;
    multi_processor_count_ = prop.multiProcessorCount;
    majorVersion_ = prop.major;
    minorVersion_ = prop.minor;
}


void cv::gpu::DeviceInfo::queryMemory(size_t& free_memory, size_t& total_memory) const
{
    int prev_device_id = getDevice();
    if (prev_device_id != device_id_)
        setDevice(device_id_);

    cudaSafeCall(cudaMemGetInfo(&free_memory, &total_memory));

    if (prev_device_id != device_id_)
        setDevice(prev_device_id);
}

////////////////////////////////////////////////////////////////////
// GpuFuncTable

BEGIN_OPENCV_DEVICE_NAMESPACE

void copy_to_with_mask(const DevMem2Db& src, DevMem2Db dst, int depth, const DevMem2Db& mask, int channels, const cudaStream_t& stream = 0);

template <typename T>
void set_to_gpu(const DevMem2Db& mat, const T* scalar, int channels, cudaStream_t stream);
template <typename T>
void set_to_gpu(const DevMem2Db& mat, const T* scalar, const DevMem2Db& mask, int channels, cudaStream_t stream);

void convert_gpu(const DevMem2Db& src, int sdepth, const DevMem2Db& dst, int ddepth, double alpha, double beta, cudaStream_t stream = 0);

END_OPENCV_DEVICE_NAMESPACE

namespace
{
    //////////////////////////////////////////////////////////////////////////
    // Convert

    template<int n> struct NPPTypeTraits;
    template<> struct NPPTypeTraits<CV_8U>  { typedef Npp8u npp_type; };
    template<> struct NPPTypeTraits<CV_16U> { typedef Npp16u npp_type; };
    template<> struct NPPTypeTraits<CV_16S> { typedef Npp16s npp_type; };
    template<> struct NPPTypeTraits<CV_32S> { typedef Npp32s npp_type; };
    template<> struct NPPTypeTraits<CV_32F> { typedef Npp32f npp_type; };

    template<int SDEPTH, int DDEPTH> struct NppConvertFunc
    {
        typedef typename NPPTypeTraits<SDEPTH>::npp_type src_t;
        typedef typename NPPTypeTraits<DDEPTH>::npp_type dst_t;

        typedef NppStatus (*func_ptr)(const src_t* pSrc, int nSrcStep, dst_t* pDst, int nDstStep, NppiSize oSizeROI);
    };
    template<int DDEPTH> struct NppConvertFunc<CV_32F, DDEPTH>
    {
        typedef typename NPPTypeTraits<DDEPTH>::npp_type dst_t;

        typedef NppStatus (*func_ptr)(const Npp32f* pSrc, int nSrcStep, dst_t* pDst, int nDstStep, NppiSize oSizeROI, NppRoundMode eRoundMode);
    };

    template<int SDEPTH, int DDEPTH, typename NppConvertFunc<SDEPTH, DDEPTH>::func_ptr func> struct NppCvt
    {
        typedef typename NPPTypeTraits<SDEPTH>::npp_type src_t;
        typedef typename NPPTypeTraits<DDEPTH>::npp_type dst_t;

        static void cvt(const GpuMat& src, GpuMat& dst)
        {
            NppiSize sz;
            sz.width = src.cols;
            sz.height = src.rows;
            nppSafeCall( func(src.ptr<src_t>(), static_cast<int>(src.step), dst.ptr<dst_t>(), static_cast<int>(dst.step), sz) );

            cudaSafeCall( cudaDeviceSynchronize() );
        }
    };
    template<int DDEPTH, typename NppConvertFunc<CV_32F, DDEPTH>::func_ptr func> struct NppCvt<CV_32F, DDEPTH, func>
    {
        typedef typename NPPTypeTraits<DDEPTH>::npp_type dst_t;

        static void cvt(const GpuMat& src, GpuMat& dst)
        {
            NppiSize sz;
            sz.width = src.cols;
            sz.height = src.rows;
            nppSafeCall( func(src.ptr<Npp32f>(), static_cast<int>(src.step), dst.ptr<dst_t>(), static_cast<int>(dst.step), sz, NPP_RND_NEAR) );

            cudaSafeCall( cudaDeviceSynchronize() );
        }
    };

    void convertToKernelCaller(const GpuMat& src, GpuMat& dst)
    {
        OPENCV_DEVICE_NAMESPACE_ convert_gpu(src.reshape(1), src.depth(), dst.reshape(1), dst.depth(), 1.0, 0.0);
    }

    //////////////////////////////////////////////////////////////////////////
    // Set
    
    template<int SDEPTH, int SCN> struct NppSetFunc
    {
        typedef typename NPPTypeTraits<SDEPTH>::npp_type src_t;

        typedef NppStatus (*func_ptr)(const src_t values[], src_t* pSrc, int nSrcStep, NppiSize oSizeROI);
    };
    template<int SDEPTH> struct NppSetFunc<SDEPTH, 1>
    {
        typedef typename NPPTypeTraits<SDEPTH>::npp_type src_t;

        typedef NppStatus (*func_ptr)(src_t val, src_t* pSrc, int nSrcStep, NppiSize oSizeROI);
    };

    template<int SDEPTH, int SCN, typename NppSetFunc<SDEPTH, SCN>::func_ptr func> struct NppSet
    {
        typedef typename NPPTypeTraits<SDEPTH>::npp_type src_t;

        static void set(GpuMat& src, Scalar s)
        {
            NppiSize sz;
            sz.width = src.cols;
            sz.height = src.rows;

            Scalar_<src_t> nppS = s;

            nppSafeCall( func(nppS.val, src.ptr<src_t>(), static_cast<int>(src.step), sz) );

            cudaSafeCall( cudaDeviceSynchronize() );
        }
    };
    template<int SDEPTH, typename NppSetFunc<SDEPTH, 1>::func_ptr func> struct NppSet<SDEPTH, 1, func>
    {
        typedef typename NPPTypeTraits<SDEPTH>::npp_type src_t;

        static void set(GpuMat& src, Scalar s)
        {
            NppiSize sz;
            sz.width = src.cols;
            sz.height = src.rows;

            Scalar_<src_t> nppS = s;

            nppSafeCall( func(nppS[0], src.ptr<src_t>(), static_cast<int>(src.step), sz) );

            cudaSafeCall( cudaDeviceSynchronize() );
        }
    };

    template <typename T>
    void kernelSet(GpuMat& src, Scalar s)
    {
        Scalar_<T> sf = s;
        OPENCV_DEVICE_NAMESPACE_ set_to_gpu(src, sf.val, src.channels(), 0);
    }

    template<int SDEPTH, int SCN> struct NppSetMaskFunc
    {
        typedef typename NPPTypeTraits<SDEPTH>::npp_type src_t;

        typedef NppStatus (*func_ptr)(const src_t values[], src_t* pSrc, int nSrcStep, NppiSize oSizeROI, const Npp8u* pMask, int nMaskStep);
    };
    template<int SDEPTH> struct NppSetMaskFunc<SDEPTH, 1>
    {
        typedef typename NPPTypeTraits<SDEPTH>::npp_type src_t;

        typedef NppStatus (*func_ptr)(src_t val, src_t* pSrc, int nSrcStep, NppiSize oSizeROI, const Npp8u* pMask, int nMaskStep);
    };

    template<int SDEPTH, int SCN, typename NppSetMaskFunc<SDEPTH, SCN>::func_ptr func> struct NppSetMask
    {
        typedef typename NPPTypeTraits<SDEPTH>::npp_type src_t;

        static void set(GpuMat& src, Scalar s, const GpuMat& mask)
        {
            NppiSize sz;
            sz.width = src.cols;
            sz.height = src.rows;

            Scalar_<src_t> nppS = s;

            nppSafeCall( func(nppS.val, src.ptr<src_t>(), static_cast<int>(src.step), sz, mask.ptr<Npp8u>(), static_cast<int>(mask.step)) );

            cudaSafeCall( cudaDeviceSynchronize() );
        }
    };
    template<int SDEPTH, typename NppSetMaskFunc<SDEPTH, 1>::func_ptr func> struct NppSetMask<SDEPTH, 1, func>
    {
        typedef typename NPPTypeTraits<SDEPTH>::npp_type src_t;

        static void set(GpuMat& src, Scalar s, const GpuMat& mask)
        {
            NppiSize sz;
            sz.width = src.cols;
            sz.height = src.rows;

            Scalar_<src_t> nppS = s;

            nppSafeCall( func(nppS[0], src.ptr<src_t>(), static_cast<int>(src.step), sz, mask.ptr<Npp8u>(), static_cast<int>(mask.step)) );

            cudaSafeCall( cudaDeviceSynchronize() );
        }
    };

    template <typename T>
    void kernelSetMask(GpuMat& src, Scalar s, const GpuMat& mask)
    {
        Scalar_<T> sf = s;
        OPENCV_DEVICE_NAMESPACE_ set_to_gpu(src, sf.val, mask, src.channels(), 0);
    }

    class CudaFuncTable : public GpuFuncTable
    {
    public:
        void copy(const Mat& src, GpuMat& dst) const 
        { 
            cudaSafeCall( cudaMemcpy2D(dst.data, dst.step, src.data, src.step, src.cols * src.elemSize(), src.rows, cudaMemcpyHostToDevice) );
        }
        void copy(const GpuMat& src, Mat& dst) const
        { 
            cudaSafeCall( cudaMemcpy2D(dst.data, dst.step, src.data, src.step, src.cols * src.elemSize(), src.rows, cudaMemcpyDeviceToHost) );
        }
        void copy(const GpuMat& src, GpuMat& dst) const
        { 
            cudaSafeCall( cudaMemcpy2D(dst.data, dst.step, src.data, src.step, src.cols * src.elemSize(), src.rows, cudaMemcpyDeviceToDevice) );
        }

        void copyWithMask(const GpuMat& src, GpuMat& dst, const GpuMat& mask) const 
        { 
            OPENCV_DEVICE_NAMESPACE_ copy_to_with_mask(src, dst, src.depth(), mask, src.channels());
        }

        void convert(const GpuMat& src, GpuMat& dst) const 
        { 
            typedef void (*caller_t)(const GpuMat& src, GpuMat& dst);
            static const caller_t callers[7][7][7] =
            {
                {                
                    /*  8U ->  8U */ {0, 0, 0, 0},
                    /*  8U ->  8S */ {convertToKernelCaller, convertToKernelCaller, convertToKernelCaller, convertToKernelCaller},
                    /*  8U -> 16U */ {NppCvt<CV_8U, CV_16U, nppiConvert_8u16u_C1R>::cvt,convertToKernelCaller,convertToKernelCaller,NppCvt<CV_8U, CV_16U, nppiConvert_8u16u_C4R>::cvt},
                    /*  8U -> 16S */ {NppCvt<CV_8U, CV_16S, nppiConvert_8u16s_C1R>::cvt,convertToKernelCaller,convertToKernelCaller,NppCvt<CV_8U, CV_16S, nppiConvert_8u16s_C4R>::cvt},
                    /*  8U -> 32S */ {convertToKernelCaller,convertToKernelCaller,convertToKernelCaller,convertToKernelCaller},
                    /*  8U -> 32F */ {NppCvt<CV_8U, CV_32F, nppiConvert_8u32f_C1R>::cvt,convertToKernelCaller,convertToKernelCaller,convertToKernelCaller},
                    /*  8U -> 64F */ {convertToKernelCaller,convertToKernelCaller,convertToKernelCaller,convertToKernelCaller}
                },
                {
                    /*  8S ->  8U */ {convertToKernelCaller,convertToKernelCaller,convertToKernelCaller,convertToKernelCaller},
                    /*  8S ->  8S */ {0,0,0,0},
                    /*  8S -> 16U */ {convertToKernelCaller,convertToKernelCaller,convertToKernelCaller,convertToKernelCaller},
                    /*  8S -> 16S */ {convertToKernelCaller,convertToKernelCaller,convertToKernelCaller,convertToKernelCaller},
                    /*  8S -> 32S */ {convertToKernelCaller,convertToKernelCaller,convertToKernelCaller,convertToKernelCaller},
                    /*  8S -> 32F */ {convertToKernelCaller,convertToKernelCaller,convertToKernelCaller,convertToKernelCaller},
                    /*  8S -> 64F */ {convertToKernelCaller,convertToKernelCaller,convertToKernelCaller,convertToKernelCaller}
                },
                {
                    /* 16U ->  8U */ {NppCvt<CV_16U, CV_8U, nppiConvert_16u8u_C1R>::cvt,convertToKernelCaller,convertToKernelCaller,NppCvt<CV_16U, CV_8U, nppiConvert_16u8u_C4R>::cvt},
                    /* 16U ->  8S */ {convertToKernelCaller,convertToKernelCaller,convertToKernelCaller,convertToKernelCaller},
                    /* 16U -> 16U */ {0,0,0,0},
                    /* 16U -> 16S */ {convertToKernelCaller,convertToKernelCaller,convertToKernelCaller,convertToKernelCaller},
                    /* 16U -> 32S */ {NppCvt<CV_16U, CV_32S, nppiConvert_16u32s_C1R>::cvt,convertToKernelCaller,convertToKernelCaller,convertToKernelCaller},
                    /* 16U -> 32F */ {NppCvt<CV_16U, CV_32F, nppiConvert_16u32f_C1R>::cvt,convertToKernelCaller,convertToKernelCaller,convertToKernelCaller},
                    /* 16U -> 64F */ {convertToKernelCaller,convertToKernelCaller,convertToKernelCaller,convertToKernelCaller}
                },
                {
                    /* 16S ->  8U */ {NppCvt<CV_16S, CV_8U, nppiConvert_16s8u_C1R>::cvt,convertToKernelCaller,convertToKernelCaller,NppCvt<CV_16S, CV_8U, nppiConvert_16s8u_C4R>::cvt},
                    /* 16S ->  8S */ {convertToKernelCaller,convertToKernelCaller,convertToKernelCaller,convertToKernelCaller},
                    /* 16S -> 16U */ {convertToKernelCaller,convertToKernelCaller,convertToKernelCaller,convertToKernelCaller},
                    /* 16S -> 16S */ {0,0,0,0},
                    /* 16S -> 32S */ {NppCvt<CV_16S, CV_32S, nppiConvert_16s32s_C1R>::cvt,convertToKernelCaller,convertToKernelCaller,convertToKernelCaller},
                    /* 16S -> 32F */ {NppCvt<CV_16S, CV_32F, nppiConvert_16s32f_C1R>::cvt,convertToKernelCaller,convertToKernelCaller,convertToKernelCaller},
                    /* 16S -> 64F */ {convertToKernelCaller,convertToKernelCaller,convertToKernelCaller,convertToKernelCaller}
                },
                {
                    /* 32S ->  8U */ {convertToKernelCaller,convertToKernelCaller,convertToKernelCaller,convertToKernelCaller},
                    /* 32S ->  8S */ {convertToKernelCaller,convertToKernelCaller,convertToKernelCaller,convertToKernelCaller},
                    /* 32S -> 16U */ {convertToKernelCaller,convertToKernelCaller,convertToKernelCaller,convertToKernelCaller},
                    /* 32S -> 16S */ {convertToKernelCaller,convertToKernelCaller,convertToKernelCaller,convertToKernelCaller},
                    /* 32S -> 32S */ {0,0,0,0},
                    /* 32S -> 32F */ {convertToKernelCaller,convertToKernelCaller,convertToKernelCaller,convertToKernelCaller},
                    /* 32S -> 64F */ {convertToKernelCaller,convertToKernelCaller,convertToKernelCaller,convertToKernelCaller}
                },
                {
                    /* 32F ->  8U */ {NppCvt<CV_32F, CV_8U, nppiConvert_32f8u_C1R>::cvt,convertToKernelCaller,convertToKernelCaller,convertToKernelCaller},
                    /* 32F ->  8S */ {convertToKernelCaller,convertToKernelCaller,convertToKernelCaller,convertToKernelCaller},
                    /* 32F -> 16U */ {NppCvt<CV_32F, CV_16U, nppiConvert_32f16u_C1R>::cvt,convertToKernelCaller,convertToKernelCaller,convertToKernelCaller},
                    /* 32F -> 16S */ {NppCvt<CV_32F, CV_16S, nppiConvert_32f16s_C1R>::cvt,convertToKernelCaller,convertToKernelCaller,convertToKernelCaller},
                    /* 32F -> 32S */ {convertToKernelCaller,convertToKernelCaller,convertToKernelCaller,convertToKernelCaller},
                    /* 32F -> 32F */ {0,0,0,0},
                    /* 32F -> 64F */ {convertToKernelCaller,convertToKernelCaller,convertToKernelCaller,convertToKernelCaller}
                },
                {
                    /* 64F ->  8U */ {convertToKernelCaller,convertToKernelCaller,convertToKernelCaller,convertToKernelCaller},
                    /* 64F ->  8S */ {convertToKernelCaller,convertToKernelCaller,convertToKernelCaller,convertToKernelCaller},
                    /* 64F -> 16U */ {convertToKernelCaller,convertToKernelCaller,convertToKernelCaller,convertToKernelCaller},
                    /* 64F -> 16S */ {convertToKernelCaller,convertToKernelCaller,convertToKernelCaller,convertToKernelCaller},
                    /* 64F -> 32S */ {convertToKernelCaller,convertToKernelCaller,convertToKernelCaller,convertToKernelCaller},
                    /* 64F -> 32F */ {convertToKernelCaller,convertToKernelCaller,convertToKernelCaller,convertToKernelCaller},
                    /* 64F -> 64F */ {0,0,0,0}
                }
            };

            caller_t func = callers[src.depth()][dst.depth()][src.channels() - 1];
            CV_DbgAssert(func != 0);

            func(src, dst);
        }

        void convert(const GpuMat& src, GpuMat& dst, double alpha, double beta) const 
        { 
            device::convert_gpu(src.reshape(1), src.depth(), dst.reshape(1), dst.depth(), alpha, beta);
        }

        void setTo(GpuMat& m, Scalar s, const GpuMat& mask) const
        {
            NppiSize sz;
            sz.width  = m.cols;
            sz.height = m.rows;

            if (mask.empty())
            {
                if (s[0] == 0.0 && s[1] == 0.0 && s[2] == 0.0 && s[3] == 0.0)
                {
                    cudaSafeCall( cudaMemset2D(m.data, m.step, 0, m.cols * m.elemSize(), m.rows) );
                    return;
                }

                if (m.depth() == CV_8U)
                {
                    int cn = m.channels();

                    if (cn == 1 || (cn == 2 && s[0] == s[1]) || (cn == 3 && s[0] == s[1] && s[0] == s[2]) || (cn == 4 && s[0] == s[1] && s[0] == s[2] && s[0] == s[3]))
                    {
                        int val = saturate_cast<uchar>(s[0]);
                        cudaSafeCall( cudaMemset2D(m.data, m.step, val, m.cols * m.elemSize(), m.rows) );
                        return;
                    }
                }

                typedef void (*caller_t)(GpuMat& src, Scalar s);
                static const caller_t callers[7][4] =
                {
                    {NppSet<CV_8U, 1, nppiSet_8u_C1R>::set,kernelSet<uchar>,kernelSet<uchar>,NppSet<CV_8U, 4, nppiSet_8u_C4R>::set},
                    {kernelSet<schar>,kernelSet<schar>,kernelSet<schar>,kernelSet<schar>},
                    {NppSet<CV_16U, 1, nppiSet_16u_C1R>::set,NppSet<CV_16U, 2, nppiSet_16u_C2R>::set,kernelSet<ushort>,NppSet<CV_16U, 4, nppiSet_16u_C4R>::set},
                    {NppSet<CV_16S, 1, nppiSet_16s_C1R>::set,NppSet<CV_16S, 2, nppiSet_16s_C2R>::set,kernelSet<short>,NppSet<CV_16S, 4, nppiSet_16s_C4R>::set},
                    {NppSet<CV_32S, 1, nppiSet_32s_C1R>::set,kernelSet<int>,kernelSet<int>,NppSet<CV_32S, 4, nppiSet_32s_C4R>::set},
                    {NppSet<CV_32F, 1, nppiSet_32f_C1R>::set,kernelSet<float>,kernelSet<float>,NppSet<CV_32F, 4, nppiSet_32f_C4R>::set},
                    {kernelSet<double>,kernelSet<double>,kernelSet<double>,kernelSet<double>}
                };

                callers[m.depth()][m.channels() - 1](m, s);
            }
            else
            {
                typedef void (*caller_t)(GpuMat& src, Scalar s, const GpuMat& mask);

                static const caller_t callers[7][4] =
                {
                    {NppSetMask<CV_8U, 1, nppiSet_8u_C1MR>::set,kernelSetMask<uchar>,kernelSetMask<uchar>,NppSetMask<CV_8U, 4, nppiSet_8u_C4MR>::set},
                    {kernelSetMask<schar>,kernelSetMask<schar>,kernelSetMask<schar>,kernelSetMask<schar>},
                    {NppSetMask<CV_16U, 1, nppiSet_16u_C1MR>::set,kernelSetMask<ushort>,kernelSetMask<ushort>,NppSetMask<CV_16U, 4, nppiSet_16u_C4MR>::set},
                    {NppSetMask<CV_16S, 1, nppiSet_16s_C1MR>::set,kernelSetMask<short>,kernelSetMask<short>,NppSetMask<CV_16S, 4, nppiSet_16s_C4MR>::set},
                    {NppSetMask<CV_32S, 1, nppiSet_32s_C1MR>::set,kernelSetMask<int>,kernelSetMask<int>,NppSetMask<CV_32S, 4, nppiSet_32s_C4MR>::set},
                    {NppSetMask<CV_32F, 1, nppiSet_32f_C1MR>::set,kernelSetMask<float>,kernelSetMask<float>,NppSetMask<CV_32F, 4, nppiSet_32f_C4MR>::set},
                    {kernelSetMask<double>,kernelSetMask<double>,kernelSetMask<double>,kernelSetMask<double>}
                };

                callers[m.depth()][m.channels() - 1](m, s, mask);
            }
        }

        void mallocPitch(void** devPtr, size_t* step, size_t width, size_t height) const
        {
            cudaSafeCall( cudaMallocPitch(devPtr, step, width, height) );
        }

        void free(void* devPtr) const
        {
            cudaFree(devPtr);
        }
    };

    class Initializer
    {
    public:
        Initializer()
        {
            static CudaFuncTable funcTable;
            setGpuFuncTable(&funcTable);
        }
    };

    Initializer init;
}

#endif

