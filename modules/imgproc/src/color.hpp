// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "precomp.hpp"
#include <limits>
#include "opencl_kernels_imgproc.hpp"
#include "hal_replacement.hpp"
#include "opencv2/core/hal/intrin.hpp"
#include "opencv2/core/softfloat.hpp"

#define  CV_DESCALE(x,n)     (((x) + (1 << ((n)-1))) >> (n))

namespace cv
{
//constants for conversion from/to RGB and Gray, YUV, YCrCb according to BT.601
const float B2YF = 0.114f;
const float G2YF = 0.587f;
const float R2YF = 0.299f;
//to YCbCr
const float YCBF = 0.564f; // == 1/2/(1-B2YF)
const float YCRF = 0.713f; // == 1/2/(1-R2YF)
const int YCBI = 9241;  // == YCBF*16384
const int YCRI = 11682; // == YCRF*16384
//to YUV
const float B2UF = 0.492f;
const float R2VF = 0.877f;
const int B2UI = 8061;  // == B2UF*16384
const int R2VI = 14369; // == R2VF*16384
//from YUV
const float U2BF = 2.032f;
const float U2GF = -0.395f;
const float V2GF = -0.581f;
const float V2RF = 1.140f;
const int U2BI = 33292;
const int U2GI = -6472;
const int V2GI = -9519;
const int V2RI = 18678;
//from YCrCb
const float CB2BF = 1.773f;
const float CB2GF = -0.344f;
const float CR2GF = -0.714f;
const float CR2RF = 1.403f;
const int CB2BI = 29049;
const int CB2GI = -5636;
const int CR2GI = -11698;
const int CR2RI = 22987;

enum
{
    yuv_shift = 14,
    xyz_shift = 12,
    R2Y = 4899, // == R2YF*16384
    G2Y = 9617, // == G2YF*16384
    B2Y = 1868, // == B2YF*16384
    BLOCK_SIZE = 256
};


template<typename _Tp> struct ColorChannel
{
    typedef float worktype_f;
    static _Tp max() { return std::numeric_limits<_Tp>::max(); }
    static _Tp half() { return (_Tp)(max()/2 + 1); }
};

template<> struct ColorChannel<float>
{
    typedef float worktype_f;
    static float max() { return 1.f; }
    static float half() { return 0.5f; }
};

/*template<> struct ColorChannel<double>
{
    typedef double worktype_f;
    static double max() { return 1.; }
    static double half() { return 0.5; }
};*/

//
// Helper functions
//

namespace {

inline bool isHSV(int code)
{
    using namespace cv;
    switch(code)
    {
    case COLOR_HSV2BGR: case COLOR_HSV2RGB: case COLOR_HSV2BGR_FULL: case COLOR_HSV2RGB_FULL:
    case COLOR_BGR2HSV: case COLOR_RGB2HSV: case COLOR_BGR2HSV_FULL: case COLOR_RGB2HSV_FULL:
        return true;
    default:
        return false;
    }
}

inline bool isLab(int code)
{
    using namespace cv;
    switch (code)
    {
    case COLOR_Lab2BGR: case COLOR_Lab2RGB: case COLOR_Lab2LBGR: case COLOR_Lab2LRGB:
    case COLOR_BGR2Lab: case COLOR_RGB2Lab: case COLOR_LBGR2Lab: case COLOR_LRGB2Lab:
        return true;
    default:
        return false;
    }
}

inline bool issRGB(int code)
{
    using namespace cv;
    switch (code)
    {
    case COLOR_BGR2Lab: case COLOR_RGB2Lab: case COLOR_BGR2Luv: case COLOR_RGB2Luv:
    case COLOR_Lab2BGR: case COLOR_Lab2RGB: case COLOR_Luv2BGR: case COLOR_Luv2RGB:
        return true;
    default:
        return false;
    }
}

inline bool swapBlue(int code)
{
    using namespace cv;
    switch (code)
    {
    case COLOR_BGR2BGRA: case COLOR_BGRA2BGR:
    case COLOR_BGR2BGR565: case COLOR_BGR2BGR555: case COLOR_BGRA2BGR565: case COLOR_BGRA2BGR555:
    case COLOR_BGR5652BGR: case COLOR_BGR5552BGR: case COLOR_BGR5652BGRA: case COLOR_BGR5552BGRA:
    case COLOR_BGR2GRAY: case COLOR_BGRA2GRAY:
    case COLOR_BGR2YCrCb: case COLOR_BGR2YUV:
    case COLOR_YCrCb2BGR: case COLOR_YUV2BGR:
    case COLOR_BGR2XYZ: case COLOR_XYZ2BGR:
    case COLOR_BGR2HSV: case COLOR_BGR2HLS: case COLOR_BGR2HSV_FULL: case COLOR_BGR2HLS_FULL:
    case COLOR_YUV2BGR_YV12: case COLOR_YUV2BGRA_YV12: case COLOR_YUV2BGR_IYUV: case COLOR_YUV2BGRA_IYUV:
    case COLOR_YUV2BGR_NV21: case COLOR_YUV2BGRA_NV21: case COLOR_YUV2BGR_NV12: case COLOR_YUV2BGRA_NV12:
    case COLOR_Lab2BGR: case COLOR_Luv2BGR: case COLOR_Lab2LBGR: case COLOR_Luv2LBGR:
    case COLOR_BGR2Lab: case COLOR_BGR2Luv: case COLOR_LBGR2Lab: case COLOR_LBGR2Luv:
    case COLOR_HSV2BGR: case COLOR_HLS2BGR: case COLOR_HSV2BGR_FULL: case COLOR_HLS2BGR_FULL:
    case COLOR_YUV2BGR_UYVY: case COLOR_YUV2BGRA_UYVY: case COLOR_YUV2BGR_YUY2:
    case COLOR_YUV2BGRA_YUY2:  case COLOR_YUV2BGR_YVYU: case COLOR_YUV2BGRA_YVYU:
    case COLOR_BGR2YUV_IYUV: case COLOR_BGRA2YUV_IYUV: case COLOR_BGR2YUV_YV12: case COLOR_BGRA2YUV_YV12:
        return false;
    default:
        return true;
    }
}

inline bool isFullRange(int code)
{
    using namespace cv;
    switch (code)
    {
    case COLOR_BGR2HSV_FULL: case COLOR_RGB2HSV_FULL: case COLOR_BGR2HLS_FULL: case COLOR_RGB2HLS_FULL:
    case COLOR_HSV2BGR_FULL: case COLOR_HSV2RGB_FULL: case COLOR_HLS2BGR_FULL: case COLOR_HLS2RGB_FULL:
        return true;
    default:
        return false;
    }
}

inline int dstChannels(int code)
{
    switch( code )
    {
        case COLOR_BGR2BGRA: case COLOR_RGB2BGRA: case COLOR_BGRA2RGBA:
        case COLOR_BGR5652BGRA: case COLOR_BGR5552BGRA: case COLOR_BGR5652RGBA: case COLOR_BGR5552RGBA:
        case COLOR_GRAY2BGRA:
        case COLOR_YUV2BGRA_NV21: case COLOR_YUV2RGBA_NV21: case COLOR_YUV2BGRA_NV12: case COLOR_YUV2RGBA_NV12:
        case COLOR_YUV2BGRA_YV12: case COLOR_YUV2RGBA_YV12: case COLOR_YUV2BGRA_IYUV: case COLOR_YUV2RGBA_IYUV:
        case COLOR_YUV2RGBA_UYVY: case COLOR_YUV2BGRA_UYVY: case COLOR_YUV2RGBA_YVYU: case COLOR_YUV2BGRA_YVYU:
        case COLOR_YUV2RGBA_YUY2: case COLOR_YUV2BGRA_YUY2:

            return 4;

        case COLOR_BGRA2BGR: case COLOR_RGBA2BGR: case COLOR_RGB2BGR:
        case COLOR_BGR5652BGR: case COLOR_BGR5552BGR: case COLOR_BGR5652RGB: case COLOR_BGR5552RGB:
        case COLOR_GRAY2BGR:
        case COLOR_YUV2BGR_NV21: case COLOR_YUV2RGB_NV21: case COLOR_YUV2BGR_NV12: case COLOR_YUV2RGB_NV12:
        case COLOR_YUV2BGR_YV12: case COLOR_YUV2RGB_YV12: case COLOR_YUV2BGR_IYUV: case COLOR_YUV2RGB_IYUV:
        case COLOR_YUV2RGB_UYVY: case COLOR_YUV2BGR_UYVY: case COLOR_YUV2RGB_YVYU: case COLOR_YUV2BGR_YVYU:
        case COLOR_YUV2RGB_YUY2: case COLOR_YUV2BGR_YUY2:

            return 3;

        default:
            return 0;
    }
}

inline int greenBits(int code)
{
    switch( code )
    {
        case COLOR_BGR2BGR565: case COLOR_RGB2BGR565: case COLOR_BGRA2BGR565: case COLOR_RGBA2BGR565:
        case COLOR_BGR5652BGR: case COLOR_BGR5652RGB: case COLOR_BGR5652BGRA: case COLOR_BGR5652RGBA:
        case COLOR_BGR5652GRAY: case COLOR_GRAY2BGR565:

            return 6;

        case COLOR_BGR2BGR555: case COLOR_RGB2BGR555: case COLOR_BGRA2BGR555: case COLOR_RGBA2BGR555:
        case COLOR_BGR5552BGR: case COLOR_BGR5552RGB: case COLOR_BGR5552BGRA: case COLOR_BGR5552RGBA:
        case COLOR_BGR5552GRAY: case COLOR_GRAY2BGR555:

            return 5;

        default:
            return 0;
    }
}

inline int uIndex(int code)
{
    switch( code )
    {
        case COLOR_RGB2YUV_YV12: case COLOR_BGR2YUV_YV12: case COLOR_RGBA2YUV_YV12: case COLOR_BGRA2YUV_YV12:

            return 2;

        case COLOR_YUV2RGB_YVYU: case COLOR_YUV2BGR_YVYU: case COLOR_YUV2RGBA_YVYU: case COLOR_YUV2BGRA_YVYU:
        case COLOR_RGB2YUV_IYUV: case COLOR_BGR2YUV_IYUV: case COLOR_RGBA2YUV_IYUV: case COLOR_BGRA2YUV_IYUV:
        case COLOR_YUV2BGR_NV21:  case COLOR_YUV2RGB_NV21: case COLOR_YUV2BGRA_NV21: case COLOR_YUV2RGBA_NV21:
        case COLOR_YUV2BGR_YV12: case COLOR_YUV2RGB_YV12: case COLOR_YUV2BGRA_YV12: case COLOR_YUV2RGBA_YV12:

            return 1;

        case COLOR_YUV2BGR_NV12:  case COLOR_YUV2RGB_NV12: case COLOR_YUV2BGRA_NV12: case COLOR_YUV2RGBA_NV12:
        case COLOR_YUV2BGR_IYUV: case COLOR_YUV2RGB_IYUV: case COLOR_YUV2BGRA_IYUV: case COLOR_YUV2RGBA_IYUV:
        case COLOR_YUV2RGB_UYVY: case COLOR_YUV2BGR_UYVY: case COLOR_YUV2RGBA_UYVY: case COLOR_YUV2BGRA_UYVY:
        case COLOR_YUV2RGB_YUY2: case COLOR_YUV2BGR_YUY2: case COLOR_YUV2RGBA_YUY2: case COLOR_YUV2BGRA_YUY2:

            return 0;

        default:
            return -1;
    }
}

} // namespace::

template<int i0, int i1 = -1, int i2 = -1>
struct ValueSet
{
    static bool contains(int i)
    {
        return (i == i0 || i == i1 || i == i2);
    }
};

template<int i0, int i1>
struct ValueSet<i0, i1, -1>
{
    static bool contains(int i)
    {
        return (i == i0 || i == i1);
    }
};

template<int i0>
struct ValueSet<i0, -1, -1>
{
    static bool contains(int i)
    {
        return (i == i0);
    }
};

template< typename VScn, typename VDcn, typename VDepth >
struct CvtHelper
{
    CvtHelper(InputArray _src, OutputArray _dst, int dcn)
    {
        int stype = _src.type();
        scn = CV_MAT_CN(stype), depth = CV_MAT_DEPTH(stype);

        CV_Assert( VScn::contains(scn) && VDcn::contains(dcn) && VDepth::contains(depth) );

        if (_src.getObj() == _dst.getObj()) // inplace processing (#6653)
            _src.copyTo(src);
        else
            src = _src.getMat();

        _dst.create(src.size(), CV_MAKETYPE(depth, dcn));
        dst = _dst.getMat();
    }
    Mat src, dst;
    int depth, scn;
};


template< typename VScn, typename VDcn, typename VDepth >
struct OclHelper
{
    OclHelper( InputArray _src, OutputArray _dst, int dcn)
    {
        src = _src.getUMat();
        Size sz = src.size(), dstSz = sz;
        int scn = src.channels();
        int depth = src.depth();

        CV_Assert( VScn::contains(scn) && VDcn::contains(dcn) && VDepth::contains(depth) );

        _dst.create(dstSz, CV_MAKETYPE(depth, dcn));
        dst = _dst.getUMat();

        srcarg = ocl::KernelArg::ReadOnlyNoSize(src);
        dstarg = ocl::KernelArg::WriteOnly(dst);
    }

    bool createKernel(cv::String name, ocl::ProgramSource& source, cv::String options)
    {
        ocl::Device dev = ocl::Device::getDefault();
        int pxPerWIy = dev.isIntel() && (dev.type() & ocl::Device::TYPE_GPU) ? 4 : 1;

        globalSize[0] = (size_t)src.cols;
        globalSize[1] = ((size_t)src.rows + pxPerWIy - 1) / pxPerWIy;

        cv::String baseOptions = format("-D depth=%d -D scn=%d -D PIX_PER_WI_Y=%d ",
                                        src.depth(), src.channels(), pxPerWIy);

        k.create(name.c_str(), source, baseOptions + options);

        if(k.empty())
            return false;

        nArgs = k.set(0, srcarg);
        nArgs = k.set(nArgs, dstarg);
        return true;
    }

    bool run()
    {
        return k.run(2, globalSize, NULL, false);
    }

    template<typename T>
    void setArg(const T& arg)
    {
        nArgs = k.set(nArgs, arg);
    }

    UMat src, dst;
    ocl::Kernel k;
    size_t globalSize[2];
    ocl::KernelArg srcarg, dstarg;
    int nArgs;
};

///////////////////////////// Top-level template function ////////////////////////////////

template <typename Cvt>
class CvtColorLoop_Invoker : public ParallelLoopBody
{
    typedef typename Cvt::channel_type _Tp;
public:

    CvtColorLoop_Invoker(const uchar * src_data_, size_t src_step_, uchar * dst_data_, size_t dst_step_, int width_, const Cvt& _cvt) :
        ParallelLoopBody(), src_data(src_data_), src_step(src_step_), dst_data(dst_data_), dst_step(dst_step_),
        width(width_), cvt(_cvt)
    {
    }

    virtual void operator()(const Range& range) const
    {
        CV_TRACE_FUNCTION();

        const uchar* yS = src_data + static_cast<size_t>(range.start) * src_step;
        uchar* yD = dst_data + static_cast<size_t>(range.start) * dst_step;

        for( int i = range.start; i < range.end; ++i, yS += src_step, yD += dst_step )
            cvt(reinterpret_cast<const _Tp*>(yS), reinterpret_cast<_Tp*>(yD), width);
    }

private:
    const uchar * src_data;
    size_t src_step;
    uchar * dst_data;
    size_t dst_step;
    int width;
    const Cvt& cvt;

    const CvtColorLoop_Invoker& operator= (const CvtColorLoop_Invoker&);
};

template <typename Cvt>
void CvtColorLoop(const uchar * src_data, size_t src_step, uchar * dst_data, size_t dst_step, int width, int height, const Cvt& cvt)
{
    parallel_for_(Range(0, height),
                  CvtColorLoop_Invoker<Cvt>(src_data, src_step, dst_data, dst_step, width, cvt),
                  (width * height) / static_cast<double>(1<<16));
}

#define NEED_IPP (defined (HAVE_IPP) && (IPP_VERSION_X100 >= 700))

#if NEED_IPP

#define MAX_IPP8u   255
#define MAX_IPP16u  65535
#define MAX_IPP32f  1.0

typedef IppStatus (CV_STDCALL* ippiReorderFunc)(const void *, int, void *, int, IppiSize, const int *);
typedef IppStatus (CV_STDCALL* ippiGeneralFunc)(const void *, int, void *, int, IppiSize);
typedef IppStatus (CV_STDCALL* ippiColor2GrayFunc)(const void *, int, void *, int, IppiSize, const Ipp32f *);

template <typename Cvt>
class CvtColorIPPLoop_Invoker :
        public ParallelLoopBody
{
public:

    CvtColorIPPLoop_Invoker(const uchar * src_data_, size_t src_step_, uchar * dst_data_, size_t dst_step_, int width_, const Cvt& _cvt, bool *_ok) :
        ParallelLoopBody(), src_data(src_data_), src_step(src_step_), dst_data(dst_data_), dst_step(dst_step_), width(width_), cvt(_cvt), ok(_ok)
    {
        *ok = true;
    }

    virtual void operator()(const Range& range) const
    {
        const void *yS = src_data + src_step * range.start;
        void *yD = dst_data + dst_step * range.start;
        if( !cvt(yS, static_cast<int>(src_step), yD, static_cast<int>(dst_step), width, range.end - range.start) )
            *ok = false;
        else
        {
            CV_IMPL_ADD(CV_IMPL_IPP|CV_IMPL_MT);
        }
    }

private:
    const uchar * src_data;
    size_t src_step;
    uchar * dst_data;
    size_t dst_step;
    int width;
    const Cvt& cvt;
    bool *ok;

    const CvtColorIPPLoop_Invoker& operator= (const CvtColorIPPLoop_Invoker&);
};


template <typename Cvt>
bool CvtColorIPPLoop(const uchar * src_data, size_t src_step, uchar * dst_data, size_t dst_step, int width, int height, const Cvt& cvt)
{
    bool ok;
    parallel_for_(Range(0, height), CvtColorIPPLoop_Invoker<Cvt>(src_data, src_step, dst_data, dst_step, width, cvt, &ok), (width * height)/(double)(1<<16) );
    return ok;
}


template <typename Cvt>
bool CvtColorIPPLoopCopy(const uchar * src_data, size_t src_step, int src_type, uchar * dst_data, size_t dst_step, int width, int height, const Cvt& cvt)
{
    Mat temp;
    Mat src(Size(width, height), src_type, const_cast<uchar*>(src_data), src_step);
    Mat source = src;
    if( src_data == dst_data )
    {
        src.copyTo(temp);
        source = temp;
    }
    bool ok;
    parallel_for_(Range(0, source.rows),
                  CvtColorIPPLoop_Invoker<Cvt>(source.data, source.step, dst_data, dst_step,
                                               source.cols, cvt, &ok),
                  source.total()/(double)(1<<16) );
    return ok;
}


struct IPPGeneralFunctor
{
    IPPGeneralFunctor(ippiGeneralFunc _func) : ippiColorConvertGeneral(_func){}
    bool operator()(const void *src, int srcStep, void *dst, int dstStep, int cols, int rows) const
    {
        return ippiColorConvertGeneral ? CV_INSTRUMENT_FUN_IPP(ippiColorConvertGeneral, src, srcStep, dst, dstStep, ippiSize(cols, rows)) >= 0 : false;
    }
private:
    ippiGeneralFunc ippiColorConvertGeneral;
};


struct IPPReorderFunctor
{
    IPPReorderFunctor(ippiReorderFunc _func, int _order0, int _order1, int _order2) : ippiColorConvertReorder(_func)
    {
        order[0] = _order0;
        order[1] = _order1;
        order[2] = _order2;
        order[3] = 3;
    }
    bool operator()(const void *src, int srcStep, void *dst, int dstStep, int cols, int rows) const
    {
        return ippiColorConvertReorder ? CV_INSTRUMENT_FUN_IPP(ippiColorConvertReorder, src, srcStep, dst, dstStep, ippiSize(cols, rows), order) >= 0 : false;
    }
private:
    ippiReorderFunc ippiColorConvertReorder;
    int order[4];
};


struct IPPReorderGeneralFunctor
{
    IPPReorderGeneralFunctor(ippiReorderFunc _func1, ippiGeneralFunc _func2, int _order0, int _order1, int _order2, int _depth) :
        ippiColorConvertReorder(_func1), ippiColorConvertGeneral(_func2), depth(_depth)
    {
        order[0] = _order0;
        order[1] = _order1;
        order[2] = _order2;
        order[3] = 3;
    }
    bool operator()(const void *src, int srcStep, void *dst, int dstStep, int cols, int rows) const
    {
        if (ippiColorConvertReorder == 0 || ippiColorConvertGeneral == 0)
            return false;

        Mat temp;
        temp.create(rows, cols, CV_MAKETYPE(depth, 3));
        if(CV_INSTRUMENT_FUN_IPP(ippiColorConvertReorder, src, srcStep, temp.ptr(), (int)temp.step[0], ippiSize(cols, rows), order) < 0)
            return false;
        return CV_INSTRUMENT_FUN_IPP(ippiColorConvertGeneral, temp.ptr(), (int)temp.step[0], dst, dstStep, ippiSize(cols, rows)) >= 0;
    }
private:
    ippiReorderFunc ippiColorConvertReorder;
    ippiGeneralFunc ippiColorConvertGeneral;
    int order[4];
    int depth;
};


struct IPPGeneralReorderFunctor
{
    IPPGeneralReorderFunctor(ippiGeneralFunc _func1, ippiReorderFunc _func2, int _order0, int _order1, int _order2, int _depth) :
        ippiColorConvertGeneral(_func1), ippiColorConvertReorder(_func2), depth(_depth)
    {
        order[0] = _order0;
        order[1] = _order1;
        order[2] = _order2;
        order[3] = 3;
    }
    bool operator()(const void *src, int srcStep, void *dst, int dstStep, int cols, int rows) const
    {
        if (ippiColorConvertGeneral == 0 || ippiColorConvertReorder == 0)
            return false;

        Mat temp;
        temp.create(rows, cols, CV_MAKETYPE(depth, 3));
        if(CV_INSTRUMENT_FUN_IPP(ippiColorConvertGeneral, src, srcStep, temp.ptr(), (int)temp.step[0], ippiSize(cols, rows)) < 0)
            return false;
        return CV_INSTRUMENT_FUN_IPP(ippiColorConvertReorder, temp.ptr(), (int)temp.step[0], dst, dstStep, ippiSize(cols, rows), order) >= 0;
    }
private:
    ippiGeneralFunc ippiColorConvertGeneral;
    ippiReorderFunc ippiColorConvertReorder;
    int order[4];
    int depth;
};


//TODO: make them external (or rewrite IPP code)

static IppStatus CV_STDCALL ippiSwapChannels_8u_C3C4Rf(const Ipp8u* pSrc, int srcStep, Ipp8u* pDst, int dstStep,
         IppiSize roiSize, const int *dstOrder)
{
    return CV_INSTRUMENT_FUN_IPP(ippiSwapChannels_8u_C3C4R, pSrc, srcStep, pDst, dstStep, roiSize, dstOrder, MAX_IPP8u);
}

static IppStatus CV_STDCALL ippiSwapChannels_16u_C3C4Rf(const Ipp16u* pSrc, int srcStep, Ipp16u* pDst, int dstStep,
         IppiSize roiSize, const int *dstOrder)
{
    return CV_INSTRUMENT_FUN_IPP(ippiSwapChannels_16u_C3C4R, pSrc, srcStep, pDst, dstStep, roiSize, dstOrder, MAX_IPP16u);
}

static IppStatus CV_STDCALL ippiSwapChannels_32f_C3C4Rf(const Ipp32f* pSrc, int srcStep, Ipp32f* pDst, int dstStep,
         IppiSize roiSize, const int *dstOrder)
{
    return CV_INSTRUMENT_FUN_IPP(ippiSwapChannels_32f_C3C4R, pSrc, srcStep, pDst, dstStep, roiSize, dstOrder, MAX_IPP32f);
}

static ippiReorderFunc ippiSwapChannelsC3C4RTab[] =
{
    (ippiReorderFunc)ippiSwapChannels_8u_C3C4Rf, 0, (ippiReorderFunc)ippiSwapChannels_16u_C3C4Rf, 0,
    0, (ippiReorderFunc)ippiSwapChannels_32f_C3C4Rf, 0, 0
};

static ippiGeneralFunc ippiCopyAC4C3RTab[] =
{
    (ippiGeneralFunc)ippiCopy_8u_AC4C3R, 0, (ippiGeneralFunc)ippiCopy_16u_AC4C3R, 0,
    0, (ippiGeneralFunc)ippiCopy_32f_AC4C3R, 0, 0
};

static ippiReorderFunc ippiSwapChannelsC4C3RTab[] =
{
    (ippiReorderFunc)ippiSwapChannels_8u_C4C3R, 0, (ippiReorderFunc)ippiSwapChannels_16u_C4C3R, 0,
    0, (ippiReorderFunc)ippiSwapChannels_32f_C4C3R, 0, 0
};

static ippiReorderFunc ippiSwapChannelsC3RTab[] =
{
    (ippiReorderFunc)ippiSwapChannels_8u_C3R, 0, (ippiReorderFunc)ippiSwapChannels_16u_C3R, 0,
    0, (ippiReorderFunc)ippiSwapChannels_32f_C3R, 0, 0
};

#if IPP_VERSION_X100 >= 810
static ippiReorderFunc ippiSwapChannelsC4RTab[] =
{
    (ippiReorderFunc)ippiSwapChannels_8u_C4R, 0, (ippiReorderFunc)ippiSwapChannels_16u_C4R, 0,
    0, (ippiReorderFunc)ippiSwapChannels_32f_C4R, 0, 0
};
#endif

#endif

// TODO: rewrite this
bool oclCvtColorBGR2Luv( InputArray _src, OutputArray _dst, int bidx, bool srgb );
bool oclCvtColorBGR2Lab( InputArray _src, OutputArray _dst, int bidx, bool srgb );
bool oclCvtColorLab2BGR( InputArray _src, OutputArray _dst, int dcn, int bidx, bool srgb);
bool oclCvtColorLuv2BGR( InputArray _src, OutputArray _dst, int dcn, int bidx, bool srgb);
bool oclCvtColorBGR2XYZ( InputArray _src, OutputArray _dst, int bidx );
bool oclCvtColorXYZ2BGR( InputArray _src, OutputArray _dst, int dcn, int bidx );

bool oclCvtColorRGBA2mRGBA( InputArray _src, OutputArray _dst);

void cvtColorBGR2Lab( InputArray _src, OutputArray _dst, bool swapb, bool srgb);
void cvtColorBGR2Luv( InputArray _src, OutputArray _dst, bool swapb, bool srgb);
void cvtColorLab2BGR( InputArray _src, OutputArray _dst, int dcn, bool swapb, bool srgb );
void cvtColorLuv2BGR( InputArray _src, OutputArray _dst, int dcn, bool swapb, bool srgb );
void cvtColorBGR2XYZ( InputArray _src, OutputArray _dst, bool swapb );
void cvtColorXYZ2BGR( InputArray _src, OutputArray _dst, int dcn, bool swapb );

void cvtColorBGR2YUV( InputArray _src, OutputArray _dst, bool swapb, bool crcb);
void cvtColorYUV2BGR( InputArray _src, OutputArray _dst, int dcn, bool swapb, bool crcb);

void cvtColorOnePlaneYUV2BGR( InputArray _src, OutputArray _dst, int dcn, bool swapb, int uidx, int ycn);
void cvtColorBGR2ThreePlaneYUV( InputArray _src, OutputArray _dst, bool swapb, int uidx);
void cvtColorYUV2Gray_420( InputArray _src, OutputArray _dst );
void cvtColorYUV2Gray_ch( InputArray _src, OutputArray _dst, int coi );
void cvtColorThreePlaneYUV2BGR( InputArray _src, OutputArray _dst, int dcn, bool swapb, int uidx );
void cvtColorTwoPlaneYUV2BGR( InputArray _src, OutputArray _dst, int dcn, bool swapb, int uidx );

void cvtColorBGR2HLS( InputArray _src, OutputArray _dst, bool swapb, bool fullRange );
void cvtColorBGR2HSV( InputArray _src, OutputArray _dst, bool swapb, bool fullRange );
void cvtColorHLS2BGR( InputArray _src, OutputArray _dst, int dcn, bool swapb, bool fullRange);
void cvtColorHSV2BGR( InputArray _src, OutputArray _dst, int dcn, bool swapb, bool fullRange);

void cvtColorBGR2BGR( InputArray _src, OutputArray _dst, int dcn, bool swapb);
void cvtColorBGR25x5( InputArray _src, OutputArray _dst, bool swapb, int gbits);
void cvtColor5x52BGR( InputArray _src, OutputArray _dst, int dcn, bool swapb, int gbits);

void cvtColorBGR2Gray( InputArray _src, OutputArray _dst, bool swapb);
void cvtColorGray2BGR( InputArray _src, OutputArray _dst, int dcn);

void cvtColor5x52Gray( InputArray _src, OutputArray _dst, int gbits);
void cvtColorGray25x5( InputArray _src, OutputArray _dst, int gbits);

void cvtColorRGBA2mRGBA(InputArray _src, OutputArray _dst);
void cvtColormRGBA2RGBA(InputArray _src, OutputArray _dst);

} //namespace cv
