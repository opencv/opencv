// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "opencv2/imgproc.hpp"
#include "hal_replacement.hpp"

namespace cv {

//
// Helper functions
//

namespace impl {

#include "color.simd_helpers.hpp"

inline bool isHSV(int code)
{
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
    switch (code)
    {
    case COLOR_Lab2BGR: case COLOR_Lab2RGB: case COLOR_Lab2LBGR: case COLOR_Lab2LRGB:
    case COLOR_BGR2Lab: case COLOR_RGB2Lab: case COLOR_LBGR2Lab: case COLOR_LRGB2Lab:
        return true;
    default:
        return false;
    }
}

inline bool is_sRGB(int code)
{
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

inline bool isFullRangeHSV(int code)
{
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
        case COLOR_HSV2RGB: case COLOR_HSV2BGR: case COLOR_RGB2HSV: case COLOR_BGR2HSV:
        case COLOR_HLS2RGB: case COLOR_HLS2BGR: case COLOR_RGB2HLS: case COLOR_BGR2HLS:
        case COLOR_HSV2RGB_FULL: case COLOR_HSV2BGR_FULL: case COLOR_RGB2HSV_FULL: case COLOR_BGR2HSV_FULL:
        case COLOR_HLS2RGB_FULL: case COLOR_HLS2BGR_FULL: case COLOR_RGB2HLS_FULL: case COLOR_BGR2HLS_FULL:
        case COLOR_YUV2RGB: case COLOR_YUV2BGR: case COLOR_RGB2YUV: case COLOR_BGR2YUV:
        case COLOR_BGR5652BGR: case COLOR_BGR5552BGR: case COLOR_BGR5652RGB: case COLOR_BGR5552RGB:
        case COLOR_GRAY2BGR:
        case COLOR_YUV2BGR_NV21: case COLOR_YUV2RGB_NV21: case COLOR_YUV2BGR_NV12: case COLOR_YUV2RGB_NV12:
        case COLOR_YUV2BGR_YV12: case COLOR_YUV2RGB_YV12: case COLOR_YUV2BGR_IYUV: case COLOR_YUV2RGB_IYUV:
        case COLOR_YUV2RGB_UYVY: case COLOR_YUV2BGR_UYVY: case COLOR_YUV2RGB_YVYU: case COLOR_YUV2BGR_YVYU:
        case COLOR_YUV2RGB_YUY2: case COLOR_YUV2BGR_YUY2:
        case COLOR_Lab2BGR: case COLOR_Lab2RGB: case COLOR_BGR2Lab: case COLOR_RGB2Lab:
        case COLOR_Lab2LBGR: case COLOR_Lab2LRGB:
        case COLOR_Luv2BGR: case COLOR_Luv2RGB: case COLOR_BGR2Luv: case COLOR_RGB2Luv:
        case COLOR_Luv2LBGR: case COLOR_Luv2LRGB:
        case COLOR_YCrCb2BGR: case COLOR_YCrCb2RGB: case COLOR_BGR2YCrCb: case COLOR_RGB2YCrCb:
        case COLOR_XYZ2BGR: case COLOR_XYZ2RGB: case COLOR_BGR2XYZ: case COLOR_RGB2XYZ:

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
using namespace impl;

/*template< typename VScn, typename VDcn, typename VDepth, SizePolicy sizePolicy = NONE >
struct CvtHelper
{
    CvtHelper(InputArray _src, OutputArray _dst, int dcn)
    {
        CV_Assert(!_src.empty());

        int stype = _src.type();
        scn = CV_MAT_CN(stype), depth = CV_MAT_DEPTH(stype);

        CV_Check(scn, VScn::contains(scn), "Invalid number of channels in input image");
        CV_Check(dcn, VDcn::contains(dcn), "Invalid number of channels in output image");
        CV_CheckDepth(depth, VDepth::contains(depth), "Unsupported depth of input image");

        if (_src.getObj() == _dst.getObj()) // inplace processing (#6653)
            _src.copyTo(src);
        else
            src = _src.getMat();
        Size sz = src.size();
        switch (sizePolicy)
        {
        case TO_YUV:
            CV_Assert( sz.width % 2 == 0 && sz.height % 2 == 0);
            dstSz = Size(sz.width, sz.height / 2 * 3);
            break;
        case FROM_YUV:
            CV_Assert( sz.width % 2 == 0 && sz.height % 3 == 0);
            dstSz = Size(sz.width, sz.height * 2 / 3);
            break;
        case NONE:
        default:
            dstSz = sz;
            break;
        }
        _dst.create(dstSz, CV_MAKETYPE(depth, dcn));
        dst = _dst.getMat();
    }
    Mat src, dst;
    int depth, scn;
    Size dstSz;
};*/

#ifdef HAVE_OPENCL

template< typename VScn, typename VDcn, typename VDepth, SizePolicy sizePolicy = NONE >
struct OclHelper
{
    OclHelper( InputArray _src, OutputArray _dst, int dcn) :
        nArgs(0)
    {
        src = _src.getUMat();
        Size sz = src.size(), dstSz;
        int scn = src.channels();
        int depth = src.depth();

        CV_Check(scn, VScn::contains(scn), "Invalid number of channels in input image");
        CV_Check(dcn, VDcn::contains(dcn), "Invalid number of channels in output image");
        CV_CheckDepth(depth, VDepth::contains(depth), "Unsupported depth of input image");

        switch (sizePolicy)
        {
        case TO_YUV:
            CV_Assert( sz.width % 2 == 0 && sz.height % 2 == 0 );
            dstSz = Size(sz.width, sz.height / 2 * 3);
            break;
        case FROM_YUV:
            CV_Assert( sz.width % 2 == 0 && sz.height % 3 == 0 );
            dstSz = Size(sz.width, sz.height * 2 / 3);
            break;
        case NONE:
        default:
            dstSz = sz;
            break;
        }

        _dst.create(dstSz, CV_MAKETYPE(depth, dcn));
        dst = _dst.getUMat();
    }

    bool createKernel(cv::String name, ocl::ProgramSource& source, cv::String options)
    {
        ocl::Device dev = ocl::Device::getDefault();
        int pxPerWIy = dev.isIntel() && (dev.type() & ocl::Device::TYPE_GPU) ? 4 : 1;
        int pxPerWIx = 1;

        cv::String baseOptions = format("-D depth=%d -D scn=%d -D PIX_PER_WI_Y=%d ",
                                        src.depth(), src.channels(), pxPerWIy);

        switch (sizePolicy)
        {
        case TO_YUV:
            if (dev.isIntel() &&
                    src.cols % 4 == 0 && src.step % 4 == 0 && src.offset % 4 == 0 &&
                    dst.step % 4 == 0 && dst.offset % 4 == 0)
            {
                pxPerWIx = 2;
            }
            globalSize[0] = (size_t)dst.cols/(2*pxPerWIx);
            globalSize[1] = ((size_t)dst.rows/3 + pxPerWIy - 1) / pxPerWIy;
            baseOptions += format("-D PIX_PER_WI_X=%d ", pxPerWIx);
            break;
        case FROM_YUV:
            globalSize[0] = (size_t)dst.cols/2;
            globalSize[1] = ((size_t)dst.rows/2 + pxPerWIy - 1) / pxPerWIy;
            break;
        case NONE:
        default:
            globalSize[0] = (size_t)src.cols;
            globalSize[1] = ((size_t)src.rows + pxPerWIy - 1) / pxPerWIy;
            break;
        }

        k.create(name.c_str(), source, baseOptions + options);

        if(k.empty())
            return false;

        nArgs = k.set(0, ocl::KernelArg::ReadOnlyNoSize(src));
        nArgs = k.set(nArgs, ocl::KernelArg::WriteOnly(dst));
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
    int nArgs;
};

#endif



#if defined (HAVE_IPP) && (IPP_VERSION_X100 >= 700)
#  define NEED_IPP 1
#else
#  define NEED_IPP 0
#endif

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

    virtual void operator()(const Range& range) const CV_OVERRIDE
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
    const size_t src_step;
    uchar * dst_data;
    const size_t dst_step;
    const int width;
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

extern ippiReorderFunc ippiSwapChannelsC3C4RTab[8];
extern ippiReorderFunc ippiSwapChannelsC4C3RTab[8];
extern ippiReorderFunc ippiSwapChannelsC3RTab[8];

#endif

#ifdef HAVE_OPENCL

bool oclCvtColorBGR2Luv( InputArray _src, OutputArray _dst, int bidx, bool srgb );
bool oclCvtColorBGR2Lab( InputArray _src, OutputArray _dst, int bidx, bool srgb );
bool oclCvtColorLab2BGR( InputArray _src, OutputArray _dst, int dcn, int bidx, bool srgb);
bool oclCvtColorLuv2BGR( InputArray _src, OutputArray _dst, int dcn, int bidx, bool srgb);
bool oclCvtColorBGR2XYZ( InputArray _src, OutputArray _dst, int bidx );
bool oclCvtColorXYZ2BGR( InputArray _src, OutputArray _dst, int dcn, int bidx );

bool oclCvtColorHSV2BGR( InputArray _src, OutputArray _dst, int dcn, int bidx, bool full );
bool oclCvtColorHLS2BGR( InputArray _src, OutputArray _dst, int dcn, int bidx, bool full );
bool oclCvtColorBGR2HLS( InputArray _src, OutputArray _dst, int bidx, bool full );
bool oclCvtColorBGR2HSV( InputArray _src, OutputArray _dst, int bidx, bool full );

bool oclCvtColorBGR2BGR( InputArray _src, OutputArray _dst, int dcn, bool reverse );
bool oclCvtColorBGR25x5( InputArray _src, OutputArray _dst, int bidx, int gbits );
bool oclCvtColor5x52BGR( InputArray _src, OutputArray _dst, int dcn, int bidx, int gbits );
bool oclCvtColor5x52Gray( InputArray _src, OutputArray _dst, int gbits );
bool oclCvtColorGray25x5( InputArray _src, OutputArray _dst, int gbits );
bool oclCvtColorBGR2Gray( InputArray _src, OutputArray _dst, int bidx );
bool oclCvtColorGray2BGR( InputArray _src, OutputArray _dst, int dcn );
bool oclCvtColorRGBA2mRGBA( InputArray _src, OutputArray _dst );
bool oclCvtColormRGBA2RGBA( InputArray _src, OutputArray _dst );

bool oclCvtColorBGR2YCrCb( InputArray _src, OutputArray _dst, int bidx);
bool oclCvtcolorYCrCb2BGR( InputArray _src, OutputArray _dst, int dcn, int bidx);
bool oclCvtColorBGR2YUV( InputArray _src, OutputArray _dst, int bidx );
bool oclCvtColorYUV2BGR( InputArray _src, OutputArray _dst, int dcn, int bidx );

bool oclCvtColorOnePlaneYUV2BGR( InputArray _src, OutputArray _dst, int dcn, int bidx, int uidx, int yidx );
bool oclCvtColorTwoPlaneYUV2BGR( InputArray _src, OutputArray _dst, int dcn, int bidx, int uidx );
bool oclCvtColorThreePlaneYUV2BGR( InputArray _src, OutputArray _dst, int dcn, int bidx, int uidx );
bool oclCvtColorBGR2ThreePlaneYUV( InputArray _src, OutputArray _dst, int bidx, int uidx );
bool oclCvtColorYUV2Gray_420( InputArray _src, OutputArray _dst );

#endif

void cvtColorBGR2Lab( InputArray _src, OutputArray _dst, bool swapb, bool srgb);
void cvtColorBGR2Luv( InputArray _src, OutputArray _dst, bool swapb, bool srgb);
void cvtColorLab2BGR( InputArray _src, OutputArray _dst, int dcn, bool swapb, bool srgb );
void cvtColorLuv2BGR( InputArray _src, OutputArray _dst, int dcn, bool swapb, bool srgb );
void cvtColorBGR2XYZ( InputArray _src, OutputArray _dst, bool swapb );
void cvtColorXYZ2BGR( InputArray _src, OutputArray _dst, int dcn, bool swapb );

void cvtColorBGR2YUV( InputArray _src, OutputArray _dst, bool swapb, bool crcb);
void cvtColorYUV2BGR( InputArray _src, OutputArray _dst, int dcn, bool swapb, bool crcb);

void cvtColorOnePlaneYUV2BGR( InputArray _src, OutputArray _dst, int dcn, bool swapb, int uidx, int ycn);
void cvtColorTwoPlaneYUV2BGR( InputArray _src, OutputArray _dst, int dcn, bool swapb, int uidx );
void cvtColorTwoPlaneYUV2BGRpair( InputArray _ysrc, InputArray _uvsrc, OutputArray _dst, int dcn, bool swapb, int uidx );
void cvtColorThreePlaneYUV2BGR( InputArray _src, OutputArray _dst, int dcn, bool swapb, int uidx );
void cvtColorBGR2ThreePlaneYUV( InputArray _src, OutputArray _dst, bool swapb, int uidx);
void cvtColorYUV2Gray_420( InputArray _src, OutputArray _dst );
void cvtColorYUV2Gray_ch( InputArray _src, OutputArray _dst, int coi );

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
