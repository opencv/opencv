// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "ipp_hal_imgproc.hpp"

#ifdef HAVE_IPP_IW

#include <opencv2/core.hpp>
#include "precomp_ipp.hpp"

#include "iw++/iw.hpp"

#include <cfloat>

#define IPP_RESIZE_PARALLEL 1

class ipp_resizeParallel: public cv::ParallelLoopBody
{
public:
    ipp_resizeParallel(::ipp::IwiImage &src, ::ipp::IwiImage &dst, bool &ok):
        m_src(src), m_dst(dst), m_ok(ok) {}
    ~ipp_resizeParallel()
    {
    }

    void Init(IppiInterpolationType inter)
    {
        iwiResize.InitAlloc(m_src.m_size, m_dst.m_size, m_src.m_dataType, m_src.m_channels, inter, ::ipp::IwiResizeParams(0, 0, 0.75, 4), ippBorderRepl);

        m_ok = true;
    }

    virtual void operator() (const cv::Range& range) const CV_OVERRIDE
    {
        if(!m_ok)
            return;

        try
        {
            ::ipp::IwiTile tile = ::ipp::IwiRoi(0, range.start, m_dst.m_size.width, range.end - range.start);
            CV_INSTRUMENT_FUN_IPP(iwiResize, m_src, m_dst, ippBorderRepl, tile);
        }
        catch(const ::ipp::IwException &)
        {
            m_ok = false;
            return;
        }
    }
private:
    ::ipp::IwiImage &m_src;
    ::ipp::IwiImage &m_dst;

    mutable ::ipp::IwiResize iwiResize;

    volatile bool &m_ok;
    const ipp_resizeParallel& operator= (const ipp_resizeParallel&);
};

class ipp_resizeAffineParallel: public cv::ParallelLoopBody
{
public:
    ipp_resizeAffineParallel(::ipp::IwiImage &src, ::ipp::IwiImage &dst, bool &ok):
        m_src(src), m_dst(dst), m_ok(ok) {}
    ~ipp_resizeAffineParallel()
    {
    }

    void Init(IppiInterpolationType inter, double scaleX, double scaleY)
    {
        double shift = (inter == ippNearest)?-1e-10:-0.5;
        double coeffs[2][3] = {
            {scaleX, 0,      shift+0.5*scaleX},
            {0,      scaleY, shift+0.5*scaleY}
        };

        iwiWarpAffine.InitAlloc(m_src.m_size, m_dst.m_size, m_src.m_dataType, m_src.m_channels, coeffs, iwTransForward, inter, ::ipp::IwiWarpAffineParams(0, 0, 0.75), ippBorderRepl);

        m_ok = true;
    }

    virtual void operator() (const cv::Range& range) const CV_OVERRIDE
    {
        if(!m_ok)
            return;

        try
        {
            ::ipp::IwiTile tile = ::ipp::IwiRoi(0, range.start, m_dst.m_size.width, range.end - range.start);
            CV_INSTRUMENT_FUN_IPP(iwiWarpAffine, m_src, m_dst, tile);
        }
        catch(const ::ipp::IwException &)
        {
            m_ok = false;
            return;
        }
    }
private:
    ::ipp::IwiImage &m_src;
    ::ipp::IwiImage &m_dst;

    mutable ::ipp::IwiWarpAffine iwiWarpAffine;

    volatile bool &m_ok;
    const ipp_resizeAffineParallel& operator= (const ipp_resizeAffineParallel&);
};

int ipp_hal_resize(int src_type, const uchar *src_data, size_t src_step, int src_width, int src_height,
                   uchar *dst_data, size_t dst_step, int dst_width, int dst_height,
                   double inv_scale_x, double inv_scale_y, int interpolation)
{
    CV_HAL_CHECK_USE_IPP();

    int depth = CV_MAT_DEPTH(src_type), channels = CV_MAT_CN(src_type);

    IppDataType           ippDataType = ippiGetDataType(depth);
    IppiInterpolationType ippInter    = ippiGetInterpolation(interpolation);
    if((int)ippInter < 0)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    // Resize which doesn't match OpenCV exactly
    if (!cv::ipp::useIPP_NotExact())
    {
        if (ippInter == ippNearest || ippInter == ippSuper || (ippDataType == ipp8u && ippInter == ippLinear))
            return CV_HAL_ERROR_NOT_IMPLEMENTED;
    }

    if(ippInter != ippLinear && ippDataType == ipp64f)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

#if IPP_VERSION_X100 < 201801
    // Degradations on int^2 linear downscale
    if (ippDataType != ipp64f && ippInter == ippLinear && inv_scale_x < 1 && inv_scale_y < 1) // if downscale
    {
        int scale_x = (int)(1 / inv_scale_x);
        int scale_y = (int)(1 / inv_scale_y);
        if (1 / inv_scale_x - scale_x < DBL_EPSILON && 1 / inv_scale_y - scale_y < DBL_EPSILON) // if integer
        {
            if (!(scale_x&(scale_x - 1)) && !(scale_y&(scale_y - 1))) // if power of 2
                return CV_HAL_ERROR_NOT_IMPLEMENTED;
        }
    }
#endif

    bool  affine = false;
    const double IPP_RESIZE_EPS = (depth == CV_64F)?0:1e-10;
    double ex = fabs((double)dst_width / src_width  - inv_scale_x) / inv_scale_x;
    double ey = fabs((double)dst_height / src_height - inv_scale_y) / inv_scale_y;

    // Use affine transform resize to allow sub-pixel accuracy
    if(ex > IPP_RESIZE_EPS || ey > IPP_RESIZE_EPS)
        affine = true;

    // Affine doesn't support Lanczos and Super interpolations
    if(affine && (ippInter == ippLanczos || ippInter == ippSuper))
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    try
    {
        ::ipp::IwiImage iwSrc(::ipp::IwiSize(src_width, src_height), ippDataType, channels, 0, (void*)src_data, src_step);
        ::ipp::IwiImage iwDst(::ipp::IwiSize(dst_width, dst_height), ippDataType, channels, 0, (void*)dst_data, dst_step);

        bool  ok;
        int   threads = ippiSuggestThreadsNum(iwDst, 1+((double)(src_width*src_height)/(dst_width*dst_height)));
        cv::Range range(0, dst_height);
        ipp_resizeParallel       invokerGeneral(iwSrc, iwDst, ok);
        ipp_resizeAffineParallel invokerAffine(iwSrc, iwDst, ok);
        cv::ParallelLoopBody    *pInvoker = NULL;
        if(affine)
        {
            pInvoker = &invokerAffine;
            invokerAffine.Init(ippInter, inv_scale_x, inv_scale_y);
        }
        else
        {
            pInvoker = &invokerGeneral;
            invokerGeneral.Init(ippInter);
        }

        if(IPP_RESIZE_PARALLEL && threads > 1)
            cv::parallel_for_(range, *pInvoker, threads*4);
        else
            pInvoker->operator()(range);

        if(!ok)
            return CV_HAL_ERROR_NOT_IMPLEMENTED;
    }
    catch(const ::ipp::IwException &)
    {
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    }

    return CV_HAL_ERROR_OK;
}

#endif // HAVE_IPP_IW
