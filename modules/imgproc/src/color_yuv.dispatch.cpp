// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "precomp.hpp"
#include "opencl_kernels_imgproc.hpp"

#include "color.hpp"

#include "color_yuv.simd.hpp"
#include "color_yuv.simd_declarations.hpp" // defines CV_CPU_DISPATCH_MODES_ALL=AVX2,...,BASELINE based on CMakeLists.txt content

namespace cv {

//
// HAL functions
//
namespace hal {

// 8u, 16u, 32f
void cvtBGRtoYUV(const uchar * src_data, size_t src_step,
                 uchar * dst_data, size_t dst_step,
                 int width, int height,
                 int depth, int scn, bool swapBlue, bool isCbCr)
{
    CV_INSTRUMENT_REGION();

    CALL_HAL(cvtBGRtoYUV, cv_hal_cvtBGRtoYUV, src_data, src_step, dst_data, dst_step, width, height, depth, scn, swapBlue, isCbCr);

#if defined(HAVE_IPP)
#if !IPP_DISABLE_RGB_YUV
    CV_IPP_CHECK()
    {
        if (scn == 3 && depth == CV_8U && swapBlue && !isCbCr)
        {
            if (CvtColorIPPLoop(src_data, src_step, dst_data, dst_step, width, height,
                                IPPGeneralFunctor((ippiGeneralFunc)ippiRGBToYUV_8u_C3R)))
                return;
        }
        else if (scn == 3 && depth == CV_8U && !swapBlue && !isCbCr)
        {
            if (CvtColorIPPLoop(src_data, src_step, dst_data, dst_step, width, height,
                                IPPReorderGeneralFunctor(ippiSwapChannelsC3RTab[depth],
                                                         (ippiGeneralFunc)ippiRGBToYUV_8u_C3R, 2, 1, 0, depth)))
                return;
        }
        else if (scn == 4 && depth == CV_8U && swapBlue && !isCbCr)
        {
            if (CvtColorIPPLoop(src_data, src_step, dst_data, dst_step, width, height,
                                IPPReorderGeneralFunctor(ippiSwapChannelsC4C3RTab[depth],
                                                         (ippiGeneralFunc)ippiRGBToYUV_8u_C3R, 0, 1, 2, depth)))
                return;
        }
        else if (scn == 4 && depth == CV_8U && !swapBlue && !isCbCr)
        {
            if (CvtColorIPPLoop(src_data, src_step, dst_data, dst_step, width, height,
                                IPPReorderGeneralFunctor(ippiSwapChannelsC4C3RTab[depth],
                                                         (ippiGeneralFunc)ippiRGBToYUV_8u_C3R, 2, 1, 0, depth)))
                return;
        }
    }
#endif
#endif

    CV_CPU_DISPATCH(cvtBGRtoYUV, (src_data, src_step, dst_data, dst_step, width, height, depth, scn, swapBlue, isCbCr),
        CV_CPU_DISPATCH_MODES_ALL);
}

void cvtYUVtoBGR(const uchar * src_data, size_t src_step,
                 uchar * dst_data, size_t dst_step,
                 int width, int height,
                 int depth, int dcn, bool swapBlue, bool isCbCr)
{
    CV_INSTRUMENT_REGION();

    CALL_HAL(cvtYUVtoBGR, cv_hal_cvtYUVtoBGR, src_data, src_step, dst_data, dst_step, width, height, depth, dcn, swapBlue, isCbCr);


#if defined(HAVE_IPP)
#if !IPP_DISABLE_YUV_RGB
    CV_IPP_CHECK()
    {
        if (dcn == 3 && depth == CV_8U && swapBlue && !isCbCr)
        {
            if (CvtColorIPPLoop(src_data, src_step, dst_data, dst_step, width, height,
                                IPPGeneralFunctor((ippiGeneralFunc)ippiYUVToRGB_8u_C3R)))
                return;
        }
        else if (dcn == 3 && depth == CV_8U && !swapBlue && !isCbCr)
        {
            if (CvtColorIPPLoop(src_data, src_step, dst_data, dst_step, width, height,
                                IPPGeneralReorderFunctor((ippiGeneralFunc)ippiYUVToRGB_8u_C3R,
                                                                   ippiSwapChannelsC3RTab[depth], 2, 1, 0, depth)))
                return;
        }
        else if (dcn == 4 && depth == CV_8U && swapBlue && !isCbCr)
        {
            if (CvtColorIPPLoop(src_data, src_step, dst_data, dst_step, width, height,
                                IPPGeneralReorderFunctor((ippiGeneralFunc)ippiYUVToRGB_8u_C3R,
                                                                   ippiSwapChannelsC3C4RTab[depth], 0, 1, 2, depth)))
                return;
        }
        else if (dcn == 4 && depth == CV_8U && !swapBlue && !isCbCr)
        {
            if (CvtColorIPPLoop(src_data, src_step, dst_data, dst_step, width, height,
                                IPPGeneralReorderFunctor((ippiGeneralFunc)ippiYUVToRGB_8u_C3R,
                                                                   ippiSwapChannelsC3C4RTab[depth], 2, 1, 0, depth)))
                return;
        }
    }
#endif
#endif

    CV_CPU_DISPATCH(cvtYUVtoBGR, (src_data, src_step, dst_data, dst_step, width, height, depth, dcn, swapBlue, isCbCr),
        CV_CPU_DISPATCH_MODES_ALL);
}

void cvtTwoPlaneYUVtoBGR(const uchar * src_data, size_t src_step,
                         uchar * dst_data, size_t dst_step,
                         int dst_width, int dst_height,
                         int dcn, bool swapBlue, int uIdx)
{
    CV_INSTRUMENT_REGION();

    CALL_HAL(cvtTwoPlaneYUVtoBGR, cv_hal_cvtTwoPlaneYUVtoBGR, src_data, src_step, dst_data, dst_step, dst_width, dst_height, dcn, swapBlue, uIdx);

    cvtTwoPlaneYUVtoBGR(
            src_data, src_step, src_data + src_step * dst_height, src_step, dst_data, dst_step,
            dst_width, dst_height, dcn, swapBlue, uIdx);
}

void cvtTwoPlaneYUVtoBGR(const uchar * y_data, const uchar * uv_data, size_t src_step,
                         uchar * dst_data, size_t dst_step,
                         int dst_width, int dst_height,
                         int dcn, bool swapBlue, int uIdx)
{
    CV_INSTRUMENT_REGION();

    cvtTwoPlaneYUVtoBGR(y_data, src_step, uv_data, src_step, dst_data, dst_step, dst_width, dst_height, dcn, swapBlue, uIdx);
}

void cvtTwoPlaneYUVtoBGR(const uchar * y_data, size_t y_step, const uchar * uv_data, size_t uv_step,
                         uchar * dst_data, size_t dst_step,
                         int dst_width, int dst_height,
                         int dcn, bool swapBlue, int uIdx)
{
    CV_INSTRUMENT_REGION();

    CALL_HAL(cvtTwoPlaneYUVtoBGREx, cv_hal_cvtTwoPlaneYUVtoBGREx,
             y_data, y_step, uv_data, uv_step, dst_data, dst_step, dst_width, dst_height, dcn, swapBlue, uIdx);

    CV_CPU_DISPATCH(cvtTwoPlaneYUVtoBGR, (y_data, y_step, uv_data, uv_step, dst_data, dst_step, dst_width, dst_height, dcn, swapBlue, uIdx),
        CV_CPU_DISPATCH_MODES_ALL);
}

void cvtThreePlaneYUVtoBGR(const uchar * src_data, size_t src_step,
                           uchar * dst_data, size_t dst_step,
                           int dst_width, int dst_height,
                           int dcn, bool swapBlue, int uIdx)
{
    CV_INSTRUMENT_REGION();

    CALL_HAL(cvtThreePlaneYUVtoBGR, cv_hal_cvtThreePlaneYUVtoBGR, src_data, src_step, dst_data, dst_step, dst_width, dst_height, dcn, swapBlue, uIdx);

    CV_CPU_DISPATCH(cvtThreePlaneYUVtoBGR, (src_data, src_step, dst_data, dst_step, dst_width, dst_height, dcn, swapBlue, uIdx),
        CV_CPU_DISPATCH_MODES_ALL);
}

void cvtBGRtoThreePlaneYUV(const uchar * src_data, size_t src_step,
                           uchar * dst_data, size_t dst_step,
                           int width, int height,
                           int scn, bool swapBlue, int uIdx)
{
    CV_INSTRUMENT_REGION();

    CALL_HAL(cvtBGRtoThreePlaneYUV, cv_hal_cvtBGRtoThreePlaneYUV, src_data, src_step, dst_data, dst_step, width, height, scn, swapBlue, uIdx);

    CV_CPU_DISPATCH(cvtBGRtoThreePlaneYUV, (src_data, src_step, dst_data, dst_step, width, height, scn, swapBlue, uIdx),
        CV_CPU_DISPATCH_MODES_ALL);
}

void cvtBGRtoTwoPlaneYUV(const uchar * src_data, size_t src_step,
                         uchar * y_data, uchar * uv_data, size_t dst_step,
                         int width, int height,
                         int scn, bool swapBlue, int uIdx)
{
    CV_INSTRUMENT_REGION();

    CALL_HAL(cvtBGRtoTwoPlaneYUV, cv_hal_cvtBGRtoTwoPlaneYUV,
             src_data, src_step, y_data, dst_step, uv_data, dst_step, width, height, scn, swapBlue, uIdx);

    CV_CPU_DISPATCH(cvtBGRtoTwoPlaneYUV, (src_data, src_step, y_data, uv_data, dst_step, width, height, scn, swapBlue, uIdx),
        CV_CPU_DISPATCH_MODES_ALL);
}

void cvtOnePlaneYUVtoBGR(const uchar * src_data, size_t src_step,
                         uchar * dst_data, size_t dst_step,
                         int width, int height,
                         int dcn, bool swapBlue, int uIdx, int ycn)
{
    CV_INSTRUMENT_REGION();

    CALL_HAL(cvtOnePlaneYUVtoBGR, cv_hal_cvtOnePlaneYUVtoBGR, src_data, src_step, dst_data, dst_step, width, height, dcn, swapBlue, uIdx, ycn);

    CV_CPU_DISPATCH(cvtOnePlaneYUVtoBGR, (src_data, src_step, dst_data, dst_step, width, height, dcn, swapBlue, uIdx, ycn),
        CV_CPU_DISPATCH_MODES_ALL);
}

} // namespace hal

//
// OCL calls
//

#ifdef HAVE_OPENCL

bool oclCvtColorYUV2BGR( InputArray _src, OutputArray _dst, int dcn, int bidx )
{
    OclHelper< Set<3>, Set<3, 4>, Set<CV_8U, CV_16U, CV_32F> > h(_src, _dst, dcn);

    if(!h.createKernel("YUV2RGB", ocl::imgproc::color_yuv_oclsrc,
                       format("-D dcn=%d -D bidx=%d", dcn, bidx)))
    {
        return false;
    }

    return h.run();
}

bool oclCvtColorBGR2YUV( InputArray _src, OutputArray _dst, int bidx )
{
    OclHelper< Set<3, 4>, Set<3>, Set<CV_8U, CV_16U, CV_32F> > h(_src, _dst, 3);

    if(!h.createKernel("RGB2YUV", ocl::imgproc::color_yuv_oclsrc,
                       format("-D dcn=3 -D bidx=%d", bidx)))
    {
        return false;
    }

    return h.run();
}

bool oclCvtcolorYCrCb2BGR( InputArray _src, OutputArray _dst, int dcn, int bidx)
{
    OclHelper< Set<3>, Set<3, 4>, Set<CV_8U, CV_16U, CV_32F> > h(_src, _dst, dcn);

    if(!h.createKernel("YCrCb2RGB", ocl::imgproc::color_yuv_oclsrc,
                       format("-D dcn=%d -D bidx=%d", dcn, bidx)))
    {
        return false;
    }

    return h.run();
}

bool oclCvtColorBGR2YCrCb( InputArray _src, OutputArray _dst, int bidx)
{
    OclHelper< Set<3, 4>, Set<3>, Set<CV_8U, CV_16U, CV_32F> > h(_src, _dst, 3);

    if(!h.createKernel("RGB2YCrCb", ocl::imgproc::color_yuv_oclsrc,
                       format("-D dcn=3 -D bidx=%d", bidx)))
    {
        return false;
    }

    return h.run();
}

bool oclCvtColorOnePlaneYUV2BGR( InputArray _src, OutputArray _dst, int dcn, int bidx, int uidx, int yidx )
{
    OclHelper< Set<2>, Set<3, 4>, Set<CV_8U> > h(_src, _dst, dcn);

    bool optimized = _src.offset() % 4 == 0 && _src.step() % 4 == 0;
    if(!h.createKernel("YUV2RGB_422", ocl::imgproc::color_yuv_oclsrc,
                       format("-D dcn=%d -D bidx=%d -D uidx=%d -D yidx=%d%s", dcn, bidx, uidx, yidx,
                       optimized ? " -D USE_OPTIMIZED_LOAD" : "")))
    {
        return false;
    }

    return h.run();
}

bool oclCvtColorYUV2Gray_420( InputArray _src, OutputArray _dst )
{
    OclHelper< Set<1>, Set<1>, Set<CV_8U>, FROM_YUV> h(_src, _dst, 1);

    h.src.rowRange(0, _dst.rows()).copyTo(_dst);
    return true;
}

bool oclCvtColorTwoPlaneYUV2BGR( InputArray _src, OutputArray _dst, int dcn, int bidx, int uidx )
{
    OclHelper< Set<1>, Set<3, 4>, Set<CV_8U>, FROM_YUV > h(_src, _dst, dcn);

    if(!h.createKernel("YUV2RGB_NVx", ocl::imgproc::color_yuv_oclsrc,
                       format("-D dcn=%d -D bidx=%d -D uidx=%d", dcn, bidx, uidx)))
    {
        return false;
    }

    return h.run();
}

bool oclCvtColorThreePlaneYUV2BGR( InputArray _src, OutputArray _dst, int dcn, int bidx, int uidx )
{
    OclHelper< Set<1>, Set<3, 4>, Set<CV_8U>, FROM_YUV > h(_src, _dst, dcn);

    if(!h.createKernel("YUV2RGB_YV12_IYUV", ocl::imgproc::color_yuv_oclsrc,
                       format("-D dcn=%d -D bidx=%d -D uidx=%d%s", dcn, bidx, uidx,
                       _src.isContinuous() ? " -D SRC_CONT" : "")))
    {
        return false;
    }

    return h.run();
}

bool oclCvtColorBGR2ThreePlaneYUV( InputArray _src, OutputArray _dst, int bidx, int uidx )
{
    OclHelper< Set<3, 4>, Set<1>, Set<CV_8U>, TO_YUV > h(_src, _dst, 1);

    if(!h.createKernel("RGB2YUV_YV12_IYUV", ocl::imgproc::color_yuv_oclsrc,
                       format("-D dcn=1 -D bidx=%d -D uidx=%d", bidx, uidx)))
    {
        return false;
    }

    return h.run();
}

#endif

//
// HAL calls
//

void cvtColorBGR2YUV(InputArray _src, OutputArray _dst, bool swapb, bool crcb)
{
    CvtHelper< Set<3, 4>, Set<3>, Set<CV_8U, CV_16U, CV_32F> > h(_src, _dst, 3);

    hal::cvtBGRtoYUV(h.src.data, h.src.step, h.dst.data, h.dst.step, h.src.cols, h.src.rows,
                     h.depth, h.scn, swapb, crcb);
}

void cvtColorYUV2BGR(InputArray _src, OutputArray _dst, int dcn, bool swapb, bool crcb)
{
    if(dcn <= 0) dcn = 3;
    CvtHelper< Set<3>, Set<3, 4>, Set<CV_8U, CV_16U, CV_32F> > h(_src, _dst, dcn);

    hal::cvtYUVtoBGR(h.src.data, h.src.step, h.dst.data, h.dst.step, h.src.cols, h.src.rows,
                     h.depth, dcn, swapb, crcb);
}

void cvtColorOnePlaneYUV2BGR( InputArray _src, OutputArray _dst, int dcn, bool swapb, int uidx, int ycn)
{
    CvtHelper< Set<2>, Set<3, 4>, Set<CV_8U>, FROM_UYVY > h(_src, _dst, dcn);

    hal::cvtOnePlaneYUVtoBGR(h.src.data, h.src.step, h.dst.data, h.dst.step, h.src.cols, h.src.rows,
                             dcn, swapb, uidx, ycn);
}

void cvtColorYUV2Gray_ch( InputArray _src, OutputArray _dst, int coi )
{
    CV_Assert( _src.channels() == 2 && _src.depth() == CV_8U );

    extractChannel(_src, _dst, coi);
}

void cvtColorBGR2ThreePlaneYUV( InputArray _src, OutputArray _dst, bool swapb, int uidx)
{
    CvtHelper< Set<3, 4>, Set<1>, Set<CV_8U>, TO_YUV > h(_src, _dst, 1);

    hal::cvtBGRtoThreePlaneYUV(h.src.data, h.src.step, h.dst.data, h.dst.step, h.src.cols, h.src.rows,
                               h.scn, swapb, uidx);
}

void cvtColorYUV2Gray_420( InputArray _src, OutputArray _dst )
{
    CvtHelper< Set<1>, Set<1>, Set<CV_8U>, FROM_YUV > h(_src, _dst, 1);

#ifdef HAVE_IPP
#if IPP_VERSION_X100 >= 201700
    if (CV_INSTRUMENT_FUN_IPP(ippiCopy_8u_C1R_L, h.src.data, (IppSizeL)h.src.step, h.dst.data, (IppSizeL)h.dst.step,
                              ippiSizeL(h.dstSz.width, h.dstSz.height)) >= 0)
        return;
#endif
#endif
    h.src(Range(0, h.dstSz.height), Range::all()).copyTo(h.dst);
}

void cvtColorThreePlaneYUV2BGR( InputArray _src, OutputArray _dst, int dcn, bool swapb, int uidx)
{
    if(dcn <= 0) dcn = 3;
    CvtHelper< Set<1>, Set<3, 4>, Set<CV_8U>, FROM_YUV> h(_src, _dst, dcn);

    hal::cvtThreePlaneYUVtoBGR(h.src.data, h.src.step, h.dst.data, h.dst.step, h.dst.cols, h.dst.rows,
                               dcn, swapb, uidx);
}

// http://www.fourcc.org/yuv.php#NV21 == yuv420sp -> a plane of 8 bit Y samples followed by an interleaved V/U plane containing 8 bit 2x2 subsampled chroma samples
// http://www.fourcc.org/yuv.php#NV12 -> a plane of 8 bit Y samples followed by an interleaved U/V plane containing 8 bit 2x2 subsampled colour difference samples

void cvtColorTwoPlaneYUV2BGR( InputArray _src, OutputArray _dst, int dcn, bool swapb, int uidx )
{
    if(dcn <= 0) dcn = 3;
    CvtHelper< Set<1>, Set<3, 4>, Set<CV_8U>, FROM_YUV> h(_src, _dst, dcn);

    hal::cvtTwoPlaneYUVtoBGR(h.src.data, h.src.step, h.dst.data, h.dst.step, h.dst.cols, h.dst.rows,
                             dcn, swapb, uidx);
}

void cvtColorTwoPlaneYUV2BGRpair( InputArray _ysrc, InputArray _uvsrc, OutputArray _dst, int dcn, bool swapb, int uidx )
{
    int stype = _ysrc.type();
    int depth = CV_MAT_DEPTH(stype);
    Size ysz = _ysrc.size(), uvs = _uvsrc.size();
    CV_Assert( dcn == 3 || dcn == 4 );
    CV_Assert( depth == CV_8U );
    CV_Assert( ysz.width == uvs.width * 2 && ysz.height == uvs.height * 2 );

    Mat ysrc = _ysrc.getMat(), uvsrc = _uvsrc.getMat();

    _dst.create( ysz, CV_MAKETYPE(depth, dcn));
    Mat dst = _dst.getMat();

    if(ysrc.step == uvsrc.step)
    {
        hal::cvtTwoPlaneYUVtoBGR(ysrc.data, uvsrc.data, ysrc.step,
                                 dst.data, dst.step, dst.cols, dst.rows,
                                 dcn, swapb, uidx);
    }
    else
    {
        hal::cvtTwoPlaneYUVtoBGR(ysrc.data, ysrc.step, uvsrc.data, uvsrc.step,
                                dst.data, dst.step, dst.cols, dst.rows,
                                dcn, swapb, uidx);
    }
}

} // namespace cv
