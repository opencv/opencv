// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "precomp.hpp"

#include "opencl_kernels_imgproc.hpp"

#include "color.hpp"

#include "color_hsv.simd.hpp"
#include "color_hsv.simd_declarations.hpp" // defines CV_CPU_DISPATCH_MODES_ALL=AVX2,...,BASELINE based on CMakeLists.txt content

namespace cv {

//
// IPP functions
//

#if NEED_IPP

#if !IPP_DISABLE_RGB_HSV
static ippiGeneralFunc ippiRGB2HSVTab[] =
{
    (ippiGeneralFunc)ippiRGBToHSV_8u_C3R, 0, (ippiGeneralFunc)ippiRGBToHSV_16u_C3R, 0,
    0, 0, 0, 0
};
#endif

static ippiGeneralFunc ippiHSV2RGBTab[] =
{
    (ippiGeneralFunc)ippiHSVToRGB_8u_C3R, 0, (ippiGeneralFunc)ippiHSVToRGB_16u_C3R, 0,
    0, 0, 0, 0
};

static ippiGeneralFunc ippiRGB2HLSTab[] =
{
    (ippiGeneralFunc)ippiRGBToHLS_8u_C3R, 0, (ippiGeneralFunc)ippiRGBToHLS_16u_C3R, 0,
    0, (ippiGeneralFunc)ippiRGBToHLS_32f_C3R, 0, 0
};

static ippiGeneralFunc ippiHLS2RGBTab[] =
{
    (ippiGeneralFunc)ippiHLSToRGB_8u_C3R, 0, (ippiGeneralFunc)ippiHLSToRGB_16u_C3R, 0,
    0, (ippiGeneralFunc)ippiHLSToRGB_32f_C3R, 0, 0
};

#endif

//
// HAL functions
//

namespace hal
{

// 8u, 32f
void cvtBGRtoHSV(const uchar * src_data, size_t src_step,
                 uchar * dst_data, size_t dst_step,
                 int width, int height,
                 int depth, int scn, bool swapBlue, bool isFullRange, bool isHSV)
{
    CV_INSTRUMENT_REGION();

    CALL_HAL(cvtBGRtoHSV, cv_hal_cvtBGRtoHSV, src_data, src_step, dst_data, dst_step, width, height, depth, scn, swapBlue, isFullRange, isHSV);

#if defined(HAVE_IPP) && IPP_VERSION_X100 >= 700
    CV_IPP_CHECK()
    {
        if(depth == CV_8U && isFullRange)
        {
            if (isHSV)
            {
#if !IPP_DISABLE_RGB_HSV // breaks OCL accuracy tests
                if(scn == 3 && !swapBlue)
                {
                    if( CvtColorIPPLoopCopy(src_data, src_step, CV_MAKE_TYPE(depth, scn), dst_data, dst_step, width, height,
                                            IPPReorderGeneralFunctor(ippiSwapChannelsC3RTab[depth], ippiRGB2HSVTab[depth], 2, 1, 0, depth)) )
                        return;
                }
                else if(scn == 4 && !swapBlue)
                {
                    if( CvtColorIPPLoop(src_data, src_step, dst_data, dst_step, width, height,
                                        IPPReorderGeneralFunctor(ippiSwapChannelsC4C3RTab[depth], ippiRGB2HSVTab[depth], 2, 1, 0, depth)) )
                        return;
                }
                else if(scn == 4 && swapBlue)
                {
                    if( CvtColorIPPLoop(src_data, src_step, dst_data, dst_step, width, height,
                                        IPPReorderGeneralFunctor(ippiSwapChannelsC4C3RTab[depth], ippiRGB2HSVTab[depth], 0, 1, 2, depth)) )
                        return;
                }
#endif
            }
            else
            {
                if(scn == 3 && !swapBlue)
                {
                    if( CvtColorIPPLoopCopy(src_data, src_step, CV_MAKE_TYPE(depth, scn), dst_data, dst_step, width, height,
                                            IPPReorderGeneralFunctor(ippiSwapChannelsC3RTab[depth], ippiRGB2HLSTab[depth], 2, 1, 0, depth)) )
                        return;
                }
                else if(scn == 4 && !swapBlue)
                {
                    if( CvtColorIPPLoop(src_data, src_step, dst_data, dst_step, width, height,
                                        IPPReorderGeneralFunctor(ippiSwapChannelsC4C3RTab[depth], ippiRGB2HLSTab[depth], 2, 1, 0, depth)) )
                        return;
                }
                else if(scn == 3 && swapBlue)
                {
                    if( CvtColorIPPLoopCopy(src_data, src_step, CV_MAKE_TYPE(depth, scn), dst_data, dst_step, width, height,
                                            IPPGeneralFunctor(ippiRGB2HLSTab[depth])) )
                        return;
                }
                else if(scn == 4 && swapBlue)
                {
                    if( CvtColorIPPLoop(src_data, src_step, dst_data, dst_step, width, height,
                                        IPPReorderGeneralFunctor(ippiSwapChannelsC4C3RTab[depth], ippiRGB2HLSTab[depth], 0, 1, 2, depth)) )
                        return;
                }
            }
        }
    }
#endif

    CV_CPU_DISPATCH(cvtBGRtoHSV, (src_data, src_step, dst_data, dst_step, width, height, depth, scn, swapBlue, isFullRange, isHSV),
        CV_CPU_DISPATCH_MODES_ALL);
}

// 8u, 32f
void cvtHSVtoBGR(const uchar * src_data, size_t src_step,
                 uchar * dst_data, size_t dst_step,
                 int width, int height,
                 int depth, int dcn, bool swapBlue, bool isFullRange, bool isHSV)
{
    CV_INSTRUMENT_REGION();

    CALL_HAL(cvtHSVtoBGR, cv_hal_cvtHSVtoBGR, src_data, src_step, dst_data, dst_step, width, height, depth, dcn, swapBlue, isFullRange, isHSV);

#if defined(HAVE_IPP) && IPP_VERSION_X100 >= 700
    CV_IPP_CHECK()
    {
        if (depth == CV_8U && isFullRange)
        {
            if (isHSV)
            {
                if(dcn == 3 && !swapBlue)
                {
                    if( CvtColorIPPLoopCopy(src_data, src_step, CV_MAKETYPE(depth, 3), dst_data, dst_step, width, height,
                                            IPPGeneralReorderFunctor(ippiHSV2RGBTab[depth], ippiSwapChannelsC3RTab[depth], 2, 1, 0, depth)) )
                        return;
                }
                else if(dcn == 4 && !swapBlue)
                {
                    if( CvtColorIPPLoop(src_data, src_step, dst_data, dst_step, width, height,
                                        IPPGeneralReorderFunctor(ippiHSV2RGBTab[depth], ippiSwapChannelsC3C4RTab[depth], 2, 1, 0, depth)) )
                        return;
                }
                else if(dcn == 3 && swapBlue)
                {
                    if( CvtColorIPPLoopCopy(src_data, src_step, CV_MAKETYPE(depth, 3), dst_data, dst_step, width, height,
                                            IPPGeneralFunctor(ippiHSV2RGBTab[depth])) )
                        return;
                }
                else if(dcn == 4 && swapBlue)
                {
                    if( CvtColorIPPLoop(src_data, src_step, dst_data, dst_step, width, height,
                                        IPPGeneralReorderFunctor(ippiHSV2RGBTab[depth], ippiSwapChannelsC3C4RTab[depth], 0, 1, 2, depth)) )
                        return;
                }
            }
            else
            {
                if(dcn == 3 && !swapBlue)
                {
                    if( CvtColorIPPLoopCopy(src_data, src_step, CV_MAKETYPE(depth, 3), dst_data, dst_step, width, height,
                                            IPPGeneralReorderFunctor(ippiHLS2RGBTab[depth], ippiSwapChannelsC3RTab[depth], 2, 1, 0, depth)) )
                        return;
                }
                else if(dcn == 4 && !swapBlue)
                {
                    if( CvtColorIPPLoop(src_data, src_step, dst_data, dst_step, width, height,
                                        IPPGeneralReorderFunctor(ippiHLS2RGBTab[depth], ippiSwapChannelsC3C4RTab[depth], 2, 1, 0, depth)) )
                        return;
                }
                else if(dcn == 3 && swapBlue)
                {
                    if( CvtColorIPPLoopCopy(src_data, src_step, CV_MAKETYPE(depth, 3), dst_data, dst_step, width, height,
                                            IPPGeneralFunctor(ippiHLS2RGBTab[depth])) )
                        return;
                }
                else if(dcn == 4 && swapBlue)
                {
                    if( CvtColorIPPLoop(src_data, src_step, dst_data, dst_step, width, height,
                                        IPPGeneralReorderFunctor(ippiHLS2RGBTab[depth], ippiSwapChannelsC3C4RTab[depth], 0, 1, 2, depth)) )
                        return;
                }
            }
        }
    }
#endif

    CV_CPU_DISPATCH(cvtHSVtoBGR, (src_data, src_step, dst_data, dst_step, width, height, depth, dcn, swapBlue, isFullRange, isHSV),
        CV_CPU_DISPATCH_MODES_ALL);
}

} // namespace hal

//
// OCL calls
//

#ifdef HAVE_OPENCL

bool oclCvtColorHSV2BGR( InputArray _src, OutputArray _dst, int dcn, int bidx, bool full )
{
    OclHelper< Set<3>, Set<3, 4>, Set<CV_8U, CV_32F> > h(_src, _dst, dcn);

    int hrange = _src.depth() == CV_32F ? 360 : (!full ? 180 : 255);

    if(!h.createKernel("HSV2RGB", ocl::imgproc::color_hsv_oclsrc,
                       format("-D DCN=%d -D BIDX=%d -D HRANGE=%d -D HSCALE=%ff", dcn, bidx, hrange, 6.f/hrange)))
    {
        return false;
    }

    return h.run();
}

bool oclCvtColorHLS2BGR( InputArray _src, OutputArray _dst, int dcn, int bidx, bool full )
{
    OclHelper< Set<3>, Set<3, 4>, Set<CV_8U, CV_32F> > h(_src, _dst, dcn);

    int hrange = _src.depth() == CV_32F ? 360 : (!full ? 180 : 255);

    if(!h.createKernel("HLS2RGB", ocl::imgproc::color_hsv_oclsrc,
                       format("-D DCN=%d -D BIDX=%d -D HRANGE=%d -D HSCALE=%ff", dcn, bidx, hrange, 6.f/hrange)))
    {
        return false;
    }

    return h.run();
}

bool oclCvtColorBGR2HLS( InputArray _src, OutputArray _dst, int bidx, bool full )
{
    OclHelper< Set<3, 4>, Set<3>, Set<CV_8U, CV_32F> > h(_src, _dst, 3);

    float hscale = (_src.depth() == CV_32F ? 360.f : (!full ? 180.f : 256.f))/360.f;

    if(!h.createKernel("RGB2HLS", ocl::imgproc::color_hsv_oclsrc,
                       format("-D HSCALE=%ff -D BIDX=%d -D DCN=3", hscale, bidx)))
    {
        return false;
    }

    return h.run();
}

static UMat init_sdiv_table()
{
    std::vector<int> sdiv(256);

    const int hsv_shift = 12;
    const int v = 255 << hsv_shift;

    sdiv[0] = 0;
    for(int i = 1; i < 256; i++ )
        sdiv[i] = saturate_cast<int>(v/(1.*i));

    return UMat(sdiv, true);
}

static UMat init_hdiv_table(int hrange)
{
    std::vector<int> hdiv(256);

    const int hsv_shift = 12;
    const int v = hrange << hsv_shift;

    hdiv[0] = 0;
    for (int i = 1; i < 256; i++ )
        hdiv[i] = saturate_cast<int>(v/(6.*i));

    return UMat(hdiv, true);
}

bool oclCvtColorBGR2HSV( InputArray _src, OutputArray _dst, int bidx, bool full )
{
    OclHelper< Set<3, 4>, Set<3>, Set<CV_8U, CV_32F> > h(_src, _dst, 3);

    int hrange = _src.depth() == CV_32F ? 360 : (!full ? 180 : 256);

    cv::String options = (_src.depth() == CV_8U ?
                          format("-D HRANGE=%d -D BIDX=%d -D DCN=3", hrange, bidx) :
                          format("-D HSCALE=%ff -D BIDX=%d -D DCN=3", hrange*(1.f/360.f), bidx));

    if(!h.createKernel("RGB2HSV", ocl::imgproc::color_hsv_oclsrc, options))
    {
        return false;
    }

    if(_src.depth() == CV_8U)
    {
        static UMat sdiv_data = init_sdiv_table();
        UMat hdiv_data;

        if (hrange == 180)
        {
            static UMat hdiv_data180 = init_hdiv_table(180);
            hdiv_data = hdiv_data180;
        }
        else
        {
            static UMat hdiv_data256 = init_hdiv_table(256);
            hdiv_data = hdiv_data256;
        }

        h.setArg(ocl::KernelArg::PtrReadOnly(sdiv_data));
        h.setArg(ocl::KernelArg::PtrReadOnly(hdiv_data));
    }

    return h.run();
}

#endif

//
// HAL calls
//

void cvtColorBGR2HLS( InputArray _src, OutputArray _dst, bool swapb, bool fullRange )
{
    CvtHelper< Set<3, 4>, Set<3>, Set<CV_8U, CV_32F> > h(_src, _dst, 3);

    hal::cvtBGRtoHSV(h.src.data, h.src.step, h.dst.data, h.dst.step, h.src.cols, h.src.rows,
                     h.depth, h.scn, swapb, fullRange, false);
}

void cvtColorBGR2HSV( InputArray _src, OutputArray _dst, bool swapb, bool fullRange )
{
    CvtHelper< Set<3, 4>, Set<3>, Set<CV_8U, CV_32F> > h(_src, _dst, 3);

    hal::cvtBGRtoHSV(h.src.data, h.src.step, h.dst.data, h.dst.step, h.src.cols, h.src.rows,
                     h.depth, h.scn, swapb, fullRange, true);
}

void cvtColorHLS2BGR( InputArray _src, OutputArray _dst, int dcn, bool swapb, bool fullRange)
{
    if(dcn <= 0) dcn = 3;
    CvtHelper< Set<3>, Set<3, 4>, Set<CV_8U, CV_32F> > h(_src, _dst, dcn);

    hal::cvtHSVtoBGR(h.src.data, h.src.step, h.dst.data, h.dst.step, h.src.cols, h.src.rows,
                     h.depth, dcn, swapb, fullRange, false);
}

void cvtColorHSV2BGR( InputArray _src, OutputArray _dst, int dcn, bool swapb, bool fullRange)
{
    if(dcn <= 0) dcn = 3;
    CvtHelper< Set<3>, Set<3, 4>, Set<CV_8U, CV_32F> > h(_src, _dst, dcn);

    hal::cvtHSVtoBGR(h.src.data, h.src.step, h.dst.data, h.dst.step, h.src.cols, h.src.rows,
                     h.depth, dcn, swapb, fullRange, true);
}


} // namespace cv
