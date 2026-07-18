// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "precomp.hpp"
#include "opencl_kernels_imgproc.hpp"

#include "color.hpp"

#include "color_rgb.simd.hpp"
#include "color_rgb.simd_declarations.hpp" // defines CV_CPU_DISPATCH_MODES_ALL=AVX2,...,BASELINE based on CMakeLists.txt content

namespace cv {

//
// HAL functions
//

namespace hal {

// 8u, 16u, 32f
void cvtBGRtoBGR(const uchar * src_data, size_t src_step,
                 uchar * dst_data, size_t dst_step,
                 int width, int height,
                 int depth, int scn, int dcn, bool swapBlue)
{
    CV_INSTRUMENT_REGION();

    CALL_HAL(cvtBGRtoBGR, cv_hal_cvtBGRtoBGR, src_data, src_step, dst_data, dst_step, width, height, depth, scn, dcn, swapBlue);

    CV_CPU_DISPATCH(cvtBGRtoBGR, (src_data, src_step, dst_data, dst_step, width, height, depth, scn, dcn, swapBlue),
        CV_CPU_DISPATCH_MODES_ALL);
}

// only 8u
void cvtBGRtoBGR5x5(const uchar * src_data, size_t src_step,
                    uchar * dst_data, size_t dst_step,
                    int width, int height,
                    int scn, bool swapBlue, int greenBits)
{
    CV_INSTRUMENT_REGION();

    CALL_HAL(cvtBGRtoBGR5x5, cv_hal_cvtBGRtoBGR5x5, src_data, src_step, dst_data, dst_step, width, height, scn, swapBlue, greenBits);

    CV_CPU_DISPATCH(cvtBGRtoBGR5x5, (src_data, src_step, dst_data, dst_step, width, height, scn, swapBlue, greenBits),
        CV_CPU_DISPATCH_MODES_ALL);
}

// only 8u
void cvtBGR5x5toBGR(const uchar * src_data, size_t src_step,
                    uchar * dst_data, size_t dst_step,
                    int width, int height,
                    int dcn, bool swapBlue, int greenBits)
{
    CV_INSTRUMENT_REGION();

    CALL_HAL(cvtBGR5x5toBGR, cv_hal_cvtBGR5x5toBGR, src_data, src_step, dst_data, dst_step, width, height, dcn, swapBlue, greenBits);

    CV_CPU_DISPATCH(cvtBGR5x5toBGR, (src_data, src_step, dst_data, dst_step, width, height, dcn, swapBlue, greenBits),
        CV_CPU_DISPATCH_MODES_ALL);
}

// 8u, 16u, 32f
void cvtBGRtoGray(const uchar * src_data, size_t src_step,
                  uchar * dst_data, size_t dst_step,
                  int width, int height,
                  int depth, int scn, bool swapBlue)
{
    CV_INSTRUMENT_REGION();

    CALL_HAL(cvtBGRtoGray, cv_hal_cvtBGRtoGray, src_data, src_step, dst_data, dst_step, width, height, depth, scn, swapBlue);

    CV_CPU_DISPATCH(cvtBGRtoGray, (src_data, src_step, dst_data, dst_step, width, height, depth, scn, swapBlue),
        CV_CPU_DISPATCH_MODES_ALL);
}

// 8u, 16u, 32f
void cvtGraytoBGR(const uchar * src_data, size_t src_step,
                  uchar * dst_data, size_t dst_step,
                  int width, int height,
                  int depth, int dcn)
{
    CV_INSTRUMENT_REGION();

    CALL_HAL(cvtGraytoBGR, cv_hal_cvtGraytoBGR, src_data, src_step, dst_data, dst_step, width, height, depth, dcn);

    CV_CPU_DISPATCH(cvtGraytoBGR, (src_data, src_step, dst_data, dst_step, width, height, depth, dcn),
        CV_CPU_DISPATCH_MODES_ALL);
}

// only 8u
void cvtBGR5x5toGray(const uchar * src_data, size_t src_step,
                     uchar * dst_data, size_t dst_step,
                     int width, int height,
                     int greenBits)
{
    CV_INSTRUMENT_REGION();

    CALL_HAL(cvtBGR5x5toGray, cv_hal_cvtBGR5x5toGray, src_data, src_step, dst_data, dst_step, width, height, greenBits);

    CV_CPU_DISPATCH(cvtBGR5x5toGray, (src_data, src_step, dst_data, dst_step, width, height, greenBits),
        CV_CPU_DISPATCH_MODES_ALL);
}

// only 8u
void cvtGraytoBGR5x5(const uchar * src_data, size_t src_step,
                     uchar * dst_data, size_t dst_step,
                     int width, int height,
                     int greenBits)
{
    CV_INSTRUMENT_REGION();

    CALL_HAL(cvtGraytoBGR5x5, cv_hal_cvtGraytoBGR5x5, src_data, src_step, dst_data, dst_step, width, height, greenBits);

    CV_CPU_DISPATCH(cvtGraytoBGR5x5, (src_data, src_step, dst_data, dst_step, width, height, greenBits),
        CV_CPU_DISPATCH_MODES_ALL);
}

void cvtRGBAtoMultipliedRGBA(const uchar * src_data, size_t src_step,
                             uchar * dst_data, size_t dst_step,
                             int width, int height)
{
    CV_INSTRUMENT_REGION();

    CALL_HAL(cvtRGBAtoMultipliedRGBA, cv_hal_cvtRGBAtoMultipliedRGBA, src_data, src_step, dst_data, dst_step, width, height);

    CV_CPU_DISPATCH(cvtRGBAtoMultipliedRGBA, (src_data, src_step, dst_data, dst_step, width, height),
        CV_CPU_DISPATCH_MODES_ALL);
}

void cvtMultipliedRGBAtoRGBA(const uchar * src_data, size_t src_step,
                             uchar * dst_data, size_t dst_step,
                             int width, int height)
{
    CV_INSTRUMENT_REGION();

    CALL_HAL(cvtMultipliedRGBAtoRGBA, cv_hal_cvtMultipliedRGBAtoRGBA, src_data, src_step, dst_data, dst_step, width, height);

    CV_CPU_DISPATCH(cvtMultipliedRGBAtoRGBA, (src_data, src_step, dst_data, dst_step, width, height),
        CV_CPU_DISPATCH_MODES_ALL);
}

} // namespace hal

//
// OCL calls
//

#ifdef HAVE_OPENCL

bool oclCvtColorBGR2BGR( InputArray _src, OutputArray _dst, int dcn, bool reverse )
{
    OclHelper< Set<3, 4>, Set<3, 4>, Set<CV_8U, CV_16U, CV_32F> > h(_src, _dst, dcn);

    if(!h.createKernel("RGB", ocl::imgproc::color_rgb_oclsrc,
                       format("-D DCN=%d -D BIDX=0 -D %s", dcn, reverse ? "REVERSE" : "ORDER")))
    {
        return false;
    }

    return h.run();
}

bool oclCvtColorBGR25x5( InputArray _src, OutputArray _dst, int bidx, int gbits )
{
    OclHelper< Set<3, 4>, Set<2>, Set<CV_8U> > h(_src, _dst, 2);

    if(!h.createKernel("RGB2RGB5x5", ocl::imgproc::color_rgb_oclsrc,
                       format("-D DCN=2 -D BIDX=%d -D GREENBITS=%d", bidx, gbits)))
    {
        return false;
    }

    return h.run();
}

bool oclCvtColor5x52BGR( InputArray _src, OutputArray _dst, int dcn, int bidx, int gbits)
{
    OclHelper< Set<2>, Set<3, 4>, Set<CV_8U> > h(_src, _dst, dcn);

    if(!h.createKernel("RGB5x52RGB", ocl::imgproc::color_rgb_oclsrc,
                       format("-D DCN=%d -D BIDX=%d -D GREENBITS=%d", dcn, bidx, gbits)))
    {
        return false;
    }

    return h.run();
}

bool oclCvtColor5x52Gray( InputArray _src, OutputArray _dst, int gbits)
{
    OclHelper< Set<2>, Set<1>, Set<CV_8U> > h(_src, _dst, 1);

    if(!h.createKernel("BGR5x52Gray", ocl::imgproc::color_rgb_oclsrc,
                       format("-D DCN=1 -D BIDX=0 -D GREENBITS=%d", gbits)))
    {
        return false;
    }

    return h.run();
}

bool oclCvtColorGray25x5( InputArray _src, OutputArray _dst, int gbits)
{
    OclHelper< Set<1>, Set<2>, Set<CV_8U> > h(_src, _dst, 2);

    if(!h.createKernel("Gray2BGR5x5", ocl::imgproc::color_rgb_oclsrc,
                        format("-D DCN=2 -D BIDX=0 -D GREENBITS=%d", gbits)))
    {
        return false;
    }

    return h.run();
}

bool oclCvtColorBGR2Gray( InputArray _src, OutputArray _dst, int bidx)
{
    OclHelper< Set<3, 4>, Set<1>, Set<CV_8U, CV_16U, CV_32F> > h(_src, _dst, 1);

    int stripeSize = 1;
    if(!h.createKernel("RGB2Gray", ocl::imgproc::color_rgb_oclsrc,
                       format("-D DCN=1 -D BIDX=%d -D STRIPE_SIZE=%d", bidx, stripeSize)))
    {
        return false;
    }

    h.globalSize[0] = (h.src.cols + stripeSize - 1)/stripeSize;
    return h.run();
}

bool oclCvtColorGray2BGR( InputArray _src, OutputArray _dst, int dcn)
{
    OclHelper< Set<1>, Set<3, 4>, Set<CV_8U, CV_16U, CV_32F> > h(_src, _dst, dcn);
    if(!h.createKernel("Gray2RGB", ocl::imgproc::color_rgb_oclsrc,
                       format("-D BIDX=0 -D DCN=%d", dcn)))
    {
        return false;
    }

    return h.run();
}

bool oclCvtColorRGBA2mRGBA( InputArray _src, OutputArray _dst)
{
    OclHelper< Set<4>, Set<4>, Set<CV_8U> > h(_src, _dst, 4);

    if(!h.createKernel("RGBA2mRGBA", ocl::imgproc::color_rgb_oclsrc,
                       "-D DCN=4 -D BIDX=3"))
    {
        return false;
    }

    return h.run();
}

bool oclCvtColormRGBA2RGBA( InputArray _src, OutputArray _dst)
{
    OclHelper< Set<4>, Set<4>, Set<CV_8U> > h(_src, _dst, 4);

    if(!h.createKernel("mRGBA2RGBA", ocl::imgproc::color_rgb_oclsrc,
                       "-D DCN=4 -D BIDX=3"))
    {
        return false;
    }

    return h.run();
}

#endif

//
// HAL calls
//

void cvtColorBGR2BGR( InputArray _src, OutputArray _dst, int dcn, bool swapb)
{
    CvtHelper< Set<3, 4>, Set<3, 4>, Set<CV_8U, CV_16U, CV_32F> > h(_src, _dst, dcn);

    hal::cvtBGRtoBGR(h.src.data, h.src.step, h.dst.data, h.dst.step, h.src.cols, h.src.rows,
                     h.depth, h.scn, dcn, swapb);
}

void cvtColorBGR25x5( InputArray _src, OutputArray _dst, bool swapb, int gbits)
{
    CvtHelper< Set<3, 4>, Set<2>, Set<CV_8U> > h(_src, _dst, 2);

    hal::cvtBGRtoBGR5x5(h.src.data, h.src.step, h.dst.data, h.dst.step, h.src.cols, h.src.rows,
                        h.scn, swapb, gbits);
}

void cvtColor5x52BGR( InputArray _src, OutputArray _dst, int dcn, bool swapb, int gbits)
{
    if(dcn <= 0) dcn = 3;
    CvtHelper< Set<2>, Set<3, 4>, Set<CV_8U> > h(_src, _dst, dcn);

    hal::cvtBGR5x5toBGR(h.src.data, h.src.step, h.dst.data, h.dst.step, h.src.cols, h.src.rows,
                        dcn, swapb, gbits);
}

void cvtColorBGR2Gray( InputArray _src, OutputArray _dst, bool swapb)
{
    CvtHelper< Set<3, 4>, Set<1>, Set<CV_8U, CV_16U, CV_32F> > h(_src, _dst, 1);

    hal::cvtBGRtoGray(h.src.data, h.src.step, h.dst.data, h.dst.step, h.src.cols, h.src.rows,
                      h.depth, h.scn, swapb);
}

void cvtColorGray2BGR( InputArray _src, OutputArray _dst, int dcn)
{
    if(dcn <= 0) dcn = 3;
    CvtHelper< Set<1>, Set<3, 4>, Set<CV_8U, CV_16U, CV_32F> > h(_src, _dst, dcn);

    hal::cvtGraytoBGR(h.src.data, h.src.step, h.dst.data, h.dst.step, h.src.cols, h.src.rows, h.depth, dcn);
}

void cvtColor5x52Gray( InputArray _src, OutputArray _dst, int gbits)
{
    CvtHelper< Set<2>, Set<1>, Set<CV_8U> > h(_src, _dst, 1);

    hal::cvtBGR5x5toGray(h.src.data, h.src.step, h.dst.data, h.dst.step, h.src.cols, h.src.rows, gbits);
}

void cvtColorGray25x5( InputArray _src, OutputArray _dst, int gbits)
{
    CvtHelper< Set<1>, Set<2>, Set<CV_8U> > h(_src, _dst, 2);

    hal::cvtGraytoBGR5x5(h.src.data, h.src.step, h.dst.data, h.dst.step, h.src.cols, h.src.rows, gbits);
}

void cvtColorRGBA2mRGBA( InputArray _src, OutputArray _dst)
{
    CvtHelper< Set<4>, Set<4>, Set<CV_8U> > h(_src, _dst, 4);

    hal::cvtRGBAtoMultipliedRGBA(h.src.data, h.src.step, h.dst.data, h.dst.step, h.src.cols, h.src.rows);
}

void cvtColormRGBA2RGBA( InputArray _src, OutputArray _dst)
{
    CvtHelper< Set<4>, Set<4>, Set<CV_8U> > h(_src, _dst, 4);

    hal::cvtMultipliedRGBAtoRGBA(h.src.data, h.src.step, h.dst.data, h.dst.step, h.src.cols, h.src.rows);
}

} // namespace cv
