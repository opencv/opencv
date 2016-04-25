#ifndef CV_IMGPROC_HAL_HPP
#define CV_IMGPROC_HAL_HPP

#include "opencv2/core/cvdef.h"
#include "opencv2/core/cvstd.hpp"
#include "opencv2/core/hal/interface.h"

namespace cv { namespace hal {

//! @addtogroup imgproc_hal_functions
//! @{

struct CV_EXPORTS Filter2D
{
    static Ptr<hal::Filter2D> create(uchar * kernel_data, size_t kernel_step, int kernel_type,
                                     int kernel_width, int kernel_height,
                                     int max_width, int max_height,
                                     int stype, int dtype,
                                     int borderType, double delta,
                                     int anchor_x, int anchor_y,
                                     bool isSubmatrix, bool isInplace);
    virtual void apply(uchar * src_data, size_t src_step,
                       uchar * dst_data, size_t dst_step,
                       int width, int height,
                       int full_width, int full_height,
                       int offset_x, int offset_y) = 0;
    virtual ~Filter2D() {}
};

struct CV_EXPORTS SepFilter2D
{
    static Ptr<hal::SepFilter2D> create(int stype, int dtype, int ktype,
                                        uchar * kernelx_data, size_t kernelx_step,
                                        int kernelx_width, int kernelx_height,
                                        uchar * kernely_data, size_t kernely_step,
                                        int kernely_width, int kernely_height,
                                        int anchor_x, int anchor_y,
                                        double delta, int borderType);
    virtual void apply(uchar * src_data, size_t src_step,
                       uchar * dst_data, size_t dst_step,
                       int width, int height,
                       int full_width, int full_height,
                       int offset_x, int offset_y) = 0;
    virtual ~SepFilter2D() {}
};


struct  CV_EXPORTS Morph
{
    static Ptr<Morph> create(int op, int src_type, int dst_type, int max_width, int max_height,
                                    int kernel_type, uchar * kernel_data, size_t kernel_step,
                                    int kernel_width, int kernel_height,
                                    int anchor_x, int anchor_y,
                                    int borderType, const double borderValue[4],
                                    int iterations, bool isSubmatrix, bool allowInplace);
    virtual void apply(uchar * src_data, size_t src_step, uchar * dst_data, size_t dst_step, int width, int height,
                       int roi_width, int roi_height, int roi_x, int roi_y,
                       int roi_width2, int roi_height2, int roi_x2, int roi_y2) = 0;
    virtual ~Morph() {}
};


CV_EXPORTS void resize(int src_type,
                       const uchar * src_data, size_t src_step, int src_width, int src_height,
                       uchar * dst_data, size_t dst_step, int dst_width, int dst_height,
                       double inv_scale_x, double inv_scale_y, int interpolation);

CV_EXPORTS void warpAffine(int src_type,
                           const uchar * src_data, size_t src_step, int src_width, int src_height,
                           uchar * dst_data, size_t dst_step, int dst_width, int dst_height,
                           const double M[6], int interpolation, int borderType, const double borderValue[4]);

CV_EXPORTS void warpPerspectve(int src_type,
                               const uchar * src_data, size_t src_step, int src_width, int src_height,
                               uchar * dst_data, size_t dst_step, int dst_width, int dst_height,
                               const double M[9], int interpolation, int borderType, const double borderValue[4]);

//! @}

}}

#endif // CV_IMGPROC_HAL_HPP
