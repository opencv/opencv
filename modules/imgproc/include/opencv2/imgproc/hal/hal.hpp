#ifndef CV_IMGPROC_HAL_HPP
#define CV_IMGPROC_HAL_HPP

#include "opencv2/core/cvdef.h"
#include "opencv2/core/cvstd.hpp"
#include "opencv2/core/hal/interface.h"

namespace cv { namespace hal {

//! @addtogroup core_hal_functions
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

//! @}

}}

#endif // CV_IMGPROC_HAL_HPP
