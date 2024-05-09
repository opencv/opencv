// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"

#include "write_mat_to_xrgb8888.hpp"
#include "write_mat_to_xrgb8888.simd.hpp"
#include "write_mat_to_xrgb8888.simd_declarations.hpp" // defines CV_CPU_DISPATCH_MODES_ALL=AVX2,...,BASELINE based on CMakeLists.txt content


namespace cv {
namespace impl {
void write_mat_to_xrgb8888(cv::Mat const &img_, void *data)
{
    // Validate destination data.
    CV_CheckFalse((data == nullptr), "Destination Address must not be nullptr.");

    // Validate source img parameters.
    CV_CheckFalse(img_.empty(), "Source Mat must not be empty.");
    const int ncn   = img_.channels();
    CV_CheckType(img_.type(),
        ( (ncn == 1) || (ncn == 3) || (ncn == 4)),
        "Unsupported channels, please convert to 1, 3 or 4 channels"
    );

    // The supported Mat depth is according to imshow() specification.
    const int depth = CV_MAT_DEPTH(img_.type());
    CV_CheckDepth(img_.type(),
        ( (depth == CV_8U)  || (depth == CV_8S)  ||
          (depth == CV_16U) || (depth == CV_16S) ||
          (depth == CV_32F) || (depth == CV_64F) ),
        "Unsupported depth, please convert to CV_8U"
    );

    // Convert to CV_8U
    cv::Mat img;
    const int mtype = CV_MAKE_TYPE(CV_8U, ncn);
    switch(CV_MAT_DEPTH(depth))
    {
    case CV_8U:
        img = img_; // do nothing.
        break;
    case CV_8S:
        // [-128,127] -> [0,255]
        img_.convertTo(img, mtype, 1.0, 128);
        break;
    case CV_16U:
        // [0,65535] -> [0,255]
        img_.convertTo(img, mtype, 1.0/255. );
        break;
    case CV_16S:
        // [-32768,32767] -> [0,255]
        img_.convertTo(img, mtype, 1.0/255. , 128);
        break;
    case CV_32F:
    case CV_64F:
        // [0, 1] -> [0,255]
        img_.convertTo(img, mtype, 255.);
        break;
    default:
        // it cannot be reachable.
        break;
    }
    CV_CheckDepthEQ(CV_MAT_DEPTH(img.type()), CV_8U, "img should be CV_8U");

    int img_rows = img.rows;
    int img_cols = img.cols;

    // Reshape to reduce calling img.ptr() and CV_CPU_DISPATCH()
    if(img.isContinuous())
    {
        img_cols *= img_rows;
        img_rows = 1;
    }

    // Convert to xrgb8888
    uint8_t* dst = (uint8_t*)data;
    for(int y = 0; y < img_rows; y++, dst+=img_cols * 4)
    {
        const uint8_t* src = img.ptr(y);
        CV_CPU_DISPATCH(write_raw_to_xrgb8888, (src, dst, img_cols, ncn), CV_CPU_DISPATCH_MODES_ALL);
    }
}

} // namespace impl
} // namespace cv
