// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2014, Itseez, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.

#include "../test_precomp.hpp"
#include "opencv2/ts/ocl_test.hpp"

#ifdef HAVE_OPENCL

namespace cvtest {
namespace ocl {

PARAM_TEST_CASE(Image2DBasicTest, int, int)
{
    int depth, ch;


};

TEST(Image2D, turnOffOpenCL)
{
    if (cv::ocl::haveOpenCL())
    {
        // save the current state
        bool useOCL = cv::ocl::useOpenCL();

        cv::ocl::setUseOpenCL(true);
        UMat um(128, 128, CV_8UC1);

        cv::ocl::setUseOpenCL(false);
        cv::ocl::Image2D image(um);
    
        // reset state to the previous one
        cv::ocl::setUseOpenCL(useOCL);
    }
}

} } // namespace cvtest::ocl

#endif // HAVE_OPENCL