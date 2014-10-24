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

TEST(Image2D, createAliasEmptyUMat)
{
    if (cv::ocl::haveOpenCL())
    {
        UMat um;
        EXPECT_FALSE(cv::ocl::Image2D::canCreateAlias(um));
    }
    else
        std::cout << "OpenCL runtime not found. Test skipped." << std::endl;
}

TEST(Image2D, createImage2DWithEmptyUMat)
{
    if (cv::ocl::haveOpenCL())
    {
        UMat um;
        EXPECT_ANY_THROW(cv::ocl::Image2D image(um));
    }
    else
        std::cout << "OpenCL runtime not found. Test skipped." << std::endl;
}

TEST(Image2D, createAlias)
{
    if (cv::ocl::haveOpenCL())
    {
        const cv::ocl::Device & d = cv::ocl::Device::getDefault();
        int minor = d.deviceVersionMinor(), major = d.deviceVersionMajor();

        // aliases is OpenCL 1.2 extension
        if (1 < major || (1 == major && 2 <= minor))
        {
            UMat um(128, 128, CV_8UC1);
            bool isFormatSupported = false, canCreateAlias = false;

            EXPECT_NO_THROW(isFormatSupported = cv::ocl::Image2D::isFormatSupported(CV_8U, 1, false));
            EXPECT_NO_THROW(canCreateAlias = cv::ocl::Image2D::canCreateAlias(um));

            if (isFormatSupported && canCreateAlias)
            {
                EXPECT_NO_THROW(cv::ocl::Image2D image(um, false, true));
            }
            else
                std::cout << "Impossible to create alias for selected image. Test skipped." << std::endl;
        }
    }
    else
        std::cout << "OpenCL runtime not found. Test skipped" << std::endl;
}

TEST(Image2D, turnOffOpenCL)
{
    if (cv::ocl::haveOpenCL())
    {
        // save the current state
        bool useOCL = cv::ocl::useOpenCL();
        bool isFormatSupported = false;

        cv::ocl::setUseOpenCL(true);
        UMat um(128, 128, CV_8UC1);

        cv::ocl::setUseOpenCL(false);
        EXPECT_NO_THROW(isFormatSupported = cv::ocl::Image2D::isFormatSupported(CV_8U, 1, true));

        if (isFormatSupported)
        {
            EXPECT_NO_THROW(cv::ocl::Image2D image(um));
        }
        else
            std::cout << "CV_8UC1 is not supported for OpenCL images. Test skipped." << std::endl;
    
        // reset state to the previous one
        cv::ocl::setUseOpenCL(useOCL);
    }
    else
        std::cout << "OpenCL runtime not found. Test skipped." << std::endl;
}

} } // namespace cvtest::ocl

#endif // HAVE_OPENCL