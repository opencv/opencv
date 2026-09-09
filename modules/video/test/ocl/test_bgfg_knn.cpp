// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../test_precomp.hpp"
#include "opencv2/ts/ocl_test.hpp"

#ifdef HAVE_OPENCL

namespace opencv_test {
namespace ocl {

OCL_TEST(BackgroundSubtractorKNN, ZeroLearningRateFreezesModel)
{
    const int imageTypes[] = { CV_8UC1, CV_8UC3 };
    for (int imageType : imageTypes)
    {
        SCOPED_TRACE(cv::typeToString(imageType));

        Ptr<BackgroundSubtractorKNN> knn =
                createBackgroundSubtractorKNN(50, 400.0, false);
        UMat background(Size(16, 16), imageType, Scalar::all(0));
        UMat foreground(background.size(), background.type(), Scalar::all(255));
        UMat fgmask;

        for (int i = 0; i < 30; ++i)
        {
            OCL_ON(knn->apply(background, fgmask, 0.2));
        }
        ASSERT_EQ(0, countNonZero(fgmask));

        for (int i = 0; i < 30; ++i)
        {
            OCL_ON(knn->apply(foreground, fgmask, 0.0));
            EXPECT_EQ((int)fgmask.total(), countNonZero(fgmask)) << "iteration " << i;
        }

        UMat model;
        OCL_ON(knn->getBackgroundImage(model));
        EXPECT_EQ(0.0, cv::norm(background, model, NORM_INF));

        for (int i = 0; i < 30; ++i)
        {
            OCL_ON(knn->apply(foreground, fgmask, 0.2));
        }
        EXPECT_EQ(0, countNonZero(fgmask));
    }
}

}} // namespace opencv_test::ocl

#endif
