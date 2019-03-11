// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "perf_precomp.hpp"
#include "opencv2/ts/ocl_perf.hpp"

namespace opencv_test
{
using namespace perf;
using namespace ::cvtest::ocl;


struct OpenCLState
{
    OpenCLState(bool useOpenCL)
    {
        isOpenCL_enabled = cv::ocl::useOpenCL();
        cv::ocl::setUseOpenCL(useOpenCL);
    }

    ~OpenCLState()
    {
        cv::ocl::setUseOpenCL(isOpenCL_enabled);
    }

private:
    bool isOpenCL_enabled;
};

typedef TestBaseWithParam< tuple<Size, bool, int> > UMatTest;

OCL_PERF_TEST_P(UMatTest, CustomPtr, Combine(Values(sz1080p, sz2160p), Bool(), ::testing::Values(4, 64, 4096)))
{
    OpenCLState s(get<1>(GetParam()));

    int type = CV_8UC1;
    cv::Size size = get<0>(GetParam());
    size_t align_base = 4096;
    const int align_offset = get<2>(GetParam());

    void* pData_allocated = new unsigned char [size.area() * CV_ELEM_SIZE(type) + (align_base + align_offset)];
    void* pData = (char*)alignPtr(pData_allocated, (int)align_base) + align_offset;
    size_t step = size.width * CV_ELEM_SIZE(type);

    OCL_TEST_CYCLE()
    {
        Mat m = Mat(size, type, pData, step);
        m.setTo(cv::Scalar::all(2));

        UMat u = m.getUMat(ACCESS_RW);
        cv::add(u, cv::Scalar::all(2), u);
        cv::add(u, cv::Scalar::all(3), u);

        Mat d = u.getMat(ACCESS_READ);
        ASSERT_EQ(7, d.at<char>(0, 0));
    }

    delete[] (unsigned char*)pData_allocated;

    SANITY_CHECK_NOTHING();
}

} // namespace
