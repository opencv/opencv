// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "test_precomp.hpp"

namespace opencv_test { namespace {

/** @brief The non-CPU devices that really hold data, or skips the test when there are none. */
static std::vector<ComputeDevice> testDevices()
{
    std::vector<ComputeDevice> devices;
    const std::vector<ComputeDevice> all = ComputeDevice::enumerate();
    for (size_t i = 1; i < all.size(); i++)
        if (all[i].backend() != "opencl" || ocl::useOpenCL())
            devices.push_back(all[i]);
    if (devices.empty())
        throw SkipTestException("no device besides the CPU");
    return devices;
}

/** @brief A matrix of random values in [-5, 5). */
static Mat randomMat(int rows, int cols, int type = CV_32FC3)
{
    Mat m(rows, cols, type);
    randu(m, -5, 5);
    return m;
}

TEST(Core_Tensor, empty_by_default)
{
    const Tensor t;
    EXPECT_TRUE(t.empty());
    EXPECT_TRUE(t.device().isCPU());
    EXPECT_TRUE(t.getMat(ACCESS_READ).empty());
}

TEST(Core_Tensor, wraps_mat_without_copy)
{
    const Mat m(3, 4, CV_32FC2, Scalar(1, 2));
    const Tensor t(m);
    EXPECT_TRUE(t.device().isCPU());
    EXPECT_EQ(m.data, t.getMat(ACCESS_READ).data);
    EXPECT_EQ(m.type(), t.type());
    EXPECT_EQ(2, t.channels());
    EXPECT_EQ(m.shape(), t.shape());
    EXPECT_EQ((size_t)12, t.total());
}

TEST(Core_Tensor, cpu_tensor_in_cv_functions)
{
    const Mat a = randomMat(5, 7), b = randomMat(5, 7);
    Mat ref;
    cv::add(a, b, ref);
    Tensor sum;
    cv::add(Tensor(a), Tensor(b), sum);
    EXPECT_TRUE(sum.device().isCPU());
    EXPECT_EQ(0, cvtest::norm(ref, sum.getMat(ACCESS_READ), NORM_INF));
}

TEST(Core_Tensor, output_written_in_place)
{
    Mat dst(5, 7, CV_32FC3, Scalar::all(0));
    Tensor t(dst);
    Tensor(Mat(5, 7, CV_32FC3, Scalar::all(3))).copyTo(t);
    EXPECT_EQ(3.f, dst.at<Vec3f>(2, 2)[1]);
}

TEST(Core_Tensor, views_share_data)
{
    const Tensor t = Tensor::zeros(MatShape({4, 6}), CV_8UC1, ComputeDevice::cpu());
    t(Rect(1, 1, 2, 2)).setTo(Scalar::all(9));
    const Mat m = t.getMat(ACCESS_READ);
    EXPECT_EQ(9, m.at<uchar>(1, 1));
    EXPECT_EQ(0, m.at<uchar>(0, 0));

    const Tensor flat = t.reshape(MatShape({24}));
    EXPECT_EQ(1, flat.dims());
    EXPECT_EQ(m.data, flat.getMat(ACCESS_READ).data);

    const std::vector<Range> rows = {Range(2, 4), Range::all()};
    EXPECT_EQ(m.ptr(2), t(rows).getMat(ACCESS_READ).data);
}

TEST(Core_Tensor, roi_as_output)
{
    const Tensor big = Tensor::zeros(MatShape({4, 4}), CV_8UC1, ComputeDevice::cpu());
    Tensor(Mat(2, 2, CV_8UC1, Scalar::all(5))).copyTo(big(Rect(2, 2, 2, 2)));
    const Mat m = big.getMat(ACCESS_READ);
    EXPECT_EQ(5, m.at<uchar>(3, 3));
    EXPECT_EQ(0, m.at<uchar>(1, 1));
}

TEST(Core_Tensor, scalar_tensor)
{
    const Tensor s(MatShape::scalar(), CV_32FC1, Scalar(2.5), ComputeDevice::cpu());
    EXPECT_EQ(0, s.dims());
    EXPECT_EQ((size_t)1, s.total());
    EXPECT_FALSE(s.empty());
    EXPECT_EQ(2.5f, *s.getMat(ACCESS_READ).ptr<float>());
}

TEST(Core_Tensor, to_same_device_shares_data)
{
    const Tensor t(Mat(2, 2, CV_8UC1, Scalar::all(1)));
    EXPECT_EQ(t.getMat(ACCESS_READ).data, t.to(ComputeDevice::cpu()).getMat(ACCESS_READ).data);
    EXPECT_EQ(t.getMat(ACCESS_READ).data, t.to("cpu").getMat(ACCESS_READ).data);
}

TEST(Core_Tensor, release_leaves_empty_tensor)
{
    const Mat m(3, 3, CV_8UC1, Scalar::all(4));
    Tensor t(m);
    t.release();
    EXPECT_TRUE(t.empty());
    EXPECT_EQ(4, m.at<uchar>(1, 1));
}

TEST(Core_Tensor, device_round_trip)
{
    const Mat src = randomMat(17, 23);
    for (const ComputeDevice& dev : testDevices())
    {
        SCOPED_TRACE(dev.name());
        const Tensor onDevice = Tensor(src).to(dev);
        EXPECT_EQ(dev, onDevice.device());
        EXPECT_EQ(src.shape(), onDevice.shape());
        EXPECT_EQ(src.type(), onDevice.type());
        EXPECT_EQ(0, cvtest::norm(src, onDevice.getMat(ACCESS_READ), NORM_INF));

        const Tensor back = onDevice.to(ComputeDevice::cpu());
        EXPECT_TRUE(back.device().isCPU());
        EXPECT_EQ(0, cvtest::norm(src, back.getMat(ACCESS_READ), NORM_INF));
    }
}

TEST(Core_Tensor, wraps_umat_without_copy)
{
    for (const ComputeDevice& dev : testDevices())
    {
        SCOPED_TRACE(dev.name());
        UMat u;
        u.allocator = dev.allocator();
        u.create(4, 4, CV_8UC1);
        const Tensor t(u);
        EXPECT_EQ(dev, t.device());
        EXPECT_EQ(u.u, t.getUMat(ACCESS_READ).u);
    }
}

TEST(Core_Tensor, device_outputs)
{
    const Mat a = randomMat(9, 11), b = randomMat(9, 11);
    Mat ref;
    cv::add(a, b, ref);
    for (const ComputeDevice& dev : testDevices())
    {
        SCOPED_TRACE(dev.name());
        const Tensor ta = Tensor(a).to(dev), tb = Tensor(b).to(dev);

        Tensor inScope;
        {
            DeviceScope scope(dev);
            cv::add(ta, tb, inScope);
        }
        EXPECT_EQ(dev, inScope.device());
        EXPECT_LE(cvtest::norm(ref, inScope.getMat(ACCESS_READ), NORM_INF), 1e-5);

        Tensor outsideScope;
        cv::add(ta, tb, outsideScope);
        EXPECT_TRUE(outsideScope.device().isCPU());
        EXPECT_LE(cvtest::norm(ref, outsideScope.getMat(ACCESS_READ), NORM_INF), 1e-5);

        Tensor preallocated = Tensor::zeros(MatShape({9, 11}), CV_32FC3, dev);
        cv::add(ta, tb, preallocated);
        EXPECT_EQ(dev, preallocated.device());
        EXPECT_LE(cvtest::norm(ref, preallocated.getMat(ACCESS_READ), NORM_INF), 1e-5);
    }
}

TEST(Core_Tensor, resized_output_keeps_device)
{
    const Mat src = randomMat(3, 5);
    Mat ref;
    cv::transpose(src, ref);
    for (const ComputeDevice& dev : testDevices())
    {
        SCOPED_TRACE(dev.name());
        const Tensor t = Tensor(src).to(dev);

        Tensor allocated = Tensor::zeros(MatShape({4, 4}), CV_8UC1, dev);
        cv::transpose(t, allocated);
        EXPECT_EQ(dev, allocated.device());
        EXPECT_EQ(0, cvtest::norm(ref, allocated.getMat(ACCESS_READ), NORM_INF));

        Tensor madeInScope;
        {
            DeviceScope scope(dev);
            cv::add(t, t, madeInScope);
        }
        cv::transpose(t, madeInScope);
        EXPECT_EQ(dev, madeInScope.device());
        EXPECT_EQ(0, cvtest::norm(ref, madeInScope.getMat(ACCESS_READ), NORM_INF));

        madeInScope.create(MatShape({8, 2}), CV_32FC1);
        EXPECT_EQ(dev, madeInScope.device());
        EXPECT_EQ(MatShape({8, 2}), madeInScope.shape());
    }
}

TEST(Core_Tensor, methods_follow_source_device)
{
    const Mat src = randomMat(6, 8);
    for (const ComputeDevice& dev : testDevices())
    {
        SCOPED_TRACE(dev.name());
        const Tensor t = Tensor(src).to(dev);

        const Tensor copy = t.clone();
        EXPECT_EQ(dev, copy.device());
        EXPECT_EQ(0, cvtest::norm(src, copy.getMat(ACCESS_READ), NORM_INF));

        Tensor copied;
        t.copyTo(copied);
        EXPECT_EQ(dev, copied.device());

        Tensor converted;
        t.convertTo(converted, CV_64F, 2.0);
        EXPECT_EQ(dev, converted.device());
        EXPECT_EQ(CV_64FC3, converted.type());
        Mat ref;
        src.convertTo(ref, CV_64F, 2.0);
        EXPECT_LE(cvtest::norm(ref, converted.getMat(ACCESS_READ), NORM_INF), 1e-6);

        EXPECT_EQ(dev, t(Rect(1, 1, 3, 3)).device());
        EXPECT_EQ(dev, t.reshape(MatShape({48})).device());
    }
}

TEST(Core_Tensor, scope_sets_device_of_new_tensors)
{
    for (const ComputeDevice& dev : testDevices())
    {
        SCOPED_TRACE(dev.name());
        DeviceScope scope(dev);
        EXPECT_EQ(dev, Tensor(MatShape({3, 3}), CV_8UC1).device());
        EXPECT_EQ(dev, Tensor::zeros(MatShape({3, 3}), CV_8UC1).device());
        EXPECT_TRUE(Tensor(Mat(3, 3, CV_8UC1)).device().isCPU());

        Tensor later;
        later.create(MatShape({2, 2}), CV_8UC1);
        EXPECT_EQ(dev, later.device());
    }
}

}} // namespace
