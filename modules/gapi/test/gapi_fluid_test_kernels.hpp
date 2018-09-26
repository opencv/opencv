// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#ifndef GAPI_FLUID_TEST_KERNELS_HPP
#define GAPI_FLUID_TEST_KERNELS_HPP

#include "opencv2/gapi/fluid/gfluidkernel.hpp"

namespace cv
{
namespace gapi_test_kernels
{

G_TYPED_KERNEL(TAddSimple, <GMat(GMat, GMat)>, "test.fluid.add_simple") {
    static cv::GMatDesc outMeta(cv::GMatDesc a, cv::GMatDesc) {
        return a;
    }
};

G_TYPED_KERNEL(TAddCSimple, <GMat(GMat,int)>, "test.fluid.addc_simple")
{
    static GMatDesc outMeta(const cv::GMatDesc &in, int) {
        return in;
    }
};

G_TYPED_KERNEL(TAddScalar, <GMat(GMat,GScalar)>, "test.fluid.addc_scalar")
{
    static GMatDesc outMeta(const cv::GMatDesc &in, const cv::GScalarDesc&) {
        return in;
    }
};

G_TYPED_KERNEL(TAddScalarToMat, <GMat(GScalar,GMat)>, "test.fluid.add_scalar_to_mat")
{
    static GMatDesc outMeta(const cv::GScalarDesc&, const cv::GMatDesc &in) {
        return in;
    }
};

G_TYPED_KERNEL(TBlur1x1, <GMat(GMat,int,Scalar)>, "org.opencv.imgproc.filters.blur1x1"){
    static GMatDesc outMeta(GMatDesc in, int, Scalar) {
        return in;
    }
};

G_TYPED_KERNEL(TBlur3x3, <GMat(GMat,int,Scalar)>, "org.opencv.imgproc.filters.blur3x3"){
    static GMatDesc outMeta(GMatDesc in, int, Scalar) {
        return in;
    }
};

G_TYPED_KERNEL(TBlur5x5, <GMat(GMat,int,Scalar)>, "org.opencv.imgproc.filters.blur5x5"){
    static GMatDesc outMeta(GMatDesc in, int, Scalar) {
        return in;
    }
};

G_TYPED_KERNEL(TBlur3x3_2lpi, <GMat(GMat,int,Scalar)>, "org.opencv.imgproc.filters.blur3x3_2lpi"){
    static GMatDesc outMeta(GMatDesc in, int, Scalar) {
        return in;
    }
};

G_TYPED_KERNEL(TBlur5x5_2lpi, <GMat(GMat,int,Scalar)>, "org.opencv.imgproc.filters.blur5x5_2lpi"){
    static GMatDesc outMeta(GMatDesc in, int, Scalar) {
        return in;
    }
};

G_TYPED_KERNEL(TId, <GMat(GMat)>, "test.fluid.identity") {
    static cv::GMatDesc outMeta(cv::GMatDesc a) {
        return a;
    }
};

G_TYPED_KERNEL(TId7x7, <GMat(GMat)>, "test.fluid.identity7x7") {
    static cv::GMatDesc outMeta(cv::GMatDesc a) {
        return a;
    }
};

G_TYPED_KERNEL(TPlusRow0, <GMat(GMat)>, "test.fluid.plus_row0") {
    static cv::GMatDesc outMeta(cv::GMatDesc a) {
        return a;
    }
};

G_TYPED_KERNEL(TSum2MatsAndScalar, <GMat(GMat,GScalar,GMat)>, "test.fluid.sum_2_mats_and_scalar")
{
    static GMatDesc outMeta(const cv::GMatDesc &in, const cv::GScalarDesc&, const cv::GMatDesc&) {
        return in;
    }
};

extern cv::gapi::GKernelPackage fluidTestPackage;

} // namespace gapi_test_kernels
} // namespace cv

#endif // GAPI_FLUID_TEST_KERNELS_HPP
