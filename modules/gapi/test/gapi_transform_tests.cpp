// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2019 Intel Corporation

#include <tuple>

#include "test_precomp.hpp"
#include "opencv2/gapi/gtransform.hpp"

namespace opencv_test
{

namespace
{
using GMat2 = std::tuple<GMat, GMat>;
using GMat3 = std::tuple<GMat, GMat, GMat>;
using GMat = cv::GMat;

GAPI_TRANSFORM(my_transform, <GMat(GMat, GMat)>, "does nothing")
{
    static GMat pattern(GMat, GMat)
    {
        return {};
    };

    static GMat substitute(GMat, GMat)
    {
        return {};
    }
};

GAPI_TRANSFORM(another_transform, <GMat3(GMat, GMat)>, "does nothing")
{
    static GMat3 pattern(GMat, GMat)
    {
        return {};
    };

    static GMat3 substitute(GMat, GMat)
    {
        return {};
    }
};

GAPI_TRANSFORM(copy_transform, <GMat3(GMat, GMat)>, "does nothing")
{
    static GMat3 pattern(GMat, GMat)
    {
        return {};
    };

    static GMat3 substitute(GMat, GMat)
    {
        return {};
    }
};
} // namespace

TEST(KernelPackageTransform, SingleOutInclude)
{
    cv::gapi::GKernelPackage pkg;
    pkg.include<my_transform>();
    EXPECT_EQ(1u, pkg.size());
}

TEST(KernelPackageTransform, MultiOutInclude)
{
    cv::gapi::GKernelPackage pkg;
    pkg.include<my_transform>();
    pkg.include<another_transform>();
    EXPECT_EQ(2u, pkg.size());
}

TEST(KernelPackageTransform, MultiOutConstructor)
{
    cv::gapi::GKernelPackage pkg = cv::gapi::kernels<my_transform,
                                                     another_transform>();
    EXPECT_EQ(2u, pkg.size());
}

TEST(KernelPackageTransform, CopyClass)
{
    cv::gapi::GKernelPackage pkg = cv::gapi::kernels<copy_transform,
                                                     another_transform>();
    EXPECT_EQ(2u, pkg.size());
}

TEST(KernelPackageTransform, Combine)
{
    cv::gapi::GKernelPackage pkg1 = cv::gapi::kernels<my_transform>();
    cv::gapi::GKernelPackage pkg2 = cv::gapi::kernels<another_transform>();
    cv::gapi::GKernelPackage pkg_comb =
        cv::gapi::combine(pkg1, pkg2);

    EXPECT_EQ(2u, pkg_comb.size());
}

TEST(KernelPackageTransform, GArgsSize)
{
    auto tr = copy_transform::transformation();
    GMat a, b;
    auto subst = tr.substitute({cv::GArg(a), cv::GArg(b)});

    // return type of 'copy_transform' is GMat3
    EXPECT_EQ(3u, subst.size());
}

} // namespace opencv_test
