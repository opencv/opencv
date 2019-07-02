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
using GMat = cv::GMat;
using GMat2 = std::tuple<GMat, GMat>;
using GMat3 = std::tuple<GMat, GMat, GMat>;
using GScalar = cv::GScalar;
template <typename T> using GArray = cv::GArray<T>;

GAPI_TRANSFORM(gmat_in_gmat_out, <GMat(GMat)>, "gmat_in_gmat_out")
{
    static GMat pattern(GMat) { return {}; }
    static GMat substitute(GMat) { return {}; }
};

GAPI_TRANSFORM(gmat2_in_gmat_out, <GMat(GMat, GMat)>, "gmat2_in_gmat_out")
{
    static GMat pattern(GMat, GMat) { return {}; }
    static GMat substitute(GMat, GMat) { return {}; }
};

GAPI_TRANSFORM(gmat2_in_gmat3_out, <GMat3(GMat, GMat)>, "gmat2_in_gmat3_out")
{
    static GMat3 pattern(GMat, GMat) { return {}; }
    static GMat3 substitute(GMat, GMat) { return {}; }
};

GAPI_TRANSFORM(gmatp_in_gmatp_out, <GMatP(GMatP)>, "gmatp_in_gmatp_out")
{
    static GMatP pattern(GMatP) { return {}; }
    static GMatP substitute(GMatP) { return {}; }
};

GAPI_TRANSFORM(gsc_in_gmat_out, <GMat(GScalar)>, "gsc_in_gmat_out")
{
    static GMat pattern(GScalar) { return {}; }
    static GMat substitute(GScalar) { return {}; }
};

GAPI_TRANSFORM(gmat_in_gsc_out, <GScalar(GMat)>, "gmat_in_gsc_out")
{
    static GScalar pattern(GMat) { return {}; }
    static GScalar substitute(GMat) { return {}; }
};

GAPI_TRANSFORM(garr_in_gmat_out, <GMat(GArray<int>)>, "garr_in_gmat_out")
{
    static GMat pattern(GArray<int>) { return {}; }
    static GMat substitute(GArray<int>) { return {}; }
};

GAPI_TRANSFORM(gmat_in_garr_out, <GArray<int>(GMat)>, "gmat_in_garr_out")
{
    static GArray<int> pattern(GMat) { return {}; }
    static GArray<int> substitute(GMat) { return {}; }
};

} // anonymous namespace

TEST(KernelPackageTransform, CreatePackage)
{
    auto pkg = cv::gapi::kernels
        < gmat_in_gmat_out
        , gmat2_in_gmat_out
        , gmat2_in_gmat3_out
        , gsc_in_gmat_out
        , gmat_in_gsc_out
        >();

    auto tr = pkg.get_transformations();
    EXPECT_EQ(5u, tr.size());
}

TEST(KernelPackageTransform, Include)
{
    cv::gapi::GKernelPackage pkg;
    pkg.include<gmat_in_gmat_out>();
    pkg.include<gmat2_in_gmat_out>();
    pkg.include<gmat2_in_gmat3_out>();
    auto tr = pkg.get_transformations();
    EXPECT_EQ(3u, tr.size());
}

TEST(KernelPackageTransform, Combine)
{
    auto pkg1 = cv::gapi::kernels<gmat_in_gmat_out>();
    auto pkg2 = cv::gapi::kernels<gmat2_in_gmat_out>();
    auto pkg_comb = cv::gapi::combine(pkg1, pkg2);
    auto tr = pkg_comb.get_transformations();
    EXPECT_EQ(2u, tr.size());
}

TEST(KernelPackageTransform, Pattern)
{
    auto tr = gmat2_in_gmat3_out::transformation();
    GMat a, b;
    auto pattern = tr.pattern({cv::GArg(a), cv::GArg(b)});

    // return type of '2gmat_in_gmat3_out' is GMat3
    EXPECT_EQ(3u, pattern.size());
    for (const auto& p : pattern)
    {
        EXPECT_NO_THROW(p.get<GMat>());
    }
}

TEST(KernelPackageTransform, Substitute)
{
    auto tr = gmat2_in_gmat3_out::transformation();
    GMat a, b;
    auto subst = tr.substitute({cv::GArg(a), cv::GArg(b)});

    EXPECT_EQ(3u, subst.size());
    for (const auto& s : subst)
    {
        EXPECT_NO_THROW(s.get<GMat>());
    }
}

template <typename Transformation, typename InType, typename OutType>
static void transformTest()
{
    auto tr = Transformation::transformation();
    InType in;
    auto pattern = tr.pattern({cv::GArg(in)});
    auto subst = tr.substitute({cv::GArg(in)});

    EXPECT_EQ(1u, pattern.size());
    EXPECT_EQ(1u, subst.size());

    auto checkOut = [](GArg& garg) {
        EXPECT_TRUE(garg.kind == cv::detail::GTypeTraits<OutType>::kind);
        EXPECT_NO_THROW(garg.get<OutType>());
    };

    checkOut(pattern[0]);
    checkOut(subst[0]);
}

TEST(KernelPackageTransform, GMat)
{
    transformTest<gmat_in_gmat_out, GMat, GMat>();
}

TEST(KernelPackageTransform, GMatP)
{
    transformTest<gmatp_in_gmatp_out, GMatP, GMatP>();
}

TEST(KernelPackageTransform, GScalarIn)
{
    transformTest<gsc_in_gmat_out, GScalar, GMat>();
}

TEST(KernelPackageTransform, GScalarOut)
{
    transformTest<gmat_in_gsc_out, GMat, GScalar>();
}

TEST(KernelPackageTransform, DISABLED_GArrayIn)
{
    transformTest<garr_in_gmat_out, GArray<int>, GMat>();
}

TEST(KernelPackageTransform, DISABLED_GArrayOut)
{
    transformTest<gmat_in_garr_out, GMat, GArray<int>>();
}

} // namespace opencv_test
