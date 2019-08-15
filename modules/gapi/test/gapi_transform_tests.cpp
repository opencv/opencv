// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2019 Intel Corporation

#include <tuple>

#include "test_precomp.hpp"
#include "opencv2/gapi/gtransform.hpp"
#include "opencv2/gapi/gtype_traits.hpp"
// explicit include to use GComputation::Priv
#include "api/gcomputation_priv.hpp"

namespace opencv_test
{

namespace
{
using GMat = cv::GMat;
using GMat2 = std::tuple<GMat, GMat>;
using GMat3 = std::tuple<GMat, GMat, GMat>;
using GScalar = cv::GScalar;
template <typename T> using GArray = cv::GArray<T>;

using ArrayT = int;
using WrongArrayT = char;

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

GAPI_TRANSFORM(garr_in_gmat_out, <GMat(GArray<ArrayT>)>, "garr_in_gmat_out")
{
    static GMat pattern(GArray<ArrayT>) { return {}; }
    static GMat substitute(GArray<ArrayT>) { return {}; }
};

GAPI_TRANSFORM(gmat_in_garr_out, <GArray<ArrayT>(GMat)>, "gmat_in_garr_out")
{
    static GArray<ArrayT> pattern(GMat) { return {}; }
    static GArray<ArrayT> substitute(GMat) { return {}; }
};

GAPI_TRANSFORM(garr_in_gscalar_out, <GScalar(GArray<ArrayT>)>, "garr_in_gscalar_out")
{
    static GScalar pattern(GArray<ArrayT>) { return {}; }
    static GScalar substitute(GArray<ArrayT>) { return {}; }
};

GAPI_TRANSFORM(gscalar_in_garr_out, <GArray<ArrayT>(GScalar)>, "gscalar_in_garr_out")
{
    static GArray<ArrayT> pattern(GScalar) { return {}; }
    static GArray<ArrayT> substitute(GScalar) { return {}; }
};

GAPI_TRANSFORM(gmat_gsc_garray_in_gmat2_out, <GMat2(GMat, GScalar, GArray<ArrayT>)>, "gmat_gsc_garray_in_gmat2_out")
{
    static GMat2 pattern(GMat, GScalar, GArray<ArrayT>) { return {}; }
    static GMat2 substitute(GMat, GScalar, GArray<ArrayT>) { return {}; }
};

} // anonymous namespace

TEST(KernelPackageTransform, CreatePackage)
{
    auto pkg = cv::gapi::kernels
        < gmat_in_gmat_out
        , gmat2_in_gmat_out
        , gmat2_in_gmat3_out
        , gmatp_in_gmatp_out
        , gsc_in_gmat_out
        , gmat_in_gsc_out
        , garr_in_gmat_out
        , gmat_in_garr_out
        , garr_in_gscalar_out
        , gscalar_in_garr_out
        , gmat_gsc_garray_in_gmat2_out
        >();

    auto tr = pkg.get_transformations();
    EXPECT_EQ(11u, tr.size());
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

namespace
{
    template <typename T>
    inline bool ProtoContainsT(const cv::GProtoArg &arg) {
        return cv::GProtoArg::index_of<T>() == arg.index();
    }
} // anonymous namespace

TEST(KernelPackageTransform, gmat_gsc_garray_in_gmat2_out)
{
    auto tr = gmat_gsc_garray_in_gmat2_out::transformation();

    auto check = [](const cv::GComputation &comp){
        const auto &p = comp.priv();
        EXPECT_EQ(3u, p.m_ins.size());
        EXPECT_EQ(2u, p.m_outs.size());

        EXPECT_TRUE(ProtoContainsT<GMat>(p.m_ins[0]));
        EXPECT_TRUE(ProtoContainsT<GScalar>(p.m_ins[1]));
        EXPECT_TRUE(ProtoContainsT<cv::detail::GArrayU>(p.m_ins[2]));
        EXPECT_TRUE(cv::util::get<cv::detail::GArrayU>(p.m_ins[2]).holds<ArrayT>());
        EXPECT_FALSE(cv::util::get<cv::detail::GArrayU>(p.m_ins[2]).holds<WrongArrayT>());

        EXPECT_TRUE(ProtoContainsT<GMat>(p.m_outs[0]));
        EXPECT_TRUE(ProtoContainsT<GMat>(p.m_outs[1]));
    };

    check(tr.pattern());
    check(tr.substitute());
}

namespace
{
    template<typename ArgT>
    typename std::enable_if<(cv::detail::GTypeTraits<ArgT>::kind == cv::detail::ArgKind::GARRAY), void>::type
    arg_check(const cv::GProtoArg &arg)
    {
        EXPECT_TRUE(ProtoContainsT<cv::detail::GArrayU>(arg));
        EXPECT_TRUE(cv::util::get<cv::detail::GArrayU>(arg).holds<ArrayT>());
        EXPECT_FALSE(cv::util::get<cv::detail::GArrayU>(arg).holds<WrongArrayT>());
    }

    template<typename ArgT>
    typename std::enable_if<(cv::detail::GTypeTraits<ArgT>::kind != cv::detail::ArgKind::GARRAY), void>::type
    arg_check(const cv::GProtoArg &arg)
    {
        EXPECT_TRUE(ProtoContainsT<ArgT>(arg));
    }

    template<typename InType, typename OutType>
    void args_check(const cv::GComputation &comp)
    {
        const auto &p = comp.priv();
        EXPECT_EQ(1u, p.m_ins.size());
        EXPECT_EQ(1u, p.m_outs.size());
        arg_check<InType>(p.m_ins[0]);
        arg_check<OutType>(p.m_outs[0]);
    }
} // anonymous namespace

template <typename Transformation, typename InType, typename OutType>
static void transformTest()
{
    auto tr = Transformation::transformation();

    args_check<InType, OutType>(tr.pattern());
    args_check<InType, OutType>(tr.substitute());
}

TEST(KernelPackageTransform, gmat_in_gmat_out)
{
    transformTest<gmat_in_gmat_out, GMat, GMat>();
}

TEST(KernelPackageTransform, gmatp_in_gmatp_out)
{
    transformTest<gmatp_in_gmatp_out, GMatP, GMatP>();
}

TEST(KernelPackageTransform, gsc_in_gmat_out)
{
    transformTest<gsc_in_gmat_out, GScalar, GMat>();
}

TEST(KernelPackageTransform, gmat_in_gsc_out)
{
    transformTest<gmat_in_gsc_out, GMat, GScalar>();
}

TEST(KernelPackageTransform, garr_in_gmat_out)
{
    transformTest<garr_in_gmat_out, GArray<ArrayT>, GMat>();
}

TEST(KernelPackageTransform, gmat_in_garr_out)
{
    transformTest<gmat_in_garr_out, GMat, GArray<ArrayT>>();
}

TEST(KernelPackageTransform, garr_in_gscalar_out)
{
    transformTest<garr_in_gscalar_out, GArray<ArrayT>, GScalar>();
}

TEST(KernelPackageTransform, gscalar_in_garr_out)
{
    transformTest<gscalar_in_garr_out, GScalar, GArray<ArrayT>>();
}

} // namespace opencv_test
