// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2019 Intel Corporation


#include "precomp.hpp"

#include <opencv2/gapi/core.hpp>

#include <opencv2/gapi/plaidml/core.hpp>

#ifdef HAVE_PLAIDML

#include <opencv2/gapi/plaidml/gplaidmlkernel.hpp>

#include <plaidml2/edsl/edsl.h>

#define GAPI_PLAIDML_LOGICAL_OP(Name, API, Op) \
GAPI_PLAIDML_KERNEL(Name, API) \
{ \
    static void run(const plaidml::edsl::Tensor& src1, \
                    const plaidml::edsl::Tensor& src2, \
                    plaidml::edsl::Tensor& dst) \
    { \
        dst = src1 Op src2; \
    }; \
}; \

#define GAPI_PLAIDML_ARITHMETIC_OP(Name, API, Op) \
GAPI_PLAIDML_KERNEL(Name, API) \
{ \
    static void run(const plaidml::edsl::Tensor& src1, \
                    const plaidml::edsl::Tensor& src2, \
                    int, /* dtype */ \
                    plaidml::edsl::Tensor& dst) \
    { \
        dst = src1 Op src2; \
    }; \
}; \

GAPI_PLAIDML_LOGICAL_OP(GPlaidMLAnd, cv::gapi::core::GAnd, &);
GAPI_PLAIDML_LOGICAL_OP(GPlaidMLXor, cv::gapi::core::GXor, ^);
GAPI_PLAIDML_LOGICAL_OP(GPlaidMLOr , cv::gapi::core::GOr , |)

GAPI_PLAIDML_ARITHMETIC_OP(GPlaidMLAdd, cv::gapi::core::GAdd, +);
GAPI_PLAIDML_ARITHMETIC_OP(GPlaidMLSub, cv::gapi::core::GSub, -);

cv::GKernelPackage cv::gapi::core::plaidml::kernels()
{
    static auto pkg = cv::gapi::kernels<GPlaidMLAdd, GPlaidMLSub, GPlaidMLAnd, GPlaidMLXor, GPlaidMLOr>();
    return pkg;
}

#else // HAVE_PLAIDML

cv::GKernelPackage cv::gapi::core::plaidml::kernels()
{
    // Still provide this symbol to avoid linking issues
    util::throw_error(std::runtime_error("G-API has been compiled without PlaidML2 support"));
}

#endif // HAVE_PLAIDML
