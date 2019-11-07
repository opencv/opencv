#if 1

#include "precomp.hpp"

#include <opencv2/gapi/core.hpp>
#include <opencv2/gapi/plaidml/core.hpp>
#include <opencv2/gapi/plaidml/gplaidmlkernel.hpp>

GAPI_PLAIDML_KERNEL(GPlaidMLAdd, cv::gapi::core::GAdd)
{
    static void run(const plaidml::edsl::Tensor& src1,
                    const plaidml::edsl::Tensor& src2,
                    int /* dtype */,
                    plaidml::edsl::Tensor& dst)
    {
        dst = src1 + src2;
    };
};

GAPI_PLAIDML_KERNEL(GPlaidMLSub, cv::gapi::core::GSub)
{
    static void run(const plaidml::edsl::Tensor& src1,
                    const plaidml::edsl::Tensor& src2,
                    int /* dtype */,
                    plaidml::edsl::Tensor& dst)
    {
        dst = src1 - src2;
    };
};

GAPI_PLAIDML_KERNEL(GPlaidMLAnd, cv::gapi::core::GAnd)
{
    static void run(const plaidml::edsl::Tensor& src1,
                    const plaidml::edsl::Tensor& src2,
                    plaidml::edsl::Tensor& dst)
    {
        dst = src1 & src2;
    };
};

GAPI_PLAIDML_KERNEL(GPlaidMLXor, cv::gapi::core::GXor)
{
    static void run(const plaidml::edsl::Tensor& src1,
                    const plaidml::edsl::Tensor& src2,
                    plaidml::edsl::Tensor& dst)
    {
        dst = src1 ^ src2;
    };
};

GAPI_PLAIDML_KERNEL(GPlaidMLOr, cv::gapi::core::GOr)
{
    static void run(const plaidml::edsl::Tensor& src1,
                    const plaidml::edsl::Tensor& src2,
                    plaidml::edsl::Tensor& dst)
    {
        dst = src1 | src2;
    };
};

cv::gapi::GKernelPackage cv::gapi::core::plaidml::kernels()
{
    static auto pkg = cv::gapi::kernels<GPlaidMLAdd, GPlaidMLSub, GPlaidMLAnd, GPlaidMLXor, GPlaidMLOr>();
    return pkg;
}

#endif
