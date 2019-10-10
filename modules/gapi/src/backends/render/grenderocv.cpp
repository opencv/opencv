#include <opencv2/imgproc.hpp>

#include "api/render_ocv.hpp"
#include "backends/render/grenderkernel.hpp"
#include "backends/render/grenderocv.hpp"

GAPI_RENDER_OCV_KERNEL(RenderBGRImpl, cv::gapi::wip::draw::GRenderBGR)
{
    static void run(cv::gapi::render::GRenderContext& ctx)
    {
        const auto& in    = to_ocv(ctx.inMat(0));
        const auto& prims = ctx.inPrims(1);
        auto        out   = to_ocv(ctx.outMatR(0));

        // FIXME Add description for this
        if (in.data != out.data) {
            in.copyTo(out);
        }

        cv::gapi::wip::draw::drawPrimitivesOCVBGR(out, prims);
    }
};

GAPI_RENDER_OCV_KERNEL(RenderNV12Impl, cv::gapi::wip::draw::GRenderNV12)
{
    static void run(cv::gapi::render::GRenderContext& ctx)
    {
        const auto& in_y   = to_ocv(ctx.inMat(0));
        const auto& in_uv  = to_ocv(ctx.inMat(1));
        const auto& prims  = ctx.inPrims(2);
        auto        out_y  = to_ocv(ctx.outMatR(0));
        auto        out_uv = to_ocv(ctx.outMatR(1));

        // FIXME Add description for this
        if (in_y.data != out_y.data) {
            in_y.copyTo(out_y);
        }

        if (in_uv.data != out_uv.data) {
            in_uv.copyTo(out_uv);
        }

        /* FIXME How to render correctly on NV12 format ?
         *
         * Rendering on NV12 via OpenCV looks like this:
         *
         * y --------> 1)(NV12 -> YUV) -> yuv -> 2)draw -> yuv -> 3)split -------> out_y
         *                  ^                                         |
         *                  |                                         |
         * uv --------------                                          `----------> out_uv
         *
         *
         * 1) Collect yuv mat from two planes, uv plain in two times less than y plane
         *    so, upsample uv in tow times, with bilinear interpolation
         *
         * 2) Render primitives on YUV
         *
         * 3) Convert yuv to NV12 (using bilinear interpolation)
         *
         */

        // NV12 -> YUV
        cv::Mat upsample_uv, yuv;
        cv::resize(in_uv, upsample_uv, in_uv.size() * 2, cv::INTER_LINEAR);
        cv::merge(std::vector<cv::Mat>{in_y, upsample_uv}, yuv);

        cv::gapi::wip::draw::drawPrimitivesOCVYUV(yuv, prims);

        // YUV -> NV12
        cv::Mat out_u, out_v, uv_plane;
        std::vector<cv::Mat> chs = {out_y, out_u, out_v};
        cv::split(yuv, chs);
        cv::merge(std::vector<cv::Mat>{chs[1], chs[2]}, uv_plane);
        cv::resize(uv_plane, out_uv, uv_plane.size() / 2, cv::INTER_LINEAR);
    }
};

cv::gapi::GKernelPackage cv::gapi::render::ocv::kernels()
{
    const static auto pkg = cv::gapi::kernels<RenderBGRImpl, RenderNV12Impl>();
    return pkg;
}
