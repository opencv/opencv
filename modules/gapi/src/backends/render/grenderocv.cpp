#include <opencv2/imgproc.hpp>

#include "api/render_ocv.hpp"
#include "backends/render/grenderocv.hpp"

#include <opencv2/gapi/cpu/gcpukernel.hpp>

GAPI_RENDER_OCV_KERNEL(RenderBGROCVImpl, cv::gapi::wip::draw::GRenderBGR)
{
    static void run(const cv::Mat& in,
                    const cv::gapi::wip::draw::Prims& prims,
                    cv::gapi::wip::draw::FTTextRender* ftpr,
                    cv::Mat& out)
    {
        // NB: If in and out cv::Mats are the same object
        // we can avoid copy and render on out cv::Mat
        // It's work if this kernel is last operation in the graph
        if (in.data != out.data) {
            in.copyTo(out);
        }

        cv::gapi::wip::draw::drawPrimitivesOCVBGR(out, prims, ftpr);
    }
};

GAPI_RENDER_OCV_KERNEL(RenderNV12OCVImpl, cv::gapi::wip::draw::GRenderNV12)
{
    static void run(const cv::Mat& in_y,
                    const cv::Mat& in_uv,
                    const cv::gapi::wip::draw::Prims& prims,
                    cv::gapi::wip::draw::FTTextRender* ftpr,
                    cv::Mat& out_y,
                    cv::Mat& out_uv)
    {
        // NB: If in and out cv::Mats are the same object
        // we can avoid copy and render on out cv::Mat
        // It's work if this kernel is last operation in the graph
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

        cv::gapi::wip::draw::drawPrimitivesOCVYUV(yuv, prims, ftpr);

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
    const static auto pkg = cv::gapi::kernels<RenderBGROCVImpl, RenderNV12OCVImpl>();
    return pkg;
}
