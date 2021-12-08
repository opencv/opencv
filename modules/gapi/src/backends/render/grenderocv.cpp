#include <opencv2/imgproc.hpp>

#include "api/render_ocv.hpp"

#include <opencv2/gapi/cpu/gcpukernel.hpp>
#include <opencv2/gapi/fluid/core.hpp>

struct RenderOCVState
{
    std::shared_ptr<cv::gapi::wip::draw::FTTextRender> ftpr;
};

GAPI_OCV_KERNEL_ST(RenderBGROCVImpl, cv::gapi::wip::draw::GRenderBGR, RenderOCVState)
{
    static void run(const cv::Mat& in,
                    const cv::gapi::wip::draw::Prims& prims,
                    cv::Mat& out,
                    RenderOCVState& state)
    {
        // NB: If in and out cv::Mats are the same object
        // we can avoid copy and render on out cv::Mat
        // It's work if this kernel is last operation in the graph
        if (in.data != out.data) {
            in.copyTo(out);
        }

        cv::gapi::wip::draw::drawPrimitivesOCVBGR(out, prims, state.ftpr);
    }

    static void setup(const cv::GMatDesc& /* in */,
                      const cv::GArrayDesc& /* prims */,
                      std::shared_ptr<RenderOCVState>& state,
                      const cv::GCompileArgs& args)
    {
        using namespace cv::gapi::wip::draw;
        auto opt_freetype_font = cv::gapi::getCompileArg<freetype_font>(args);
        state = std::make_shared<RenderOCVState>();

        if (opt_freetype_font.has_value())
        {
            state->ftpr = std::make_shared<FTTextRender>(opt_freetype_font->path);
        }
    }
};

GAPI_OCV_KERNEL_ST(RenderNV12OCVImpl, cv::gapi::wip::draw::GRenderNV12, RenderOCVState)
{
    static void run(const cv::Mat& in_y,
                    const cv::Mat& in_uv,
                    const cv::gapi::wip::draw::Prims& prims,
                    cv::Mat& out_y,
                    cv::Mat& out_uv,
                    RenderOCVState& state)
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

        cv::gapi::wip::draw::drawPrimitivesOCVYUV(yuv, prims, state.ftpr);

        // YUV -> NV12
        cv::Mat out_u, out_v, uv_plane;
        std::vector<cv::Mat> chs = {out_y, out_u, out_v};
        cv::split(yuv, chs);
        cv::merge(std::vector<cv::Mat>{chs[1], chs[2]}, uv_plane);
        cv::resize(uv_plane, out_uv, uv_plane.size() / 2, cv::INTER_LINEAR);
    }

    static void setup(const cv::GMatDesc&   /* in_y  */,
                      const cv::GMatDesc&   /* in_uv */,
                      const cv::GArrayDesc& /* prims */,
                      std::shared_ptr<RenderOCVState>& state,
                      const cv::GCompileArgs& args)
    {
        using namespace cv::gapi::wip::draw;
        auto has_freetype_font = cv::gapi::getCompileArg<freetype_font>(args);
        state = std::make_shared<RenderOCVState>();

        if (has_freetype_font)
        {
            state->ftpr = std::make_shared<FTTextRender>(has_freetype_font->path);
        }
    }
};

GAPI_OCV_KERNEL_ST(RenderFrameOCVImpl, cv::gapi::wip::draw::GRenderFrame, RenderOCVState)
{
    static void run(const cv::MediaFrame & in,
                    const cv::gapi::wip::draw::Prims & prims,
                    cv::MediaFrame & out,
                    RenderOCVState & state)
    {
        GAPI_Assert(in.desc().fmt == cv::MediaFormat::NV12);

        // FIXME: consider a better approach (aka native inplace operation)
        // Non-intuitive logic with shared_ptr Priv class
        out = in;

        auto desc = out.desc();
        cv::Mat upsample_uv, yuv;
        {
            auto r_in = in.access(cv::MediaFrame::Access::R);

            auto in_y = cv::Mat(desc.size, CV_8UC1, r_in.ptr[0], r_in.stride[0]);
            auto in_uv = cv::Mat(desc.size / 2, CV_8UC2, r_in.ptr[1], r_in.stride[1]);

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
         *    so, upsample uv in two times, with bilinear interpolation
         *
         * 2) Render primitives on YUV
         *
         * 3) Convert yuv to NV12 (using bilinear interpolation)
         *
         */

            // NV12 -> YUV
            cv::resize(in_uv, upsample_uv, in_uv.size() * 2, cv::INTER_LINEAR);
            cv::merge(std::vector<cv::Mat>{in_y, upsample_uv}, yuv);
        }

        cv::gapi::wip::draw::drawPrimitivesOCVYUV(yuv, prims, state.ftpr);

        // YUV -> NV12
        {
            auto w_out = out.access(cv::MediaFrame::Access::W);

            auto out_y = cv::Mat(desc.size, CV_8UC1, w_out.ptr[0], w_out.stride[0]);
            auto out_uv = cv::Mat(desc.size / 2, CV_8UC2, w_out.ptr[1], w_out.stride[1]);

            cv::Mat out_u, out_v, uv_plane;
            std::vector<cv::Mat> chs = { out_y, out_u, out_v };
            cv::split(yuv, chs);
            cv::merge(std::vector<cv::Mat>{chs[1], chs[2]}, uv_plane);
            cv::resize(uv_plane, out_uv, uv_plane.size() / 2, cv::INTER_LINEAR);
        }
    }

    static void setup(const cv::GFrameDesc&   /* in_nv12  */,
        const cv::GArrayDesc& /* prims */,
        std::shared_ptr<RenderOCVState>&state,
        const cv::GCompileArgs & args)
    {
        using namespace cv::gapi::wip::draw;
        auto has_freetype_font = cv::gapi::getCompileArg<freetype_font>(args);
        state = std::make_shared<RenderOCVState>();

        if (has_freetype_font)
        {
            state->ftpr = std::make_shared<FTTextRender>(has_freetype_font->path);
        }
    }
};


cv::gapi::GKernelPackage cv::gapi::render::ocv::kernels()
{
    const static auto pkg = cv::gapi::kernels<RenderBGROCVImpl, RenderNV12OCVImpl, RenderFrameOCVImpl>();
    return pkg;
}
