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

GAPI_OCV_KERNEL_ST(RenderFrameOCVImpl, cv::gapi::wip::draw::GRenderFrame, RenderOCVState)
{
    static void run(const cv::MediaFrame& in,
                    const cv::gapi::wip::draw::Prims& prims,
                    cv::MediaFrame& out,
                    RenderOCVState& state)
    {
        const auto& in_desc  = in.desc();
        const auto& out_desc = out.desc();
        auto vout = out.access(cv::MediaFrame::Access::W);
        auto vin  = in.access(cv::MediaFrame::Access::R);

        GAPI_Assert(in_desc.fmt == out_desc.fmt &&
                    "Input and output frame should have the same format");

        switch (in_desc.fmt)
        {
            case cv::MediaFormat::BGR:
            {
                cv::Mat in_bgr(in_desc.size, CV_8UC3, vin.ptr[0]);
                cv::Mat out_bgr(out_desc.size, CV_8UC3, vout.ptr[0]);

                // NB: If in and out cv::Mats are the same object
                // we can avoid copy and render on out cv::Mat
                // It's work if this kernel is last operation in the graph
                if (in_bgr.data != out_bgr.data)
                {
                    in_bgr.copyTo(out_bgr);
                }
                cv::gapi::wip::draw::drawPrimitivesOCVBGR(out_bgr, prims, state.ftpr);
                break;
            }
            case cv::MediaFormat::NV12:
            {
               cv::Mat in_y(in_desc.size       , CV_8UC1, vin.ptr[0]);
               cv::Mat out_y(out_desc.size     , CV_8UC1, vout.ptr[0]);
               cv::Mat in_uv(in_desc.size   / 2, CV_8UC2, vin.ptr[1]);
               cv::Mat out_uv(out_desc.size / 2, CV_8UC2, vout.ptr[1]);

               // NB: If in and out cv::Mats are the same object
               // we can avoid copy and render on out cv::Mat
               // It's work if this kernel is last operation in the graph
               if (in_y.data != out_y.data)
               {
                   in_y.copyTo(out_y);
               }

               if (in_uv.data != out_uv.data)
               {
                   in_uv.copyTo(out_uv);
               }

               cv::Mat yuv;
               // NV12 -> YUV
               cv::gapi::wip::draw::cvtNV12ToYUV(out_y, out_uv, yuv);
               cv::gapi::wip::draw::drawPrimitivesOCVYUV(yuv, prims, state.ftpr);
               // YUV -> NV12
               cv::gapi::wip::draw::cvtYUVToNV12(yuv, out_y, out_uv);

               break;
            }
            default:
                cv::util::throw_error(std::logic_error("Unsupported MediaFrame format for render"));
        }
    }

    static void setup(const cv::GFrameDesc& /* in */,
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
         cv::Mat yuv;
         // NV12 -> YUV
         cv::gapi::wip::draw::cvtNV12ToYUV(out_y, out_uv, yuv);
         cv::gapi::wip::draw::drawPrimitivesOCVYUV(yuv, prims, state.ftpr);
         // YUV -> NV12
         cv::gapi::wip::draw::cvtYUVToNV12(yuv, out_y, out_uv);
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

cv::gapi::GKernelPackage cv::gapi::render::ocv::kernels()
{
    const static auto pkg = cv::gapi::kernels<RenderBGROCVImpl
                                            , RenderNV12OCVImpl
                                            , RenderFrameOCVImpl>();
    return pkg;
}
