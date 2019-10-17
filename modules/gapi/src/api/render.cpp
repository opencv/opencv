#include <opencv2/imgproc.hpp>
#include <opencv2/gapi/render/render.hpp>
#include <opencv2/gapi/own/assert.hpp>

#include "api/render_priv.hpp"

void cv::gapi::wip::draw::render(cv::Mat &bgr,
                                 const cv::gapi::wip::draw::Prims &prims,
                                 cv::GCompileArgs&& args)
{
    cv::GMat in;
    cv::GArray<Prim> arr;

    cv::GComputation comp(cv::GIn(in, arr),
                          cv::GOut(cv::gapi::wip::draw::render3ch(in, arr)));
    comp.apply(cv::gin(bgr, prims), cv::gout(bgr), std::move(args));
}

void cv::gapi::wip::draw::render(cv::Mat &y_plane,
                                 cv::Mat &uv_plane,
                                 const Prims &prims,
                                 cv::GCompileArgs&& args)
{
    cv::GMat y_in, uv_in, y_out, uv_out;
    cv::GArray<Prim> arr;
    std::tie(y_out, uv_out) = cv::gapi::wip::draw::renderNV12(y_in, uv_in, arr);

    cv::GComputation comp(cv::GIn(y_in, uv_in, arr), cv::GOut(y_out, uv_out));
    comp.apply(cv::gin(y_plane, uv_plane, prims),
               cv::gout(y_plane, uv_plane), std::move(args));
}

void cv::gapi::wip::draw::BGR2NV12(const cv::Mat &bgr,
                                   cv::Mat &y_plane,
                                   cv::Mat &uv_plane)
{
    GAPI_Assert(bgr.size().width  % 2 == 0);
    GAPI_Assert(bgr.size().height % 2 == 0);

    cv::Mat yuv;
    cvtColor(bgr, yuv, cv::COLOR_BGR2YUV);

    std::vector<cv::Mat> chs(3);
    cv::split(yuv, chs);
    y_plane = chs[0];

    cv::merge(std::vector<cv::Mat>{chs[1], chs[2]}, uv_plane);
    cv::resize(uv_plane, uv_plane, uv_plane.size() / 2, cv::INTER_LINEAR);
}

namespace cv
{
namespace detail
{
    template<> struct CompileArgTag<cv::gapi::wip::draw::use_freetype>
    {
        static const char* tag() { return "gapi.use_freetype"; }
    };

} // namespace detail

GMat cv::gapi::wip::draw::render3ch(const GMat& src, const GArray<Prim>& prims)
{
    return cv::gapi::wip::draw::GRenderBGR::on(src, prims);
}

std::tuple<GMat, GMat> cv::gapi::wip::draw::renderNV12(const GMat& y,
                                                       const GMat& uv,
                                                       const GArray<cv::gapi::wip::draw::Prim>& prims)
{
    return cv::gapi::wip::draw::GRenderNV12::on(y, uv, prims);
}

} // namespace cv
