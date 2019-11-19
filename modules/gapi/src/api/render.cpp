#include <opencv2/imgproc.hpp>
#include <opencv2/gapi/render.hpp>
#include <opencv2/gapi/own/assert.hpp>

#include "api/render_priv.hpp"

void cv::gapi::wip::draw::render(cv::Mat &bgr,
                                 const cv::gapi::wip::draw::Prims &prims,
                                 const cv::gapi::GKernelPackage& pkg)
{
    cv::GMat in;
    cv::GArray<Prim> arr;

    cv::GComputation comp(cv::GIn(in, arr),
                          cv::GOut(cv::gapi::wip::draw::GRenderBGR::on(in, arr)));
    comp.apply(cv::gin(bgr, prims), cv::gout(bgr), cv::compile_args(pkg));
}

void cv::gapi::wip::draw::render(cv::Mat &y_plane,
                                 cv::Mat &uv_plane,
                                 const Prims &prims,
                                 const GKernelPackage& pkg)
{
    cv::GMat y_in, uv_in, y_out, uv_out;
    cv::GArray<Prim> arr;
    std::tie(y_out, uv_out) = cv::gapi::wip::draw::GRenderNV12::on(y_in, uv_in, arr);

    cv::GComputation comp(cv::GIn(y_in, uv_in, arr), cv::GOut(y_out, uv_out));
    comp.apply(cv::gin(y_plane, uv_plane, prims),
               cv::gout(y_plane, uv_plane),
               cv::compile_args(pkg));
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
