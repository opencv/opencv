#include <stdexcept>

#include <opencv2/imgproc.hpp>
#include <opencv2/gapi/render.hpp>
#include <opencv2/gapi/own/assert.hpp>

#include "api/render_priv.hpp"

#include "api/ocv_mask_creator.hpp"
#include "api/freetype_mask_creator.hpp"

using namespace cv::gapi::wip::draw;

void cv::gapi::wip::draw::render(cv::Mat &bgr, const Prims &prims, const cv::gapi::GKernelPackage& pkg)
{
    cv::GMat in;
    cv::GArray<Prim> arr;

    cv::GComputation comp(cv::GIn(in, arr),
                          cv::GOut(GRenderBGR::on(in, arr)));
    comp.apply(cv::gin(bgr, prims), cv::gout(bgr), cv::compile_args(pkg));
}

void cv::gapi::wip::draw::render(cv::Mat &y_plane,
                                 cv::Mat &uv_plane,
                                 const Prims &prims,
                                 const cv::gapi::GKernelPackage& pkg)
{
    cv::GMat y_in, uv_in, y_out, uv_out;
    cv::GArray<Prim> arr;
    std::tie(y_out, uv_out) = GRenderNV12::on(y_in, uv_in, arr);

    cv::GComputation comp(cv::GIn(y_in, uv_in, arr), cv::GOut(y_out, uv_out));
    comp.apply(cv::gin(y_plane, uv_plane, prims),
               cv::gout(y_plane, uv_plane),
               cv::compile_args(pkg));
}

void cv::gapi::wip::draw::BGR2NV12(const cv::Mat &bgr, cv::Mat &y_plane, cv::Mat &uv_plane)
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

std::unique_ptr<IBitmaskCreator> cv::gapi::wip::draw::IBitmaskCreator::create(BackendT type)
{
    switch(type)
    {
        case BackendT::OpenCV:
            return std::unique_ptr<IBitmaskCreator>(new OCVBitmaskCreator());
        case BackendT::FreeType:
#ifdef HAVE_FREETYPE
            return std::unique_ptr<IBitmaskCreator>(new FreeTypeBitmaskCreator());
#else
            throw std::logic_error("FreeType library not found");
#endif
        default: throw std::logic_error("Invalid backend type");
    }
}
