#include <opencv2/imgproc.hpp>

#include "opencv2/gapi/render.hpp"
#include "opencv2/gapi/own/assert.hpp"

#include "api/render_priv.hpp"

using namespace cv::gapi::wip::draw;
// FXIME util::visitor ?
void cv::gapi::wip::draw::render(cv::Mat& bgr, const Prims& prims)
{
    for (const auto& p : prims)
    {
        switch (p.index())
        {
            case Prim::index_of<Rect>():
            {
                const auto& t_p = cv::util::get<Rect>(p);
                cv::rectangle(bgr, t_p.rect, t_p.color , t_p.thick, t_p.lt, t_p.shift);
                break;
            }

            case Prim::index_of<Text>():
            {
                const auto& t_p = cv::util::get<Text>(p);
                cv::putText(bgr, t_p.text, t_p.org, t_p.ff, t_p.fs,
                            t_p.color, t_p.thick, t_p.lt, t_p.bottom_left_origin);
                break;
            }

            case Prim::index_of<Circle>():
            {
                const auto& c_p = cv::util::get<Circle>(p);
                cv::circle(bgr, c_p.center, c_p.radius, c_p.color, c_p.thick, c_p.lt, c_p.shift);
                break;
            }

            case Prim::index_of<Line>():
            {
                const auto& l_p = cv::util::get<Line>(p);
                cv::line(bgr, l_p.pt1, l_p.pt2, l_p.color, l_p.thick, l_p.lt, l_p.shift);
                break;
            }

            default: util::throw_error(std::logic_error("Unsupported draw operation"));
        }
    }
}

void cv::gapi::wip::draw::render(cv::Mat& y_plane, cv::Mat& uv_plane , const Prims& prims)
{
    cv::Mat bgr;
    cv::cvtColorTwoPlane(y_plane, uv_plane, bgr, cv::COLOR_YUV2BGR_NV12);
    render(bgr, prims);
    BGR2NV12(bgr, y_plane, uv_plane);
}

void cv::gapi::wip::draw::splitNV12TwoPlane(const cv::Mat& yuv, cv::Mat& y_plane, cv::Mat& uv_plane) {
    y_plane.create(yuv.size(),      CV_8UC1);
    uv_plane.create(yuv.size() / 2, CV_8UC2);

    // Fill Y plane
    for (int i = 0; i < yuv.rows; ++i)
    {
        const uchar* in  = yuv.ptr<uchar>(i);
        uchar* out       = y_plane.ptr<uchar>(i);
        for (int j = 0; j < yuv.cols; j++) {
            out[j] = in[3 * j];
        }
    }

    // Fill UV plane
    for (int i = 0; i < uv_plane.rows; i++)
    {
        const uchar* in = yuv.ptr<uchar>(2 * i);
        uchar* out      = uv_plane.ptr<uchar>(i);
        for (int j = 0; j < uv_plane.cols; j++) {
            out[j * 2    ] = in[6 * j + 1];
            out[j * 2 + 1] = in[6 * j + 2];
        }
    }
}

void cv::gapi::wip::draw::BGR2NV12(const cv::Mat& bgr, cv::Mat& y_plane, cv::Mat& uv_plane)
{
    GAPI_Assert(bgr.size().width  % 2 == 0);
    GAPI_Assert(bgr.size().height % 2 == 0);

    cvtColor(bgr, bgr, cv::COLOR_BGR2YUV);
    splitNV12TwoPlane(bgr, y_plane, uv_plane);
}
