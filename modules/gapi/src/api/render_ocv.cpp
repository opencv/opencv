#include <opencv2/gapi/cpu/gcpukernel.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/gapi/render.hpp> // Kernel API's

#include "api/render_ocv.hpp"

namespace cv
{
namespace gapi
{

namespace ocv
{

GAPI_OCV_KERNEL(GOCVRenderNV12, cv::gapi::wip::draw::GRenderNV12)
{
    static void run(const cv::Mat& y, const cv::Mat& uv, const cv::gapi::wip::draw::Prims& prims,
                    cv::Mat& out_y, cv::Mat& out_uv)
    {
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
        cv::resize(uv, upsample_uv, uv.size() * 2, cv::INTER_LINEAR);
        cv::merge(std::vector<cv::Mat>{y, upsample_uv}, yuv);

        cv::gapi::wip::draw::drawPrimitivesOCVYUV(yuv, prims);

        // YUV -> NV12
        cv::Mat out_u, out_v, uv_plane;
        std::vector<cv::Mat> chs = {out_y, out_u, out_v};
        cv::split(yuv, chs);
        cv::merge(std::vector<cv::Mat>{chs[1], chs[2]}, uv_plane);
        cv::resize(uv_plane, out_uv, uv_plane.size() / 2, cv::INTER_LINEAR);
    }
};

GAPI_OCV_KERNEL(GOCVRenderBGR, cv::gapi::wip::draw::GRenderBGR)
{
    static void run(const cv::Mat&, const cv::gapi::wip::draw::Prims& prims, cv::Mat& out)
    {
        cv::gapi::wip::draw::drawPrimitivesOCVBGR(out, prims);
    }
};

cv::gapi::GKernelPackage kernels()
{
    static const auto pkg = cv::gapi::kernels<GOCVRenderNV12, GOCVRenderBGR>();
    return pkg;
}

} // namespace ocv

namespace wip
{
namespace draw
{

void mosaic(cv::Mat mat, const cv::Rect &rect, int cellSz);
void image(cv::Mat mat, cv::Point org, cv::Mat img, cv::Mat alpha);
void poly(cv::Mat mat, std::vector<cv::Point>, cv::Scalar color, int lt, int shift);

void mosaic(cv::Mat mat, const cv::Rect &rect, int cellSz)
{
    cv::Mat msc_roi = mat(rect);
    int crop_x = msc_roi.cols - msc_roi.cols % cellSz;
    int crop_y = msc_roi.rows - msc_roi.rows % cellSz;

    for(int i = 0; i < crop_y; i += cellSz )
        for(int j = 0; j < crop_x; j += cellSz) {
            auto cell_roi = msc_roi(cv::Rect(j, i, cellSz, cellSz));
            cell_roi = cv::mean(cell_roi);
        }

};

void image(cv::Mat mat, cv::Point org, cv::Mat img, cv::Mat alpha)
{
    auto roi = mat(cv::Rect(org.x, org.y, img.size().width, img.size().height));
    cv::Mat img32f_w;
    cv::merge(std::vector<cv::Mat>(3, alpha), img32f_w);

    cv::Mat roi32f_w(roi.size(), CV_32FC3, cv::Scalar::all(1.0));
    roi32f_w -= img32f_w;

    cv::Mat img32f, roi32f;
    img.convertTo(img32f, CV_32F, 1.0/255);
    roi.convertTo(roi32f, CV_32F, 1.0/255);

    cv::multiply(img32f, img32f_w, img32f);
    cv::multiply(roi32f, roi32f_w, roi32f);
    roi32f += img32f;

    roi32f.convertTo(roi, CV_8U, 255.0);
};

void poly(cv::Mat mat, std::vector<cv::Point> points, cv::Scalar color, int lt, int shift)
{
    std::vector<std::vector<cv::Point>> pp{points};
    cv::fillPoly(mat, pp, color, lt, shift);
};

struct BGR2YUVConverter
{
    cv::Scalar cvtColor(const cv::Scalar& bgr) const
    {
        double y = bgr[2] *  0.299000 + bgr[1] *  0.587000 + bgr[0] *  0.114000;
        double u = bgr[2] * -0.168736 + bgr[1] * -0.331264 + bgr[0] *  0.500000 + 128;
        double v = bgr[2] *  0.500000 + bgr[1] * -0.418688 + bgr[0] * -0.081312 + 128;

        return {y, u, v};
    }

    void cvtImg(const cv::Mat& in, cv::Mat& out) { cv::cvtColor(in, out, cv::COLOR_BGR2YUV); };
};

struct EmptyConverter
{
    cv::Scalar cvtColor(const cv::Scalar& bgr)   const { return bgr; };
    void cvtImg(const cv::Mat& in, cv::Mat& out) const { out = in;   };
};

// FIXME util::visitor ?
template <typename ColorConverter>
void drawPrimitivesOCV(cv::Mat &in, const Prims &prims)
{
    ColorConverter converter;
    for (const auto &p : prims)
    {
        switch (p.index())
        {
            case Prim::index_of<Rect>():
            {
                const auto& t_p = cv::util::get<Rect>(p);
                const auto color = converter.cvtColor(t_p.color);
                cv::rectangle(in, t_p.rect, color , t_p.thick, t_p.lt, t_p.shift);
                break;
            }

            case Prim::index_of<Text>():
            {
                const auto& t_p = cv::util::get<Text>(p);
                const auto color = converter.cvtColor(t_p.color);
                cv::putText(in, t_p.text, t_p.org, t_p.ff, t_p.fs,
                            color, t_p.thick, t_p.lt, t_p.bottom_left_origin);
                break;
            }

            case Prim::index_of<Circle>():
            {
                const auto& c_p = cv::util::get<Circle>(p);
                const auto color = converter.cvtColor(c_p.color);
                cv::circle(in, c_p.center, c_p.radius, color, c_p.thick, c_p.lt, c_p.shift);
                break;
            }

            case Prim::index_of<Line>():
            {
                const auto& l_p = cv::util::get<Line>(p);
                const auto color = converter.cvtColor(l_p.color);
                cv::line(in, l_p.pt1, l_p.pt2, color, l_p.thick, l_p.lt, l_p.shift);
                break;
            }

            case Prim::index_of<Mosaic>():
            {
                const auto& l_p = cv::util::get<Mosaic>(p);
                mosaic(in, l_p.mos, l_p.cellSz);
                break;
            }

            case Prim::index_of<Image>():
            {
                const auto& i_p = cv::util::get<Image>(p);

                cv::Mat img;
                converter.cvtImg(i_p.img, img);

                image(in, i_p.org, img, i_p.alpha);
                break;
            }

            case Prim::index_of<Poly>():
            {
                const auto& p_p = cv::util::get<Poly>(p);
                const auto color = converter.cvtColor(p_p.color);
                poly(in, p_p.points, color, p_p.lt, p_p.shift);
                break;
            }

            default: cv::util::throw_error(std::logic_error("Unsupported draw operation"));
        }
    }
}

void drawPrimitivesOCVBGR(cv::Mat &in, const Prims &prims)
{
    drawPrimitivesOCV<EmptyConverter>(in, prims);
}

void drawPrimitivesOCVYUV(cv::Mat &in, const Prims &prims)
{
    drawPrimitivesOCV<BGR2YUVConverter>(in, prims);
}

} // namespace draw
} // namespace wip
} // namespace gapi
} // namespace cv
