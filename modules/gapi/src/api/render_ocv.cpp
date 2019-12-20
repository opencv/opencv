#include <opencv2/imgproc.hpp>
#include <opencv2/gapi/render/render.hpp> // Kernel API's

#include "api/render_ocv.hpp"
#include "api/ft_render.hpp"

namespace cv
{
namespace gapi
{
namespace wip
{
namespace draw
{

// FIXME Support `decim` mosaic parameter
inline void mosaic(cv::Mat& mat, const cv::Rect &rect, int cellSz)
{
    cv::Rect mat_rect(0, 0, mat.cols, mat.rows);
    auto intersection = mat_rect & rect;

    cv::Mat msc_roi = mat(intersection);

    bool has_crop_x = false;
    bool has_crop_y = false;

    int cols = msc_roi.cols;
    int rows = msc_roi.rows;

    if (msc_roi.cols % cellSz != 0)
    {
        has_crop_x = true;
        cols -= msc_roi.cols % cellSz;
    }

    if (msc_roi.rows % cellSz != 0)
    {
        has_crop_y = true;
        rows -= msc_roi.rows % cellSz;
    }

    cv::Mat cell_roi;
    for(int i = 0; i < rows; i += cellSz )
    {
        for(int j = 0; j < cols; j += cellSz)
        {
            cell_roi = msc_roi(cv::Rect(j, i, cellSz, cellSz));
            cell_roi = cv::mean(cell_roi);
        }
        if (has_crop_x)
        {
            cell_roi = msc_roi(cv::Rect(cols, i, msc_roi.cols - cols, cellSz));
            cell_roi = cv::mean(cell_roi);
        }
    }

    if (has_crop_y)
    {
        for(int j = 0; j < cols; j += cellSz)
        {
            cell_roi = msc_roi(cv::Rect(j, rows, cellSz, msc_roi.rows - rows));
            cell_roi = cv::mean(cell_roi);
        }
        if (has_crop_x)
        {
            cell_roi = msc_roi(cv::Rect(cols, rows, msc_roi.cols - cols, msc_roi.rows - rows));
            cell_roi = cv::mean(cell_roi);
        }
    }
};

inline void blendImage(const cv::Mat& img,
                       const cv::Mat& alpha,
                       const cv::Point& org,
                       cv::Mat background)
{
    GAPI_Assert(alpha.type() == CV_32FC1);
    GAPI_Assert(background.channels() == 3u);

    cv::Mat roi = background(cv::Rect(org, img.size()));
    cv::Mat img32f_w;
    cv::merge(std::vector<cv::Mat>(3, alpha), img32f_w);

    cv::Mat roi32f_w(roi.size(), CV_32FC3, cv::Scalar::all(1.0));
    roi32f_w -= img32f_w;

    cv::Mat img32f, roi32f;
    if (img.type() == CV_32FC3) {
        img.copyTo(img32f);
    } else {
        img.convertTo(img32f, CV_32F, 1.0/255);
    }

    roi.convertTo(roi32f, CV_32F, 1.0/255);

    cv::multiply(img32f, img32f_w, img32f);
    cv::multiply(roi32f, roi32f_w, roi32f);
    roi32f += img32f;

    roi32f.convertTo(roi, CV_8U, 255.0);
}

inline void blendTextMask(cv::Mat& img,
                          cv::Mat& mask,
                          const cv::Point& tl,
                          const cv::Scalar& color)
{
    mask.convertTo(mask, CV_32FC1, 1 / 255.0);
    cv::Mat color_mask;

    cv::merge(std::vector<cv::Mat>(3, mask), color_mask);
    cv::Scalar color32f = color / 255.0;
    cv::multiply(color_mask, color32f, color_mask);

    blendImage(color_mask, mask, tl, img);
}

inline void poly(cv::Mat& mat,
                 const cv::gapi::wip::draw::Poly& pp)
{
    std::vector<std::vector<cv::Point>> points{pp.points};
    cv::fillPoly(mat, points, pp.color, pp.lt, pp.shift);
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
void drawPrimitivesOCV(cv::Mat& in,
                       const cv::gapi::wip::draw::Prims& prims,
                       cv::gapi::wip::draw::FTTextRender* ftpr)
{
#ifndef HAVE_FREETYPE
    cv::util::suppress_unused_warning(ftpr);
#endif

    using namespace cv::gapi::wip::draw;

    ColorConverter converter;
    for (const auto &p : prims)
    {
        switch (p.index())
        {
            case Prim::index_of<Rect>():
            {
                const auto& rp = cv::util::get<Rect>(p);
                const auto color = converter.cvtColor(rp.color);
                cv::rectangle(in, rp.rect, color , rp.thick);
                break;
            }

            // FIXME avoid code duplicate for Text and FText
            case Prim::index_of<Text>():
            {
                auto tp = cv::util::get<Text>(p);
                tp.color = converter.cvtColor(tp.color);

                int baseline = 0;
                auto size    = cv::getTextSize(tp.text, tp.ff, tp.fs, tp.thick, &baseline);
                baseline    += tp.thick;
                size.height += baseline;

                // Allocate mask outside
                cv::Mat mask(size, CV_8UC1, cv::Scalar::all(0));
                // Org it's bottom left position for baseline
                cv::Point org(0, mask.rows - baseline);
                cv::putText(mask, tp.text, org, tp.ff, tp.fs, 255, tp.thick);

                // Org is bottom left point, trasform it to top left point for blendImage
                cv::Point tl(tp.org.x, tp.org.y - mask.size().height + baseline);

                blendTextMask(in, mask, tl, tp.color);
                break;
            }

            case Prim::index_of<FText>():
            {
#ifdef HAVE_FREETYPE
                const auto& ftp  = cv::util::get<FText>(p);
                const auto color = converter.cvtColor(ftp.color);

                GAPI_Assert(ftpr && "You must pass cv::gapi::wip::draw::freetype_font"
                                    " to the graph compile arguments");
                int baseline = 0;
                auto size    = ftpr->getTextSize(ftp.text, ftp.fh, &baseline);

                // Allocate mask outside
                cv::Mat mask(size, CV_8UC1, cv::Scalar::all(0));
                // Org it's bottom left position for baseline
                cv::Point org(0, mask.rows - baseline);
                ftpr->putText(mask, ftp.text, org, ftp.fh);

                // Org is bottom left point, trasform it to top left point for blendImage
                cv::Point tl(ftp.org.x, ftp.org.y - mask.size().height + baseline);

                blendTextMask(in, mask, tl, color);
#else
                cv::util::throw_error(std::runtime_error("FreeType not found !"));
#endif
                break;
            }

            case Prim::index_of<Circle>():
            {
                const auto& cp = cv::util::get<Circle>(p);
                const auto color = converter.cvtColor(cp.color);
                cv::circle(in, cp.center, cp.radius, color, cp.thick);
                break;
            }

            case Prim::index_of<Line>():
            {
                const auto& lp = cv::util::get<Line>(p);
                const auto color = converter.cvtColor(lp.color);
                cv::line(in, lp.pt1, lp.pt2, color, lp.thick);
                break;
            }

            case Prim::index_of<Mosaic>():
            {
                const auto& mp = cv::util::get<Mosaic>(p);
                GAPI_Assert(mp.decim == 0 && "Only decim = 0 supported now");
                mosaic(in, mp.mos, mp.cellSz);
                break;
            }

            case Prim::index_of<Image>():
            {
                const auto& ip = cv::util::get<Image>(p);

                cv::Mat img;
                converter.cvtImg(ip.img, img);

                img.convertTo(img, CV_32FC1, 1.0 / 255);
                blendImage(img, ip.alpha, ip.org, in);
                break;
            }

            case Prim::index_of<Poly>():
            {
                auto pp = cv::util::get<Poly>(p);
                pp.color = converter.cvtColor(pp.color);
                poly(in, pp);
                break;
            }

            default: cv::util::throw_error(std::logic_error("Unsupported draw operation"));
        }
    }
}

void drawPrimitivesOCVBGR(cv::Mat &in,
                          const cv::gapi::wip::draw::Prims &prims,
                          cv::gapi::wip::draw::FTTextRender* ftpr)
{
    drawPrimitivesOCV<EmptyConverter>(in, prims, ftpr);
}

void drawPrimitivesOCVYUV(cv::Mat &in,
                          const cv::gapi::wip::draw::Prims &prims,
                          cv::gapi::wip::draw::FTTextRender* ftpr)
{
    drawPrimitivesOCV<BGR2YUVConverter>(in, prims, ftpr);
}

} // namespace draw
} // namespace wip
} // namespace gapi
} // namespace cv
