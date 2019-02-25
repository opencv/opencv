#ifndef OPENCV_GAPI_RENDER_HPP
#define OPENCV_GAPI_RENDER_HPP

#include <stdexcept>
#include <memory>

#include "opencv2/gapi/util/variant.hpp"

namespace cv
{

class OCVRender;

struct PutTextEvent
{
    std::string text_;
    cv::Point org_;
    int font_face_;
    double font_scale_;
    cv::Scalar color_;
    int thickness_;
    int line_type_;
    bool bottom_left_origin_;
};

struct RectangleEvent
{
    cv::Point2f p1_;
    cv::Point2f p2_;
    cv::Scalar color_;
    int thickness_;
    int line_type_;
    int shift_;
};

using DrawEvent  = util::variant<PutTextEvent, RectangleEvent>;
using DrawEvents = std::vector<DrawEvent>;

class GAPI_EXPORTS Render
{
public:
    enum class BackendType
    {
        MOVIDIOUS,
        OCV
    };

    static std::unique_ptr<Render> create(BackendType type);

    void putText(const std::string& text, cv::Point org, int font_face, double font_scale,
                 cv::Scalar color, int thickness = 1, int line_type = cv::LINE_8,
                 bool bottom_left_origin = false);

    void rectangle(cv::Point2f p1, cv::Point2f p2, cv::Scalar color,
                   int thickness = 1, int line_type = cv::LINE_8, int shift = 0);

    virtual void run(cv::Mat&) = 0;
    virtual void run(cv::Mat&, cv::Mat&) = 0;

    virtual ~Render() = default;

protected:
    DrawEvents events_;
};

class OCVRender : public Render
{
public:
    void run(cv::Mat&) override;
    void run(cv::Mat&, cv::Mat&) override;
};

} // namespace cv

#endif // OPENCV_GAPI_RENDER_HPP
