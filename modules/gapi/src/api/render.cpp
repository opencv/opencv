#include <opencv2/gapi/render.hpp>

std::unique_ptr<cv::Render> cv::RenderCreator::create(cv::RenderCreator::BackendType type)
{
    switch (type)
    {
        case BackendType::OCV:       return std::unique_ptr<Render>(new cv::OCVRender());
        default: throw std::logic_error("Incorrect backend type");
    }
    //assert(false);
}

void cv::Render::putText(const std::string& text, cv::Point org, int font_face, double font_scale,
                         cv::Scalar color, int thickness, int line_type, bool bottom_left_origin)
{
    events_.push_back(cv::DrawEvent{ cv::PutTextEvent{ text, org, font_face, font_scale, color,
                                                       thickness, line_type, bottom_left_origin } });
}

void cv::Render::rectangle(cv::Point2f p1, cv::Point2f p2, cv::Scalar color,
                           int thickness, int line_type, int shift)
{
    events_.push_back(cv::DrawEvent{ cv::RectangleEvent{ p1, p2, color, thickness, line_type, shift } });
}

/************************************** OCV Render ************************************************/

void cv::OCVRender::run(cv::Mat& bgr_plane)
{
    process(bgr_plane);
}

void cv::OCVRender::run(cv::Mat& y_plane, cv::Mat& uv_plane)
{
    cv::Mat bgr_plane;
    cv::cvtColorTwoPlane(y_plane, uv_plane, bgr_plane, cv::COLOR_YUV2BGR_NV12);
    process(bgr_plane);
}

void cv::OCVRender::process(cv::Mat& bgr_plane)
{
    for (const auto& event : events_)
    {
        switch (event.index())
        {
            case cv::DrawEvent::index_of<cv::RectangleEvent>():
            {
                auto rect_event = cv::util::get<RectangleEvent>(event);
                cv::rectangle(bgr_plane, rect_event.p1_, rect_event.p2_, rect_event.color_,
                              rect_event.thickness_, rect_event.line_type_, rect_event.shift_);
                break;
            }

            case cv::DrawEvent::index_of<cv::PutTextEvent>():
            {
                auto text_event = cv::util::get<PutTextEvent>(event);
                cv::putText(bgr_plane, text_event.text_, text_event.org_, text_event.font_face_,
                            text_event.font_scale_, text_event.color_, text_event.thickness_,
                            text_event.bottom_left_origin_);

                auto pts = text2Points(text_event.text_, text_event.org_, text_event.font_face_,
                                       text_event.font_scale_, text_event.bottom_left_origin_);

                for (int i = 0; i < pts.size(); ++i)
                {
                    std::cout << pts[i].size() << std::endl;
                    for (int j = 0; j < pts[i].size(); ++j)
                    {
                        auto p0 = pts[i][j];

                        p0.x = (p0.x + ((1<<16)>>1)) >> 16;
                        p0.y = (p0.y + ((1<<16)>>1)) >> 16;

                        cv::circle(bgr_plane, p0, 1, cv::Scalar(0, 0, 255), 1);
                    }
                }
                break;
            }

            default: util::throw_error(std::logic_error("Unsupported draw event"));
        }
    }
}
