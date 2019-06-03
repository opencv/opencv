#include <opencv2/imgproc.hpp>
#include "opencv2/gapi/render.hpp"

// FXIME util::visitor ?
void cv::gapi::render(cv::Mat& bgr, const std::vector<cv::gapi::DrawEvent>& events)
{
    for (const auto& e : events)
    {
        switch (e.index())
        {
            case cv::gapi::DrawEvent::index_of<cv::gapi::RectEvent>():
            {
                auto r_e = cv::util::get<cv::gapi::RectEvent>(e);
                cv::Scalar color{r_e.color[0], r_e.color[1], r_e.color[2]};
                cv::rectangle(bgr, cv::Rect{r_e.x, r_e.y, r_e.widht, r_e.height},
                              color , r_e.thick, r_e.lt, r_e.shift);
                break;
            }

            case cv::gapi::DrawEvent::index_of<cv::gapi::TextEvent>():
            {
                auto t_e = cv::util::get<cv::gapi::TextEvent>(e);
                cv::Scalar color{t_e.color[0], t_e.color[1], t_e.color[2]};
                cv::putText(bgr, t_e.text, cv::Point{t_e.x, t_e.y}, t_e.ff, t_e.fs,
                            color, t_e.thick, t_e.bottom_left_origin);
                break;
            }

            default: util::throw_error(std::logic_error("Unsupported draw event"));
        }
    }
}
