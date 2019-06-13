#include <opencv2/imgproc.hpp>
#include "opencv2/gapi/render.hpp"

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
                auto t_p = cv::util::get<Rect>(p);
                cv::rectangle(bgr, t_p.rect, t_p.color , t_p.thick, t_p.lt, t_p.shift);
                break;
            }

            case Prim::index_of<Text>():
            {
                auto t_p = cv::util::get<Text>(p);
                cv::putText(bgr, t_p.text, t_p.point, t_p.ff, t_p.fs,
                            t_p.color, t_p.thick, t_p.bottom_left_origin);
                break;
            }

            default: util::throw_error(std::logic_error("Unsupported draw event"));
        }
    }
}
