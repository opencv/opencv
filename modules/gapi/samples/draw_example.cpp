#include <opencv2/imgproc.hpp> // cv::FONT*, cv::LINE*, cv::FILLED
#include <opencv2/highgui.hpp> // imwrite

#include <opencv2/gapi.hpp>
#include <opencv2/gapi/render.hpp>

int main(int argc, char *argv[])
{
    if (argc < 2) {
        std::cerr << "Filename requried" << std::endl;
        return 1;
    }

    const auto font  = cv::FONT_HERSHEY_DUPLEX;
    const auto blue  = cv::Scalar{ 255,   0,   0}; // B/G/R
    const auto green = cv::Scalar{   0, 255,   0};
    const auto coral = cv::Scalar{0x81,0x81,0xF1};
    const auto white = cv::Scalar{ 255, 255, 255};
    cv::Mat test(cv::Size(480, 160), CV_8UC3, white);

    namespace draw = cv::gapi::wip::draw;
    std::vector<draw::Prim> prims;
    prims.emplace_back(draw::Circle{   // CIRCLE primitive
            {400,72},                  // Position (a cv::Point)
            32,                        // Radius
            coral,                     // Color
            cv::FILLED,                // Thickness/fill type
            cv::LINE_8,                // Line type
            0                          // Shift
        });
    prims.emplace_back(draw::Text{     // TEXT primitive
            "Hello from G-API!",       // Text
            {64,96},                   // Position (a cv::Point)
            font,                      // Font
            1.0,                       // Scale (size)
            blue,                      // Color
            2,                         // Thickness
            cv::LINE_8,                // Line type
            false                      // Bottom left origin flag
        });
    prims.emplace_back(draw::Rect{     // RECTANGLE primitive
            {16,48,400,72},            // Geometry (a cv::Rect)
            green,                     // Color
            2,                         // Thickness
            cv::LINE_8,                // Line type
            0                          // Shift
        });
    prims.emplace_back(draw::Mosaic{   // MOSAIC primitive
            {320,96,128,32},           // Geometry (a cv::Rect)
            16,                        // Cell size
            0                          // Decimation
        });
    draw::render(test, prims);
    cv::imwrite(argv[1], test);
    return 0;
}
