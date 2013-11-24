#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;

int main(int, char* [])
{
    VideoCapture video(0);
    Mat frame, curr, prev, curr64f, prev64f, hann;
    int key = 0;

    do
    {
        video >> frame;
        cvtColor(frame, curr, COLOR_RGB2GRAY);

        if(prev.empty())
        {
            prev = curr.clone();
            createHanningWindow(hann, curr.size(), CV_64F);
        }

        prev.convertTo(prev64f, CV_64F);
        curr.convertTo(curr64f, CV_64F);

        Point2d shift = phaseCorrelate(prev64f, curr64f, hann);
        double radius = std::sqrt(shift.x*shift.x + shift.y*shift.y);

        if(radius > 5)
        {
            // draw a circle and line indicating the shift direction...
            Point center(curr.cols >> 1, curr.rows >> 1);
            circle(frame, center, (int)radius, Scalar(0, 255, 0), 3, LINE_AA);
            line(frame, center, Point(center.x + (int)shift.x, center.y + (int)shift.y), Scalar(0, 255, 0), 3, LINE_AA);
        }

        imshow("phase shift", frame);
        key = waitKey(2);

        prev = curr.clone();
    } while((char)key != 27); // Esc to exit...

    return 0;
}
