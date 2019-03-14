
#include "opencv2/core/utility.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/video.hpp"

using namespace std;
using namespace cv;

static void help()
{
    printf("Usage: dis_optflow <video_file>\n");
}

int main(int argc, char **argv)
{
    VideoCapture cap;

    if (argc < 2)
    {
        help();
        exit(1);
    }

    cap.open(argv[1]);
    if(!cap.isOpened())
    {
        printf("ERROR: Cannot open file %s\n", argv[1]);
        return -1;
    }

    Mat prevgray, gray, rgb, frame;
    Mat flow, flow_uv[2];
    Mat mag, ang;
    Mat hsv_split[3], hsv;
    char ret;

    namedWindow("flow", 1);
    namedWindow("orig", 1);

    Ptr<DenseOpticalFlow> algorithm = DISOpticalFlow::create(DISOpticalFlow::PRESET_MEDIUM);

    while(true)
    {
        cap >> frame;
        if (frame.empty())
            break;

        cvtColor(frame, gray, COLOR_BGR2GRAY);

        if (!prevgray.empty())
        {
            algorithm->calc(prevgray, gray, flow);
            split(flow, flow_uv);
            multiply(flow_uv[1], -1, flow_uv[1]);
            cartToPolar(flow_uv[0], flow_uv[1], mag, ang, true);
            normalize(mag, mag, 0, 1, NORM_MINMAX);
            hsv_split[0] = ang;
            hsv_split[1] = mag;
            hsv_split[2] = Mat::ones(ang.size(), ang.type());
            merge(hsv_split, 3, hsv);
            cvtColor(hsv, rgb, COLOR_HSV2BGR);
            imshow("flow", rgb);
            imshow("orig", frame);
        }

        if ((ret = (char)waitKey(20)) > 0)
            break;
        std::swap(prevgray, gray);
    }

    return 0;
}
