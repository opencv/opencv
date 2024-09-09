#include "opencv2/core/utility.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/video.hpp"

using namespace std;
using namespace cv;

int main(int argc, char **argv)
{
    CommandLineParser parser(argc, argv, "{ @video  | vtest.avi  | use video as input }");
    string filename = samples::findFileOrKeep(parser.get<string>("@video"));

    VideoCapture cap;
    cap.open(filename);

    if (!cap.isOpened())
    {
        printf("ERROR: Cannot open file %s\n", filename.c_str());
        parser.printMessage();
        return -1;
    }

    Mat prevgray, gray, rgb, frame;
    Mat flow, flow_uv[2];
    Mat mag, ang;
    Mat hsv_split[3], hsv;

    Ptr<DenseOpticalFlow> algorithm = DISOpticalFlow::create(DISOpticalFlow::PRESET_MEDIUM);

    while (true)
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

            // 注释掉图像显示部分
            // imshow("flow", rgb);
            // imshow("orig", frame);
        }

        // 注释掉等待按键部分
        // if ((ret = (char)waitKey(20)) > 0)
        //     break;
        std::swap(prevgray, gray);
    }

    printf("Processing completed.\n");
    return 0;
}

