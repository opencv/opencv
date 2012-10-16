#include <vector>
#include <iostream>
#include <string>

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/gpu/gpu.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/contrib/contrib.hpp"

using namespace std;
using namespace cv;
using namespace cv::gpu;

static Mat loadImage(const string& name)
{
    Mat image = imread(name, IMREAD_GRAYSCALE);
    if (image.empty())
    {
        cerr << "Can't load image - " << name << endl;
        exit(-1);
    }
    return image;
}

int main(int argc, const char* argv[])
{
    CommandLineParser cmd(argc, argv,
        "{ i | image          | pic1.png  | input image }"
        "{ t | template       | templ.png | template image }"
        "{ s | scale          |           | estimate scale }"
        "{ r | rotation       |           | estimate rotation }"
        "{   | gpu            |           | use gpu version }"
        "{   | minDist        | 100       | minimum distance between the centers of the detected objects }"
        "{   | levels         | 360       | R-Table levels }"
        "{   | votesThreshold | 30        | the accumulator threshold for the template centers at the detection stage. The smaller it is, the more false positions may be detected }"
        "{   | angleThresh    | 10000     | angle votes treshold }"
        "{   | scaleThresh    | 1000      | scale votes treshold }"
        "{   | posThresh      | 100       | position votes threshold }"
        "{   | dp             | 2         | inverse ratio of the accumulator resolution to the image resolution }"
        "{   | minScale       | 0.5       | minimal scale to detect }"
        "{   | maxScale       | 2         | maximal scale to detect }"
        "{   | scaleStep      | 0.05      | scale step }"
        "{   | minAngle       | 0         | minimal rotation angle to detect in degrees }"
        "{   | maxAngle       | 360       | maximal rotation angle to detect in degrees }"
        "{   | angleStep      | 1         | angle step in degrees }"
        "{   | maxSize        | 1000      | maximal size of inner buffers }"
        "{ h | help           |           | print help message }"
    );

    //cmd.about("This program demonstrates arbitary object finding with the Generalized Hough transform.");

    if (cmd.get<bool>("help"))
    {
        cmd.printParams();
        return 0;
    }

    const string templName = cmd.get<string>("template");
    const string imageName = cmd.get<string>("image");
    const bool estimateScale = cmd.get<bool>("scale");
    const bool estimateRotation = cmd.get<bool>("rotation");
    const bool useGpu = cmd.get<bool>("gpu");
    const double minDist = cmd.get<double>("minDist");
    const int levels = cmd.get<int>("levels");
    const int votesThreshold = cmd.get<int>("votesThreshold");
    const int angleThresh = cmd.get<int>("angleThresh");
    const int scaleThresh = cmd.get<int>("scaleThresh");
    const int posThresh = cmd.get<int>("posThresh");
    const double dp = cmd.get<double>("dp");
    const double minScale = cmd.get<double>("minScale");
    const double maxScale = cmd.get<double>("maxScale");
    const double scaleStep = cmd.get<double>("scaleStep");
    const double minAngle = cmd.get<double>("minAngle");
    const double maxAngle = cmd.get<double>("maxAngle");
    const double angleStep = cmd.get<double>("angleStep");
    const int maxSize = cmd.get<int>("maxSize");

    Mat templ = loadImage(templName);
    Mat image = loadImage(imageName);

    int method = GHT_POSITION;
    if (estimateScale)
        method += GHT_SCALE;
    if (estimateRotation)
        method += GHT_ROTATION;

    vector<Vec4f> position;
    cv::TickMeter tm;

    if (useGpu)
    {
        GpuMat d_templ(templ);
        GpuMat d_image(image);
        GpuMat d_position;

        Ptr<GeneralizedHough_GPU> d_hough = GeneralizedHough_GPU::create(method);
        d_hough->set("minDist", minDist);
        d_hough->set("levels", levels);
        d_hough->set("dp", dp);
        d_hough->set("maxSize", maxSize);
        if (estimateScale && estimateRotation)
        {
            d_hough->set("angleThresh", angleThresh);
            d_hough->set("scaleThresh", scaleThresh);
            d_hough->set("posThresh", posThresh);
        }
        else
        {
            d_hough->set("votesThreshold", votesThreshold);
        }
        if (estimateScale)
        {
            d_hough->set("minScale", minScale);
            d_hough->set("maxScale", maxScale);
            d_hough->set("scaleStep", scaleStep);
        }
        if (estimateRotation)
        {
            d_hough->set("minAngle", minAngle);
            d_hough->set("maxAngle", maxAngle);
            d_hough->set("angleStep", angleStep);
        }

        d_hough->setTemplate(d_templ);

        tm.start();

        d_hough->detect(d_image, d_position);
        d_hough->download(d_position, position);

        tm.stop();
    }
    else
    {
        Ptr<GeneralizedHough> hough = GeneralizedHough::create(method);
        hough->set("minDist", minDist);
        hough->set("levels", levels);
        hough->set("dp", dp);
        if (estimateScale && estimateRotation)
        {
            hough->set("angleThresh", angleThresh);
            hough->set("scaleThresh", scaleThresh);
            hough->set("posThresh", posThresh);
            hough->set("maxSize", maxSize);
        }
        else
        {
            hough->set("votesThreshold", votesThreshold);
        }
        if (estimateScale)
        {
            hough->set("minScale", minScale);
            hough->set("maxScale", maxScale);
            hough->set("scaleStep", scaleStep);
        }
        if (estimateRotation)
        {
            hough->set("minAngle", minAngle);
            hough->set("maxAngle", maxAngle);
            hough->set("angleStep", angleStep);
        }

        hough->setTemplate(templ);

        tm.start();

        hough->detect(image, position);

        tm.stop();
    }

    cout << "Found : " << position.size() << " objects" << endl;
    cout << "Detection time : " << tm.getTimeMilli() << " ms" << endl;

    Mat out;
    cvtColor(image, out, COLOR_GRAY2BGR);

    for (size_t i = 0; i < position.size(); ++i)
    {
        Point2f pos(position[i][0], position[i][1]);
        float scale = position[i][2];
        float angle = position[i][3];

        RotatedRect rect;
        rect.center = pos;
        rect.size = Size2f(templ.cols * scale, templ.rows * scale);
        rect.angle = angle;

        Point2f pts[4];
        rect.points(pts);

        line(out, pts[0], pts[1], Scalar(0, 0, 255), 3);
        line(out, pts[1], pts[2], Scalar(0, 0, 255), 3);
        line(out, pts[2], pts[3], Scalar(0, 0, 255), 3);
        line(out, pts[3], pts[0], Scalar(0, 0, 255), 3);
    }

    imshow("out", out);
    waitKey();

    return 0;
}
