#include <vector>
#include <iostream>
#include <string>

#include "opencv2/core.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/cudaimgproc.hpp"
#include "opencv2/highgui.hpp"

#include "tick_meter.hpp"

using namespace std;
using namespace cv;

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
        "{ image i        | ../data/pic1.png  | input image }"
        "{ template t     | templ.png | template image }"
        "{ full           |           | estimate scale and rotation }"
        "{ gpu            |           | use gpu version }"
        "{ minDist        | 100       | minimum distance between the centers of the detected objects }"
        "{ levels         | 360       | R-Table levels }"
        "{ votesThreshold | 30        | the accumulator threshold for the template centers at the detection stage. The smaller it is, the more false positions may be detected }"
        "{ angleThresh    | 10000     | angle votes treshold }"
        "{ scaleThresh    | 1000      | scale votes treshold }"
        "{ posThresh      | 100       | position votes threshold }"
        "{ dp             | 2         | inverse ratio of the accumulator resolution to the image resolution }"
        "{ minScale       | 0.5       | minimal scale to detect }"
        "{ maxScale       | 2         | maximal scale to detect }"
        "{ scaleStep      | 0.05      | scale step }"
        "{ minAngle       | 0         | minimal rotation angle to detect in degrees }"
        "{ maxAngle       | 360       | maximal rotation angle to detect in degrees }"
        "{ angleStep      | 1         | angle step in degrees }"
        "{ maxBufSize     | 1000      | maximal size of inner buffers }"
        "{ help h ?       |           | print help message }"
    );

    cmd.about("This program demonstrates arbitary object finding with the Generalized Hough transform.");

    if (cmd.has("help"))
    {
        cmd.printMessage();
        return 0;
    }

    const string templName = cmd.get<string>("template");
    const string imageName = cmd.get<string>("image");
    const bool full = cmd.has("full");
    const bool useGpu = cmd.has("gpu");
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
    const int maxBufSize = cmd.get<int>("maxBufSize");

    if (!cmd.check())
    {
        cmd.printErrors();
        return -1;
    }

    Mat templ = loadImage(templName);
    Mat image = loadImage(imageName);

    Ptr<GeneralizedHough> alg;

    if (!full)
    {
        Ptr<GeneralizedHoughBallard> ballard = useGpu ? cuda::createGeneralizedHoughBallard() : createGeneralizedHoughBallard();

        ballard->setMinDist(minDist);
        ballard->setLevels(levels);
        ballard->setDp(dp);
        ballard->setMaxBufferSize(maxBufSize);
        ballard->setVotesThreshold(votesThreshold);

        alg = ballard;
    }
    else
    {
        Ptr<GeneralizedHoughGuil> guil = useGpu ? cuda::createGeneralizedHoughGuil() : createGeneralizedHoughGuil();

        guil->setMinDist(minDist);
        guil->setLevels(levels);
        guil->setDp(dp);
        guil->setMaxBufferSize(maxBufSize);

        guil->setMinAngle(minAngle);
        guil->setMaxAngle(maxAngle);
        guil->setAngleStep(angleStep);
        guil->setAngleThresh(angleThresh);

        guil->setMinScale(minScale);
        guil->setMaxScale(maxScale);
        guil->setScaleStep(scaleStep);
        guil->setScaleThresh(scaleThresh);

        guil->setPosThresh(posThresh);

        alg = guil;
    }

    vector<Vec4f> position;
    TickMeter tm;

    if (useGpu)
    {
        cuda::GpuMat d_templ(templ);
        cuda::GpuMat d_image(image);
        cuda::GpuMat d_position;

        alg->setTemplate(d_templ);

        tm.start();

        alg->detect(d_image, d_position);
        d_position.download(position);

        tm.stop();
    }
    else
    {
        alg->setTemplate(templ);

        tm.start();

        alg->detect(image, position);

        tm.stop();
    }

    cout << "Found : " << position.size() << " objects" << endl;
    cout << "Detection time : " << tm.getTimeMilli() << " ms" << endl;

    Mat out;
    cv::cvtColor(image, out, COLOR_GRAY2BGR);

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
