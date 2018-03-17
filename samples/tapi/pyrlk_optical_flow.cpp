#include <iostream>
#include <vector>
#include <iomanip>

#include "opencv2/core/utility.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core/ocl.hpp"
#include "opencv2/video/video.hpp"

using namespace std;
using namespace cv;

typedef unsigned char uchar;
#define LOOP_NUM 10
int64 work_begin = 0;
int64 work_end = 0;

static void workBegin()
{
    work_begin = getTickCount();
}
static void workEnd()
{
    work_end += (getTickCount() - work_begin);
}
static double getTime()
{
    return work_end * 1000. / getTickFrequency();
}

static void drawArrows(UMat& _frame, const vector<Point2f>& prevPts, const vector<Point2f>& nextPts, const vector<uchar>& status,
                       Scalar line_color = Scalar(0, 0, 255))
{
    Mat frame = _frame.getMat(ACCESS_WRITE);
    for (size_t i = 0; i < prevPts.size(); ++i)
    {
        if (status[i])
        {
            int line_thickness = 1;

            Point p = prevPts[i];
            Point q = nextPts[i];

            double angle = atan2((double) p.y - q.y, (double) p.x - q.x);

            double hypotenuse = sqrt( (double)(p.y - q.y)*(p.y - q.y) + (double)(p.x - q.x)*(p.x - q.x) );

            if (hypotenuse < 1.0)
                continue;

            // Here we lengthen the arrow by a factor of three.
            q.x = (int) (p.x - 3 * hypotenuse * cos(angle));
            q.y = (int) (p.y - 3 * hypotenuse * sin(angle));

            // Now we draw the main line of the arrow.
            line(frame, p, q, line_color, line_thickness);

            // Now draw the tips of the arrow. I do some scaling so that the
            // tips look proportional to the main line of the arrow.

            p.x = (int) (q.x + 9 * cos(angle + CV_PI / 4));
            p.y = (int) (q.y + 9 * sin(angle + CV_PI / 4));
            line(frame, p, q, line_color, line_thickness);

            p.x = (int) (q.x + 9 * cos(angle - CV_PI / 4));
            p.y = (int) (q.y + 9 * sin(angle - CV_PI / 4));
            line(frame, p, q, line_color, line_thickness);
        }
    }
}


int main(int argc, const char* argv[])
{
    const char* keys =
        "{ h help           |                 | print help message }"
        "{ l left           |                 | specify left image }"
        "{ r right          |                 | specify right image }"
        "{ c camera         | 0               | enable camera capturing }"
        "{ v video          |                 | use video as input }"
        "{ o output         | pyrlk_output.jpg| specify output save path when input is images }"
        "{ points           | 1000            | specify points count [GoodFeatureToTrack] }"
        "{ min_dist         | 0               | specify minimal distance between points [GoodFeatureToTrack] }"
        "{ m cpu_mode       | false           | run without OpenCL }";

    CommandLineParser cmd(argc, argv, keys);

    if (cmd.has("help"))
    {
        cout << "Usage: pyrlk_optical_flow [options]" << endl;
        cout << "Available options:" << endl;
        cmd.printMessage();
        return EXIT_SUCCESS;
    }

    bool defaultPicturesFail = true;
    string fname0 = cmd.get<string>("left");
    string fname1 = cmd.get<string>("right");
    string vdofile = cmd.get<string>("video");
    string outfile = cmd.get<string>("output");
    int points = cmd.get<int>("points");
    double minDist = cmd.get<double>("min_dist");
    int inputName = cmd.get<int>("c");

    UMat frame0;
    imread(fname0, cv::IMREAD_GRAYSCALE).copyTo(frame0);
    UMat frame1;
    imread(fname1, cv::IMREAD_GRAYSCALE).copyTo(frame1);

    vector<cv::Point2f> pts(points);
    vector<cv::Point2f> nextPts(points);
    vector<unsigned char> status(points);
    vector<float> err;

    cout << "Points count : " << points << endl << endl;

    if (frame0.empty() || frame1.empty())
    {
        VideoCapture capture;
        UMat frame, frameCopy;
        UMat frame0Gray, frame1Gray;
        UMat ptr0, ptr1;

        if(vdofile.empty())
            capture.open( inputName );
        else
            capture.open(vdofile.c_str());

        int c = inputName ;
        if(!capture.isOpened())
        {
            if(vdofile.empty())
                cout << "Capture from CAM " << c << " didn't work" << endl;
            else
                cout << "Capture from file " << vdofile << " failed" <<endl;
            if (defaultPicturesFail)
                return EXIT_FAILURE;
            goto nocamera;
        }

        cout << "In capture ..." << endl;
        for(int i = 0;; i++)
        {
            if( !capture.read(frame) )
                break;

            if (i == 0)
            {
                frame.copyTo( frame0 );
                cvtColor(frame0, frame0Gray, COLOR_BGR2GRAY);
            }
            else
            {
                if (i%2 == 1)
                {
                    frame.copyTo(frame1);
                    cvtColor(frame1, frame1Gray, COLOR_BGR2GRAY);
                    ptr0 = frame0Gray;
                    ptr1 = frame1Gray;
                }
                else
                {
                    frame.copyTo(frame0);
                    cvtColor(frame0, frame0Gray, COLOR_BGR2GRAY);
                    ptr0 = frame1Gray;
                    ptr1 = frame0Gray;
                }


                pts.clear();
                goodFeaturesToTrack(ptr0, pts, points, 0.01, 0.0);
                if(pts.size() == 0)
                    continue;
                calcOpticalFlowPyrLK(ptr0, ptr1, pts, nextPts, status, err);

                if (i%2 == 1)
                    frame1.copyTo(frameCopy);
                else
                    frame0.copyTo(frameCopy);
                drawArrows(frameCopy, pts, nextPts, status, Scalar(255, 0, 0));
                imshow("PyrLK [Sparse]", frameCopy);
            }
            char key = (char)waitKey(10);

            if (key == 27)
                break;
            else if (key == 'm' || key == 'M')
            {
                ocl::setUseOpenCL(!cv::ocl::useOpenCL());
                cout << "Switched to " << (ocl::useOpenCL() ? "OpenCL" : "CPU") << " mode\n";
            }
        }
        capture.release();
    }
    else
    {
nocamera:
        if (cmd.has("cpu_mode"))
        {
            ocl::setUseOpenCL(false);
            std::cout << "OpenCL was disabled" << std::endl;
        }
        for(int i = 0; i <= LOOP_NUM; i ++)
        {
            cout << "loop" << i << endl;
            if (i > 0) workBegin();

            goodFeaturesToTrack(frame0, pts, points, 0.01, minDist);
            calcOpticalFlowPyrLK(frame0, frame1, pts, nextPts, status, err);

            if (i > 0 && i <= LOOP_NUM)
                workEnd();

            if (i == LOOP_NUM)
            {
                cout << "average time (noCamera) : ";

                cout << getTime() / LOOP_NUM << " ms" << endl;

                drawArrows(frame0, pts, nextPts, status, Scalar(255, 0, 0));
                imshow("PyrLK [Sparse]", frame0);
                imwrite(outfile, frame0);
            }
        }
    }

    waitKey();

    return EXIT_SUCCESS;
}
