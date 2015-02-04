#include <iostream>
#include <vector>

#include "opencv2/core.hpp"
#include <opencv2/core/utility.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/video.hpp"
#include "opencv2/cudaoptflow.hpp"
#include "opencv2/cudaimgproc.hpp"

using namespace std;
using namespace cv;
using namespace cv::cuda;

static void download(const GpuMat& d_mat, vector<Point2f>& vec)
{
    vec.resize(d_mat.cols);
    Mat mat(1, d_mat.cols, CV_32FC2, (void*)&vec[0]);
    d_mat.download(mat);
}

static void download(const GpuMat& d_mat, vector<uchar>& vec)
{
    vec.resize(d_mat.cols);
    Mat mat(1, d_mat.cols, CV_8UC1, (void*)&vec[0]);
    d_mat.download(mat);
}

static void drawArrows(Mat& frame, const vector<Point2f>& prevPts, const vector<Point2f>& nextPts, const vector<uchar>& status, Scalar line_color = Scalar(0, 0, 255))
{
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

template <typename T> inline T clamp (T x, T a, T b)
{
    return ((x) > (a) ? ((x) < (b) ? (x) : (b)) : (a));
}

template <typename T> inline T mapValue(T x, T a, T b, T c, T d)
{
    x = clamp(x, a, b);
    return c + (d - c) * (x - a) / (b - a);
}

int main(int argc, const char* argv[])
{
    const char* keys =
        "{ h             help   |       | print help message }"
        "{ l             left   | ../data/pic1.png       | specify left image }"
        "{ r             right  | ../data/pic2.png       | specify right image }"
        "{ gray                 |       | use grayscale sources [PyrLK Sparse] }"
        "{ win_size             | 21    | specify windows size [PyrLK] }"
        "{ max_level            | 3     | specify max level [PyrLK] }"
        "{ iters                | 30    | specify iterations count [PyrLK] }"
        "{ points               | 4000  | specify points count [GoodFeatureToTrack] }"
        "{ min_dist             | 0     | specify minimal distance between points [GoodFeatureToTrack] }";

    CommandLineParser cmd(argc, argv, keys);

    if (cmd.has("help") || !cmd.check())
    {
        cmd.printMessage();
        cmd.printErrors();
        return 0;
    }

    string fname0 = cmd.get<string>("left");
    string fname1 = cmd.get<string>("right");

    if (fname0.empty() || fname1.empty())
    {
        cerr << "Missing input file names" << endl;
        return -1;
    }

    bool useGray = cmd.has("gray");
    int winSize = cmd.get<int>("win_size");
    int maxLevel = cmd.get<int>("max_level");
    int iters = cmd.get<int>("iters");
    int points = cmd.get<int>("points");
    double minDist = cmd.get<double>("min_dist");

    Mat frame0 = imread(fname0);
    Mat frame1 = imread(fname1);

    if (frame0.empty() || frame1.empty())
    {
        cout << "Can't load input images" << endl;
        return -1;
    }

    namedWindow("PyrLK [Sparse]", WINDOW_NORMAL);
    namedWindow("PyrLK [Dense] Flow Field", WINDOW_NORMAL);

    cout << "Image size : " << frame0.cols << " x " << frame0.rows << endl;
    cout << "Points count : " << points << endl;

    cout << endl;

    Mat frame0Gray;
    cv::cvtColor(frame0, frame0Gray, COLOR_BGR2GRAY);
    Mat frame1Gray;
    cv::cvtColor(frame1, frame1Gray, COLOR_BGR2GRAY);

    // goodFeaturesToTrack

    GpuMat d_frame0Gray(frame0Gray);
    GpuMat d_prevPts;

    Ptr<cuda::CornersDetector> detector = cuda::createGoodFeaturesToTrackDetector(d_frame0Gray.type(), points, 0.01, minDist);

    detector->detect(d_frame0Gray, d_prevPts);

    // Sparse

    Ptr<cuda::SparsePyrLKOpticalFlow> d_pyrLK = cuda::SparsePyrLKOpticalFlow::create(
                Size(winSize, winSize), maxLevel, iters);

    GpuMat d_frame0(frame0);
    GpuMat d_frame1(frame1);
    GpuMat d_frame1Gray(frame1Gray);
    GpuMat d_nextPts;
    GpuMat d_status;

    d_pyrLK->calc(useGray ? d_frame0Gray : d_frame0, useGray ? d_frame1Gray : d_frame1, d_prevPts, d_nextPts, d_status);

    // Draw arrows

    vector<Point2f> prevPts(d_prevPts.cols);
    download(d_prevPts, prevPts);

    vector<Point2f> nextPts(d_nextPts.cols);
    download(d_nextPts, nextPts);

    vector<uchar> status(d_status.cols);
    download(d_status, status);

    drawArrows(frame0, prevPts, nextPts, status, Scalar(255, 0, 0));

    imshow("PyrLK [Sparse]", frame0);

    waitKey();

    return 0;
}
