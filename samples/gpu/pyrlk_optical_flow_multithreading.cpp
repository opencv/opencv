#include <iostream>
#include <vector>
#include <sstream>

#include "opencv2/core/core.hpp"
#include "cvconfig.h"

#ifdef HAVE_TBB
#include <tbb/parallel_for_each.h>
#include <tbb/task_scheduler_init.h>

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/video/video.hpp"
#include "opencv2/gpu/gpu.hpp"

using namespace std;
using namespace cv;
using namespace cv::gpu;

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

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
struct S_Thread_data
{
    Size winSize;
    int maxLevel;
    int iters;

    Stream stream;
    Mat frame0;
    Mat frame1;
    Mat frame1Gray;

    GpuMat d_frame0Gray;
    GpuMat d_prevPts;
    bool useGray;
};

struct pyrLK_task
{
    pyrLK_task(size_t n):
        _n(n),
        _thread_data(NULL){}

    void operator()()
    {
        // Sparse

        PyrLKOpticalFlow d_pyrLK;

        d_pyrLK.winSize.width = _thread_data->winSize.width;
        d_pyrLK.winSize.height = _thread_data->winSize.height;
        d_pyrLK.maxLevel = _thread_data->maxLevel;
        d_pyrLK.iters = _thread_data->iters;

        GpuMat d_frame0(_thread_data->frame0);
        GpuMat d_frame1(_thread_data->frame1);
        GpuMat d_frame1Gray(_thread_data->frame1Gray);
        GpuMat d_nextPts;
        GpuMat d_status;

        bool useGray = _thread_data->useGray;

        d_pyrLK.sparse_multi(useGray ? _thread_data->d_frame0Gray : d_frame0,
                    useGray ? d_frame1Gray : d_frame1,
                            _thread_data->d_prevPts, d_nextPts,
                    d_status, _thread_data->stream, NULL);

        // Draw arrows

        vector<Point2f> prevPts(_thread_data->d_prevPts.cols);
        download(_thread_data->d_prevPts, prevPts);

        vector<Point2f> nextPts(d_nextPts.cols);
        download(d_nextPts, nextPts);

        vector<uchar> status(d_status.cols);
        download(d_status, status);

        drawArrows(_thread_data->frame0, prevPts, nextPts, status, Scalar(255, 0, 0));
    }

    size_t _n;
    struct S_Thread_data* _thread_data;
};

template <typename T> struct invoker {
  void operator()(T& it) const {it();}
};

#define THREADS_NB    12

int main(int argc, const char* argv[])
{
    const char* keys =
        "{ h            | help           | false | print help message }"
        "{ l            | left           |       | specify left image }"
        "{ r            | right          |       | specify right image }"
        "{ gray         | gray           | false | use grayscale sources [PyrLK Sparse] }"
        "{ win_size     | win_size       | 21    | specify windows size [PyrLK] }"
        "{ max_level    | max_level      | 3     | specify max level [PyrLK] }"
        "{ iters        | iters          | 30    | specify iterations count [PyrLK] }"
        "{ points       | points         | 4000  | specify points count [GoodFeatureToTrack] }"
        "{ min_dist     | min_dist       | 0     | specify minimal distance between points [GoodFeatureToTrack] }";

    CommandLineParser cmd(argc, argv, keys);

    if (cmd.get<bool>("help"))
    {
        cout << "Usage: pyrlk_optical_flow_multithreading [options]" << endl;
        cout << "Avaible options:" << endl;
        cmd.printParams();
        return 0;
    }

    string fname0 = cmd.get<string>("left");
    string fname1 = cmd.get<string>("right");

    if (fname0.empty() || fname1.empty())
    {
        cerr << "Missing input file names" << endl;
        return -1;
    }

    bool useGray = cmd.get<bool>("gray");
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

    cout << "Image size : " << frame0.cols << " x " << frame0.rows << endl;
    cout << "Points count : " << points << endl;

    cout << endl;

    Mat frame0Gray;
    cvtColor(frame0, frame0Gray, COLOR_BGR2GRAY);
    Mat frame1Gray;
    cvtColor(frame1, frame1Gray, COLOR_BGR2GRAY);

    // goodFeaturesToTrack

    GoodFeaturesToTrackDetector_GPU detector(points, 0.01, minDist);

    GpuMat d_frame0Gray(frame0Gray);
    GpuMat d_prevPts;

    detector(d_frame0Gray, d_prevPts);

    // Sparse

    tbb::task_scheduler_init init(THREADS_NB);

    std::vector<pyrLK_task> tasks;

    S_Thread_data s_thread_data[THREADS_NB];

    for (unsigned int uiI = 0; uiI < THREADS_NB; ++uiI)
    {
        s_thread_data[uiI].stream = Stream();
        s_thread_data[uiI].frame0 = frame0.clone();
        s_thread_data[uiI].frame1 = frame1.clone();
        s_thread_data[uiI].frame1Gray = frame0Gray.clone();

        s_thread_data[uiI].iters = iters;
        s_thread_data[uiI].useGray = useGray;
        s_thread_data[uiI].maxLevel = maxLevel;
        s_thread_data[uiI].winSize.height = winSize;
        s_thread_data[uiI].winSize.width = winSize;
        s_thread_data[uiI].d_frame0Gray = d_frame0Gray.clone();
        s_thread_data[uiI].d_prevPts = d_prevPts.clone();

        tasks.push_back(pyrLK_task(uiI));
        tasks.back()._thread_data = &(s_thread_data[uiI]);
    }

    tbb::parallel_for_each(tasks.begin(),tasks.end(),invoker<pyrLK_task>());

    for (unsigned int uiI = 0; uiI < THREADS_NB; ++uiI)
    {
        stringstream ss;
        ss << "PyrLK MultiThreading [Sparse] " << uiI;

        imshow(ss.str(), s_thread_data[uiI].frame0);
        ss.str("");
    }

    waitKey();

    return 0;
}
#else
int main(int , const char* [])
{
    std::cout << "This example pyrlk_optical_flow_multithreading must be compiled with TBB Option" << std::endl;
    return 0;
}
#endif // HAVE_TBB
