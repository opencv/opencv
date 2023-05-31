#include <iostream>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>
#include <opencv2/cudaoptflow.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>

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

inline bool isFlowCorrect(Point2f u)
{
    return !cvIsNaN(u.x) && !cvIsNaN(u.y) && fabs(u.x) < 1e9 && fabs(u.y) < 1e9;
}

static Vec3b computeColor(float fx, float fy)
{
    static bool first = true;

    // relative lengths of color transitions:
    // these are chosen based on perceptual similarity
    // (e.g. one can distinguish more shades between red and yellow
    //  than between yellow and green)
    const int RY = 15;
    const int YG = 6;
    const int GC = 4;
    const int CB = 11;
    const int BM = 13;
    const int MR = 6;
    const int NCOLS = RY + YG + GC + CB + BM + MR;
    static Vec3i colorWheel[NCOLS];

    if (first)
    {
        int k = 0;

        for (int i = 0; i < RY; ++i, ++k)
            colorWheel[k] = Vec3i(255, 255 * i / RY, 0);

        for (int i = 0; i < YG; ++i, ++k)
            colorWheel[k] = Vec3i(255 - 255 * i / YG, 255, 0);

        for (int i = 0; i < GC; ++i, ++k)
            colorWheel[k] = Vec3i(0, 255, 255 * i / GC);

        for (int i = 0; i < CB; ++i, ++k)
            colorWheel[k] = Vec3i(0, 255 - 255 * i / CB, 255);

        for (int i = 0; i < BM; ++i, ++k)
            colorWheel[k] = Vec3i(255 * i / BM, 0, 255);

        for (int i = 0; i < MR; ++i, ++k)
            colorWheel[k] = Vec3i(255, 0, 255 - 255 * i / MR);

        first = false;
    }

    const float rad = sqrt(fx * fx + fy * fy);
    const float a = atan2(-fy, -fx) / (float)CV_PI;

    const float fk = (a + 1.0f) / 2.0f * (NCOLS - 1);
    const int k0 = static_cast<int>(fk);
    const int k1 = (k0 + 1) % NCOLS;
    const float f = fk - k0;

    Vec3b pix;

    for (int b = 0; b < 3; b++)
    {
        const float col0 = colorWheel[k0][b] / 255.0f;
        const float col1 = colorWheel[k1][b] / 255.0f;

        float col = (1 - f) * col0 + f * col1;

        if (rad <= 1)
            col = 1 - rad * (1 - col); // increase saturation with radius
        else
            col *= .75; // out of range

        pix[2 - b] = static_cast<uchar>(255.0 * col);
    }

    return pix;
}

static void drawOpticalFlow(const Mat_<float>& flowx, const Mat_<float>& flowy, Mat& dst, float maxmotion = -1)
{
    dst.create(flowx.size(), CV_8UC3);
    dst.setTo(Scalar::all(0));

    // determine motion range:
    float maxrad = maxmotion;

    if (maxmotion <= 0)
    {
        maxrad = 1;
        for (int y = 0; y < flowx.rows; ++y)
        {
            for (int x = 0; x < flowx.cols; ++x)
            {
                Point2f u(flowx(y, x), flowy(y, x));

                if (!isFlowCorrect(u))
                    continue;

                maxrad = max(maxrad, sqrt(u.x * u.x + u.y * u.y));
            }
        }
    }

    for (int y = 0; y < flowx.rows; ++y)
    {
        for (int x = 0; x < flowx.cols; ++x)
        {
            Point2f u(flowx(y, x), flowy(y, x));

            if (isFlowCorrect(u))
                dst.at<Vec3b>(y, x) = computeColor(u.x / maxrad, u.y / maxrad);
        }
    }
}

static void showFlow(const char* name, const GpuMat& d_flow)
{
    GpuMat planes[2];
    cuda::split(d_flow, planes);

    Mat flowx(planes[0]);
    Mat flowy(planes[1]);

    Mat out;
    drawOpticalFlow(flowx, flowy, out, 10);

    imshow(name, out);
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
        "{ h             help   |        | print help message }"
        "{ l             left   | ../data/pic1.png       | specify left image }"
        "{ r             right  | ../data/pic2.png       | specify right image }"
        "{ flow                 | sparse | specify flow type [PyrLK] }"
        "{ gray                 |        | use grayscale sources [PyrLK Sparse] }"
        "{ win_size             | 21     | specify windows size [PyrLK] }"
        "{ max_level            | 3      | specify max level [PyrLK] }"
        "{ iters                | 30     | specify iterations count [PyrLK] }"
        "{ points               | 4000   | specify points count [GoodFeatureToTrack] }"
        "{ min_dist             | 0      | specify minimal distance between points [GoodFeatureToTrack] }";

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

    string flow_type = cmd.get<string>("flow");
    bool is_sparse = true;
    if (flow_type == "sparse")
    {
        is_sparse = true;
    }
    else if (flow_type == "dense")
    {
        is_sparse = false;
    }
    else
    {
        cerr << "please specify 'sparse' or 'dense' as flow type" << endl;
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

    GpuMat d_frame0(frame0);
    GpuMat d_frame1(frame1);
    GpuMat d_frame1Gray(frame1Gray);
    GpuMat d_nextPts;
    GpuMat d_status;
    GpuMat d_flow(frame0.size(), CV_32FC2);

    if (is_sparse)
    {
        // Sparse
        Ptr<cuda::SparsePyrLKOpticalFlow> d_pyrLK_sparse = cuda::SparsePyrLKOpticalFlow::create(
            Size(winSize, winSize), maxLevel, iters);
        d_pyrLK_sparse->calc(useGray ? d_frame0Gray : d_frame0, useGray ? d_frame1Gray : d_frame1, d_prevPts, d_nextPts, d_status);

        // Draw arrows
        vector<Point2f> prevPts(d_prevPts.cols);
        download(d_prevPts, prevPts);

        vector<Point2f> nextPts(d_nextPts.cols);
        download(d_nextPts, nextPts);

        vector<uchar> status(d_status.cols);
        download(d_status, status);

        namedWindow("PyrLK [Sparse]", WINDOW_NORMAL);
        drawArrows(frame0, prevPts, nextPts, status, Scalar(255, 0, 0));
        imshow("PyrLK [Sparse]", frame0);
    }
    else
    {
        // Dense
        Ptr<cuda::DensePyrLKOpticalFlow> d_pyrLK_dense = cuda::DensePyrLKOpticalFlow::create(
            Size(winSize, winSize), maxLevel, iters);
        d_pyrLK_dense->calc(d_frame0Gray, d_frame1Gray, d_flow);

        // Draw flows
        namedWindow("PyrLK [Dense] Flow Field", WINDOW_NORMAL);
        showFlow("PyrLK [Dense] Flow Field", d_flow);
    }

    waitKey(0);

    return 0;
}