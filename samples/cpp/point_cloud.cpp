#include <cstring>
#include <cmath>
#include <iostream>
#include <sstream>
#include "opencv2/core/core.hpp"
#include "opencv2/core/opengl_interop.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/contrib/contrib.hpp"

using namespace std;
using namespace cv;
using namespace cv::gpu;

class PointCloudRenderer
{
public:
    PointCloudRenderer(const Mat& points, const Mat& img, double scale);

    void onMouseEvent(int event, int x, int y, int flags);
    void draw();
    void update(int key, double aspect);

    int fov_;

private:
    int mouse_dx_;
    int mouse_dy_;

    double yaw_;
    double pitch_;
    Point3d pos_;

    TickMeter tm_;
    static const int step_;
    int frame_;

    GlCamera camera_;
    GlArrays pointCloud_;
    string fps_;
};

bool stop = false;

void mouseCallback(int event, int x, int y, int flags, void* userdata)
{
    if (stop)
        return;

    PointCloudRenderer* renderer = static_cast<PointCloudRenderer*>(userdata);
    renderer->onMouseEvent(event, x, y, flags);
}

void openGlDrawCallback(void* userdata)
{
    if (stop)
        return;

    PointCloudRenderer* renderer = static_cast<PointCloudRenderer*>(userdata);
    renderer->draw();
}

int main(int argc, const char* argv[])
{
    const char* keys =
       "{ l | left      |       | left image file name }"
       "{ r | right     |       | right image file name }"
       "{ i | intrinsic |       | intrinsic camera parameters file name }"
       "{ e | extrinsic |       | extrinsic camera parameters file name }"
       "{ d | ndisp     | 256   | number of disparities }"
       "{ s | scale     | 1.0   | scale factor for point cloud }"
       "{ h | help      | false | print help message }";

    CommandLineParser cmd(argc, argv, keys);

    if (cmd.get<bool>("help"))
    {
        cout << "Avaible options:" << endl;
        cmd.printParams();
        return 0;
    }

    string left = cmd.get<string>("left");
    string right = cmd.get<string>("right");
    string intrinsic = cmd.get<string>("intrinsic");
    string extrinsic = cmd.get<string>("extrinsic");
    int ndisp = cmd.get<int>("ndisp");
    double scale = cmd.get<double>("scale");

    if (left.empty() || right.empty())
    {
        cout << "Missed input images" << endl;
        cout << "Avaible options:" << endl;
        cmd.printParams();
        return 0;
    }

    if (intrinsic.empty() ^ extrinsic.empty())
    {
        cout << "Boss camera parameters must be specified" << endl;
        cout << "Avaible options:" << endl;
        cmd.printParams();
        return 0;
    }

    Mat imgLeftColor = imread(left, IMREAD_COLOR);
    Mat imgRightColor = imread(right, IMREAD_COLOR);

    if (imgLeftColor.empty())
    {
        cout << "Can't load image " << left << endl;
        return -1;
    }

    if (imgRightColor.empty())
    {
        cout << "Can't load image " << right << endl;
        return -1;
    }

    Mat Q = Mat::eye(4, 4, CV_32F);
    if (!intrinsic.empty() && !extrinsic.empty())
    {
        FileStorage fs;

        // reading intrinsic parameters
        fs.open(intrinsic, CV_STORAGE_READ);
        if (!fs.isOpened())
        {
            cout << "Failed to open file " << intrinsic << endl;
            return -1;
        }

        Mat M1, D1, M2, D2;
        fs["M1"] >> M1;
        fs["D1"] >> D1;
        fs["M2"] >> M2;
        fs["D2"] >> D2;

        // reading extrinsic parameters
        fs.open(extrinsic, CV_STORAGE_READ);
        if (!fs.isOpened())
        {
            cout << "Failed to open file " << extrinsic << endl;
            return -1;
        }

        Mat R, T, R1, P1, R2, P2;
        fs["R"] >> R;
        fs["T"] >> T;

        Size img_size = imgLeftColor.size();

        Rect roi1, roi2;
        stereoRectify(M1, D1, M2, D2, img_size, R, T, R1, R2, P1, P2, Q, CALIB_ZERO_DISPARITY, -1, img_size, &roi1, &roi2);

        Mat map11, map12, map21, map22;
        initUndistortRectifyMap(M1, D1, R1, P1, img_size, CV_16SC2, map11, map12);
        initUndistortRectifyMap(M2, D2, R2, P2, img_size, CV_16SC2, map21, map22);

        Mat img1r, img2r;
        remap(imgLeftColor, img1r, map11, map12, INTER_LINEAR);
        remap(imgRightColor, img2r, map21, map22, INTER_LINEAR);

        imgLeftColor = img1r(roi1);
        imgRightColor = img2r(roi2);
    }

    Mat imgLeftGray, imgRightGray;
    cvtColor(imgLeftColor, imgLeftGray, COLOR_BGR2GRAY);
    cvtColor(imgRightColor, imgRightGray, COLOR_BGR2GRAY);

    cvtColor(imgLeftColor, imgLeftColor, COLOR_BGR2RGB);

    Mat disp, points;

    StereoBM bm(0, ndisp);

    bm(imgLeftGray, imgRightGray, disp);
    disp.convertTo(disp, CV_8U, 1.0 / 16.0);

    disp = disp(Range(21, disp.rows - 21), Range(ndisp, disp.cols - 21)).clone();
    imgLeftColor = imgLeftColor(Range(21, imgLeftColor.rows - 21), Range(ndisp, imgLeftColor.cols - 21)).clone();

    reprojectImageTo3D(disp, points, Q);

    const string windowName = "OpenGL Sample";

    namedWindow(windowName, WINDOW_OPENGL);
    resizeWindow(windowName, 400, 400);

    PointCloudRenderer renderer(points, imgLeftColor, scale);

    createTrackbar("Fov", windowName, &renderer.fov_, 100);
    setMouseCallback(windowName, mouseCallback, &renderer);
    setOpenGlDrawCallback(windowName, openGlDrawCallback, &renderer);

    for(;;)
    {
        int key = waitKey(10);

        if (key >= 0)
            key = key & 0xff;

        if (key == 27)
        {
            stop = true;
            break;
        }

        double aspect = getWindowProperty(windowName, WND_PROP_ASPECT_RATIO);

        key = tolower(key);

        renderer.update(key, aspect);

        updateWindow(windowName);
    }

    return 0;
}

const int PointCloudRenderer::step_ = 20;

PointCloudRenderer::PointCloudRenderer(const Mat& points, const Mat& img, double scale)
{
    mouse_dx_ = 0;
    mouse_dy_ = 0;

    fov_ = 0;
    yaw_ = 0.0;
    pitch_ = 0.0;

    frame_ = 0;

    camera_.setScale(Point3d(scale, scale, scale));

    pointCloud_.setVertexArray(points);
    pointCloud_.setColorArray(img, false);

    tm_.start();
}

inline int clamp(int val, int minVal, int maxVal)
{
    return max(min(val, maxVal), minVal);
}

void PointCloudRenderer::onMouseEvent(int event, int x, int y, int /*flags*/)
{
    static int oldx = x;
    static int oldy = y;
    static bool moving = false;

    if (event == EVENT_LBUTTONDOWN)
    {
        oldx = x;
        oldy = y;
        moving = true;
    }
    else if (event == EVENT_LBUTTONUP)
    {
        moving = false;
    }

    if (moving)
    {
        mouse_dx_ = oldx - x;
        mouse_dy_ = oldy - y;
    }
    else
    {
        mouse_dx_ = 0;
        mouse_dy_ = 0;
    }

    const int mouseClamp = 300;
    mouse_dx_ = clamp(mouse_dx_, -mouseClamp, mouseClamp);
    mouse_dy_ = clamp(mouse_dy_, -mouseClamp, mouseClamp);
}

Point3d rotate(Point3d v, double yaw, double pitch)
{
    Point3d t1;
    t1.x = v.x * cos(-yaw / 180.0 * CV_PI) - v.z * sin(-yaw / 180.0 * CV_PI);
    t1.y = v.y;
    t1.z = v.x * sin(-yaw / 180.0 * CV_PI) + v.z * cos(-yaw / 180.0 * CV_PI);

    Point3d t2;
    t2.x = t1.x;
    t2.y = t1.y * cos(pitch / 180.0 * CV_PI) - t1.z * sin(pitch / 180.0 * CV_PI);
    t2.z = t1.y * sin(pitch / 180.0 * CV_PI) + t1.z * cos(pitch / 180.0 * CV_PI);

    return t2;
}

void PointCloudRenderer::update(int key, double aspect)
{
    const Point3d dirVec(0.0, 0.0, -1.0);
    const Point3d upVec(0.0, 1.0, 0.0);
    const Point3d leftVec(-1.0, 0.0, 0.0);

    const double posStep = 0.1;

    const double mouseStep = 0.001;

    camera_.setPerspectiveProjection(30.0 + fov_ / 100.0 * 40.0, aspect, 0.1, 1000.0);

    yaw_ += mouse_dx_ * mouseStep;
    pitch_ += mouse_dy_ * mouseStep;

    if (key == 'w')
        pos_ += posStep * rotate(dirVec, yaw_, pitch_);
    else if (key == 's')
        pos_ -= posStep * rotate(dirVec, yaw_, pitch_);
    else if (key == 'a')
        pos_ += posStep * rotate(leftVec, yaw_, pitch_);
    else if (key == 'd')
        pos_ -= posStep * rotate(leftVec, yaw_, pitch_);
    else if (key == 'q')
        pos_ += posStep * rotate(upVec, yaw_, pitch_);
    else if (key == 'e')
        pos_ -= posStep * rotate(upVec, yaw_, pitch_);

    camera_.setCameraPos(pos_, yaw_, pitch_, 0.0);

    tm_.stop();

    if (frame_++ >= step_)
    {
        ostringstream ostr;
        ostr << "FPS: " << step_ / tm_.getTimeSec();
        fps_ = ostr.str();

        frame_ = 0;
        tm_.reset();
    }

    tm_.start();
}

void PointCloudRenderer::draw()
{
    camera_.setupProjectionMatrix();
    camera_.setupModelViewMatrix();

    render(pointCloud_);

    render(fps_, GlFont::get("Courier New", 16), Scalar::all(255), Point2d(3.0, 0.0));
}
