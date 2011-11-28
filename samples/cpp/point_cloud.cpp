#include <cstring>
#include <cmath>
#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/core/gpumat.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"

using namespace std;
using namespace cv;
using namespace cv::gpu;

void mouseCallback(int event, int x, int y, int flags, void* userdata)
{
    int* dx = static_cast<int*>(userdata);
    int* dy = dx + 1;

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
        *dx = oldx - x;
        *dy = oldy - y;
    }
    else
    {
        *dx = 0;
        *dy = 0;
    }
}

inline int clamp(int val, int minVal, int maxVal)
{
    return max(min(val, maxVal), minVal);
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

    namedWindow("OpenGL Sample", WINDOW_OPENGL);

    int fov = 0;
    createTrackbar("Fov", "OpenGL Sample", &fov, 100);

    int mouse[2] = {0, 0};
    setMouseCallback("OpenGL Sample", mouseCallback, mouse);

    GlArrays pointCloud;

    pointCloud.setVertexArray(points);
    pointCloud.setColorArray(imgLeftColor, false);

    GlCamera camera;
    camera.setScale(Point3d(scale, scale, scale));

    double yaw = 0.0;
    double pitch = 0.0;

    const Point3d dirVec(0.0, 0.0, -1.0);
    const Point3d upVec(0.0, 1.0, 0.0);
    const Point3d leftVec(-1.0, 0.0, 0.0);
    Point3d pos;

    while (true)
    {
        int key = waitKey(1);
        if (key >= 0)
            key = key & 0xff;

        if (key == 27)
            break;

        double aspect = getWindowProperty("OpenGL Sample", WND_PROP_ASPECT_RATIO);

        const double posStep = 0.1;
        
        #ifdef _WIN32
        const double mouseStep = 0.001;
        #else
        const double mouseStep = 0.000001;
        #endif
        
        const int mouseClamp = 300;

        camera.setPerspectiveProjection(30.0 + fov / 100.0 * 40.0, aspect, 0.1, 1000.0);

        int mouse_dx = clamp(mouse[0], -mouseClamp, mouseClamp);
        int mouse_dy = clamp(mouse[1], -mouseClamp, mouseClamp);

        yaw += mouse_dx * mouseStep;
        pitch += mouse_dy * mouseStep;

        key = tolower(key);
        if (key == 'w')
            pos += posStep * rotate(dirVec, yaw, pitch);
        else if (key == 's')
            pos -= posStep * rotate(dirVec, yaw, pitch);
        else if (key == 'a')
            pos += posStep * rotate(leftVec, yaw, pitch);
        else if (key == 'd')
            pos -= posStep * rotate(leftVec, yaw, pitch);
        else if (key == 'q')
            pos += posStep * rotate(upVec, yaw, pitch);
        else if (key == 'e')
            pos -= posStep * rotate(upVec, yaw, pitch);

        camera.setCameraPos(pos, yaw, pitch, 0.0);

        pointCloudShow("OpenGL Sample", camera, pointCloud);
    }

    return 0;
}
