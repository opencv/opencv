#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <vector>
#include <algorithm>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include "utility.h"

using namespace std;
using namespace cv;
using namespace cv::gpu;

struct Object
{
    string name;

    Mat imgColor;
    Mat imgGray;
    GpuMat d_imgGray;

    Scalar color;

    vector<KeyPoint> keypoints;

    Mat descriptors;
    GpuMat d_descriptors;

    Object(const string& name_, const Mat& imgColor_, Scalar color_) :
        name(name_), imgColor(imgColor_), color(color_)
    {
        cvtColor(imgColor, imgGray, COLOR_BGR2GRAY);
        d_imgGray.upload(imgGray);
    }
};

class App : public BaseApp
{
public:
    App();

protected:
    void process();
    bool processKey(int key);
    bool parseCmdArgs(int& i, int argc, const char* argv[]);
    void printHelp();

private:
    void calcKeypoints(const Mat& img, const GpuMat& d_img, vector<KeyPoint>& keypoints, Mat& descriptors, GpuMat& d_descriptors);

    void match(const Mat& descriptors1, const GpuMat& d_descriptors1,
               const Mat& descriptors2, const GpuMat& d_descriptors2,
               vector<DMatch>& matches);

    void displayState(Mat& frame, double detect_fps, double match_fps, double total_fps);

    vector<Object> objects;

    bool useGPU;
    bool showCorrespondences;
    int curSource;

    SURF_GPU d_surf;
    SURF surf;

    BFMatcher matcher;
    BFMatcher_GPU d_matcher;
    GpuMat trainIdx, distance, allDist;
};

App::App() :
    d_surf(500),
    surf(500),
    matcher(NORM_L2),
    d_matcher(NORM_L2)
{
    useGPU = true;
    showCorrespondences = true;
    curSource = 0;
}

void App::process()
{
    if (objects.empty())
    {
        cout << "Loading default objects..." << endl;

        objects.push_back(Object("opengl", imread("data/features2d/objects/opengl.jpg"), CV_RGB(0, 255, 0)));
        objects.push_back(Object("java", imread("data/features2d/objects/java.jpg"), CV_RGB(255, 0, 0)));
        objects.push_back(Object("qt4", imread("data/features2d/objects/qt4.jpg"), CV_RGB(0, 0, 255)));
    }

    if (sources.empty())
    {
        cout << "Loading default frames source..." << endl;

        sources.push_back(new ImageSource("data/features2d/frames/1.jpg"));
        sources.push_back(new ImageSource("data/features2d/frames/2.jpg"));
        sources.push_back(new ImageSource("data/features2d/frames/3.jpg"));
        sources.push_back(new ImageSource("data/features2d/frames/4.jpg"));
        sources.push_back(new ImageSource("data/features2d/frames/5.jpg"));
        sources.push_back(new ImageSource("data/features2d/frames/6.jpg"));
        sources.push_back(new ImageSource("data/features2d/frames/7.jpg"));
    }

    Mat frame;
    Mat frameGray;
    GpuMat d_frameGray;

    vector<KeyPoint> frameKeypoints;
    Mat frameDescriptors;
    GpuMat d_frameDescriptors;

    vector< vector<DMatch> > matches(objects.size());

    Mat img_to_show;

    namedWindow("Features2D Demo", WINDOW_NORMAL);
    setWindowProperty("Features2D Demo", WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
    setWindowProperty("Features2D Demo", WND_PROP_ASPECT_RATIO, CV_WINDOW_FREERATIO);

    while (!exited)
    {
        int64 start = getTickCount();

        sources[curSource]->next(frame);
        cvtColor(frame, frameGray, COLOR_BGR2GRAY);
        d_frameGray.upload(frameGray);

        int64 detect_start = getTickCount();
        {
            for (size_t i = 0; i < objects.size(); ++i)
                calcKeypoints(objects[i].imgGray, objects[i].d_imgGray, objects[i].keypoints, objects[i].descriptors, objects[i].d_descriptors);

            calcKeypoints(frameGray, d_frameGray, frameKeypoints, frameDescriptors, d_frameDescriptors);
        }
        double detect_fps = getTickFrequency() / (getTickCount() - detect_start);

        int64 match_start = cv::getTickCount();
        {
            for (size_t i = 0; i < objects.size(); ++i)
                match(objects[i].descriptors, objects[i].d_descriptors, frameDescriptors, d_frameDescriptors, matches[i]);
        }
        double match_fps = getTickFrequency() / (getTickCount() - match_start);

        const int offset = 350;

        Size outSize = frame.size();
        int max_height = 0;
        int sum_width = offset;
        for (size_t i = 0; i < objects.size(); ++i)
        {
            sum_width += objects[i].imgColor.cols;
            max_height = std::max(max_height, objects[i].imgColor.rows);
        }
        outSize.height += max_height;
        outSize.width = std::max(outSize.width, sum_width);

        img_to_show.create(outSize, CV_8UC3);
        img_to_show.setTo(0);
        frame.copyTo(img_to_show(Rect(0, max_height, frame.cols, frame.rows)));

        int objX = offset;
        for (size_t i = 0; i < objects.size(); ++i)
        {
            objects[i].imgColor.copyTo(img_to_show(Rect(objX, 0, objects[i].imgColor.cols, objects[i].imgColor.rows)));

            putText(img_to_show, objects[i].name, Point(objX, 15), FONT_HERSHEY_DUPLEX, 0.8, objects[i].color);

            if (matches[i].size() >= 10)
            {
                static vector<Point2f> pt1;
                static vector<Point2f> pt2;

                pt1.resize(matches[i].size());
                pt2.resize(matches[i].size());

                for (size_t j = 0; j < matches[i].size(); ++j)
                {
                    DMatch m = matches[i][j];

                    KeyPoint objKp = objects[i].keypoints[m.queryIdx];
                    KeyPoint frameKp = frameKeypoints[m.trainIdx];

                    pt1[j] = objKp.pt;
                    pt2[j] = frameKp.pt;

                    if (showCorrespondences)
                    {
                        Point objCenter(cvRound(objKp.pt.x) + objX, cvRound(objKp.pt.y));
                        Point frameCenter(cvRound(frameKp.pt.x), cvRound(frameKp.pt.y) + max_height);

                        circle(img_to_show, objCenter, 3, objects[i].color);
                        circle(img_to_show, frameCenter, 3, objects[i].color);
                        line(img_to_show, objCenter, frameCenter, objects[i].color);
                    }
                }

                Mat H = findHomography(pt1, pt2, RANSAC);

                if (H.empty())
                    continue;

                Point src_corners[] =
                {
                    Point(0, 0),
                    Point(objects[i].imgColor.cols, 0),
                    Point(objects[i].imgColor.cols, objects[i].imgColor.rows),
                    Point(0, objects[i].imgColor.rows)
                };
                Point dst_corners[5];

                for (int j = 0; j < 4; ++j)
                {
                    double x = src_corners[j].x;
                    double y = src_corners[j].y;

                    double Z = 1.0 / (H.at<double>(2, 0) * x + H.at<double>(2, 1) * y + H.at<double>(2, 2));
                    double X = (H.at<double>(0, 0) * x + H.at<double>(0, 1) * y + H.at<double>(0, 2)) * Z;
                    double Y = (H.at<double>(1, 0) * x + H.at<double>(1, 1) * y + H.at<double>(1, 2)) * Z;

                    dst_corners[j] = Point(cvRound(X), cvRound(Y));
                }

                for (int j = 0; j < 4; ++j)
                {
                    Point r1 = dst_corners[j % 4];
                    Point r2 = dst_corners[(j + 1) % 4];

                    line(img_to_show, Point(r1.x, r1.y + max_height), Point(r2.x, r2.y + max_height), objects[i].color, 3);
                }

                putText(img_to_show, objects[i].name, Point(dst_corners[0].x, dst_corners[0].y + max_height), FONT_HERSHEY_DUPLEX, 0.8, objects[i].color);
            }

            objX += objects[i].imgColor.cols;
        }

        double total_fps = getTickFrequency() / (getTickCount() - start);

        displayState(img_to_show, detect_fps, match_fps, total_fps);

        imshow("Features2D Demo", img_to_show);

        processKey(waitKey(3));
    }
}

void App::calcKeypoints(const Mat& img, const GpuMat& d_img, vector<KeyPoint>& keypoints, Mat& descriptors, GpuMat& d_descriptors)
{
    keypoints.clear();

    if (useGPU)
        d_surf(d_img, GpuMat(), keypoints, d_descriptors);
    else
        surf(img, noArray(), keypoints, descriptors);
}

struct DMatchCmp
{
    inline bool operator ()(const DMatch& m1, const DMatch& m2) const
    {
        return m1.distance < m2.distance;
    }
};

void App::match(const Mat& descriptors1, const GpuMat& d_descriptors1,
                const Mat& descriptors2, const GpuMat& d_descriptors2,
                vector<DMatch>& matches)
{
    static vector< vector<DMatch> > temp;

    matches.clear();

    if (useGPU)
    {
        d_matcher.knnMatchSingle(d_descriptors1, d_descriptors2, trainIdx, distance, allDist, 2);
        d_matcher.knnMatchDownload(trainIdx, distance, temp);
    }
    else
    {
        matcher.knnMatch(descriptors1, descriptors2, temp, 2);
    }

    for (size_t i = 0; i < temp.size(); ++i)
    {
        if (temp[i].size() != 2)
            continue;

        DMatch m1 = temp[i][0];
        DMatch m2 = temp[i][1];

        if (m1.distance < 0.55 * m2.distance)
            matches.push_back(m1);
    }

    if (useGPU)
        sort(matches.begin(), matches.end(), DMatchCmp());
}

void App::displayState(Mat& frame, double detect_fps, double match_fps, double total_fps)
{
    const Scalar fontColorRed = CV_RGB(255, 0, 0);

    int i = 0;

    ostringstream txt;
    txt.str(""); txt << "Source size: " << frame.cols << 'x' << frame.rows;
    printText(frame, txt.str(), i++);

    printText(frame, useGPU ? "Mode: CUDA" : "Mode: CPU", i++);

    txt.str(""); txt << "FPS (Detect): " << std::fixed << std::setprecision(1) << detect_fps;
    printText(frame, txt.str(), i++);

    txt.str(""); txt << "FPS (Match): " << std::fixed << std::setprecision(1) << match_fps;
    printText(frame, txt.str(), i++);

    txt.str(""); txt << "FPS (Total): " << std::fixed << std::setprecision(1) << total_fps;
    printText(frame, txt.str(), i++);

    printText(frame, "Space - switch method", i++, fontColorRed);
    printText(frame, "S - show correspondences", i++, fontColorRed);
    printText(frame, "N - next source", i++, fontColorRed);
}

bool App::processKey(int key)
{
    if (BaseApp::processKey(key))
        return true;

    switch (toupper(key & 0xff))
    {
    case 32 /*space*/:
        useGPU = !useGPU;
        cout << "Switched to " << (useGPU ? "CUDA" : "CPU") << " mode\n";
        break;

    case 'S':
        showCorrespondences = !showCorrespondences;
        break;

    case 'N':
        curSource = (curSource + 1) % sources.size();
        sources[curSource]->reset();
        cout << "Switch source to " << curSource << endl;
        break;

    default:
        return false;
    }

    return true;
}

bool App::parseCmdArgs(int& i, int argc, const char* argv[])
{
    string arg = argv[i];

    if (arg == "--object")
    {
        ++i;

        if (i >= argc)
            THROW_EXCEPTION("Missing file name after " << arg);

        RNG& rng = theRNG();
        objects.push_back(Object(argv[i], imread(argv[i]), CV_RGB(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255))));
    }
    else
        return false;

    return true;
}

void App::printHelp()
{
    cout << "This sample demonstrates Object Detection via keypoints matching" << endl;
    cout << "Usage: demo_features2d [--object <image>]* [options]" << endl;
    cout << "Options:" << endl;
    BaseApp::printHelp();
}

RUN_APP(App)
