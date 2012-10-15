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

#include "utility_lib/utility_lib.h"

enum Method
{
    SURF_GPU,
    SURF_CPU,
    ORB_GPU,
    ORB_CPU
};

const char* method_str[] =
{
    "SURF CUDA",
    "SURF CPU",
    "ORB CUDA",
    "ORB CPU"
};

struct Object
{
    std::string name;

    cv::Mat imgColor;
    cv::Mat imgGray;
    cv::gpu::GpuMat d_imgGray;

    cv::Scalar color;

    std::vector<cv::KeyPoint> keypoints;

    cv::Mat descriptors;
    cv::gpu::GpuMat d_descriptors;

    Object(const std::string& name_, const cv::Mat& imgColor_, cv::Scalar color_) :
        name(name_), imgColor(imgColor_), color(color_)
    {
        cv::cvtColor(imgColor, imgGray, cv::COLOR_BGR2GRAY);
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
    void calcKeypoints(const cv::Mat& img, const cv::gpu::GpuMat& d_img, std::vector<cv::KeyPoint>& keypoints,
                       cv::Mat& descriptors, cv::gpu::GpuMat& d_descriptors);

    void match(const cv::Mat& descriptors1, const cv::gpu::GpuMat& d_descriptors1,
               const cv::Mat& descriptors2, const cv::gpu::GpuMat& d_descriptors2,
               std::vector<cv::DMatch>& matches);

    void displayState(cv::Mat& frame, double detect_fps, double match_fps, double total_fps);

    std::vector<Object> objects_;

    Method method;
    bool showCorrespondences;

    cv::gpu::SURF_GPU d_surf_;
    cv::SURF surf_;
    cv::gpu::ORB_GPU d_orb_;
    cv::ORB orb_;

    cv::BFMatcher matcher_;
    cv::gpu::BFMatcher_GPU d_matcher_;
    cv::gpu::GpuMat trainIdx_, distance_, allDist_;

    int sourceIdx;
};

App::App() :
    d_surf_(500),
    surf_(500),
    d_orb_(5000),
    orb_(5000),

    matcher_(cv::NORM_L2),
    d_matcher_(cv::NORM_L2)
{
    method = SURF_GPU;
    showCorrespondences = false;

    sourceIdx = 0;
}

void App::process()
{
    if (objects_.empty())
    {
        std::cout << "Loading default objects..." << std::endl;

        objects_.push_back(Object("opengl", cv::imread("data/features2d/objects/opengl.jpg"), CV_RGB(0, 255, 0)));
        objects_.push_back(Object("java", cv::imread("data/features2d/objects/java.jpg"), CV_RGB(255, 0, 0)));
        objects_.push_back(Object("qt4", cv::imread("data/features2d/objects/qt4.jpg"), CV_RGB(0, 0, 255)));
    }

    if (sources.empty())
    {
        std::cout << "Loading default frames source..." << std::endl;

        sources.push_back(new ImageSource("data/features2d/frames/1.jpg"));
        sources.push_back(new ImageSource("data/features2d/frames/2.jpg"));
        sources.push_back(new ImageSource("data/features2d/frames/3.jpg"));
        sources.push_back(new ImageSource("data/features2d/frames/4.jpg"));
        sources.push_back(new ImageSource("data/features2d/frames/5.jpg"));
        sources.push_back(new ImageSource("data/features2d/frames/6.jpg"));
        sources.push_back(new ImageSource("data/features2d/frames/7.jpg"));
    }

    cv::Mat frame;
    cv::Mat frameGray;
    cv::gpu::GpuMat d_frameGray;

    std::vector<cv::KeyPoint> frameKeypoints;
    cv::Mat frameDescriptors;
    cv::gpu::GpuMat d_frameDescriptors;

    std::vector< std::vector<cv::DMatch> > matches(objects_.size());

    cv::Mat img_to_show;

    cv::namedWindow("demo_features2d", cv::WINDOW_NORMAL);
    cv::setWindowProperty("demo_features2d", cv::WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
    cv::setWindowProperty("demo_features2d", cv::WND_PROP_ASPECT_RATIO, CV_WINDOW_FREERATIO);

    while (!exited)
    {
        int64 start = cv::getTickCount();

        sources[sourceIdx]->next(frame);
        cv::cvtColor(frame, frameGray, cv::COLOR_BGR2GRAY);
        d_frameGray.upload(frameGray);

        int64 detect_start = cv::getTickCount();
        {
            for (size_t i = 0; i < objects_.size(); ++i)
                calcKeypoints(objects_[i].imgGray, objects_[i].d_imgGray, objects_[i].keypoints, objects_[i].descriptors, objects_[i].d_descriptors);

            calcKeypoints(frameGray, d_frameGray, frameKeypoints, frameDescriptors, d_frameDescriptors);
        }
        double detect_fps = cv::getTickFrequency() / (cv::getTickCount() - detect_start);

        int64 match_start = cv::getTickCount();
        {
            for (size_t i = 0; i < objects_.size(); ++i)
                match(objects_[i].descriptors, objects_[i].d_descriptors, frameDescriptors, d_frameDescriptors, matches[i]);
        }
        double match_fps = cv::getTickFrequency() / (cv::getTickCount() - match_start);

        const int offset = 350;

        cv::Size outSize = frame.size();
        int max_height = 0;
        int sum_width = offset;
        for (size_t i = 0; i < objects_.size(); ++i)
        {
            sum_width += objects_[i].imgColor.cols;
            max_height = std::max(max_height, objects_[i].imgColor.rows);
        }
        outSize.height += max_height;
        outSize.width = std::max(outSize.width, sum_width);

        img_to_show.create(outSize, CV_8UC3);
        img_to_show.setTo(0);
        frame.copyTo(img_to_show(cv::Rect(0, max_height, frame.cols, frame.rows)));

        int objX = offset;
        for (size_t i = 0; i < objects_.size(); ++i)
        {
            objects_[i].imgColor.copyTo(img_to_show(cv::Rect(objX, 0, objects_[i].imgColor.cols, objects_[i].imgColor.rows)));

            cv::putText(img_to_show, objects_[i].name, cv::Point(objX, 15), cv::FONT_HERSHEY_DUPLEX, 0.8, objects_[i].color);

            if (matches[i].size() >= 10)
            {
                static std::vector<cv::Point2f> pt1;
                static std::vector<cv::Point2f> pt2;

                pt1.resize(matches[i].size());
                pt2.resize(matches[i].size());

                for (size_t j = 0; j < matches[i].size(); ++j)
                {
                    cv::DMatch m = matches[i][j];

                    cv::KeyPoint objKp = objects_[i].keypoints[m.queryIdx];
                    cv::KeyPoint frameKp = frameKeypoints[m.trainIdx];

                    pt1[j] = objKp.pt;
                    pt2[j] = frameKp.pt;

                    if (showCorrespondences)
                    {
                        cv::Point objCenter(cvRound(objKp.pt.x) + objX, cvRound(objKp.pt.y));
                        cv::Point frameCenter(cvRound(frameKp.pt.x), cvRound(frameKp.pt.y) + max_height);

                        cv::circle(img_to_show, objCenter, 3, objects_[i].color);
                        cv::circle(img_to_show, frameCenter, 3, objects_[i].color);
                        cv::line(img_to_show, objCenter, frameCenter, objects_[i].color);
                    }
                }

                cv::Mat H = cv::findHomography(pt1, pt2, cv::RANSAC);

                if (H.empty())
                    continue;

                cv::Point src_corners[] =
                {
                    cv::Point(0, 0),
                    cv::Point(objects_[i].imgColor.cols, 0),
                    cv::Point(objects_[i].imgColor.cols, objects_[i].imgColor.rows),
                    cv::Point(0, objects_[i].imgColor.rows)
                };
                cv::Point dst_corners[5];

                for (int j = 0; j < 4; ++j)
                {
                    double x = src_corners[j].x;
                    double y = src_corners[j].y;

                    double Z = 1.0 / (H.at<double>(2, 0) * x + H.at<double>(2, 1) * y + H.at<double>(2, 2));
                    double X = (H.at<double>(0, 0) * x + H.at<double>(0, 1) * y + H.at<double>(0, 2)) * Z;
                    double Y = (H.at<double>(1, 0) * x + H.at<double>(1, 1) * y + H.at<double>(1, 2)) * Z;

                    dst_corners[j] = cv::Point(cvRound(X), cvRound(Y));
                }

                for (int j = 0; j < 4; ++j)
                {
                    cv::Point r1 = dst_corners[j % 4];
                    cv::Point r2 = dst_corners[(j + 1) % 4];

                    cv::line(img_to_show, cv::Point(r1.x, r1.y + max_height), cv::Point(r2.x, r2.y + max_height), objects_[i].color, 3);
                }

                cv::putText(img_to_show, objects_[i].name, cv::Point(dst_corners[0].x, dst_corners[0].y + max_height), cv::FONT_HERSHEY_DUPLEX, 0.8,
                            objects_[i].color);
            }

            objX += objects_[i].imgColor.cols;
        }

        double total_fps = cv::getTickFrequency() / (cv::getTickCount() - start);

        displayState(img_to_show, detect_fps, match_fps, total_fps);

        cv::imshow("demo_features2d", img_to_show);

        processKey(cv::waitKey(3) & 0xff);
    }
}

void App::calcKeypoints(const cv::Mat& img, const cv::gpu::GpuMat& d_img, std::vector<cv::KeyPoint>& keypoints,
                       cv::Mat& descriptors, cv::gpu::GpuMat& d_descriptors)
{
    keypoints.clear();

    switch (method)
    {
    case SURF_GPU:
        d_surf_(d_img, cv::gpu::GpuMat(), keypoints, d_descriptors);
        break;

    case SURF_CPU:
        surf_(img, cv::noArray(), keypoints, descriptors);
        break;

    case ORB_GPU:
        d_orb_(d_img, cv::gpu::GpuMat(), keypoints, d_descriptors);
        break;

    case ORB_CPU:
        orb_(img, cv::noArray(), keypoints, descriptors);
        break;
    }
}

void App::match(const cv::Mat& descriptors1, const cv::gpu::GpuMat& d_descriptors1,
                const cv::Mat& descriptors2, const cv::gpu::GpuMat& d_descriptors2,
                std::vector<cv::DMatch>& matches)
{
    static std::vector< std::vector<cv::DMatch> > temp;

    matches.clear();

    switch (method)
    {
    case SURF_GPU:
        d_matcher_.norm = cv::NORM_L2;
        d_matcher_.knnMatchSingle(d_descriptors1, d_descriptors2, trainIdx_, distance_, allDist_, 2);
        d_matcher_.knnMatchDownload(trainIdx_, distance_, temp);
        break;

    case SURF_CPU:
        matcher_ = cv::BFMatcher(cv::NORM_L2);
        matcher_.knnMatch(descriptors1, descriptors2, temp, 2);
        break;

    case ORB_GPU:
        d_matcher_.norm = cv::NORM_HAMMING;
        d_matcher_.knnMatchSingle(d_descriptors1, d_descriptors2, trainIdx_, distance_, allDist_, 2);
        d_matcher_.knnMatchDownload(trainIdx_, distance_, temp);
        break;

    case ORB_CPU:
        matcher_ = cv::BFMatcher(cv::NORM_HAMMING);
        matcher_.knnMatch(descriptors1, descriptors2, temp, 2);
        break;
    }

    for (size_t i = 0; i < temp.size(); ++i)
    {
        if (temp[i].size() != 2)
            continue;

        cv::DMatch m1 = temp[i][0];
        cv::DMatch m2 = temp[i][1];

        switch (method)
        {
        case SURF_GPU:
        case SURF_CPU:
            if (m1.distance < 0.55 * m2.distance)
                matches.push_back(m1);
            break;

        case ORB_CPU:
        case ORB_GPU:
            if (std::abs(m1.distance - m2.distance) > 30)
                matches.push_back(m1);
            break;
        }
    }
}

void App::displayState(cv::Mat& frame, double detect_fps, double match_fps, double total_fps)
{
    const cv::Scalar fontColorRed = CV_RGB(255, 0, 0);

    int i = 0;

    std::ostringstream txt;
    txt.str(""); txt << "Source size: " << frame.cols << 'x' << frame.rows;
    printText(frame, txt.str(), i++);

    txt.str(""); txt << "Method: " << method_str[method];
    printText(frame, txt.str(), i++);

    txt.str(""); txt << "FPS (Detect): " << std::fixed << std::setprecision(1) << detect_fps;
    printText(frame, txt.str(), i++);

    txt.str(""); txt << "FPS (Match): " << std::fixed << std::setprecision(1) << match_fps;
    printText(frame, txt.str(), i++);

    txt.str(""); txt << "FPS (Total): " << std::fixed << std::setprecision(1) << total_fps;
    printText(frame, txt.str(), i++);

    printText(frame, "Space - switch method", i++, fontColorRed);
    printText(frame, "S - show correspondences", i++, fontColorRed);
    printText(frame, "I - swith source", i++, fontColorRed);
}

bool App::processKey(int key)
{
    if (BaseApp::processKey(key))
        return true;

    switch (toupper(key & 0xff))
    {
    case 32:
        switch (method)
        {
        case SURF_GPU:
            method = SURF_CPU;
            break;
        case SURF_CPU:
            method = ORB_GPU;
            break;
        case ORB_GPU:
            method = ORB_CPU;
            break;
        case ORB_CPU:
            method = SURF_GPU;
            break;
        }
        std::cout << "method: " << method_str[method] << std::endl;
        break;

    case 'S':
        showCorrespondences = !showCorrespondences;
        break;

    case 'I':
        sourceIdx = (sourceIdx + 1) % sources.size();
        break;

    default:
        return false;
    }

    return true;
}

bool App::parseCmdArgs(int& i, int argc, const char* argv[])
{
    std::string key = argv[i];

     if (key == "--object")
    {
        ++i;

        if (i >= argc)
            throw std::runtime_error("Missing value after --object");

        cv::RNG& rng = cv::theRNG();
        objects_.push_back(Object(argv[i], cv::imread(argv[i]), CV_RGB(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255))));
    }
    else
        return false;

    return true;
}

void App::printHelp()
{
    std::cout << "\nUsage: demo_features2d <frames_source>\n"
              << "  [--object <image>] # object image\n";
    BaseApp::printHelp();
}

RUN_APP(App)
