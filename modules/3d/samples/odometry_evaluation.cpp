// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include <opencv2/3d.hpp>

#include <opencv2/highgui.hpp>
#include <opencv2/3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/core/quaternion.hpp>

#include <iostream>
#include <fstream>

using namespace std;
using namespace cv;

#define BILATERAL_FILTER 0// if 1 then bilateral filter will be used for the depth

static
void writeResults( const string& filename, const vector<string>& timestamps, const vector<Mat>& Rt )
{
    CV_Assert( timestamps.size() == Rt.size() );

    ofstream file( filename.c_str() );
    if( !file.is_open() )
        return;

    cout.precision(4);
    for( size_t i = 0; i < Rt.size(); i++ )
    {
        const Mat& Rt_curr = Rt[i];
        if( Rt_curr.empty() )
            continue;

        CV_Assert( Rt_curr.type() == CV_64FC1 );

        Quatd rot = Quatd::createFromRotMat(Rt_curr(Rect(0, 0, 3, 3)));

        // timestamp tx ty tz qx qy qz qw
        file << timestamps[i] << " " << fixed
             << Rt_curr.at<double>(0,3) << " " << Rt_curr.at<double>(1,3) << " " << Rt_curr.at<double>(2,3) << " "
             << rot.x << " " << rot.y << " " << rot.z << " " << rot.w << endl;

    }
    file.close();
}

static
void setCameraMatrixFreiburg1(float& fx, float& fy, float& cx, float& cy)
{
    fx = 517.3f; fy = 516.5f; cx = 318.6f; cy = 255.3f;
}

static
void setCameraMatrixFreiburg2(float& fx, float& fy, float& cx, float& cy)
{
    fx = 520.9f; fy = 521.0f; cx = 325.1f; cy = 249.7f;
}

/*
 * This sample helps to evaluate odometry on TUM datasets and benchmark http://vision.in.tum.de/data/datasets/rgbd-dataset.
 * At this link you can find instructions for evaluation. The sample runs some opencv odometry and saves a camera trajectory
 * to file of format that the benchmark requires. Saved file can be used for online evaluation.
 */
int main(int argc, char** argv)
{
    if(argc != 4)
    {
        cout << "Format: file_with_rgb_depth_pairs trajectory_file odometry_name [Rgbd or ICP or RgbdICP or FastICP]" << endl;
        return -1;
    }

    vector<string> timestamps;
    vector<Mat> Rts;

    const string filename = argv[1];
    ifstream file( filename.c_str() );
    if( !file.is_open() )
        return -1;

    char dlmrt1 = '/';
    char dlmrt2 = '\\';

    size_t pos1 = filename.rfind(dlmrt1);
    size_t pos2 = filename.rfind(dlmrt2);
    size_t pos = pos1 < pos2 ? pos1 : pos2;
    char dlmrt = pos1 < pos2 ? dlmrt1 : dlmrt2;

    string dirname = pos == string::npos ? "" : filename.substr(0, pos) + dlmrt;

    const int timestampLength = 17;
    const int rgbPathLehgth = 17+8;
    const int depthPathLehgth = 17+10;

    float fx = 525.0f, // default
          fy = 525.0f,
          cx = 319.5f,
          cy = 239.5f;
    if(filename.find("freiburg1") != string::npos)
        setCameraMatrixFreiburg1(fx, fy, cx, cy);
    if(filename.find("freiburg2") != string::npos)
        setCameraMatrixFreiburg2(fx, fy, cx, cy);
    Mat cameraMatrix = Mat::eye(3,3,CV_32FC1);
    {
        cameraMatrix.at<float>(0,0) = fx;
        cameraMatrix.at<float>(1,1) = fy;
        cameraMatrix.at<float>(0,2) = cx;
        cameraMatrix.at<float>(1,2) = cy;
    }

    OdometrySettings ods;
    ods.setCameraMatrix(cameraMatrix);
    Odometry odometry;
    String odname = string(argv[3]);
    if (odname == "Rgbd")
        odometry = Odometry(OdometryType::RGB, ods, OdometryAlgoType::COMMON);
    else if (odname == "ICP")
        odometry = Odometry(OdometryType::DEPTH, ods, OdometryAlgoType::COMMON);
    else if (odname == "RgbdICP")
        odometry = Odometry(OdometryType::RGB_DEPTH, ods, OdometryAlgoType::COMMON);
    else if (odname == "FastICP")
        odometry = Odometry(OdometryType::DEPTH, ods, OdometryAlgoType::FAST);
    else
    {
        std::cout << "Can not create Odometry algorithm. Check the passed odometry name." << std::endl;
        return -1;
    }

    OdometryFrame frame_prev, frame_curr;

    TickMeter gtm;
    int count = 0;
    for(int i = 0; !file.eof(); i++)
    {
        string str;
        std::getline(file, str);
        if(str.empty()) break;
        if(str.at(0) == '#') continue; /* comment */

        Mat image, depth;
        // Read one pair (rgb and depth)
        // example: 1305031453.359684 rgb/1305031453.359684.png 1305031453.374112 depth/1305031453.374112.png
#if BILATERAL_FILTER
        TickMeter tm_bilateral_filter;
#endif
        {
            string rgbFilename = str.substr(timestampLength + 1, rgbPathLehgth );
            string timestap = str.substr(0, timestampLength);
            string depthFilename = str.substr(2*timestampLength + rgbPathLehgth + 3, depthPathLehgth );

            image = imread(dirname + rgbFilename);
            depth = imread(dirname + depthFilename, -1);

            CV_Assert(!image.empty());
            CV_Assert(!depth.empty());
            CV_Assert(depth.type() == CV_16UC1);

            // scale depth
            Mat depth_flt;
            depth.convertTo(depth_flt, CV_32FC1, 1.f/5000.f);
#if !BILATERAL_FILTER
            depth_flt.setTo(std::numeric_limits<float>::quiet_NaN(), depth == 0);
            depth = depth_flt;
#else
            tm_bilateral_filter.start();
            depth = Mat(depth_flt.size(), CV_32FC1, Scalar(0));
            const double depth_sigma = 0.03;
            const double space_sigma = 4.5;  // in pixels
            Mat invalidDepthMask = depth_flt == 0.f;
            depth_flt.setTo(-5*depth_sigma, invalidDepthMask);
            bilateralFilter(depth_flt, depth, -1, depth_sigma, space_sigma);
            depth.setTo(std::numeric_limits<float>::quiet_NaN(), invalidDepthMask);
            tm_bilateral_filter.stop();
            cout << "Time filter " << tm_bilateral_filter.getTimeSec() << endl;
#endif
            timestamps.push_back( timestap );
        }

        {
            Mat gray;
            cvtColor(image, gray, COLOR_BGR2GRAY);
            frame_curr = OdometryFrame(depth, gray);

            Mat Rt;
            if(!Rts.empty())
            {
                TickMeter tm;
                tm.start();
                gtm.start();
                odometry.prepareFrames(frame_curr, frame_prev);
                bool res = odometry.compute(frame_curr, frame_prev, Rt);
                gtm.stop();
                tm.stop();
                count++;
                cout << "Time " << tm.getTimeSec() << endl;
#if BILATERAL_FILTER
                cout << "Time ratio " << tm_bilateral_filter.getTimeSec() / tm.getTimeSec() << endl;
#endif
                if(!res)
                    Rt = Mat::eye(4,4,CV_64FC1);
            }

            if( Rts.empty() )
                Rts.push_back(Mat::eye(4,4,CV_64FC1));
            else
            {
                Mat& prevRt = *Rts.rbegin();
                cout << "Rt " << Rt << endl;
                Rts.push_back( prevRt * Rt );
            }

            //if (!frame_prev.empty())
            //    frame_prev.release();
            frame_prev = frame_curr;
            frame_curr = OdometryFrame();
            //std::swap(frame_prev, frame_curr);
        }
    }

    std::cout << "Average time " << gtm.getAvgTimeSec() << std::endl;
    writeResults(argv[2], timestamps, Rts);

    return 0;
}
