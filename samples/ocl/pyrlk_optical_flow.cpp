#include <iostream>
#include <vector>
#include <iomanip>

//#include "opencv2/core/opengl_interop.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/ocl/ocl.hpp"
#include "opencv2/video/video.hpp"

using namespace std;
using namespace cv;
using namespace cv::ocl;

typedef unsigned char uchar;

static void download(const oclMat& d_mat, vector<Point2f>& vec)
{
    vec.resize(d_mat.cols);
    Mat mat(1, d_mat.cols, CV_32FC2, (void*)&vec[0]);
    d_mat.download(mat);
}

static void download(const oclMat& d_mat, vector<uchar>& vec)
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

static void getFlowField(const Mat& u, const Mat& v, Mat& flowField)
{
    float maxDisplacement = 1.0f;

    for (int i = 0; i < u.rows; ++i)
    {
        const float* ptr_u = u.ptr<float>(i);
        const float* ptr_v = v.ptr<float>(i);

        for (int j = 0; j < u.cols; ++j)
        {
            float d = max(fabsf(ptr_u[j]), fabsf(ptr_v[j]));

            if (d > maxDisplacement)
                maxDisplacement = d;
        }
    }

    flowField.create(u.size(), CV_8UC4);

    for (int i = 0; i < flowField.rows; ++i)
    {
        const float* ptr_u = u.ptr<float>(i);
        const float* ptr_v = v.ptr<float>(i);


        Vec4b* row = flowField.ptr<Vec4b>(i);

        for (int j = 0; j < flowField.cols; ++j)
        {
            row[j][0] = 0;
            row[j][1] = static_cast<unsigned char> (mapValue (-ptr_v[j], -maxDisplacement, maxDisplacement, 0.0f, 255.0f));
            row[j][2] = static_cast<unsigned char> (mapValue ( ptr_u[j], -maxDisplacement, maxDisplacement, 0.0f, 255.0f));
            row[j][3] = 255;
        }
    }
}

int main(int argc, const char* argv[])
{
    static std::vector<Info> ocl_info;
    ocl::getDevice(ocl_info);
    //if you want to use undefault device, set it here
    //setDevice(oclinfo[0]);

    //set this to save kernel compile time from second time you run
    ocl::setBinpath("./");
    const char* keys =
        "{ h            | help           | false | print help message }"
        "{ l            | left           |       | specify left image }"
        "{ r            | right          |       | specify right image }"
        "{ c            | camera         | 0     | enable camera capturing }"
        "{ s            | use_cpu        | false | use cpu or gpu to process the image }"
        "{ v            | video          |       | use video as input }"
        "{ gray         | gray           | false | use grayscale sources [PyrLK Sparse] }"
        "{ win_size     | win_size       | 21    | specify windows size [PyrLK] }"
        "{ max_level    | max_level      | 3     | specify max level [PyrLK] }"
        "{ iters        | iters          | 30    | specify iterations count [PyrLK] }"
        "{ points       | points         | 1000  | specify points count [GoodFeatureToTrack] }"
        "{ min_dist     | min_dist       | 0     | specify minimal distance between points [GoodFeatureToTrack] }";

    CommandLineParser cmd(argc, argv, keys);

    if (cmd.get<bool>("help"))
    {
        cout << "Usage: pyrlk_optical_flow [options]" << endl;
        cout << "Avaible options:" << endl;
        cmd.printParams();
        return 0;
    }

    bool defaultPicturesFail = false;
    string fname0 = cmd.get<string>("left");
    string fname1 = cmd.get<string>("right");
    string vdofile = cmd.get<string>("video");

    bool useGray = cmd.get<bool>("gray");
    int winSize = cmd.get<int>("win_size");
    int maxLevel = cmd.get<int>("max_level");
    int iters = cmd.get<int>("iters");
    int points = cmd.get<int>("points");
    double minDist = cmd.get<double>("min_dist");
    bool useCPU = cmd.get<bool>("s");
    bool useCamera = cmd.get<bool>("c");
    int inputName = cmd.get<int>("c");

    (void)minDist;
    if (fname0.empty() || fname1.empty() || vdofile != "")
    {
        fname0 = "rubberwhale1.png";
        fname1 = "rubberwhale2.png";
        useCamera = true;
    }

    Mat frame0 = imread(fname0, useGray ? cv::IMREAD_GRAYSCALE : cv::IMREAD_COLOR);
    Mat frame1 = imread(fname1, useGray ? cv::IMREAD_GRAYSCALE : cv::IMREAD_COLOR);

    if (frame0.empty() || frame1.empty())
    {
        useCamera = true;
        defaultPicturesFail = true;
        CvCapture* capture = 0;
        capture = cvCaptureFromCAM( inputName );
        if (!capture)
        {
            cout << "Can't load input images" << endl;
            return -1;
        }
    }

    cout << "Points count : " << points << endl << endl;

    PyrLKOpticalFlow d_pyrLK;
    std::vector<cv::Point2f> pts;

    Mat frame0Gray;
    Mat frame1Gray;

    if (useCamera)
    {
        CvCapture* capture = 0;
        Mat frame, frameCopy, image;

        if(vdofile == "")
            capture = cvCaptureFromCAM( inputName );
        else
            capture = cvCreateFileCapture(vdofile.c_str());

        int c = inputName ;
        if(!capture)
        {
            if(vdofile == "")
                cout << "Capture from CAM " << c << " didn't work" << endl;
            else
                cout << "Capture from file " << vdofile << " failed" <<endl;
            if (defaultPicturesFail)
            {
                return -1;
            }
            goto nocamera;
        }

        cout << "In capture ..." << endl;
        for(int i = 0;; i++)
        {
            frame = cvQueryFrame( capture );
            if( frame.empty() )
                break;

            if (i == 0)
            {
                frame.copyTo( frame0 );
            }
            else
            {
                std::vector<cv::Point2f> nextPts_gold;
                std::vector<unsigned char> status_gold;
                std::vector<float> err_gold;

                if (i % 2 == 1)
                {
                    frame.copyTo( frame1 );
                }
                else
                {
                    frame.copyTo( frame0 );
                }

                cvtColor(frame0, frame0Gray, COLOR_BGR2GRAY);
                cvtColor(frame1, frame1Gray, COLOR_BGR2GRAY);

                pts.clear();
                cv::goodFeaturesToTrack(frame0Gray, pts, points, 0.01, 0.0);

                if (pts.size() == 0)
                {
                    continue;
                }

                if (useCPU)
                {
                    if (i % 2 == 1)
                    {
                        cv::calcOpticalFlowPyrLK(frame0, frame1, pts, nextPts_gold, status_gold, err_gold);
                    }
                    else
                    {
                        cv::calcOpticalFlowPyrLK(frame1, frame0, pts, nextPts_gold, status_gold, err_gold);
                    }

                    frame0.copyTo(frameCopy);
                    drawArrows(frameCopy, pts, nextPts_gold, status_gold, Scalar(255, 0, 0));

                    imshow("calcOpticalFlowPyrLK [Sparse]", frameCopy);
                }
                else
                {
                    oclMat d_prevPts;

                    cv::Mat pts_mat(1, (int)pts.size(), CV_32FC2, (void*)&pts[0]);
                    d_prevPts.upload(pts_mat);

                    d_pyrLK.winSize.width = winSize;
                    d_pyrLK.winSize.height = winSize;
                    d_pyrLK.maxLevel = maxLevel;
                    d_pyrLK.iters = iters;

                    oclMat d_frame0Gray(frame0Gray);
                    oclMat d_frame1Gray(frame1Gray);

                    oclMat d_frame0(frame0);
                    oclMat d_frame1(frame1);
                    oclMat d_nextPts;
                    oclMat d_status;

                    if (i % 2 == 1)
                    {
                        d_pyrLK.sparse(useGray ? d_frame0Gray : d_frame0, useGray ? d_frame1Gray : d_frame1, d_prevPts, d_nextPts, d_status);
                    }
                    else
                    {
                        d_pyrLK.sparse(useGray ? d_frame1Gray : d_frame1, useGray ? d_frame0Gray : d_frame0, d_prevPts, d_nextPts, d_status);
                    }

                    // Draw arrows

                    vector<Point2f> prevPts(d_prevPts.cols);
                    download(d_prevPts, prevPts);

                    vector<Point2f> nextPts(d_nextPts.cols);
                    download(d_nextPts, nextPts);

                    vector<uchar> status(d_status.cols);
                    download(d_status, status);

                    frame0.copyTo(frameCopy);
                    drawArrows(frameCopy, prevPts, nextPts, status, Scalar(255, 0, 0));

                    imshow("PyrLK [Sparse]", frameCopy);
                }
            }

            if( waitKey( 10 ) >= 0 )
                goto _cleanup_;
        }

        waitKey(0);

_cleanup_:
        cvReleaseCapture( &capture );
    }
    else
    {
nocamera:
        if (!useGray)
        {
            cvtColor(frame0, frame0Gray, COLOR_BGR2GRAY);
            cvtColor(frame1, frame1Gray, COLOR_BGR2GRAY);
        }
        else
        {
            frame0.copyTo(frame0Gray);
            frame1.copyTo(frame1Gray);
        }

        cv::goodFeaturesToTrack(frame0Gray, pts, points, 0.01, 0.0);

        if (useCPU)
        {
            std::vector<cv::Point2f> nextPts_gold;
            std::vector<unsigned char> status_gold;
            std::vector<float> err_gold;
            cv::calcOpticalFlowPyrLK(frame0, frame1, pts, nextPts_gold, status_gold, err_gold);

            imshow("calcOpticalFlowPyrLK [Sparse]", frame0);

            drawArrows(frame0, pts, nextPts_gold, status_gold, Scalar(255, 0, 0));

            imshow("calcOpticalFlowPyrLK [Sparse]", frame0);
        }
        else
        {
            oclMat d_prevPts;

            cv::Mat pts_mat(1, (int)pts.size(), CV_32FC2, (void*)&pts[0]);
            d_prevPts.upload(pts_mat);

            d_pyrLK.winSize.width = winSize;
            d_pyrLK.winSize.height = winSize;
            d_pyrLK.maxLevel = maxLevel;
            d_pyrLK.iters = iters;

            oclMat d_frame0Gray(frame0Gray);
            oclMat d_frame1Gray(frame1Gray);

            oclMat d_frame0(frame0);
            oclMat d_frame1(frame1);
            oclMat d_nextPts;
            oclMat d_status;

            d_pyrLK.sparse(useGray ? d_frame0Gray : d_frame0, useGray ? d_frame1Gray : d_frame1, d_prevPts, d_nextPts, d_status);

            // Draw arrows
            vector<Point2f> prevPts(d_prevPts.cols);
            download(d_prevPts, prevPts);

            vector<Point2f> nextPts(d_nextPts.cols);
            download(d_nextPts, nextPts);

            vector<uchar> status(d_status.cols);
            download(d_status, status);

            imshow("PyrLK [Sparse]", frame0);

            drawArrows(frame0, prevPts, nextPts, status, Scalar(255, 0, 0));

            imshow("PyrLK [Sparse]", frame0);
        }
    }

    if (!useCPU && !useCamera)
    {
        // Dense

        float timeStep = 0.1f;

        cout << "\tForward..." << endl;

        oclMat d_fu, d_fv;

        if (!useGray)
        {
            cvtColor(frame0, frame0Gray, COLOR_BGR2GRAY);
            cvtColor(frame1, frame1Gray, COLOR_BGR2GRAY);
        }
        else
        {
            frame0.copyTo(frame0Gray);
            frame1.copyTo(frame1Gray);
        }

        oclMat d_frame0Gray(frame0Gray);
        oclMat d_frame1Gray(frame1Gray);

        d_pyrLK.dense(d_frame0Gray, d_frame1Gray, d_fu, d_fv);

        Mat flowFieldForward;
        getFlowField(Mat(d_fu), Mat(d_fv), flowFieldForward);

        cout << "\tBackward..." << endl;

        oclMat d_bu, d_bv;

        d_pyrLK.dense(d_frame1Gray, d_frame0Gray, d_bu, d_bv);

        Mat flowFieldBackward;
        getFlowField(Mat(d_bu), Mat(d_bv), flowFieldBackward);

        // first frame color components
        oclMat d_b, d_g, d_r;

        // second frame color components
        oclMat d_bt, d_gt, d_rt;

        // prepare color components on host and copy them to device memory
        Mat channels[3];

        // temporary buffer
        oclMat d_buf;

        // intermediate frame color components (GPU memory)
        oclMat d_rNew, d_gNew, d_bNew;

        oclMat d_newFrame;

        vector<Mat> frames;
        // This interface can't run on Intel OCL now
        if (ocl_info[0].DeviceName[0].find("Intel(R) HD Graphics") == string::npos)
        {
            cv::split(frame0, channels);

            d_b.upload(channels[0]);
            d_g.upload(channels[1]);
            d_r.upload(channels[2]);

            cv::split(frame1, channels);

            d_bt.upload(channels[0]);
            d_gt.upload(channels[1]);
            d_rt.upload(channels[2]);

            frames.reserve(static_cast<int>(1.0f / timeStep) + 2);

            frames.push_back(frame0);

            d_b.convertTo(d_b, CV_32F, 1.0 / 255.0);
            d_g.convertTo(d_g, CV_32F, 1.0 / 255.0);
            d_r.convertTo(d_r, CV_32F, 1.0 / 255.0);

            d_bt.convertTo(d_bt, CV_32F, 1.0 / 255.0);
            d_gt.convertTo(d_gt, CV_32F, 1.0 / 255.0);
            d_rt.convertTo(d_rt, CV_32F, 1.0 / 255.0);

            // compute interpolated frames
            for (float timePos = timeStep; timePos < 1.0f; timePos += timeStep)
            {
                // interpolate blue channel
                interpolateFrames(d_b, d_bt, d_fu, d_fv, d_bu, d_bv, timePos, d_bNew, d_buf);

                // interpolate green channel
                interpolateFrames(d_g, d_gt, d_fu, d_fv, d_bu, d_bv, timePos, d_gNew, d_buf);

                // interpolate red channel
                interpolateFrames(d_r, d_rt, d_fu, d_fv, d_bu, d_bv, timePos, d_rNew, d_buf);

                oclMat dchannels[] = {d_bNew, d_gNew, d_rNew};
                merge(dchannels, 3, d_newFrame);

                frames.push_back(Mat(d_newFrame));

                cout << setprecision(4) << timePos * 100.0f << "%\r";
            }

            frames.push_back(frame1);
        }

        cout << setw(5) << "100%" << endl;

        cout << "Done" << endl;

        imshow("Forward flow", flowFieldForward);
        imshow("Backward flow", flowFieldBackward);

        if (ocl_info[0].DeviceName[0].find("Intel(R) HD Graphics") == string::npos)
        {
            // Draw flow field

            int currentFrame = 0;

            imshow("Interpolated frame", frames[currentFrame]);

            for(;;)
            {
                int key = toupper(waitKey(10) & 0xff);

                switch (key)
                {
                case 27:
                    break;

                case 'A':
                    if (currentFrame > 0)
                        --currentFrame;

                    imshow("Interpolated frame", frames[currentFrame]);
                    break;

                case 'S':
                    if (currentFrame < static_cast<int>(frames.size()) - 1)
                        ++currentFrame;

                    imshow("Interpolated frame", frames[currentFrame]);
                    break;
                }
            }
        }
    }

    waitKey();

    return 0;
}
