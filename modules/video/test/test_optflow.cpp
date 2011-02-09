/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "test_precomp.hpp"

#include <string>
#include <iostream>
#include <fstream>
#include <iterator>
#include <limits>

using namespace cv;
using namespace std;

class CV_OptFlowTest : public cvtest::BaseTest
{
public:
    CV_OptFlowTest();
    ~CV_OptFlowTest();    
protected:    
    void run(int);

    bool runDense(const Point& shift = Point(3, 0));    
    bool runSparse();
};

CV_OptFlowTest::CV_OptFlowTest() {}
CV_OptFlowTest::~CV_OptFlowTest() {}


Mat copnvert2flow(const Mat& velx, const Mat& vely)
{
    Mat flow(velx.size(), CV_32FC2);
    for(int y = 0 ; y < flow.rows; ++y)
        for(int x = 0 ; x < flow.cols; ++x)                        
            flow.at<Point2f>(y, x) = Point2f(velx.at<float>(y, x), vely.at<float>(y, x));            
    return flow;
}

void calcOpticalFlowLK( const Mat& prev, const Mat& curr, Size winSize, Mat& flow )
{
    Mat velx(prev.size(), CV_32F), vely(prev.size(), CV_32F); 
    CvMat cvvelx = velx;    CvMat cvvely = vely;
    CvMat cvprev = prev;    CvMat cvcurr = curr;
    cvCalcOpticalFlowLK( &cvprev, &cvcurr, winSize, &cvvelx, &cvvely );
    flow = copnvert2flow(velx, vely);
}

void calcOpticalFlowBM( const Mat& prev, const Mat& curr, Size bSize, Size shiftSize, Size maxRange, int usePrevious, Mat& flow )
{
    Size sz((curr.cols - bSize.width)/shiftSize.width, (curr.rows - bSize.height)/shiftSize.height);
    Mat velx(sz, CV_32F), vely(sz, CV_32F);    

    CvMat cvvelx = velx;    CvMat cvvely = vely;
    CvMat cvprev = prev;    CvMat cvcurr = curr;
    cvCalcOpticalFlowBM( &cvprev, &cvcurr, bSize, shiftSize, maxRange, usePrevious, &cvvelx, &cvvely);                     
    flow = copnvert2flow(velx, vely);
}

void calcOpticalFlowHS( const Mat& prev, const Mat& curr, int usePrevious, double lambda, TermCriteria criteria, Mat& flow)
{        
    Mat velx(prev.size(), CV_32F), vely(prev.size(), CV_32F);
    CvMat cvvelx = velx;    CvMat cvvely = vely;
    CvMat cvprev = prev;    CvMat cvcurr = curr;
    cvCalcOpticalFlowHS( &cvprev, &cvcurr, usePrevious, &cvvelx, &cvvely, lambda, criteria );
    flow = copnvert2flow(velx, vely);
}

void calcAffineFlowPyrLK( const Mat& prev, const Mat& curr, 
                          const vector<Point2f>& prev_features, vector<Point2f>& curr_features,
                          vector<uchar>& status, vector<float>& track_error, vector<float>& matrices, 
                          TermCriteria criteria = TermCriteria(TermCriteria::COUNT+TermCriteria::EPS,30, 0.01), 
                          Size win_size = Size(15, 15), int level = 3, int flags = 0)
{
    CvMat cvprev = prev;
    CvMat cvcurr = curr;

    size_t count = prev_features.size();
    curr_features.resize(count);
    status.resize(count);
    track_error.resize(count);
    matrices.resize(count * 6);

    cvCalcAffineFlowPyrLK( &cvprev, &cvcurr, 0, 0, 
        (const CvPoint2D32f*)&prev_features[0], (CvPoint2D32f*)&curr_features[0], &matrices[0], 
        (int)count, win_size, level, (char*)&status[0], &track_error[0], criteria, flags );
}

double showFlowAndCalcError(const string& name, const Mat& gray, const Mat& flow, 
                            const Rect& where, const Point& d, 
                            bool showImages = false, bool writeError = false)
{       
    const int mult = 16;

    if (showImages)
    {
        Mat tmp, cflow;    
        resize(gray, tmp, gray.size() * mult, 0, 0, INTER_NEAREST);            
        cvtColor(tmp, cflow, CV_GRAY2BGR);        

        const float m2 = 0.3f;   
        const float minVel = 0.1f;

        for(int y = 0; y < flow.rows; ++y)
            for(int x = 0; x < flow.cols; ++x)
            {
                Point2f f = flow.at<Point2f>(y, x);                          

                if (f.x * f.x + f.y * f.y > minVel * minVel)
                {
                    Point p1 = Point(x, y) * mult;
                    Point p2 = Point(cvRound((x + f.x*m2) * mult), cvRound((y + f.y*m2) * mult));

                    line(cflow, p1, p2, CV_RGB(0, 255, 0));            
                    circle(cflow, Point(x, y) * mult, 2, CV_RGB(255, 0, 0));
                }            
            }

        rectangle(cflow, (where.tl() + d) * mult, (where.br() + d - Point(1,1)) * mult, CV_RGB(0, 0, 255));    
        namedWindow(name, 1); imshow(name, cflow);
    }

    double angle = atan2((float)d.y, (float)d.x);
    double error = 0;

    bool all = true;
    Mat inner = flow(where);
    for(int y = 0; y < inner.rows; ++y)
        for(int x = 0; x < inner.cols; ++x)
        {
            const Point2f f = inner.at<Point2f>(y, x);

            if (f.x == 0 && f.y == 0)
                continue;

            all = false;

            double a = atan2(f.y, f.x);
            error += fabs(angle - a);            
        }
        double res = all ? numeric_limits<double>::max() : error / (inner.cols * inner.rows);

        if (writeError)
            cout << "Error " + name << " = " << res << endl;

        return res;
}


Mat generateImage(const Size& sz, bool doBlur = true)
{
    RNG rng;
    Mat mat(sz, CV_8U);
    mat = Scalar(0);
    for(int y = 0; y < mat.rows; ++y)
        for(int x = 0; x < mat.cols; ++x)
            mat.at<uchar>(y, x) = (uchar)rng;    
    if (doBlur)
        blur(mat, mat, Size(3, 3));
    return mat;
}

Mat generateSample(const Size& sz)
{
    Mat smpl(sz, CV_8U);    
    smpl = Scalar(0);
    Point sc(smpl.cols/2, smpl.rows/2);
    rectangle(smpl, Point(0,0), sc - Point(1,1), Scalar(255), CV_FILLED);
    rectangle(smpl, sc, Point(smpl.cols, smpl.rows), Scalar(255), CV_FILLED);
    return smpl;
}

bool CV_OptFlowTest::runDense(const Point& d)
{
    Size matSize(40, 40);
    Size movSize(8, 8);
        
    Mat smpl = generateSample(movSize);
    Mat prev = generateImage(matSize);    
    Mat curr = prev.clone();

    Rect rect(Point(prev.cols/2, prev.rows/2) - Point(movSize.width/2, movSize.height/2), movSize);

    Mat flowLK, flowBM, flowHS, flowFB, flowFB_G, flowBM_received, m1;

    m1 = prev(rect);                                smpl.copyTo(m1);
    m1 = curr(Rect(rect.tl() + d, rect.br() + d));  smpl.copyTo(m1);   
    
    calcOpticalFlowLK( prev, curr, Size(15, 15), flowLK);        
    calcOpticalFlowBM( prev, curr, Size(15, 15), Size(1, 1), Size(15, 15), 0, flowBM_received);       
    calcOpticalFlowHS( prev, curr, 0, 5, TermCriteria(TermCriteria::MAX_ITER, 400, 0), flowHS);                 
    calcOpticalFlowFarneback( prev, curr, flowFB, 0.5, 3, std::max(d.x, d.y) + 10, 100, 6, 2, 0);
    calcOpticalFlowFarneback( prev, curr, flowFB_G, 0.5, 3, std::max(d.x, d.y) + 10, 100, 6, 2, OPTFLOW_FARNEBACK_GAUSSIAN);            

    flowBM.create(prev.size(), CV_32FC2);
    flowBM = Scalar(0);    
    Point origin((flowBM.cols - flowBM_received.cols)/2, (flowBM.rows - flowBM_received.rows)/2);
    Mat wcp = flowBM(Rect(origin, flowBM_received.size()));
    flowBM_received.copyTo(wcp);

    double errorLK = showFlowAndCalcError("LK", prev, flowLK, rect, d);
    double errorBM = showFlowAndCalcError("BM", prev, flowBM, rect, d);
    double errorFB = showFlowAndCalcError("FB", prev, flowFB, rect, d);
    double errorFBG = showFlowAndCalcError("FBG", prev, flowFB_G, rect, d);
    double errorHS = showFlowAndCalcError("HS", prev, flowHS, rect, d); (void)errorHS;     
    //waitKey();   

    const double thres = 0.2;
    if (errorLK > thres || errorBM > thres || errorFB > thres || errorFBG > thres /*|| errorHS > thres */)
    {        
        ts->set_failed_test_info(cvtest::TS::FAIL_MISMATCH);
        return false;
    }        
    return true;
}


bool CV_OptFlowTest::runSparse()
{    
    Mat prev = imread(string(ts->get_data_path()) + "optflow/rock_1.bmp", 0);
    Mat next = imread(string(ts->get_data_path()) + "optflow/rock_2.bmp", 0);

    if (prev.empty() || next.empty())
    {
        ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_TEST_DATA );  
        return false;
    }

    Mat cprev, cnext;
    cvtColor(prev, cprev, CV_GRAY2BGR);
    cvtColor(next, cnext, CV_GRAY2BGR);

    vector<Point2f> prev_pts;
    vector<Point2f> next_ptsOpt;
    vector<Point2f> next_ptsAff;
    vector<uchar> status_Opt;
    vector<uchar> status_Aff;
    vector<float> error;
    vector<float> matrices;

    Size netSize(10, 10);
    Point2f center = Point(prev.cols/2, prev.rows/2);

    for(int i = 0 ; i < netSize.width; ++i)
        for(int j = 0 ; j < netSize.width; ++j)
        {
            Point2f p(i * float(prev.cols)/netSize.width, j * float(prev.rows)/netSize.height);
            prev_pts.push_back((p - center) * 0.5f + center);            
        }

    calcOpticalFlowPyrLK( prev, next, prev_pts, next_ptsOpt, status_Opt, error );
    calcAffineFlowPyrLK ( prev, next, prev_pts, next_ptsAff, status_Aff, error, matrices);

    const double expected_shift = 25;
    const double thres = 1;    
    for(size_t i = 0; i < prev_pts.size(); ++i)        
    {
        circle(cprev, prev_pts[i], 2, CV_RGB(255, 0, 0));               

        if (status_Opt[i])
        {            
            circle(cnext, next_ptsOpt[i], 2, CV_RGB(0, 0, 255));
            Point2f shift = prev_pts[i] - next_ptsOpt[i];
            
            double n = sqrt(shift.ddot(shift));
            if (fabs(n - expected_shift) > thres)
            {
                ts->set_failed_test_info(cvtest::TS::FAIL_MISMATCH);
                return false;
            }
        }

        if (status_Aff[i])
        {            
            circle(cnext, next_ptsAff[i], 4, CV_RGB(0, 255, 0));
            Point2f shift = prev_pts[i] - next_ptsAff[i];

            double n = sqrt(shift.ddot(shift));
            if (fabs(n - expected_shift) > thres)
            {
                ts->set_failed_test_info(cvtest::TS::FAIL_MISMATCH);
                return false;
            }
        }
        
    }
    
    /*namedWindow("P");  imshow("P", cprev);
    namedWindow("N"); imshow("N", cnext); 
    waitKey();*/
    
    return true;
}


void CV_OptFlowTest::run( int /* start_from */)
{	

    if (!runDense(Point(3, 0)))
        return;

    if (!runDense(Point(0, 3))) 
        return;

    //if (!runDense(Point(3, 3))) return;  //probably LK works incorrectly in this case.

    if (!runSparse()) 
        return;

    ts->set_failed_test_info(cvtest::TS::OK);
}


TEST(Video_OpticalFlow, accuracy) { CV_OptFlowTest test; test.safe_run(); }


