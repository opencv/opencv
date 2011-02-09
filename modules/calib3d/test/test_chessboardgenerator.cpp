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
#include "test_chessboardgenerator.hpp"

#include <vector>
#include <iterator>
#include <algorithm>

using namespace cv;
using namespace std;

ChessBoardGenerator::ChessBoardGenerator(const Size& _patternSize) : sensorWidth(32), sensorHeight(24),
    squareEdgePointsNum(200), min_cos(sqrt(2.f)*0.5f), cov(0.5), 
    patternSize(_patternSize), rendererResolutionMultiplier(4), tvec(Mat::zeros(1, 3, CV_32F))
{    
    Rodrigues(Mat::eye(3, 3, CV_32F), rvec);
}

void cv::ChessBoardGenerator::generateEdge(const Point3f& p1, const Point3f& p2, vector<Point3f>& out) const
{    
    Point3f step = (p2 - p1) * (1.f/squareEdgePointsNum);    
    for(size_t n = 0; n < squareEdgePointsNum; ++n)
        out.push_back( p1 + step * (float)n);
}    

Size cv::ChessBoardGenerator::cornersSize() const
{
    return Size(patternSize.width-1, patternSize.height-1);
}

struct Mult
{
    float m;
    Mult(int mult) : m((float)mult) {}
    Point2f operator()(const Point2f& p)const { return p * m; }    
};

void cv::ChessBoardGenerator::generateBasis(Point3f& pb1, Point3f& pb2) const
{
    RNG& rng = theRNG();

    Vec3f n;
    for(;;)
    {        
        n[0] = rng.uniform(-1.f, 1.f);
        n[1] = rng.uniform(-1.f, 1.f);
        n[2] = rng.uniform(-1.f, 1.f);        
        float len = (float)norm(n);    
        n[0]/=len;  
        n[1]/=len;  
        n[2]/=len;
        
        if (n[2] > min_cos)
            break;
    }

    Vec3f n_temp = n; n_temp[0] += 100;
    Vec3f b1 = n.cross(n_temp); 
    Vec3f b2 = n.cross(b1);
    float len_b1 = (float)norm(b1);
    float len_b2 = (float)norm(b2);    

    pb1 = Point3f(b1[0]/len_b1, b1[1]/len_b1, b1[2]/len_b1);
    pb2 = Point3f(b2[0]/len_b1, b2[1]/len_b2, b2[2]/len_b2);
}


Mat cv::ChessBoardGenerator::generateChessBoard(const Mat& bg, const Mat& camMat, const Mat& distCoeffs, 
                                                const Point3f& zero, const Point3f& pb1, const Point3f& pb2, 
                                                float sqWidth, float sqHeight, const vector<Point3f>& whole,
                                                vector<Point2f>& corners) const
{    
    vector< vector<Point> > squares_black;    
    for(int i = 0; i < patternSize.width; ++i)
        for(int j = 0; j < patternSize.height; ++j)
            if ( (i % 2 == 0 && j % 2 == 0) || (i % 2 != 0 && j % 2 != 0) ) 
            {            
                vector<Point3f> pts_square3d;
                vector<Point2f> pts_square2d;

                Point3f p1 = zero + (i + 0) * sqWidth * pb1 + (j + 0) * sqHeight * pb2;
                Point3f p2 = zero + (i + 1) * sqWidth * pb1 + (j + 0) * sqHeight * pb2;
                Point3f p3 = zero + (i + 1) * sqWidth * pb1 + (j + 1) * sqHeight * pb2;
                Point3f p4 = zero + (i + 0) * sqWidth * pb1 + (j + 1) * sqHeight * pb2;
                generateEdge(p1, p2, pts_square3d);
                generateEdge(p2, p3, pts_square3d);
                generateEdge(p3, p4, pts_square3d);
                generateEdge(p4, p1, pts_square3d);  
                
                projectPoints(Mat(pts_square3d), rvec, tvec, camMat, distCoeffs, pts_square2d);
                squares_black.resize(squares_black.size() + 1);  
                vector<Point2f> temp; 
                approxPolyDP(Mat(pts_square2d), temp, 1.0, true); 
                transform(temp.begin(), temp.end(), back_inserter(squares_black.back()), Mult(rendererResolutionMultiplier));             
            }  

    /* calculate corners */    
    corners3d.clear();
    for(int j = 0; j < patternSize.height - 1; ++j)
        for(int i = 0; i < patternSize.width - 1; ++i)
            corners3d.push_back(zero + (i + 1) * sqWidth * pb1 + (j + 1) * sqHeight * pb2);
    corners.clear();
    projectPoints(Mat(corners3d), rvec, tvec, camMat, distCoeffs, corners);
 
    vector<Point3f> whole3d;
    vector<Point2f> whole2d;
    generateEdge(whole[0], whole[1], whole3d);
    generateEdge(whole[1], whole[2], whole3d);
    generateEdge(whole[2], whole[3], whole3d);
    generateEdge(whole[3], whole[0], whole3d);
    projectPoints(Mat(whole3d), rvec, tvec, camMat, distCoeffs, whole2d);
    vector<Point2f> temp_whole2d; 
    approxPolyDP(Mat(whole2d), temp_whole2d, 1.0, true); 

    vector< vector<Point > > whole_contour(1);
    transform(temp_whole2d.begin(), temp_whole2d.end(), 
        back_inserter(whole_contour.front()), Mult(rendererResolutionMultiplier));    

    Mat result;
    if (rendererResolutionMultiplier == 1)
    {        
        result = bg.clone();
        drawContours(result, whole_contour, -1, Scalar::all(255), CV_FILLED, CV_AA);       
        drawContours(result, squares_black, -1, Scalar::all(0), CV_FILLED, CV_AA);
    }
    else
    {
        Mat tmp;        
        resize(bg, tmp, bg.size() * rendererResolutionMultiplier);
        drawContours(tmp, whole_contour, -1, Scalar::all(255), CV_FILLED, CV_AA);       
        drawContours(tmp, squares_black, -1, Scalar::all(0), CV_FILLED, CV_AA);
        resize(tmp, result, bg.size(), 0, 0, INTER_AREA);
    }        

    return result;
}

Mat cv::ChessBoardGenerator::operator ()(const Mat& bg, const Mat& camMat, const Mat& distCoeffs, vector<Point2f>& corners) const
{      
    cov = min(cov, 0.8);
    double fovx, fovy, focalLen;
    Point2d principalPoint;
    double aspect;
    calibrationMatrixValues( camMat, bg.size(), sensorWidth, sensorHeight, 
        fovx, fovy, focalLen, principalPoint, aspect);

    RNG& rng = theRNG();

    float d1 = static_cast<float>(rng.uniform(0.1, 10.0)); 
    float ah = static_cast<float>(rng.uniform(-fovx/2 * cov, fovx/2 * cov) * CV_PI / 180);
    float av = static_cast<float>(rng.uniform(-fovy/2 * cov, fovy/2 * cov) * CV_PI / 180);        
    
    Point3f p;
    p.z = cos(ah) * d1;
    p.x = sin(ah) * d1;
    p.y = p.z * tan(av);  

    Point3f pb1, pb2;    
    generateBasis(pb1, pb2);
            
    float cbHalfWidth = static_cast<float>(norm(p) * sin( min(fovx, fovy) * 0.5 * CV_PI / 180));
    float cbHalfHeight = cbHalfWidth * patternSize.height / patternSize.width;    

    float cbHalfWidthEx  =  cbHalfWidth * ( patternSize.width + 1) / patternSize.width;
    float cbHalfHeightEx = cbHalfHeight * (patternSize.height + 1) / patternSize.height;
    
    vector<Point3f> pts3d(4);
    vector<Point2f> pts2d(4);
    for(;;)
    {        
        pts3d[0] = p + pb1 * cbHalfWidthEx + cbHalfHeightEx * pb2;
        pts3d[1] = p + pb1 * cbHalfWidthEx - cbHalfHeightEx * pb2;
        pts3d[2] = p - pb1 * cbHalfWidthEx - cbHalfHeightEx * pb2;
        pts3d[3] = p - pb1 * cbHalfWidthEx + cbHalfHeightEx * pb2;
        
        /* can remake with better perf */
        projectPoints(Mat(pts3d), rvec, tvec, camMat, distCoeffs, pts2d);

        bool inrect1 = pts2d[0].x < bg.cols && pts2d[0].y < bg.rows && pts2d[0].x > 0 && pts2d[0].y > 0;
        bool inrect2 = pts2d[1].x < bg.cols && pts2d[1].y < bg.rows && pts2d[1].x > 0 && pts2d[1].y > 0;
        bool inrect3 = pts2d[2].x < bg.cols && pts2d[2].y < bg.rows && pts2d[2].x > 0 && pts2d[2].y > 0;
        bool inrect4 = pts2d[3].x < bg.cols && pts2d[3].y < bg.rows && pts2d[3].x > 0 && pts2d[3].y > 0;
        
        if (inrect1 && inrect2 && inrect3 && inrect4)
            break;

        cbHalfWidth*=0.8f;
        cbHalfHeight = cbHalfWidth * patternSize.height / patternSize.width;   
        
        cbHalfWidthEx  = cbHalfWidth * ( patternSize.width + 1) / patternSize.width;
        cbHalfHeightEx = cbHalfHeight * (patternSize.height + 1) / patternSize.height;
    }
    
    Point3f zero = p - pb1 * cbHalfWidth - cbHalfHeight * pb2;
    float sqWidth  = 2 * cbHalfWidth/patternSize.width;
    float sqHeight = 2 * cbHalfHeight/patternSize.height;
        
    return generateChessBoard(bg, camMat, distCoeffs, zero, pb1, pb2, sqWidth, sqHeight,  pts3d, corners);      
}


Mat cv::ChessBoardGenerator::operator ()(const Mat& bg, const Mat& camMat, const Mat& distCoeffs, 
                                         const Size2f& squareSize, vector<Point2f>& corners) const
{        
    cov = min(cov, 0.8);
    double fovx, fovy, focalLen;
    Point2d principalPoint;
    double aspect;
    calibrationMatrixValues( camMat, bg.size(), sensorWidth, sensorHeight, 
        fovx, fovy, focalLen, principalPoint, aspect);

    RNG& rng = theRNG();

    float d1 = static_cast<float>(rng.uniform(0.1, 10.0)); 
    float ah = static_cast<float>(rng.uniform(-fovx/2 * cov, fovx/2 * cov) * CV_PI / 180);
    float av = static_cast<float>(rng.uniform(-fovy/2 * cov, fovy/2 * cov) * CV_PI / 180);        
    
    Point3f p;
    p.z = cos(ah) * d1;
    p.x = sin(ah) * d1;
    p.y = p.z * tan(av);  

    Point3f pb1, pb2;    
    generateBasis(pb1, pb2);   

    float cbHalfWidth  =  squareSize.width *  patternSize.width * 0.5f;
    float cbHalfHeight = squareSize.height * patternSize.height * 0.5f;

    float cbHalfWidthEx  =  cbHalfWidth * ( patternSize.width + 1) / patternSize.width;
    float cbHalfHeightEx = cbHalfHeight * (patternSize.height + 1) / patternSize.height;
    
    vector<Point3f> pts3d(4);
    vector<Point2f> pts2d(4);
    for(;;)
    {        
        pts3d[0] = p + pb1 * cbHalfWidthEx + cbHalfHeightEx * pb2;
        pts3d[1] = p + pb1 * cbHalfWidthEx - cbHalfHeightEx * pb2;
        pts3d[2] = p - pb1 * cbHalfWidthEx - cbHalfHeightEx * pb2;
        pts3d[3] = p - pb1 * cbHalfWidthEx + cbHalfHeightEx * pb2;
        
        /* can remake with better perf */
        projectPoints(Mat(pts3d), rvec, tvec, camMat, distCoeffs, pts2d);

        bool inrect1 = pts2d[0].x < bg.cols && pts2d[0].y < bg.rows && pts2d[0].x > 0 && pts2d[0].y > 0;
        bool inrect2 = pts2d[1].x < bg.cols && pts2d[1].y < bg.rows && pts2d[1].x > 0 && pts2d[1].y > 0;
        bool inrect3 = pts2d[2].x < bg.cols && pts2d[2].y < bg.rows && pts2d[2].x > 0 && pts2d[2].y > 0;
        bool inrect4 = pts2d[3].x < bg.cols && pts2d[3].y < bg.rows && pts2d[3].x > 0 && pts2d[3].y > 0;
        
        if ( inrect1 && inrect2 && inrect3 && inrect4)
            break;

        p.z *= 1.1f;
    }
    
    Point3f zero = p - pb1 * cbHalfWidth - cbHalfHeight * pb2;
            
    return generateChessBoard(bg, camMat, distCoeffs, zero, pb1, pb2, 
        squareSize.width, squareSize.height, pts3d, corners);      
}

Mat cv::ChessBoardGenerator::operator ()(const Mat& bg, const Mat& camMat, const Mat& distCoeffs, 
                                         const Size2f& squareSize, const Point3f& pos, vector<Point2f>& corners) const
{           
    cov = min(cov, 0.8);
    Point3f p = pos;    
    Point3f pb1, pb2;    
    generateBasis(pb1, pb2);   

    float cbHalfWidth  =  squareSize.width *  patternSize.width * 0.5f;
    float cbHalfHeight = squareSize.height * patternSize.height * 0.5f;

    float cbHalfWidthEx  =  cbHalfWidth * ( patternSize.width + 1) / patternSize.width;
    float cbHalfHeightEx = cbHalfHeight * (patternSize.height + 1) / patternSize.height;
    
    vector<Point3f> pts3d(4);
    vector<Point2f> pts2d(4);
      
    pts3d[0] = p + pb1 * cbHalfWidthEx + cbHalfHeightEx * pb2;
    pts3d[1] = p + pb1 * cbHalfWidthEx - cbHalfHeightEx * pb2;
    pts3d[2] = p - pb1 * cbHalfWidthEx - cbHalfHeightEx * pb2;
    pts3d[3] = p - pb1 * cbHalfWidthEx + cbHalfHeightEx * pb2;
        
    /* can remake with better perf */
    projectPoints(Mat(pts3d), rvec, tvec, camMat, distCoeffs, pts2d);
    
    Point3f zero = p - pb1 * cbHalfWidth - cbHalfHeight * pb2;
            
    return generateChessBoard(bg, camMat, distCoeffs, zero, pb1, pb2, 
        squareSize.width, squareSize.height,  pts3d, corners);      
}

