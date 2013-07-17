/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
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
//   * The name of Intel Corporation may not be used to endorse or promote products
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

#include "precomp.hpp"

/*
 * Implementation of the paper Shape Matching and Object Recognition Using Shape Contexts
 * Belongie et al., 2002 by Juan Manuel Perez for GSoC 2013. 
 */
namespace cv
{
/* Constructors */
ThinPlateSplineTransform::ThinPlateSplineTransform()
{
    beta=0.0;
}

ThinPlateSplineTransform::ThinPlateSplineTransform(double _beta)
{
    beta=_beta;
}

/* Setters/Getters */
void ThinPlateSplineTransform::setRegularizationParam(double _beta)
{
    beta=_beta;
}

double ThinPlateSplineTransform::getRegularizationParam()
{
    return beta;
}

/****************************************************************************************\
*                                  TPS Class                                            *
\****************************************************************************************/

/* Public methods */
void ThinPlateSplineTransform::applyTransformation(InputArray _pts1, InputArray _pts2,
                                                   std::vector<DMatch>&_matches, std::vector<Point2f>& outPts)
{
    Mat pts1 = _pts1.getMat();
    Mat pts2 = _pts2.getMat();
    CV_Assert((pts1.channels()==2) & (pts1.cols>0) & (pts2.channels()==2) & (pts2.cols>0) & (pts2.cols>=pts1.cols));
    CV_Assert(_matches.size()>1);

    /* Use only valid matchings */
    std::vector<DMatch> matches;
    for (size_t i=0; i<_matches.size(); i++)
    {
        if (_matches[i].queryIdx<(int)pts1.cols &&
            _matches[i].trainIdx<(int)pts2.cols)
        {
            matches.push_back(_matches[i]);
        }
    }

    /* Organizing the correspondent points in matrix style */
    Mat shape1 = Mat::zeros(matches.size(),2,CV_32F); // transforming shape
    Mat shape2 = Mat::zeros(matches.size(),2,CV_32F); // target shape
    for (size_t i=0; i<matches.size(); i++)
    {
        Point2f pt1=pts1.at<Point2f>(0,matches[i].queryIdx);
        shape1.at<float>(i,0) = pt1.x;
        shape1.at<float>(i,1) = pt1.y;

        Point2f pt2=pts2.at<Point2f>(0,matches[i].trainIdx);
        shape2.at<float>(i,0) = pt2.x;
        shape2.at<float>(i,1) = pt2.y;
    }

    /* Building the matrices for solving the L*(w|a)=(v|0) problem
     * with L={[K|P];[P'|0]}
     */
    //Building K
    //K=d2*log(d2)
    Mat matK(matches.size(),matches.size(),CV_32F);
    for (int i=0; i<matK.rows; i++)
    {
        for (int j=0; j<matK.cols; j++)
        {
            if (i==j)
            {
                matK.at<float>(i,j)=1;
                continue;
            }
            matK.at<float>(i,j) = distance(Point2f(shape1.at<float>(i,0),shape1.at<float>(i,1)),
                                           Point2f(shape1.at<float>(j,0),shape1.at<float>(j,1)));
        }
    }

    normalize(matK, matK,0,1, NORM_MINMAX);
    Mat logMatK;
    log(matK,logMatK);
    matK = matK.mul(logMatK);
    matK += Mat::eye(matK.rows, matK.cols, CV_32F)*beta; // Regularization

    //Building P
    //rowi = (1,xi,yi)
    Mat matP(matches.size(),3,CV_32F);
    for (int i=0; i<matP.rows; i++)
    {
        matP.at<float>(i,0) = 1;
        matP.at<float>(i,1) = shape1.at<float>(i,0);
        matP.at<float>(i,2) = shape1.at<float>(i,1);
    }

    //Building B (v|0)
    Mat matB = Mat::zeros(matches.size()+3,2,CV_32F);
    for (size_t i=0; i<matches.size(); i++)
    {
        matB.at<float>(i,0) = shape2.at<float>(i,0); //x's
        matB.at<float>(i,1) = shape2.at<float>(i,1); //y's
    }

    //Building L
    Mat matL;
    Mat up, down=Mat::zeros(3,matches.size()+3,CV_32F);
    std::vector<Mat> matVec;
    matVec.push_back(matK);
    matVec.push_back(matP);
    hconcat(matVec, up);
    matVec.clear();

    for (size_t i=0; i<matches.size(); i++)
    {   //down = P'|0
        down.at<float>(0,i) = 1;
        down.at<float>(1,i) = shape1.at<float>(i,0);
        down.at<float>(2,i) = shape1.at<float>(i,1);
    }
    matVec.push_back(up);
    matVec.push_back(down);
    vconcat(matVec, matL);

    //Obtaining transformation params (w|a)
    Mat matX; //params for fx and fy respectively
    solve(matL, matB, matX, DECOMP_LU);

    //Apply transformation in the complete set of points
    Mat complete_shape1 = Mat::zeros(pts1.cols,2,CV_32F); // transforming shape
    for (int i=0; i<pts1.cols; i++)
    {
        Point2f pt1=pts1.at<Point2f>(0,i);
        complete_shape1.at<float>(i,0) = pt1.x;
        complete_shape1.at<float>(i,1) = pt1.y;
    }

    //Creating U=r^2*log(r^2)
    Mat matU(complete_shape1.rows,complete_shape1.rows,CV_32F);
    for (int i=0; i<matU.rows; i++)
    {
        for (int j=0; j<matU.cols; j++)
        {
            if (i==j)
            {
                matU.at<float>(i,j)=std::numeric_limits<float>::min();
                continue;
            }
            matU.at<float>(i,j) = distance(Point2f(complete_shape1.at<float>(i,0),complete_shape1.at<float>(i,1)),
                                           Point2f(complete_shape1.at<float>(j,0),complete_shape1.at<float>(j,1)));
        }
    }
    normalize(matU, matU,0,1, NORM_MINMAX);
    Mat logMatU;
    log(matU,logMatU);
    matU = matU.mul(logMatU);

    //creating fxAffine
    Mat afparam(1,3,CV_32F);
    afparam.at<float>(0,0)=matX.at<float>(matX.rows-3,0);
    afparam.at<float>(0,1)=matX.at<float>(matX.rows-2,0);
    afparam.at<float>(0,2)=matX.at<float>(matX.rows-1,0);
    Mat auxaf=Mat::ones(3,matU.rows,CV_32F);

    for (int i=0; i<matU.cols; i++)
    {
        auxaf.at<float>(1,i)=complete_shape1.at<float>(i,0);
        auxaf.at<float>(2,i)=complete_shape1.at<float>(i,1);
    }

    Mat fxAffine=afparam*auxaf;
    //creating fxWarp
    Mat warpaux(1,fxAffine.cols,CV_32F);
    for (int i=0; i<warpaux.cols; i++)
    {
        warpaux.at<float>(0,i)=matX.at<float>(i,0);
    }
    Mat fxWarp=warpaux*matU;
    //fx
    Mat fx=fxWarp+fxAffine;

    //Creating fyAffine
    afparam.at<float>(0,0)=matX.at<float>(matX.rows-3,1);
    afparam.at<float>(0,1)=matX.at<float>(matX.rows-2,1);
    afparam.at<float>(0,2)=matX.at<float>(matX.rows-1,1);
    Mat fyAffine=afparam*auxaf;

    //creating fyWarp
    for (int i=0; i<warpaux.cols; i++)
    {
        warpaux.at<float>(0,i)=matX.at<float>(i,1);
    }
    Mat fyWarp=warpaux*matU;

    //fx
    Mat fy=fyWarp+fyAffine;

    //Ensambling output
    outPts.clear();
    for (int i=0; i<fy.cols; i++)
    {
        outPts.push_back(Point2f(fx.at<float>(0,i), fy.at<float>(0,i)));
    }

    //Setting transform Cost
    Mat wt(2,matX.rows-3,CV_32F);
    Mat w;
    for (int i=0; i<wt.cols; i++)
    {
        wt.at<float>(0,i)=matX.at<float>(i,0);
        wt.at<float>(1,i)=matX.at<float>(i,1);
    }
    transpose(wt,w);
    Mat Q=wt*matK*w;
    transformCost=mean(Q.diag(0))[0];
}

/* getters */
float ThinPlateSplineTransform::getTranformCost(void) const
{
    return transformCost;
}

/* private methods */
double ThinPlateSplineTransform::distance(Point2f p, Point2f q) const
{
    Point2f diff = p - q;
    return sqrt(diff.x*diff.x + diff.y*diff.y);
}
/****************************************************************************************\
*                                  Affine Class                                          *
\****************************************************************************************/
/* Constructors */
AffineTransform::AffineTransform()
{
    fullAffine=true;
}

AffineTransform::AffineTransform(bool _fullAffine)
{
    fullAffine=_fullAffine;
}

/* Setters/Getters */
void AffineTransform::setFullAffine(bool _fullAffine)
{
    fullAffine=_fullAffine;
}

bool AffineTransform::getFullAffine()
{
    return fullAffine;
}

float AffineTransform::getTranformCost(void) const
{
    return transformCost;
}

/* Public methods */
void AffineTransform::applyTransformation(InputArray _pts1, InputArray _pts2,
                                          std::vector<DMatch>& _matches, std::vector<Point2f>& outPts)
{
    Mat pts1 = _pts1.getMat();
    Mat pts2 = _pts2.getMat();
    CV_Assert((pts1.channels()==2) & (pts1.cols>0) & (pts2.channels()==2) & (pts2.cols>0));
    CV_Assert(_matches.size()>1);

    /* Use only valid matchings */
    std::vector<DMatch> matches;
    for (size_t i=0; i<_matches.size(); i++)
    {
        if (_matches[i].queryIdx<(int)pts1.cols &&
            _matches[i].trainIdx<(int)pts2.cols)
        {
            matches.push_back(_matches[i]);
        }
    }

    /* Organizing the correspondent points in vector style */
    std::vector<Point2f> shape1; // transforming shape
    std::vector<Point2f> shape2; // target shape
    for (size_t i=0; i<matches.size(); i++)
    {
        Point2f pt1=pts1.at<Point2f>(0,matches[i].queryIdx);
        shape1.push_back(pt1);

        Point2f pt2=pts2.at<Point2f>(0,matches[i].trainIdx);
        shape2.push_back(pt2);
    }

    /* estimateRigidTransform */
    //Apply transformation in the complete set of points
    Mat complete_shape1 = Mat::zeros(pts1.cols,2,CV_32F); // transforming shape
    for (int i=0; i<pts1.cols; i++)
    {
        Point2f pt1=pts1.at<Point2f>(0,i);
        complete_shape1.at<float>(i,0) = pt1.x;
        complete_shape1.at<float>(i,1) = pt1.y;
    }

    Mat affine;
    estimateRigidTransform(shape1, shape2, fullAffine).convertTo(affine, CV_32F);

    Mat auxaf=Mat::ones(3,complete_shape1.rows,CV_32F);
    for (int i=0; i<complete_shape1.rows; i++)
    {
        auxaf.at<float>(0,i)=complete_shape1.at<float>(i,0);
        auxaf.at<float>(1,i)=complete_shape1.at<float>(i,1);
    }

    if (affine.cols==0)
    {   /* add a LLS solution here */
        affine = Mat::ones(2,3,CV_32F);
    }

    Mat fAffine=affine*auxaf;
    /* Ensambling output */
    for (int i=0; i<fAffine.cols; i++)
    {
        outPts.push_back(Point2f(fAffine.at<float>(0,i), fAffine.at<float>(1,i)));
    }
}
}//cv




