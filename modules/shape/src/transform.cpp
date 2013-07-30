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
#include <limits>

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

float ThinPlateSplineTransform::getTranformCost() const
{
    return transformCost;
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
    CV_Assert((pts1.channels()==2) & (pts1.cols>0) & (pts2.channels()==2) & (pts2.cols>0));
    CV_Assert(_matches.size()>1);

    // Use only valid matchings //
    std::vector<DMatch> matches;
    for (size_t i=0; i<_matches.size(); i++)
    {
        if (_matches[i].queryIdx<pts1.cols &&
            _matches[i].trainIdx<pts2.cols)
        {
            matches.push_back(_matches[i]);
        }
    }

    // Organizing the correspondent points in matrix style //
    Mat shape1(matches.size(),2,CV_32F); // transforming shape
    Mat shape2(matches.size(),2,CV_32F); // target shape
    for (size_t i=0; i<matches.size(); i++)
    {
        Point2f pt1=pts1.at<Point2f>(0,matches[i].queryIdx);
        shape1.at<float>(i,0) = pt1.x;
        shape1.at<float>(i,1) = pt1.y;

        Point2f pt2=pts2.at<Point2f>(0,matches[i].trainIdx);
        shape2.at<float>(i,0) = pt2.x;
        shape2.at<float>(i,1) = pt2.y;
    }

    // Building the matrices for solving the L*(w|a)=(v|0) problem with L={[K|P];[P'|0]}

    //Building K and P (Neede to buil L)
    Mat matK(matches.size(),matches.size(),CV_32F);
    Mat matP(matches.size(),3,CV_32F);
    for (size_t i=0; i<matches.size(); i++)
    {
        for (size_t j=0; j<matches.size(); j++)
        {
            if (i==j)
            {
                matK.at<float>(i,j)=beta; //regularization
            }
            else
            {
                matK.at<float>(i,j) = distance(Point2f(shape1.at<float>(i,0),shape1.at<float>(i,1)),
                                               Point2f(shape1.at<float>(j,0),shape1.at<float>(j,1)));
            }
        }
        matP.at<float>(i,0) = 1;
        matP.at<float>(i,1) = shape1.at<float>(i,0);
        matP.at<float>(i,2) = shape1.at<float>(i,1);
    }

    //Building L
    Mat matL=Mat::zeros(matches.size()+3,matches.size()+3,CV_32F);
    Mat matLroi(matL, Rect(0,0,matches.size(),matches.size())); //roi for K
    matK.copyTo(matLroi);
    matLroi = Mat(matL,Rect(matches.size(),0,3,matches.size())); //roi for P
    matP.copyTo(matLroi);
    Mat matPt;
    transpose(matP,matPt);
    matLroi = Mat(matL,Rect(0,matches.size(),matches.size(),3)); //roi for P'
    matPt.copyTo(matLroi);

    //Building B (v|0)
    Mat matB = Mat::zeros(matches.size()+3,2,CV_32F);
    for (size_t i=0; i<matches.size(); i++)
    {
        matB.at<float>(i,0) = shape2.at<float>(i,0); //x's
        matB.at<float>(i,1) = shape2.at<float>(i,1); //y's
    }

    //Obtaining transformation params (w|a)
    solve(matL, matB, tpsParameters, DECOMP_LU);
    //tpsParameters = matL.inv()*matB;

    //Apply transformation in the complete set of points
    outPts.clear();
    for (int i=0; i<pts1.cols; i++)
    {
        Point2f pt=pts1.at<Point2f>(0,i);
        outPts.push_back(_applyTransformation(shape1, pt));
    }

    //Setting transform Cost and Shape reference
    Mat wt(2,tpsParameters.rows-3,CV_32F);
    Mat w;
    for (int i=0; i<wt.cols; i++)
    {
        wt.at<float>(0,i)=tpsParameters.at<float>(i,0);
        wt.at<float>(1,i)=tpsParameters.at<float>(i,1);
    }
    transpose(wt,w);
    Mat Q=wt*matK*w;
    transformCost=mean(Q.diag(0))[0];
    shapeReference=shape1;

    //Adding affine cost to the total transformation cost
    /*Mat Af(2, 2, CV_32F);
    Af.at<float>(0,0)=tpsParameters.at<float>(tpsParameters.rows-2,0);
    Af.at<float>(0,1)=tpsParameters.at<float>(tpsParameters.rows-2,1);
    Af.at<float>(1,0)=tpsParameters.at<float>(tpsParameters.rows-1,0);
    Af.at<float>(1,1)=tpsParameters.at<float>(tpsParameters.rows-1,1);
    SVD mysvd(Af, SVD::NO_UV);
    Mat singVals=mysvd.w;
    transformCost+=std::log((singVals.at<float>(0,0)+FLT_MIN)/(singVals.at<float>(1,0)+FLT_MIN));*/
}

void ThinPlateSplineTransform::warpImage(InputArray input, OutputArray output) const
{
    CV_Assert((!tpsParameters.empty()) & (!input.empty()));

    Mat theinput = input.getMat();
    Mat mapX(theinput.rows, theinput.cols, CV_32FC1);
    Mat mapY(theinput.rows, theinput.cols, CV_32FC1);

    for (int row = 0; row < theinput.rows; row++)
    {
        for (int col = 0; col < theinput.cols; col++)
        {
            Point2f pt = _applyTransformation(shapeReference, Point2f(float(col), float(row)));
            mapX.at<float>(row, col) = pt.x;
            mapY.at<float>(row, col) = pt.y;
        }
    }
    remap(input, output, mapX, mapY, INTER_CUBIC, BORDER_CONSTANT, Scalar(128,128,128));
}

/* private methods */
double ThinPlateSplineTransform::distance(Point2f p, Point2f q) const
{
    Point2f diff = p - q;
    float norma = diff.x*diff.x + diff.y*diff.y;// - 2*diff.x*diff.y;
    if (norma<0) norma=0;
    norma = norma*std::log(norma+FLT_EPSILON);
    return norma;
}

Point2f ThinPlateSplineTransform::_applyTransformation(const Mat &shapeRef, const Point2f point) const
{
    Point2f out;
    for (int i=0; i<2; i++)
    {
        float a1=tpsParameters.at<float>(tpsParameters.rows-3,i);
        float ax=tpsParameters.at<float>(tpsParameters.rows-2,i);
        float ay=tpsParameters.at<float>(tpsParameters.rows-1,i);

        float affine=a1+ax*point.x+ay*point.y;
        float nonrigid=0;
        for (int j=0; j<shapeRef.rows; j++)
        {
            nonrigid+=tpsParameters.at<float>(j,i)*
                    distance(Point2f(shapeRef.at<float>(j,0),shapeRef.at<float>(j,1)),
                            point);
        }
        if (i==0)
        {
            out.x=affine+nonrigid;
        }
        if (i==1)
        {
            out.y=affine+nonrigid;
        }
    }
    return out;
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

float AffineTransform::getTranformCost() const
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

    // Use only valid matchings //
    std::vector<DMatch> matches;
    for (size_t i=0; i<_matches.size(); i++)
    {
        if (_matches[i].queryIdx<pts1.cols &&
            _matches[i].trainIdx<pts2.cols)
        {
            matches.push_back(_matches[i]);
        }
    }

    // Organizing the correspondent points in vector style //
    std::vector<Point2f> shape1; // transforming shape
    std::vector<Point2f> shape2; // target shape
    for (size_t i=0; i<matches.size(); i++)
    {
        Point2f pt1=pts1.at<Point2f>(0,matches[i].queryIdx);
        shape1.push_back(pt1);

        Point2f pt2=pts2.at<Point2f>(0,matches[i].trainIdx);
        shape2.push_back(pt2);
    }

    // estimateRigidTransform //
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
    {   // add a LLS solution here //
        affine = Mat::ones(2,3,CV_32F);
    }

    Mat fAffine=affine*auxaf;
    // Ensambling output //
    for (int i=0; i<fAffine.cols; i++)
    {
        outPts.push_back(Point2f(fAffine.at<float>(0,i), fAffine.at<float>(1,i)));
    }

    // Updating Transform Cost //
    Mat Af(2, 2, CV_32F);
    Af.at<float>(0,0)=affine.at<float>(0,0);
    Af.at<float>(0,1)=affine.at<float>(1,0);
    Af.at<float>(1,0)=affine.at<float>(0,1);
    Af.at<float>(1,1)=affine.at<float>(1,1);
    SVD mysvd(Af, SVD::NO_UV);
    Mat singVals=mysvd.w;
    transformCost=std::log((singVals.at<float>(0,0)+FLT_MIN)/(singVals.at<float>(1,0)+FLT_MIN));
}
}//cv




