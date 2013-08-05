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

/*
 * Implementation of the paper Shape Matching and Object Recognition Using Shape Contexts
 * Belongie et al., 2002 by Juan Manuel Perez for GSoC 2013. 
 */
#include "precomp.hpp"

namespace cv
{
// Constructors //
SCD::SCD(int _nAngularBins, int _nRadialBins, double _innerRadius, double _outerRadius, bool _rotationInvariant)
{
    setAngularBins(_nAngularBins);
    setRadialBins(_nRadialBins);
    setInnerRadius(_innerRadius);
    setOuterRadius(_outerRadius);
    setRotationInvariant(_rotationInvariant);
}

// Public methods //
int SCD::descriptorSize() const 
{ 
    return nAngularBins*nRadialBins; 
}

void SCD::extractSCD(InputArray contour /* Vector of points */,
                     Mat& descriptors /* Mat containing the descriptor */,
                     const std::vector<int> &queryInliers /* used to avoid outliers */,
                     const float _meanDistance)
{
    Mat contourMat = contour.getMat();
    CV_Assert((contourMat.channels()==2) & (contourMat.cols>0));
    
    Mat disMatrix = Mat::zeros(contourMat.cols, contourMat.cols, CV_32F);
    Mat angleMatrix = Mat::zeros(contourMat.cols, contourMat.cols, CV_32F);
    
    std::vector<double> logspaces, angspaces;
    logarithmicSpaces(logspaces);
    angularSpaces(angspaces);
    
    buildNormalizedDistanceMatrix(contourMat, disMatrix, queryInliers, _meanDistance);
    buildAngleMatrix(contourMat, angleMatrix);

    // Now, build the descriptor matrix (each row is a point) //
    descriptors = Mat::zeros(contourMat.cols, descriptorSize(), CV_32F);
       
    for (int ptidx=0; ptidx<contourMat.cols; ptidx++)
    {
        for (int cmp=0; cmp<contourMat.cols; cmp++)
        {
            if (ptidx==cmp) continue;
            if ((int)queryInliers.size()>0)
            {
                if (queryInliers[ptidx]==0 || queryInliers[cmp]==0) continue; //avoid outliers
            }

            int angidx=-1, radidx=-1;
            for (int i=0; i<nRadialBins; i++)
            {
                if (disMatrix.at<float>(ptidx, cmp)<=logspaces[i])
                {
                    radidx=i;
                    break;
                }
            }
            for (int i=0; i<nAngularBins; i++)
            {
                if (angleMatrix.at<float>(ptidx, cmp)<=angspaces[i])
                {
                    angidx=i;
                    break;
                }
            }
            if (angidx!=-1 && radidx!=-1)
            {
                int idx = angidx+radidx*nAngularBins;
                descriptors.at<float>(ptidx, idx)++;
            }
        }        
    }
}

// Protected methods //
void SCD::buildAngleMatrix(InputArray contour, 
                      Mat& angleMatrix) const
{
    Mat contourMat = contour.getMat();
    
    // if descriptor is rotationInvariant compute massCenter //
    Point2f massCenter(0,0);
    if (rotationInvariant)
    {
        for (int i=0; i<contourMat.cols; i++)
        {
            massCenter+=contourMat.at<Point2f>(0,i);
        }
        massCenter.x=massCenter.x/(float)contourMat.cols;
        massCenter.y=massCenter.y/(float)contourMat.cols;
    }


    for (int i=0; i<contourMat.cols; i++)
    {
        for (int j=0; j<contourMat.cols; j++)
        {
            if (i==j) angleMatrix.at<float>(i,j)=0;
            Point2f dif = contourMat.at<Point2f>(0,i) - contourMat.at<Point2f>(0,j);
            angleMatrix.at<float>(i,j) = std::atan2(dif.y, dif.x);

            if (rotationInvariant)
            {
                Point2f refPt = contourMat.at<Point2f>(0,i) - massCenter;
                float refAngle = atan2(refPt.y, refPt.x);
                angleMatrix.at<float>(i,j) -= refAngle;
            }
            angleMatrix.at<float>(i,j) = fmod(fmod(angleMatrix.at<float>(i,j)+FLT_EPSILON,2*CV_PI)+2*CV_PI,2*CV_PI);
            angleMatrix.at<float>(i,j) = floor( angleMatrix.at<float>(i,j)*nAngularBins/(2*CV_PI) );
        }
    }
}

void SCD::buildNormalizedDistanceMatrix(InputArray contour,
                      Mat& disMatrix, const std::vector<int> &queryInliers, const float _meanDistance)
{
    Mat contourMat = contour.getMat();
    Mat mask(disMatrix.rows, disMatrix.cols, CV_8U);

    for (int i=0; i<contourMat.cols; i++)
    {
        for (int j=0; j<contourMat.cols; j++)
        {
            disMatrix.at<float>(i,j) = distance(contourMat.at<Point2f>(0,i),
                                                 contourMat.at<Point2f>(0,j));
            if (_meanDistance<0)
            {
                if (queryInliers.size()>0)
                {
                    mask.at<char>(i,j)=char(queryInliers[j] & queryInliers[i]);
                }
                else
                {
                    mask.at<char>(i,j)=1;
                }
            }
        }
    }

    if (_meanDistance<0)
    {
        meanDistance=mean(disMatrix, mask)[0];
    }
    else
    {
        meanDistance=_meanDistance;
    }
    disMatrix/=meanDistance+FLT_EPSILON;
}

void SCD::logarithmicSpaces(std::vector<double>& vecSpaces) const
{
   double logmin=log10(innerRadius);
   double logmax=log10(outerRadius);
   double delta=(logmax-logmin)/(nRadialBins-1);
   double accdelta=0;
   
   for (int i=0; i<nRadialBins; i++)
   {
       double val = std::pow(10,logmin+accdelta);
       vecSpaces.push_back(val);
       accdelta += delta;
   }
}

void SCD::angularSpaces(std::vector<double>& vecSpaces) const
{
   double delta=2*CV_PI/nAngularBins;
   double val=0;
   
   for (int i=0; i<nAngularBins; i++)
   {
       val += delta;
       vecSpaces.push_back(val);
   }
}

double SCD::distance(Point2f p, Point2f q) const
{
    Point2f diff = p - q;
    double d = diff.x*diff.x + diff.y*diff.y;// - 2*diff.x*diff.y;
    if (d<0) d=0;
    d = std::sqrt(d);
    return d;
}

/* Setters and Getters*/
void SCD::setAngularBins(int n)
{
    CV_Assert(n>0);
    nAngularBins=n;
}

void SCD::setRadialBins(int n)
{
    CV_Assert(n>0);
    nRadialBins=n;
}

void SCD::setInnerRadius(double r)
{
    CV_Assert(r>0);
    innerRadius=r;
}

void SCD::setOuterRadius(double r)
{
    CV_Assert(r>0);
    outerRadius=r;
}

void SCD::setRotationInvariant(bool _rotationInvariant)
{
    rotationInvariant=_rotationInvariant;
}

int SCD::getAngularBins()
{
    return nAngularBins;
}

int SCD::getRadialBins()
{
    return nRadialBins;
}

double SCD::getInnerRadius()
{
    return innerRadius;
}

double SCD::getOuterRadius()
{
    return outerRadius;
}

bool SCD::getRotationInvariant()
{
    return rotationInvariant;
}

float SCD::getMeanDistance()
{
    return meanDistance;
}

} /* namespace cv */
