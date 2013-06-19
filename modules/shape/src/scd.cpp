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
#include <cmath>
#include <iostream>

namespace cv
{
/* Constructors */
SCD::SCD()
{
    setAngularBins(12);
    setAngularBins(5);
    setInnerRadius(0.1);
    setOuterRadius(1);
}

SCD::SCD(int _nAngularBins, int _nRadialBins, double _innerRadius, double _outerRadius)
{
    setAngularBins(_nAngularBins);
    setRadialBins(_nRadialBins);
    setInnerRadius(_innerRadius);
    setOuterRadius(_outerRadius);
}

/* Public methods */
int SCD::descriptorSize() const 
{ 
    return nAngularBins*nRadialBins; 
}

void SCD::extractSCD(std::vector<Point> contour /* Vector of points */, 
                      Mat& descriptors /* Mat containing the descriptor */) const
{
    CV_Assert(contour.size()>0);
    
    
    Mat disMatrix = Mat::zeros(contour.size(), contour.size(), CV_32F);
    Mat angleMatrix = Mat::zeros(contour.size(), contour.size(), CV_32F);
    
    std::vector<double> logspaces, angspaces;
    logarithmicSpaces(logspaces);
    angularSpaces(angspaces);
    
    buildNormalizedDistanceMatrix(contour, disMatrix);
    buildAngleMatrix(contour, angleMatrix);
    
    /* Now, build the descriptor matrix (each row is a point descriptor) 
     * ask if the correspondent points belong to a given bin.*/
    descriptors = Mat::zeros(contour.size(), descriptorSize(), CV_32F);
    for (uint ptidx=0; ptidx<contour.size(); ++ptidx)
    {
        for (uint cmp=0; cmp<contour.size(); ++cmp)
        {
            if (ptidx==cmp) continue;
            
            int angidx=0, radidx=0;
            for (int i=0; i<nRadialBins; i++)
            {
                if (disMatrix.at<float>(ptidx,cmp)<=logspaces[i])
                {
                    radidx=i;
                    break;
                }
            }
            for (int i=0; i<nAngularBins; i++)
            {
                if (angleMatrix.at<float>(ptidx,cmp)<=angspaces[i])
                {
                    angidx=i;
                    break;
                }
            }
            int idx=angidx+radidx*nAngularBins;
            descriptors.at<float>(ptidx,idx)+=1; 
        }        
    }
}

/* Protected methods */
void SCD::buildAngleMatrix(std::vector<Point> contour, 
                      Mat& angleMatrix) const
{
    for (uint i=0;i<contour.size();++i)
    {
        for (uint j=0;j<contour.size();++j)
        {
            if (i==j) continue;
            Point dif = contour[i] - contour[j];
            angleMatrix.at<float>(i,j) = std::atan2(dif.y, dif.x);
        }
    }
}

void SCD::buildNormalizedDistanceMatrix(std::vector<Point> contour, 
                      Mat& disMatrix) const
{
    for (uint i=0;i<contour.size();++i)
    {
        for (uint j=0;j<contour.size();++j)
        {
            disMatrix.at<float>(i,j) = distance(contour[i],contour[j]);
        }
    }
    
    /* Now normalizing according to the mean for scale invariance.
     * However, the paper recommends to avoid using outliers in the 
     * mean computation. Short term future work.*/
    
    normalize(disMatrix, disMatrix,0,1, NORM_MINMAX);
    sqrt(disMatrix, disMatrix);
    Scalar m = mean(disMatrix);
    if (m[0]!=0)
    {
        disMatrix=disMatrix/m[0];
    }
}

void SCD::logarithmicSpaces(std::vector<double>& vecSpaces) const
{
   double logmin = log10(innerRadius);
   double logmax = log10(outerRadius);
   double delta = (logmax-logmin)/(nRadialBins-1);
   double accdelta = 0;
   
   for (int i=0; i<nRadialBins; ++i)
   {
       double val = std::pow(10,logmin+accdelta);
       vecSpaces.push_back(val);
       accdelta += delta;
   }
}

void SCD::angularSpaces(std::vector<double>& vecSpaces) const
{
   double delta = 2*CV_PI/nAngularBins;
   double val = -CV_PI;
   
   for (int i=0; i<nAngularBins; ++i)
   {
       val += delta;
       vecSpaces.push_back(val);
   }
}

double SCD::distance(Point p, Point q) const
{
    Point diff = p - q;
    return (diff.x*diff.x + diff.y*diff.y)/2;
}

/* Setters and Getters*/
void SCD::setAngularBins(int n)
{
    nAngularBins = n;
}

void SCD::setRadialBins(int n)
{
    nRadialBins = n;
}

void SCD::setInnerRadius(double r)
{
    innerRadius = r;
}

void SCD::setOuterRadius(double r)
{
    outerRadius = r;
}

int SCD::getAngularBins(void)
{
    return nAngularBins;
}

int SCD::getRadialBins(void)
{
    return nRadialBins;
}

double SCD::getInnerRadius(void)
{
    return innerRadius;
}

double SCD::getOuterRadius(void)
{
    return outerRadius;
}

} /* namespace cv */
