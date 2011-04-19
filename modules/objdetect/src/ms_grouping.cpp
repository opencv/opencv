/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
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

#include "precomp.hpp"
using namespace cv;

MeanshiftGrouping::MeanshiftGrouping(const Point3d& densKer, const vector<Point3d>& posV, 
					 const vector<double>& wV, double modeEps, int maxIter)
{
	densityKernel = densKer;
    weightsV = wV;
    positionsV = posV;
    positionsCount = posV.size();
	meanshiftV.resize(positionsCount);
    distanceV.resize(positionsCount);
    modeEps = modeEps;
	iterMax = maxIter;
    
	for (unsigned i=0; i<positionsV.size(); i++)
	{
		meanshiftV[i] = getNewValue(positionsV[i]);

		distanceV[i] = moveToMode(meanshiftV[i]);

		meanshiftV[i] -= positionsV[i];
	}
}

void MeanshiftGrouping::getModes(vector<Point3d>& modesV, vector<double>& resWeightsV, double eps)
{
	for (size_t i=0; i <distanceV.size(); i++)
	{
		bool is_found = false;
		for(size_t j=0; j<modesV.size(); j++)
		{
			if ( getDistance(distanceV[i], modesV[j]) < eps)
			{
				is_found=true;
				break;
			}
		}
		if (!is_found)
		{
			modesV.push_back(distanceV[i]);
		}
	}
	
	resWeightsV.resize(modesV.size());

	for (size_t i=0; i<modesV.size(); i++)
	{
		resWeightsV[i] = getResultWeight(modesV[i]);
	}
}

Point3d MeanshiftGrouping::moveToMode(Point3d aPt) const
{
	Point3d bPt;
	for (int i = 0; i<iterMax; i++)
	{
		bPt = aPt;
		aPt = getNewValue(bPt);
		if ( getDistance(aPt, bPt) <= modeEps )
		{
			break;
		}
	}
	return aPt;
}

Point3d MeanshiftGrouping::getNewValue(const Point3d& inPt) const
{
	Point3d resPoint(.0);
	Point3d ratPoint(.0);
	for (size_t i=0; i<positionsV.size(); i++)
	{
		Point3d aPt= positionsV[i];
		Point3d bPt = inPt;
		Point3d sPt = densityKernel;
		
		sPt.x *= exp(aPt.z);
		sPt.y *= exp(aPt.z);
		
		aPt.x /= sPt.x;
		aPt.y /= sPt.y;
		aPt.z /= sPt.z;

		bPt.x /= sPt.x;
		bPt.y /= sPt.y;
		bPt.z /= sPt.z;
		
		double w = (weightsV[i])*std::exp(-((aPt-bPt).dot(aPt-bPt))/2)/std::sqrt(sPt.dot(Point3d(1,1,1)));
		
		resPoint += w*aPt;

		ratPoint.x += w/sPt.x;
		ratPoint.y += w/sPt.y;
		ratPoint.z += w/sPt.z;
	}
	resPoint.x /= ratPoint.x;
	resPoint.y /= ratPoint.y;
	resPoint.z /= ratPoint.z;
	return resPoint;
} 

double MeanshiftGrouping::getResultWeight(const Point3d& inPt) const
{
	double sumW=0;
	for (size_t i=0; i<positionsV.size(); i++)
	{
		Point3d aPt = positionsV[i];
		Point3d sPt = densityKernel;

		sPt.x *= exp(aPt.z);
		sPt.y *= exp(aPt.z);

		aPt -= inPt;
		
		aPt.x /= sPt.x;
		aPt.y /= sPt.y;
		aPt.z /= sPt.z;
		
		sumW+=(weightsV[i])*std::exp(-(aPt.dot(aPt))/2)/std::sqrt(sPt.dot(Point3d(1,1,1)));
	}
	return sumW;
}

double MeanshiftGrouping::getDistance(Point3d p1, Point3d p2) const 
{
	Point3d ns = densityKernel;
	ns.x *= exp(p2.z);
	ns.y *= exp(p2.z);
	p2 -= p1;
	p2.x /= ns.x;
	p2.y /= ns.y;
	p2.z /= ns.z;
	return p2.dot(p2);
}
