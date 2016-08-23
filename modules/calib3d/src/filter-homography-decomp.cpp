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
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
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
#include "filter-homography-decomp.h"

using namespace std;

namespace cv
{
	vector<int> filterHomographyDecompSolutionsByPointNormals(
		const vector<Mat>& rotations,
		const vector<Mat>& normals,
		const vector<Point2f>& beforeRectifiedPoints,
		const vector<Point2f>& afterRectifiedPoints,
		const Mat& mask)
	{
		using namespace std;

		vector<int> prevPointBehindCameraCount, currPointBehindCameraCount;
		for (int solutionIdx = 0; solutionIdx < rotations.size(); solutionIdx++)
		{
			prevPointBehindCameraCount.push_back(0);
			currPointBehindCameraCount.push_back(0);
		}

		for (int pointIdx = 0; pointIdx < beforeRectifiedPoints.size(); pointIdx++) {
			if (mask.at<bool>(pointIdx))
			{
				for (int solutionIdx = 0; solutionIdx < rotations.size(); solutionIdx++)
				{
					Mat tempAddMat = Mat(1, 1, CV_64F, double(1));

					Mat tempPrevPointMat = Mat(beforeRectifiedPoints.at(pointIdx));
					tempPrevPointMat.convertTo(tempPrevPointMat, CV_64F);
					tempPrevPointMat.push_back(tempAddMat);

					Mat tempCurrPointMat = Mat(afterRectifiedPoints.at(pointIdx));
					tempCurrPointMat.convertTo(tempCurrPointMat, CV_64F);
					tempCurrPointMat.push_back(tempAddMat);

					double prevNormDot = tempPrevPointMat.dot(normals.at(solutionIdx));
					double currNormDot = tempCurrPointMat.dot(rotations.at(solutionIdx) * normals.at(solutionIdx));

					if (prevNormDot <= 0)
						prevPointBehindCameraCount[solutionIdx]++;

					if (currNormDot <= 0)
						currPointBehindCameraCount[solutionIdx]++;
				}
			}
		}

		vector<int> possibleSolutions;

		for (int solutionIdx = 0; solutionIdx < rotations.size(); solutionIdx++)
		{
			if (prevPointBehindCameraCount[solutionIdx] == 0 && currPointBehindCameraCount[solutionIdx] == 0)
			{
				possibleSolutions.push_back(solutionIdx);
			}
		}

		return possibleSolutions;
	}

}