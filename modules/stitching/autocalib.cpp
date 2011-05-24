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
#include "autocalib.hpp"
#include "util.hpp"

using namespace std;
using namespace cv;

void focalsFromHomography(const Mat& H, double &f0, double &f1, bool &f0_ok, bool &f1_ok)
{
    CV_Assert(H.type() == CV_64F && H.size() == Size(3, 3));

    const double* h = reinterpret_cast<const double*>(H.data);

    double d1, d2; // Denominators
    double v1, v2; // Focal squares value candidates

    f1_ok = true;
    d1 = h[6] * h[7];
    d2 = (h[7] - h[6]) * (h[7] + h[6]);
    v1 = -(h[0] * h[1] + h[3] * h[4]) / d1;
    v2 = (h[0] * h[0] + h[3] * h[3] - h[1] * h[1] - h[4] * h[4]) / d2;
    if (v1 < v2) swap(v1, v2);
    if (v1 > 0 && v2 > 0) f1 = sqrt(abs(d1) > abs(d2) ? v1 : v2);
    else if (v1 > 0) f1 = sqrt(v1);
    else f1_ok = false;

    f0_ok = true;
    d1 = h[0] * h[3] + h[1] * h[4];
    d2 = h[0] * h[0] + h[1] * h[1] - h[3] * h[3] - h[4] * h[4];
    v1 = -h[2] * h[5] / d1;
    v2 = (h[5] * h[5] - h[2] * h[2]) / d2;
    if (v1 < v2) swap(v1, v2);
    if (v1 > 0 && v2 > 0) f0 = sqrt(abs(d1) > abs(d2) ? v1 : v2);
    else if (v1 > 0) f0 = sqrt(v1);
    else f0_ok = false;
}


void estimateFocal(const vector<ImageFeatures> &features, const vector<MatchesInfo> &pairwise_matches, 
                   vector<double> &focals)
{
    const int num_images = static_cast<int>(features.size());
    focals.resize(num_images);

    vector<double> all_focals;

    for (int i = 0; i < num_images; ++i)
    {        
        for (int j = 0; j < num_images; ++j)
        {
            const MatchesInfo &m = pairwise_matches[i*num_images + j];
            if (m.H.empty())
                continue;
            double f0, f1;
            bool f0ok, f1ok;
            focalsFromHomography(m.H, f0, f1, f0ok, f1ok);
            if (f0ok && f1ok) 
                all_focals.push_back(sqrt(f0 * f1));
        }
    }

    if (static_cast<int>(all_focals.size()) < num_images - 1)
    {
        LOGLN("Can't estimate focal length, will use anaive approach");
        double focals_sum = 0;
        for (int i = 0; i < num_images; ++i)
            focals_sum += features[i].img_size.width + features[i].img_size.height;
        for (int i = 0; i < num_images; ++i)
            focals[i] = focals_sum / num_images;
    }
    else
    {
        nth_element(all_focals.begin(), all_focals.end(), all_focals.begin() + all_focals.size()/2);
        for (int i = 0; i < num_images; ++i)
            focals[i] = all_focals[all_focals.size()/2];
    }
}
