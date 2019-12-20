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

namespace cv { namespace ml {

struct PairDI
{
    double d;
    int    i;
};

struct CmpPairDI
{
    bool operator ()(const PairDI& e1, const PairDI& e2) const
    {
        return (e1.d < e2.d) || (e1.d == e2.d && e1.i < e2.i);
    }
};

void createConcentricSpheresTestSet( int num_samples, int num_features, int num_classes,
                                     OutputArray _samples, OutputArray _responses)
{
    if( num_samples < 1 )
        CV_Error( CV_StsBadArg, "num_samples parameter must be positive" );

    if( num_features < 1 )
        CV_Error( CV_StsBadArg, "num_features parameter must be positive" );

    if( num_classes < 1 )
        CV_Error( CV_StsBadArg, "num_classes parameter must be positive" );

    int i, cur_class;

    _samples.create( num_samples, num_features, CV_32F );
    _responses.create( 1, num_samples, CV_32S );

    Mat responses = _responses.getMat();

    Mat mean = Mat::zeros(1, num_features, CV_32F);
    Mat cov = Mat::eye(num_features, num_features, CV_32F);

    // fill the feature values matrix with random numbers drawn from standard normal distribution
    randMVNormal( mean, cov, num_samples, _samples );
    Mat samples = _samples.getMat();

    // calculate distances from the origin to the samples and put them
    // into the sequence along with indices
    std::vector<PairDI> dis(samples.rows);

    for( i = 0; i < samples.rows; i++ )
    {
        PairDI& elem = dis[i];
        elem.i = i;
        elem.d = norm(samples.row(i), NORM_L2);
    }

    std::sort(dis.begin(), dis.end(), CmpPairDI());

    // assign class labels
    num_classes = std::min( num_samples, num_classes );
    for( i = 0, cur_class = 0; i < num_samples; ++cur_class )
    {
        int last_idx = num_samples * (cur_class + 1) / num_classes - 1;
        double max_dst = dis[last_idx].d;
        max_dst = std::max( max_dst, dis[i].d );

        for( ; i < num_samples && dis[i].d <= max_dst; ++i )
            responses.at<int>(dis[i].i) = cur_class;
    }
}

}}

/* End of file. */
