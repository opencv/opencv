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
// Copyright (C) 2008-2012, Willow Garage Inc., all rights reserved.
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

// Trating application for Soft Cascades.

#include <sft/common.hpp>
#include <sft/octave.hpp>

int main(int argc, char** argv)
{
// hard coded now
    int nfeatures  = 50;
    int npositives = 10;
    int nnegatives = 10;

    int shrinkage  = 4;
    int octave = 0;

    int nsamples = npositives + nnegatives;
    cv::Size model(64, 128);
    std::string path = "/home/kellan/cuda-dev/opencv_extra/testdata/sctrain/rescaled-train-2012-10-27-19-02-52";

    cv::Rect boundingBox(5, 5 ,16, 32);
    sft::Octave boost(boundingBox, npositives, nnegatives, octave, shrinkage);

    sft::FeaturePool pool(model, nfeatures);
    sft::Dataset dataset(path, boost.logScale);

    boost.train(dataset, pool);

    cv::Mat train_data(nfeatures, nsamples, CV_32FC1);
    cv::RNG rng;

    for (int y = 0; y < nfeatures; ++y)
        for (int x = 0; x < nsamples; ++x)
            train_data.at<float>(y, x) = rng.uniform(0.f, 1.f);

    int tflag = CV_COL_SAMPLE;
    cv::Mat responses(nsamples, 1, CV_32FC1);
    for (int y = 0; y < nsamples; ++y)
        responses.at<float>(y, 0) = (y < npositives) ? 1.f : 0.f;


    cv::Mat var_idx(1, nfeatures, CV_32SC1);
    for (int x = 0; x < nfeatures; ++x)
        var_idx.at<int>(0, x) = x;

    // Mat sample_idx;
    cv::Mat sample_idx(1, nsamples, CV_32SC1);
    for (int x = 0; x < nsamples; ++x)
        sample_idx.at<int>(0, x) = x;

    cv::Mat var_type(1, nfeatures + 1, CV_8UC1);
    for (int x = 0; x < nfeatures; ++x)
        var_type.at<uchar>(0, x) = CV_VAR_ORDERED;

    var_type.at<uchar>(0, nfeatures) = CV_VAR_CATEGORICAL;

    cv::Mat missing_mask;

    CvBoostParams params;
    {
        params.max_categories       = 10;
        params.max_depth            = 2;
        params.min_sample_count     = 2;
        params.cv_folds             = 0;
        params.truncate_pruned_tree = false;

        /// ??????????????????
        params.regression_accuracy = 0.01;
        params.use_surrogates      = false;
        params.use_1se_rule        = false;

        ///////// boost params
        params.boost_type       = CvBoost::GENTLE;
        params.weak_count       = 1;
        params.split_criteria   = CvBoost::SQERR;
        params.weight_trim_rate = 0.95;
    }

    bool update = false;

    // boost.train(train_data, responses, var_idx, sample_idx, var_type, missing_mask);

    // CvFileStorage* fs = cvOpenFileStorage( "/home/kellan/train_res.xml", 0, CV_STORAGE_WRITE );
    // boost.write(fs, "test_res");

    // cvReleaseFileStorage( &fs );
}