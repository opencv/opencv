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

#ifndef __SFT_CONFIG_HPP__
#define __SFT_CONFIG_HPP__

#include <sft/common.hpp>

#include <ostream>

namespace sft {

struct Config
{
    Config();

    void write(cv::FileStorage& fs) const;

    void read(const cv::FileNode& node);

    // Scaled and shrunk model size.
    cv::Size model(ivector::const_iterator it) const
    {
        float octave = powf(2.f, (float)(*it));
        return cv::Size( cvRound(modelWinSize.width  * octave) / shrinkage,
                         cvRound(modelWinSize.height * octave) / shrinkage );
    }

    // Scaled but, not shrunk bounding box for object in sample image.
    cv::Rect bbox(ivector::const_iterator it) const
    {
        float octave = powf(2.f, (float)(*it));
        return cv::Rect( cvRound(offset.x * octave), cvRound(offset.y * octave),
            cvRound(modelWinSize.width  * octave), cvRound(modelWinSize.height * octave));
    }

    string resPath(ivector::const_iterator it) const
    {
        return cv::format("%s%d.xml",cascadeName.c_str(), *it);
    }

    // Paths to a rescaled data
    string trainPath;
    string testPath;

    // Original model size.
    cv::Size modelWinSize;

    // example offset into positive image
    cv::Point2i offset;

    // List of octaves for which have to be trained cascades (a list of powers of two)
    ivector octaves;

    // Maximum number of positives that should be used during training
    int positives;

    // Initial number of negatives used during training.
    int negatives;

    // Number of weak negatives to add each bootstrapping step.
    int btpNegatives;

    // Inverse of scale for feature resizing
    int shrinkage;

    // Depth on weak classifier's decision tree
    int treeDepth;

    // Weak classifiers number in resulted cascade
    int weaks;

    // Feature random pool size
    int poolSize;

    // file name to store cascade
    string cascadeName;

    // path to resulting cascade
    string outXmlPath;

    // seed for random generation
    int seed;

    // channel feature type
    string featureType;

    // // bounding rectangle for actual example into example window
    // cv::Rect exampleWindow;
};

// required for cv::FileStorage serialization
void write(cv::FileStorage& fs, const string&, const Config& x);
void read(const cv::FileNode& node, Config& x, const Config& default_value);
std::ostream& operator<<(std::ostream& out, const Config& m);

}

#endif
