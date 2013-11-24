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

// Training application for Soft Cascades.

#include <sft/common.hpp>
#include <iostream>
#include <sft/dataset.hpp>
#include <sft/config.hpp>

#include <opencv2/core/core_c.h>

int main(int argc, char** argv)
{
    using namespace sft;

    const string keys =
        "{help h usage ? |      | print this message              }"
        "{config c       |      | path to configuration xml       }"
    ;

    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("Soft cascade training application.");

    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }

    if (!parser.check())
    {
        parser.printErrors();
        return 1;
    }

    string configPath = parser.get<string>("config");
    if (configPath.empty())
    {
        std::cout << "Configuration file is missing or empty. Could not start training." << std::endl;
        return 0;
    }

    std::cout << "Read configuration from file " << configPath << std::endl;
    cv::FileStorage fs(configPath, cv::FileStorage::READ);
    if(!fs.isOpened())
    {
        std::cout << "Configuration file " << configPath << " can't be opened." << std::endl;
        return 1;
    }

    // 1. load config
    sft::Config cfg;
    fs["config"] >> cfg;
    std::cout << std::endl << "Training will be executed for configuration:" << std::endl << cfg << std::endl;

    // 2. check and open output file
    cv::FileStorage fso(cfg.outXmlPath, cv::FileStorage::WRITE);
    if(!fso.isOpened())
    {
        std::cout << "Training stopped. Output classifier Xml file " << cfg.outXmlPath << " can't be opened." << std::endl;
        return 1;
    }

    fso << cfg.cascadeName
        << "{"
        << "stageType"   << "BOOST"
        << "featureType" << cfg.featureType
        << "octavesNum"  << (int)cfg.octaves.size()
        << "width"       << cfg.modelWinSize.width
        << "height"      << cfg.modelWinSize.height
        << "shrinkage"   << cfg.shrinkage
        << "octaves"     << "[";

    // 3. Train all octaves
    for (ivector::const_iterator it = cfg.octaves.begin(); it != cfg.octaves.end(); ++it)
    {
        // a. create random feature pool
        int nfeatures  = cfg.poolSize;
        cv::Size model = cfg.model(it);
        std::cout << "Model " << model << std::endl;

        int nchannels = (cfg.featureType == "HOG6MagLuv") ? 10: 8;

        std::cout << "number of feature channels is " << nchannels << std::endl;

        cv::Ptr<cv::FeaturePool> pool = cv::FeaturePool::create(model, nfeatures, nchannels);
        nfeatures = pool->size();


        int npositives = cfg.positives;
        int nnegatives = cfg.negatives;
        int shrinkage  = cfg.shrinkage;
        cv::Rect boundingBox = cfg.bbox(it);
        std::cout << "Object bounding box" << boundingBox << std::endl;

        typedef cv::Octave Octave;

        cv::Ptr<cv::ChannelFeatureBuilder> builder = cv::ChannelFeatureBuilder::create(cfg.featureType);
        std::cout << "Channel builder " << builder->info()->name() << std::endl;
        cv::Ptr<Octave> boost = Octave::create(boundingBox, npositives, nnegatives, *it, shrinkage, builder);

        std::string path = cfg.trainPath;
        sft::ScaledDataset dataset(path, *it);

        if (boost->train(&dataset, pool, cfg.weaks, cfg.treeDepth))
        {
            CvFileStorage* fout = cvOpenFileStorage(cfg.resPath(it).c_str(), 0, CV_STORAGE_WRITE);
            boost->write(fout, cfg.cascadeName);

            cvReleaseFileStorage( &fout);

            cv::Mat thresholds;
            boost->setRejectThresholds(thresholds);

            boost->write(fso, pool, thresholds);

            cv::FileStorage tfs(("thresholds." + cfg.resPath(it)).c_str(), cv::FileStorage::WRITE);
            tfs << "thresholds" << thresholds;

            std::cout << "Octave " << *it << " was successfully trained..." << std::endl;
        }
    }

    fso << "]" << "}";
    fso.release();
    std::cout << "Training complete..." << std::endl;
    return 0;
}
