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

#include <sft/config.hpp>
#include <iomanip>

sft::Config::Config(): seed(0) {}

void sft::Config::write(cv::FileStorage& fs) const
{
    fs << "{"
       << "trainPath"    << trainPath
       << "testPath"     << testPath

       << "modelWinSize" << modelWinSize
       << "offset"       << offset
       << "octaves"      << octaves

       << "positives"    << positives
       << "negatives"    << negatives
       << "btpNegatives" << btpNegatives

       << "shrinkage"    << shrinkage

       << "treeDepth"    << treeDepth
       << "weaks"        << weaks
       << "poolSize"     << poolSize

       << "cascadeName"  << cascadeName
       << "outXmlPath"   << outXmlPath

       << "seed"         << seed
       << "featureType"  << featureType
       << "}";
}

void sft::Config::read(const cv::FileNode& node)
{
    trainPath = (string)node["trainPath"];
    testPath  = (string)node["testPath"];

    cv::FileNodeIterator  nIt = node["modelWinSize"].end();
    modelWinSize = cv::Size((int)*(--nIt), (int)*(--nIt));

    nIt = node["offset"].end();
    offset = cv::Point2i((int)*(--nIt), (int)*(--nIt));

    node["octaves"] >> octaves;

    positives =    (int)node["positives"];
    negatives =    (int)node["negatives"];
    btpNegatives = (int)node["btpNegatives"];

    shrinkage = (int)node["shrinkage"];

    treeDepth = (int)node["treeDepth"];
    weaks =     (int)node["weaks"];
    poolSize =  (int)node["poolSize"];

    cascadeName = (std::string)node["cascadeName"];
    outXmlPath =  (std::string)node["outXmlPath"];

    seed = (int)node["seed"];
    featureType = (std::string)node["featureType"];
}

void sft::write(cv::FileStorage& fs, const string&, const Config& x)
{
    x.write(fs);
}

void sft::read(const cv::FileNode& node, Config& x, const Config& default_value)
{
    x = default_value;

    if(!node.empty())
        x.read(node);
}

namespace {

struct Out
{
    Out(std::ostream& _out): out(_out) {}
    template<typename T>
    void operator ()(const T a) const {out << a << " ";}

    std::ostream& out;
private:
    Out& operator=(Out const& other);
};
}

std::ostream& sft::operator<<(std::ostream& out, const Config& m)
{
    out << std::setw(14) << std::left << "trainPath"    << m.trainPath     << std::endl
        << std::setw(14) << std::left << "testPath"     << m.testPath      << std::endl

        << std::setw(14) << std::left << "modelWinSize" << m.modelWinSize  << std::endl
        << std::setw(14) << std::left << "offset"       << m.offset        << std::endl
        << std::setw(14) << std::left << "octaves";

    Out o(out);
    for_each(m.octaves.begin(), m.octaves.end(), o);

    out << std::endl
        << std::setw(14) << std::left  << "positives"    << m.positives    << std::endl
        << std::setw(14) << std::left  << "negatives"    << m.negatives    << std::endl
        << std::setw(14) << std::left  << "btpNegatives" << m.btpNegatives << std::endl

        << std::setw(14) << std::left  << "shrinkage"    << m.shrinkage    << std::endl

        << std::setw(14) << std::left  << "treeDepth"    << m.treeDepth    << std::endl
        << std::setw(14) << std::left  << "weaks"        << m.weaks        << std::endl
        << std::setw(14) << std::left  << "poolSize"     << m.poolSize     << std::endl

        << std::setw(14) << std::left  << "cascadeName"  << m.cascadeName  << std::endl
        << std::setw(14) << std::left  << "outXmlPath"   << m.outXmlPath   << std::endl
        << std::setw(14) << std::left  << "seed"         << m.seed         << std::endl
        << std::setw(14) << std::left  << "featureType"  << m.featureType  << std::endl;

    return out;
}
