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

#ifndef _OPENCV_MINIFLANN_HPP_
#define _OPENCV_MINIFLANN_HPP_

#ifdef __cplusplus

#include "opencv2/core/core.hpp"
#include "opencv2/flann/defines.h"

namespace cv
{
 
namespace flann
{

struct CV_EXPORTS IndexParams
{
    IndexParams();
    ~IndexParams();
    
    std::string getString(const std::string& key, const std::string& defaultVal=std::string()) const;
    int getInt(const std::string& key, int defaultVal=-1) const;
    double getDouble(const std::string& key, double defaultVal=-1) const;
    
    void setString(const std::string& key, const std::string& value);
    void setInt(const std::string& key, int value);
    void setDouble(const std::string& key, double value);
    
    void getAll(std::vector<std::string>& names,
                std::vector<int>& types,
                std::vector<std::string>& strValues,
                std::vector<double>& numValues) const;
    
    void* params;
};    

struct CV_EXPORTS KDTreeIndexParams : public IndexParams
{
    KDTreeIndexParams(int trees=4);
};
    
struct CV_EXPORTS LinearIndexParams : public IndexParams
{
    LinearIndexParams();
};

struct CV_EXPORTS CompositeIndexParams : public IndexParams
{
    CompositeIndexParams(int trees = 4, int branching = 32, int iterations = 11,
                         flann_centers_init_t centers_init = FLANN_CENTERS_RANDOM, float cb_index = 0.2 );
};

struct CV_EXPORTS AutotunedIndexParams : public IndexParams
{
    AutotunedIndexParams(float target_precision = 0.8, float build_weight = 0.01,
                         float memory_weight = 0, float sample_fraction = 0.1);
};
    
struct CV_EXPORTS KMeansIndexParams : public IndexParams
{
    KMeansIndexParams(int branching = 32, int iterations = 11,
                      flann_centers_init_t centers_init = FLANN_CENTERS_RANDOM, float cb_index = 0.2 );
};

struct CV_EXPORTS LshIndexParams : public IndexParams
{
    LshIndexParams(int table_number, int key_size, int multi_probe_level);
};
    
struct CV_EXPORTS SavedIndexParams : public IndexParams
{
    SavedIndexParams(const std::string& filename);
};    
    
struct CV_EXPORTS SearchParams : public IndexParams
{
    SearchParams( int checks = 32, float eps = 0, bool sorted = true );
};    
    
class CV_EXPORTS_W Index
{
public:
    CV_WRAP Index();
    CV_WRAP Index(InputArray features, const IndexParams& params, flann_distance_t distType=FLANN_DIST_L2);
    virtual ~Index();
    
    CV_WRAP virtual void build(InputArray features, const IndexParams& params, flann_distance_t distType=FLANN_DIST_L2);
    CV_WRAP virtual void knnSearch(InputArray query, OutputArray indices, 
                   OutputArray dists, int knn, const SearchParams& params=SearchParams());
    
    CV_WRAP virtual int radiusSearch(InputArray query, OutputArray indices,
                             OutputArray dists, double radius, int maxResults,
                             const SearchParams& params=SearchParams());
    
    CV_WRAP virtual void save(const std::string& filename) const;
    CV_WRAP virtual bool load(InputArray features, const std::string& filename);
    CV_WRAP virtual void release();
    CV_WRAP flann_distance_t getDistance() const;
    CV_WRAP flann_algorithm_t getAlgorithm() const;
    
protected:
    flann_distance_t distType;
    flann_algorithm_t algo;
    int featureType;
    void* index;
};
        
} } // namespace cv::flann

#endif // __cplusplus

#endif
