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

#ifndef __OPENCV_CORE_FLANN_HPP__
#define __OPENCV_CORE_FLANN_HPP__

#ifdef __cplusplus

namespace flann
{
	class Index;
}

namespace cv {

namespace flann {

/* Nearest neighbor index algorithms */
enum flann_algorithm_t {
	LINEAR = 0,
	KDTREE = 1,
	KMEANS = 2,
	COMPOSITE = 3,
	SAVED = 254,
	AUTOTUNED = 255
};

enum flann_centers_init_t {
	CENTERS_RANDOM = 0,
	CENTERS_GONZALES = 1,
	CENTERS_KMEANSPP = 2
};


enum flann_log_level_t {
	LOG_NONE = 0,
	LOG_FATAL = 1,
	LOG_ERROR = 2,
	LOG_WARN = 3,
	LOG_INFO = 4
};

enum flann_distance_t {
	EUCLIDEAN = 1,
	MANHATTAN = 2,
	MINKOWSKI = 3
};

class CV_EXPORTS IndexFactory
{
public:
    virtual ~IndexFactory() {}
	virtual ::flann::Index* createIndex(const Mat& dataset) const = 0;
};

struct CV_EXPORTS IndexParams : public IndexFactory {
protected:
	IndexParams() {};

};

struct CV_EXPORTS LinearIndexParams : public IndexParams {
	LinearIndexParams() {};

	::flann::Index* createIndex(const Mat& dataset) const;
};



struct CV_EXPORTS KDTreeIndexParams : public IndexParams {
	KDTreeIndexParams(int trees_ = 4) : trees(trees_) {};

	int trees;                 // number of randomized trees to use (for kdtree)

	::flann::Index* createIndex(const Mat& dataset) const;
};

struct CV_EXPORTS KMeansIndexParams : public IndexParams {
	KMeansIndexParams(int branching_ = 32, int iterations_ = 11,
			flann_centers_init_t centers_init_ = CENTERS_RANDOM, float cb_index_ = 0.2 ) :
		branching(branching_),
		iterations(iterations_),
		centers_init(centers_init_),
		cb_index(cb_index_) {};

	int branching;             // branching factor (for kmeans tree)
	int iterations;            // max iterations to perform in one kmeans clustering (kmeans tree)
	flann_centers_init_t centers_init;          // algorithm used for picking the initial cluster centers for kmeans tree
    float cb_index;            // cluster boundary index. Used when searching the kmeans tree

    ::flann::Index* createIndex(const Mat& dataset) const;
};


struct CV_EXPORTS CompositeIndexParams : public IndexParams {
	CompositeIndexParams(int trees_ = 4, int branching_ = 32, int iterations_ = 11,
			flann_centers_init_t centers_init_ = CENTERS_RANDOM, float cb_index_ = 0.2 ) :
		trees(trees_),
		branching(branching_),
		iterations(iterations_),
		centers_init(centers_init_),
		cb_index(cb_index_) {};

	int trees;                 // number of randomized trees to use (for kdtree)
	int branching;             // branching factor (for kmeans tree)
	int iterations;            // max iterations to perform in one kmeans clustering (kmeans tree)
	flann_centers_init_t centers_init;          // algorithm used for picking the initial cluster centers for kmeans tree
    float cb_index;            // cluster boundary index. Used when searching the kmeans tree

    ::flann::Index* createIndex(const Mat& dataset) const;
};


struct CV_EXPORTS AutotunedIndexParams : public IndexParams {
	AutotunedIndexParams( float target_precision_ = 0.9, float build_weight_ = 0.01,
			float memory_weight_ = 0, float sample_fraction_ = 0.1) :
		target_precision(target_precision_),
		build_weight(build_weight_),
		memory_weight(memory_weight_),
		sample_fraction(sample_fraction_) {};

	float target_precision;    // precision desired (used for autotuning, -1 otherwise)
	float build_weight;        // build tree time weighting factor
	float memory_weight;       // index memory weighting factor
    float sample_fraction;     // what fraction of the dataset to use for autotuning

    ::flann::Index* createIndex(const Mat& dataset) const;
};


struct CV_EXPORTS SavedIndexParams : public IndexParams {
	SavedIndexParams() {}
	SavedIndexParams(std::string filename_) : filename(filename_) {}

	std::string filename;		// filename of the stored index

	::flann::Index* createIndex(const Mat& dataset) const;
};


struct CV_EXPORTS SearchParams {
	SearchParams(int checks_ = 32) :
		checks(checks_) {};

	int checks;
};



class CV_EXPORTS Index {
	::flann::Index* nnIndex;

public:
	Index(const Mat& features, const IndexParams& params);

	~Index();

	void knnSearch(const vector<float>& queries, vector<int>& indices, vector<float>& dists, int knn, const SearchParams& params);
	void knnSearch(const Mat& queries, Mat& indices, Mat& dists, int knn, const SearchParams& params);

	int radiusSearch(const vector<float>& query, vector<int>& indices, vector<float>& dists, float radius, const SearchParams& params);
	int radiusSearch(const Mat& query, Mat& indices, Mat& dists, float radius, const SearchParams& params);

	void save(std::string filename);

	int veclen() const;

	int size() const;
};


CV_EXPORTS int hierarchicalClustering(const Mat& features, Mat& centers,
                                      const KMeansIndexParams& params);

}

}

#endif // __cplusplus

#endif
