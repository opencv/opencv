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

#ifndef _OPENCV_FLANN_HPP_
#define _OPENCV_FLANN_HPP_

#ifdef __cplusplus

#include "opencv2/flann/flann_base.hpp"

namespace cv
{
namespace flann
{

template <typename T> struct CvType {};
template <> struct CvType<unsigned char> { static int type() { return CV_8U; } };
template <> struct CvType<char> { static int type() { return CV_8S; } };
template <> struct CvType<unsigned short> { static int type() { return CV_16U; } };
template <> struct CvType<short> { static int type() { return CV_16S; } };
template <> struct CvType<int> { static int type() { return CV_32S; } };
template <> struct CvType<float> { static int type() { return CV_32F; } };
template <> struct CvType<double> { static int type() { return CV_64F; } };

    
using ::cvflann::IndexParams;
using ::cvflann::LinearIndexParams;
using ::cvflann::KDTreeIndexParams;
using ::cvflann::KMeansIndexParams;
using ::cvflann::CompositeIndexParams;
using ::cvflann::AutotunedIndexParams;
using ::cvflann::SavedIndexParams;

using ::cvflann::SearchParams;


template <typename T>
class CV_EXPORTS Index_ {
	::cvflann::Index<T>* nnIndex;

public:
	Index_(const Mat& features, const IndexParams& params);

	~Index_();

	void knnSearch(const vector<T>& query, vector<int>& indices, vector<float>& dists, int knn, const SearchParams& params);
	void knnSearch(const Mat& queries, Mat& indices, Mat& dists, int knn, const SearchParams& params);

	int radiusSearch(const vector<T>& query, vector<int>& indices, vector<float>& dists, float radius, const SearchParams& params);
	int radiusSearch(const Mat& query, Mat& indices, Mat& dists, float radius, const SearchParams& params);

	void save(std::string filename) { nnIndex->save(filename); }

	int veclen() const { return nnIndex->veclen(); }

	int size() const { return nnIndex->size(); }

	const IndexParams* getIndexParameters() { return nnIndex->getParameters(); }

};


template <typename T>
Index_<T>::Index_(const Mat& dataset, const IndexParams& params)
{
    CV_Assert(dataset.type() == CvType<T>::type());
    CV_Assert(dataset.isContinuous());
    ::cvflann::Matrix<float> m_dataset((T*)dataset.ptr<T>(0), dataset.rows, dataset.cols);
    
    nnIndex = new ::cvflann::Index<T>(m_dataset, params);
    nnIndex->buildIndex();
}

template <typename T>
Index_<T>::~Index_()
{
    delete nnIndex;
}

template <typename T>
void Index_<T>::knnSearch(const vector<T>& query, vector<int>& indices, vector<float>& dists, int knn, const SearchParams& searchParams)
{
    ::cvflann::Matrix<T> m_query((T*)&query[0], 1, (int)query.size());
    ::cvflann::Matrix<int> m_indices(&indices[0], 1, (int)indices.size());
    ::cvflann::Matrix<float> m_dists(&dists[0], 1, (int)dists.size());
    
    nnIndex->knnSearch(m_query,m_indices,m_dists,knn,searchParams);
}


template <typename T>
void Index_<T>::knnSearch(const Mat& queries, Mat& indices, Mat& dists, int knn, const SearchParams& searchParams)
{
    CV_Assert(queries.type() == CvType<T>::type());
    CV_Assert(queries.isContinuous());
    ::cvflann::Matrix<T> m_queries((T*)queries.ptr<T>(0), queries.rows, queries.cols);
    
    CV_Assert(indices.type() == CV_32S);
    CV_Assert(indices.isContinuous());
    ::cvflann::Matrix<int> m_indices((int*)indices.ptr<int>(0), indices.rows, indices.cols);
    
    CV_Assert(dists.type() == CV_32F);
    CV_Assert(dists.isContinuous());
    ::cvflann::Matrix<float> m_dists((float*)dists.ptr<float>(0), dists.rows, dists.cols);
    
    nnIndex->knnSearch(m_queries,m_indices,m_dists,knn, searchParams);
}

template <typename T>
int Index_<T>::radiusSearch(const vector<T>& query, vector<int>& indices, vector<float>& dists, float radius, const SearchParams& searchParams)
{
    ::cvflann::Matrix<T> m_query((T*)&query[0], 1, (int)query.size());
    ::cvflann::Matrix<int> m_indices(&indices[0], 1, (int)indices.size());
    ::cvflann::Matrix<float> m_dists(&dists[0], 1, (int)dists.size());
    
    return nnIndex->radiusSearch(m_query,m_indices,m_dists,radius,searchParams);
}

template <typename T>
int Index_<T>::radiusSearch(const Mat& query, Mat& indices, Mat& dists, float radius, const SearchParams& searchParams)
{
    CV_Assert(query.type() == CvType<T>::type());
    CV_Assert(query.isContinuous());
    ::cvflann::Matrix<T> m_query((T*)query.ptr<T>(0), query.rows, query.cols);
    
    CV_Assert(indices.type() == CV_32S);
    CV_Assert(indices.isContinuous());
    ::cvflann::Matrix<int> m_indices((int*)indices.ptr<int>(0), indices.rows, indices.cols);
    
    CV_Assert(dists.type() == CV_32F);
    CV_Assert(dists.isContinuous());
    ::cvflann::Matrix<float> m_dists((float*)dists.ptr<float>(0), dists.rows, dists.cols);
    
    return nnIndex->radiusSearch(m_query,m_indices,m_dists,radius,searchParams);
}

typedef Index_<float> Index;

template <typename ELEM_TYPE, typename DIST_TYPE>
int hierarchicalClustering(const Mat& features, Mat& centers, const KMeansIndexParams& params)
{
    CV_Assert(features.type() == CvType<ELEM_TYPE>::type());
    CV_Assert(features.isContinuous());
    ::cvflann::Matrix<ELEM_TYPE> m_features((ELEM_TYPE*)features.ptr<ELEM_TYPE>(0), features.rows, features.cols);
    
    CV_Assert(centers.type() == CvType<DIST_TYPE>::type());
    CV_Assert(centers.isContinuous());
    ::cvflann::Matrix<DIST_TYPE> m_centers((DIST_TYPE*)centers.ptr<DIST_TYPE>(0), centers.rows, centers.cols);
    
    return ::cvflann::hierarchicalClustering<ELEM_TYPE,DIST_TYPE>(m_features, m_centers, params);
}

} } // namespace cv::flann

#endif // __cplusplus

#endif
