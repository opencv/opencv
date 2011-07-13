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

#include "opencv2/core/types_c.h"
#include "opencv2/core/core.hpp"
#include "opencv2/flann/flann_base.hpp"
#include "opencv2/flann/miniflann.hpp"

namespace cvflann
{
    CV_EXPORTS flann_distance_t flann_distance_type();
    FLANN_DEPRECATED CV_EXPORTS void set_distance_type(flann_distance_t distance_type, int order);
}


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


// bring the flann parameters into this namespace
using ::cvflann::get_param;
using ::cvflann::print_params;

// bring the flann distances into this namespace
using ::cvflann::L2_Simple;
using ::cvflann::L2;
using ::cvflann::L1;
using ::cvflann::MinkowskiDistance;
using ::cvflann::MaxDistance;
using ::cvflann::HammingLUT;
using ::cvflann::Hamming;
using ::cvflann::Hamming2;
using ::cvflann::HistIntersectionDistance;
using ::cvflann::HellingerDistance;
using ::cvflann::ChiSquareDistance;
using ::cvflann::KL_Divergence;



template <typename Distance>
class GenericIndex 
{
public:
        typedef typename Distance::ElementType ElementType;
        typedef typename Distance::ResultType DistanceType;

        GenericIndex(const Mat& features, const IndexParams& params, Distance distance = Distance());

        ~GenericIndex();

        void knnSearch(const vector<ElementType>& query, vector<int>& indices, 
                       vector<DistanceType>& dists, int knn, const SearchParams& params);
        void knnSearch(const Mat& queries, Mat& indices, Mat& dists, int knn, const SearchParams& params);

        int radiusSearch(const vector<ElementType>& query, vector<int>& indices, 
                         vector<DistanceType>& dists, DistanceType radius, const SearchParams& params);
        int radiusSearch(const Mat& query, Mat& indices, Mat& dists, 
                         DistanceType radius, const SearchParams& params);

        void save(std::string filename) { nnIndex->save(filename); }

        int veclen() const { return nnIndex->veclen(); }

        int size() const { return nnIndex->size(); }

        IndexParams getParameters() { return nnIndex->getParameters(); }

        FLANN_DEPRECATED const IndexParams* getIndexParameters() { return nnIndex->getIndexParameters(); }

private:
        ::cvflann::Index<Distance>* nnIndex;
};


#define FLANN_DISTANCE_CHECK \
    if ( ::cvflann::flann_distance_type() != FLANN_DIST_L2) { \
        printf("[WARNING] You are using cv::flann::Index (or cv::flann::GenericIndex) and have also changed "\
        "the distance using cvflann::set_distance_type. This is no longer working as expected "\
        "(cv::flann::Index always uses L2). You should create the index templated on the distance, "\
        "for example for L1 distance use: GenericIndex< L1<float> > \n"); \
    }
    

template <typename Distance>
GenericIndex<Distance>::GenericIndex(const Mat& dataset, const IndexParams& params, Distance distance)
{
    CV_Assert(dataset.type() == CvType<ElementType>::type());
    CV_Assert(dataset.isContinuous());
    ::cvflann::Matrix<ElementType> m_dataset((ElementType*)dataset.ptr<ElementType>(0), dataset.rows, dataset.cols);
    
    nnIndex = new ::cvflann::Index<Distance>(m_dataset, params, distance);
    
    FLANN_DISTANCE_CHECK
    
    nnIndex->buildIndex();
}

template <typename Distance>
GenericIndex<Distance>::~GenericIndex()
{
    delete nnIndex;
}

template <typename Distance>
void GenericIndex<Distance>::knnSearch(const vector<ElementType>& query, vector<int>& indices, vector<DistanceType>& dists, int knn, const SearchParams& searchParams)
{
    ::cvflann::Matrix<ElementType> m_query((ElementType*)&query[0], 1, query.size());
    ::cvflann::Matrix<int> m_indices(&indices[0], 1, indices.size());
    ::cvflann::Matrix<DistanceType> m_dists(&dists[0], 1, dists.size());

    FLANN_DISTANCE_CHECK

    nnIndex->knnSearch(m_query,m_indices,m_dists,knn,searchParams);
}


template <typename Distance>
void GenericIndex<Distance>::knnSearch(const Mat& queries, Mat& indices, Mat& dists, int knn, const SearchParams& searchParams)
{
    CV_Assert(queries.type() == CvType<ElementType>::type());
    CV_Assert(queries.isContinuous());
    ::cvflann::Matrix<ElementType> m_queries((ElementType*)queries.ptr<ElementType>(0), queries.rows, queries.cols);
    
    CV_Assert(indices.type() == CV_32S);
    CV_Assert(indices.isContinuous());
    ::cvflann::Matrix<int> m_indices((int*)indices.ptr<int>(0), indices.rows, indices.cols);
    
    CV_Assert(dists.type() == CvType<DistanceType>::type());
    CV_Assert(dists.isContinuous());
    ::cvflann::Matrix<DistanceType> m_dists((DistanceType*)dists.ptr<DistanceType>(0), dists.rows, dists.cols);

    FLANN_DISTANCE_CHECK
    
    nnIndex->knnSearch(m_queries,m_indices,m_dists,knn, searchParams);
}

template <typename Distance>
int GenericIndex<Distance>::radiusSearch(const vector<ElementType>& query, vector<int>& indices, vector<DistanceType>& dists, DistanceType radius, const SearchParams& searchParams)
{
    ::cvflann::Matrix<ElementType> m_query((ElementType*)&query[0], 1, query.size());
    ::cvflann::Matrix<int> m_indices(&indices[0], 1, indices.size());
    ::cvflann::Matrix<DistanceType> m_dists(&dists[0], 1, dists.size());

    FLANN_DISTANCE_CHECK
    
    return nnIndex->radiusSearch(m_query,m_indices,m_dists,radius,searchParams);
}

template <typename Distance>
int GenericIndex<Distance>::radiusSearch(const Mat& query, Mat& indices, Mat& dists, DistanceType radius, const SearchParams& searchParams)
{
    CV_Assert(query.type() == CvType<ElementType>::type());
    CV_Assert(query.isContinuous());
    ::cvflann::Matrix<ElementType> m_query((ElementType*)query.ptr<ElementType>(0), query.rows, query.cols);
    
    CV_Assert(indices.type() == CV_32S);
    CV_Assert(indices.isContinuous());
    ::cvflann::Matrix<int> m_indices((int*)indices.ptr<int>(0), indices.rows, indices.cols);
    
    CV_Assert(dists.type() == CvType<DistanceType>::type());
    CV_Assert(dists.isContinuous());
    ::cvflann::Matrix<DistanceType> m_dists((DistanceType*)dists.ptr<DistanceType>(0), dists.rows, dists.cols);
    
    FLANN_DISTANCE_CHECK
    
    return nnIndex->radiusSearch(m_query,m_indices,m_dists,radius,searchParams);
}

/**
 * @deprecated Use GenericIndex class instead
 */
template <typename T>
class FLANN_DEPRECATED Index_ {
public:
        typedef typename L2<T>::ElementType ElementType;
        typedef typename L2<T>::ResultType DistanceType;

	Index_(const Mat& features, const IndexParams& params);

	~Index_();

	void knnSearch(const vector<ElementType>& query, vector<int>& indices, vector<DistanceType>& dists, int knn, const SearchParams& params);
	void knnSearch(const Mat& queries, Mat& indices, Mat& dists, int knn, const SearchParams& params);

	int radiusSearch(const vector<ElementType>& query, vector<int>& indices, vector<DistanceType>& dists, DistanceType radius, const SearchParams& params);
	int radiusSearch(const Mat& query, Mat& indices, Mat& dists, DistanceType radius, const SearchParams& params);

	void save(std::string filename) 
        { 
            if (nnIndex_L1) nnIndex_L1->save(filename);
            if (nnIndex_L2) nnIndex_L2->save(filename);
        }

	int veclen() const 
	{ 
            if (nnIndex_L1) return nnIndex_L1->veclen();
            if (nnIndex_L2) return nnIndex_L2->veclen();            
        }

	int size() const 
	{ 
            if (nnIndex_L1) return nnIndex_L1->size();
            if (nnIndex_L2) return nnIndex_L2->size(); 
        }

        IndexParams getParameters() 
        { 
            if (nnIndex_L1) return nnIndex_L1->getParameters();
            if (nnIndex_L2) return nnIndex_L2->getParameters();
            
        }

        FLANN_DEPRECATED const IndexParams* getIndexParameters() 
        { 
            if (nnIndex_L1) return nnIndex_L1->getIndexParameters();
            if (nnIndex_L2) return nnIndex_L2->getIndexParameters(); 
        }

private:
        // providing backwards compatibility for L2 and L1 distances (most common)
        ::cvflann::Index< L2<ElementType> >* nnIndex_L2;
        ::cvflann::Index< L1<ElementType> >* nnIndex_L1;
};


template <typename T>
Index_<T>::Index_(const Mat& dataset, const IndexParams& params)
{
    printf("[WARNING] The cv::flann::Index_<T> class is deperecated, use cv::flann::GenericIndex<Distance> instead\n");
    
    CV_Assert(dataset.type() == CvType<ElementType>::type());
    CV_Assert(dataset.isContinuous());
    ::cvflann::Matrix<ElementType> m_dataset((ElementType*)dataset.ptr<ElementType>(0), dataset.rows, dataset.cols);
    
    if ( ::cvflann::flann_distance_type() == FLANN_DIST_L2 ) {
        nnIndex_L1 = NULL;
        nnIndex_L2 = new ::cvflann::Index< L2<ElementType> >(m_dataset, params);
    }
    else if ( ::cvflann::flann_distance_type() == FLANN_DIST_L1 ) {
        nnIndex_L1 = new ::cvflann::Index< L1<ElementType> >(m_dataset, params);
        nnIndex_L2 = NULL;        
    }
    else {
        printf("[ERROR] cv::flann::Index_<T> only provides backwards compatibility for the L1 and L2 distances. "
        "For other distance types you must use cv::flann::GenericIndex<Distance>\n");
        CV_Assert(0);
    }
    if (nnIndex_L1) nnIndex_L1->buildIndex();
    if (nnIndex_L2) nnIndex_L2->buildIndex();
}

template <typename T>
Index_<T>::~Index_()
{
    if (nnIndex_L1) delete nnIndex_L1;
    if (nnIndex_L2) delete nnIndex_L2;
}

template <typename T>
void Index_<T>::knnSearch(const vector<ElementType>& query, vector<int>& indices, vector<DistanceType>& dists, int knn, const SearchParams& searchParams)
{
    ::cvflann::Matrix<ElementType> m_query((ElementType*)&query[0], 1, query.size());
    ::cvflann::Matrix<int> m_indices(&indices[0], 1, indices.size());
    ::cvflann::Matrix<DistanceType> m_dists(&dists[0], 1, dists.size());
    
    if (nnIndex_L1) nnIndex_L1->knnSearch(m_query,m_indices,m_dists,knn,searchParams);
    if (nnIndex_L2) nnIndex_L2->knnSearch(m_query,m_indices,m_dists,knn,searchParams);
}


template <typename T>
void Index_<T>::knnSearch(const Mat& queries, Mat& indices, Mat& dists, int knn, const SearchParams& searchParams)
{
    CV_Assert(queries.type() == CvType<ElementType>::type());
    CV_Assert(queries.isContinuous());
    ::cvflann::Matrix<ElementType> m_queries((ElementType*)queries.ptr<ElementType>(0), queries.rows, queries.cols);
    
    CV_Assert(indices.type() == CV_32S);
    CV_Assert(indices.isContinuous());
    ::cvflann::Matrix<int> m_indices((int*)indices.ptr<int>(0), indices.rows, indices.cols);
    
    CV_Assert(dists.type() == CvType<DistanceType>::type());
    CV_Assert(dists.isContinuous());
    ::cvflann::Matrix<DistanceType> m_dists((DistanceType*)dists.ptr<DistanceType>(0), dists.rows, dists.cols);

    if (nnIndex_L1) nnIndex_L1->knnSearch(m_queries,m_indices,m_dists,knn, searchParams);
    if (nnIndex_L2) nnIndex_L2->knnSearch(m_queries,m_indices,m_dists,knn, searchParams);
}

template <typename T>
int Index_<T>::radiusSearch(const vector<ElementType>& query, vector<int>& indices, vector<DistanceType>& dists, DistanceType radius, const SearchParams& searchParams)
{
    ::cvflann::Matrix<ElementType> m_query((ElementType*)&query[0], 1, query.size());
    ::cvflann::Matrix<int> m_indices(&indices[0], 1, indices.size());
    ::cvflann::Matrix<DistanceType> m_dists(&dists[0], 1, dists.size());
    
    if (nnIndex_L1) return nnIndex_L1->radiusSearch(m_query,m_indices,m_dists,radius,searchParams);
    if (nnIndex_L2) return nnIndex_L2->radiusSearch(m_query,m_indices,m_dists,radius,searchParams);
}

template <typename T>
int Index_<T>::radiusSearch(const Mat& query, Mat& indices, Mat& dists, DistanceType radius, const SearchParams& searchParams)
{
    CV_Assert(query.type() == CvType<ElementType>::type());
    CV_Assert(query.isContinuous());
    ::cvflann::Matrix<ElementType> m_query((ElementType*)query.ptr<ElementType>(0), query.rows, query.cols);
    
    CV_Assert(indices.type() == CV_32S);
    CV_Assert(indices.isContinuous());
    ::cvflann::Matrix<int> m_indices((int*)indices.ptr<int>(0), indices.rows, indices.cols);
    
    CV_Assert(dists.type() == CvType<DistanceType>::type());
    CV_Assert(dists.isContinuous());
    ::cvflann::Matrix<DistanceType> m_dists((DistanceType*)dists.ptr<DistanceType>(0), dists.rows, dists.cols);
    
    if (nnIndex_L1) return nnIndex_L1->radiusSearch(m_query,m_indices,m_dists,radius,searchParams);
    if (nnIndex_L2) return nnIndex_L2->radiusSearch(m_query,m_indices,m_dists,radius,searchParams);
}


template <typename Distance>
int hierarchicalClustering(const Mat& features, Mat& centers, const KMeansIndexParams& params,
                           Distance d = Distance())
{
    typedef typename Distance::ElementType ElementType;
    typedef typename Distance::ResultType DistanceType;
    
    CV_Assert(features.type() == CvType<ElementType>::type());
    CV_Assert(features.isContinuous());
    ::cvflann::Matrix<ElementType> m_features((ElementType*)features.ptr<ElementType>(0), features.rows, features.cols);
    
    CV_Assert(centers.type() == CvType<DistanceType>::type());
    CV_Assert(centers.isContinuous());
    ::cvflann::Matrix<DistanceType> m_centers((DistanceType*)centers.ptr<DistanceType>(0), centers.rows, centers.cols);

    return ::cvflann::hierarchicalClustering<Distance>(m_features, m_centers, params, d);
}


template <typename ELEM_TYPE, typename DIST_TYPE>
FLANN_DEPRECATED int hierarchicalClustering(const Mat& features, Mat& centers, const KMeansIndexParams& params)
{
    printf("[WARNING] cv::flann::hierarchicalClustering<ELEM_TYPE,DIST_TYPE> is deprecated, use "
        "cv::flann::hierarchicalClustering<Distance> instead\n");
        
    if ( ::cvflann::flann_distance_type() == FLANN_DIST_L2 ) {
        return hierarchicalClustering< L2<ELEM_TYPE> >(features, centers, params);
    }
    else if ( ::cvflann::flann_distance_type() == FLANN_DIST_L1 ) {
        return hierarchicalClustering< L1<ELEM_TYPE> >(features, centers, params);
    }
    else {
        printf("[ERROR] cv::flann::hierarchicalClustering<ELEM_TYPE,DIST_TYPE> only provides backwards "
        "compatibility for the L1 and L2 distances. "
        "For other distance types you must use cv::flann::hierarchicalClustering<Distance>\n");
        CV_Assert(0);
    }
}

} } // namespace cv::flann

#endif // __cplusplus

#endif
