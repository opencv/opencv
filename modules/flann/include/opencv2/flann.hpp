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

#ifndef OPENCV_FLANN_HPP
#define OPENCV_FLANN_HPP

#include "opencv2/core.hpp"
#include "opencv2/flann/miniflann.hpp"
#include "opencv2/flann/flann_base.hpp"

/**
@defgroup flann Clustering and Search in Multi-Dimensional Spaces

This section documents OpenCV's interface to the FLANN library. FLANN (Fast Library for Approximate
Nearest Neighbors) is a library that contains a collection of algorithms optimized for fast nearest
neighbor search in large datasets and for high dimensional features. More information about FLANN
can be found in @cite Muja2009 .
*/

namespace cvflann
{
    CV_EXPORTS flann_distance_t flann_distance_type();
    CV_DEPRECATED CV_EXPORTS void set_distance_type(flann_distance_t distance_type, int order);
}


namespace cv
{
namespace flann
{


//! @addtogroup flann
//! @{

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


/** @brief The FLANN nearest neighbor index class. This class is templated with the type of elements for which
the index is built.

`Distance` functor specifies the metric to be used to calculate the distance between two points.
There are several `Distance` functors that are readily available:

@link cvflann::L2_Simple cv::flann::L2_Simple @endlink- Squared Euclidean distance functor.
This is the simpler, unrolled version. This is preferable for very low dimensionality data (eg 3D points)

@link cvflann::L2 cv::flann::L2 @endlink- Squared Euclidean distance functor, optimized version.

@link cvflann::L1 cv::flann::L1 @endlink - Manhattan distance functor, optimized version.

@link cvflann::MinkowskiDistance cv::flann::MinkowskiDistance @endlink -  The Minkowsky distance functor.
This is highly optimised with loop unrolling.
The computation of squared root at the end is omitted for efficiency.

@link cvflann::MaxDistance cv::flann::MaxDistance @endlink - The max distance functor. It computes the
maximum distance between two vectors. This distance is not a valid kdtree distance, it's not
dimensionwise additive.

@link cvflann::HammingLUT cv::flann::HammingLUT @endlink -  %Hamming distance functor. It counts the bit
differences between two strings using a lookup table implementation.

@link cvflann::Hamming cv::flann::Hamming @endlink - %Hamming distance functor. Population count is
performed using library calls, if available. Lookup table implementation is used as a fallback.

@link cvflann::Hamming2 cv::flann::Hamming2 @endlink- %Hamming distance functor. Population count is
implemented in 12 arithmetic operations (one of which is multiplication).

@link cvflann::HistIntersectionDistance cv::flann::HistIntersectionDistance @endlink - The histogram
intersection distance functor.

@link cvflann::HellingerDistance cv::flann::HellingerDistance @endlink - The Hellinger distance functor.

@link cvflann::ChiSquareDistance cv::flann::ChiSquareDistance @endlink - The chi-square distance functor.

@link cvflann::KL_Divergence cv::flann::KL_Divergence @endlink - The Kullback-Leibler divergence functor.

Although the provided implementations cover a vast range of cases, it is also possible to use
a custom implementation. The distance functor is a class whose `operator()` computes the distance
between two features. If the distance is also a kd-tree compatible distance, it should also provide an
`accum_dist()` method that computes the distance between individual feature dimensions.

In addition to `operator()` and `accum_dist()`, a distance functor should also define the
`ElementType` and the `ResultType` as the types of the elements it operates on and the type of the
result it computes. If a distance functor can be used as a kd-tree distance (meaning that the full
distance between a pair of features can be accumulated from the partial distances between the
individual dimensions) a typedef `is_kdtree_distance` should be present inside the distance functor.
If the distance is not a kd-tree distance, but it's a distance in a vector space (the individual
dimensions of the elements it operates on can be accessed independently) a typedef
`is_vector_space_distance` should be defined inside the functor. If neither typedef is defined, the
distance is assumed to be a metric distance and will only be used with indexes operating on
generic metric distances.
 */
template <typename Distance>
class GenericIndex
{
public:
        typedef typename Distance::ElementType ElementType;
        typedef typename Distance::ResultType DistanceType;

        /** @brief Constructs a nearest neighbor search index for a given dataset.

        @param features Matrix of containing the features(points) to index. The size of the matrix is
        num_features x feature_dimensionality and the data type of the elements in the matrix must
        coincide with the type of the index.
        @param params Structure containing the index parameters. The type of index that will be
        constructed depends on the type of this parameter. See the description.
        @param distance

        The method constructs a fast search structure from a set of features using the specified algorithm
        with specified parameters, as defined by params. params is a reference to one of the following class
        IndexParams descendants:

        - **LinearIndexParams** When passing an object of this type, the index will perform a linear,
        brute-force search. :
        @code
        struct LinearIndexParams : public IndexParams
        {
        };
        @endcode
        - **KDTreeIndexParams** When passing an object of this type the index constructed will consist of
        a set of randomized kd-trees which will be searched in parallel. :
        @code
        struct KDTreeIndexParams : public IndexParams
        {
            KDTreeIndexParams( int trees = 4 );
        };
        @endcode
        - **KMeansIndexParams** When passing an object of this type the index constructed will be a
        hierarchical k-means tree. :
        @code
        struct KMeansIndexParams : public IndexParams
        {
            KMeansIndexParams(
                int branching = 32,
                int iterations = 11,
                flann_centers_init_t centers_init = CENTERS_RANDOM,
                float cb_index = 0.2 );
        };
        @endcode
        - **CompositeIndexParams** When using a parameters object of this type the index created
        combines the randomized kd-trees and the hierarchical k-means tree. :
        @code
        struct CompositeIndexParams : public IndexParams
        {
            CompositeIndexParams(
                int trees = 4,
                int branching = 32,
                int iterations = 11,
                flann_centers_init_t centers_init = CENTERS_RANDOM,
                float cb_index = 0.2 );
        };
        @endcode
        - **LshIndexParams** When using a parameters object of this type the index created uses
        multi-probe LSH (by Multi-Probe LSH: Efficient Indexing for High-Dimensional Similarity Search
        by Qin Lv, William Josephson, Zhe Wang, Moses Charikar, Kai Li., Proceedings of the 33rd
        International Conference on Very Large Data Bases (VLDB). Vienna, Austria. September 2007) :
        @code
        struct LshIndexParams : public IndexParams
        {
            LshIndexParams(
                unsigned int table_number,
                unsigned int key_size,
                unsigned int multi_probe_level );
        };
        @endcode
        - **AutotunedIndexParams** When passing an object of this type the index created is
        automatically tuned to offer the best performance, by choosing the optimal index type
        (randomized kd-trees, hierarchical kmeans, linear) and parameters for the dataset provided. :
        @code
        struct AutotunedIndexParams : public IndexParams
        {
            AutotunedIndexParams(
                float target_precision = 0.9,
                float build_weight = 0.01,
                float memory_weight = 0,
                float sample_fraction = 0.1 );
        };
        @endcode
        - **SavedIndexParams** This object type is used for loading a previously saved index from the
        disk. :
        @code
        struct SavedIndexParams : public IndexParams
        {
            SavedIndexParams( String filename );
        };
        @endcode
         */
        GenericIndex(const Mat& features, const ::cvflann::IndexParams& params, Distance distance = Distance());

        ~GenericIndex();

        /** @brief Performs a K-nearest neighbor search for a given query point using the index.

        @param query The query point
        @param indices Vector that will contain the indices of the K-nearest neighbors found. It must have
        at least knn size.
        @param dists Vector that will contain the distances to the K-nearest neighbors found. It must have
        at least knn size.
        @param knn Number of nearest neighbors to search for.
        @param params SearchParams
         */
        void knnSearch(const std::vector<ElementType>& query, std::vector<int>& indices,
                       std::vector<DistanceType>& dists, int knn, const ::cvflann::SearchParams& params);
        void knnSearch(const Mat& queries, Mat& indices, Mat& dists, int knn, const ::cvflann::SearchParams& params);

        /** @brief Performs a radius nearest neighbor search for a given query point using the index.

        @param query The query point.
        @param indices Vector that will contain the indices of the nearest neighbors found.
        @param dists Vector that will contain the distances to the nearest neighbors found. It has the same
        number of elements as indices.
        @param radius The search radius.
        @param params SearchParams

        This function returns the number of nearest neighbors found.
        */
        int radiusSearch(const std::vector<ElementType>& query, std::vector<int>& indices,
                         std::vector<DistanceType>& dists, DistanceType radius, const ::cvflann::SearchParams& params);
        int radiusSearch(const Mat& query, Mat& indices, Mat& dists,
                         DistanceType radius, const ::cvflann::SearchParams& params);

        void save(String filename) { nnIndex->save(filename); }

        int veclen() const { return nnIndex->veclen(); }

        int size() const { return (int)nnIndex->size(); }

        ::cvflann::IndexParams getParameters() { return nnIndex->getParameters(); }

        CV_DEPRECATED const ::cvflann::IndexParams* getIndexParameters() { return nnIndex->getIndexParameters(); }

private:
        ::cvflann::Index<Distance>* nnIndex;
        Mat _dataset;
};

//! @cond IGNORED

#define FLANN_DISTANCE_CHECK \
    if ( ::cvflann::flann_distance_type() != cvflann::FLANN_DIST_L2) { \
        printf("[WARNING] You are using cv::flann::Index (or cv::flann::GenericIndex) and have also changed "\
        "the distance using cvflann::set_distance_type. This is no longer working as expected "\
        "(cv::flann::Index always uses L2). You should create the index templated on the distance, "\
        "for example for L1 distance use: GenericIndex< L1<float> > \n"); \
    }


template <typename Distance>
GenericIndex<Distance>::GenericIndex(const Mat& dataset, const ::cvflann::IndexParams& params, Distance distance)
: _dataset(dataset)
{
    CV_Assert(dataset.type() == CvType<ElementType>::type());
    CV_Assert(dataset.isContinuous());
    ::cvflann::Matrix<ElementType> m_dataset((ElementType*)_dataset.ptr<ElementType>(0), _dataset.rows, _dataset.cols);

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
void GenericIndex<Distance>::knnSearch(const std::vector<ElementType>& query, std::vector<int>& indices, std::vector<DistanceType>& dists, int knn, const ::cvflann::SearchParams& searchParams)
{
    ::cvflann::Matrix<ElementType> m_query((ElementType*)&query[0], 1, query.size());
    ::cvflann::Matrix<int> m_indices(&indices[0], 1, indices.size());
    ::cvflann::Matrix<DistanceType> m_dists(&dists[0], 1, dists.size());

    FLANN_DISTANCE_CHECK

    nnIndex->knnSearch(m_query,m_indices,m_dists,knn,searchParams);
}


template <typename Distance>
void GenericIndex<Distance>::knnSearch(const Mat& queries, Mat& indices, Mat& dists, int knn, const ::cvflann::SearchParams& searchParams)
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
int GenericIndex<Distance>::radiusSearch(const std::vector<ElementType>& query, std::vector<int>& indices, std::vector<DistanceType>& dists, DistanceType radius, const ::cvflann::SearchParams& searchParams)
{
    ::cvflann::Matrix<ElementType> m_query((ElementType*)&query[0], 1, query.size());
    ::cvflann::Matrix<int> m_indices(&indices[0], 1, indices.size());
    ::cvflann::Matrix<DistanceType> m_dists(&dists[0], 1, dists.size());

    FLANN_DISTANCE_CHECK

    return nnIndex->radiusSearch(m_query,m_indices,m_dists,radius,searchParams);
}

template <typename Distance>
int GenericIndex<Distance>::radiusSearch(const Mat& query, Mat& indices, Mat& dists, DistanceType radius, const ::cvflann::SearchParams& searchParams)
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

//! @endcond

/**
 * @deprecated Use GenericIndex class instead
 */
template <typename T>
class Index_
{
public:
    typedef typename L2<T>::ElementType ElementType;
    typedef typename L2<T>::ResultType DistanceType;

    CV_DEPRECATED Index_(const Mat& dataset, const ::cvflann::IndexParams& params)
    {
        printf("[WARNING] The cv::flann::Index_<T> class is deperecated, use cv::flann::GenericIndex<Distance> instead\n");

        CV_Assert(dataset.type() == CvType<ElementType>::type());
        CV_Assert(dataset.isContinuous());
        ::cvflann::Matrix<ElementType> m_dataset((ElementType*)dataset.ptr<ElementType>(0), dataset.rows, dataset.cols);

        if ( ::cvflann::flann_distance_type() == cvflann::FLANN_DIST_L2 ) {
            nnIndex_L1 = NULL;
            nnIndex_L2 = new ::cvflann::Index< L2<ElementType> >(m_dataset, params);
        }
        else if ( ::cvflann::flann_distance_type() == cvflann::FLANN_DIST_L1 ) {
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
    CV_DEPRECATED ~Index_()
    {
        if (nnIndex_L1) delete nnIndex_L1;
        if (nnIndex_L2) delete nnIndex_L2;
    }

    CV_DEPRECATED void knnSearch(const std::vector<ElementType>& query, std::vector<int>& indices, std::vector<DistanceType>& dists, int knn, const ::cvflann::SearchParams& searchParams)
    {
        ::cvflann::Matrix<ElementType> m_query((ElementType*)&query[0], 1, query.size());
        ::cvflann::Matrix<int> m_indices(&indices[0], 1, indices.size());
        ::cvflann::Matrix<DistanceType> m_dists(&dists[0], 1, dists.size());

        if (nnIndex_L1) nnIndex_L1->knnSearch(m_query,m_indices,m_dists,knn,searchParams);
        if (nnIndex_L2) nnIndex_L2->knnSearch(m_query,m_indices,m_dists,knn,searchParams);
    }
    CV_DEPRECATED void knnSearch(const Mat& queries, Mat& indices, Mat& dists, int knn, const ::cvflann::SearchParams& searchParams)
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

    CV_DEPRECATED int radiusSearch(const std::vector<ElementType>& query, std::vector<int>& indices, std::vector<DistanceType>& dists, DistanceType radius, const ::cvflann::SearchParams& searchParams)
    {
        ::cvflann::Matrix<ElementType> m_query((ElementType*)&query[0], 1, query.size());
        ::cvflann::Matrix<int> m_indices(&indices[0], 1, indices.size());
        ::cvflann::Matrix<DistanceType> m_dists(&dists[0], 1, dists.size());

        if (nnIndex_L1) return nnIndex_L1->radiusSearch(m_query,m_indices,m_dists,radius,searchParams);
        if (nnIndex_L2) return nnIndex_L2->radiusSearch(m_query,m_indices,m_dists,radius,searchParams);
    }

    CV_DEPRECATED int radiusSearch(const Mat& query, Mat& indices, Mat& dists, DistanceType radius, const ::cvflann::SearchParams& searchParams)
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

    CV_DEPRECATED void save(String filename)
    {
        if (nnIndex_L1) nnIndex_L1->save(filename);
        if (nnIndex_L2) nnIndex_L2->save(filename);
    }

    CV_DEPRECATED int veclen() const
    {
        if (nnIndex_L1) return nnIndex_L1->veclen();
        if (nnIndex_L2) return nnIndex_L2->veclen();
    }

    CV_DEPRECATED int size() const
    {
        if (nnIndex_L1) return nnIndex_L1->size();
        if (nnIndex_L2) return nnIndex_L2->size();
    }

    CV_DEPRECATED ::cvflann::IndexParams getParameters()
    {
        if (nnIndex_L1) return nnIndex_L1->getParameters();
        if (nnIndex_L2) return nnIndex_L2->getParameters();

    }

    CV_DEPRECATED const ::cvflann::IndexParams* getIndexParameters()
    {
        if (nnIndex_L1) return nnIndex_L1->getIndexParameters();
        if (nnIndex_L2) return nnIndex_L2->getIndexParameters();
    }

private:
    // providing backwards compatibility for L2 and L1 distances (most common)
    ::cvflann::Index< L2<ElementType> >* nnIndex_L2;
    ::cvflann::Index< L1<ElementType> >* nnIndex_L1;
};


/** @brief Clusters features using hierarchical k-means algorithm.

@param features The points to be clustered. The matrix must have elements of type
Distance::ElementType.
@param centers The centers of the clusters obtained. The matrix must have type
Distance::ResultType. The number of rows in this matrix represents the number of clusters desired,
however, because of the way the cut in the hierarchical tree is chosen, the number of clusters
computed will be the highest number of the form (branching-1)\*k+1 that's lower than the number of
clusters desired, where branching is the tree's branching factor (see description of the
KMeansIndexParams).
@param params Parameters used in the construction of the hierarchical k-means tree.
@param d Distance to be used for clustering.

The method clusters the given feature vectors by constructing a hierarchical k-means tree and
choosing a cut in the tree that minimizes the cluster's variance. It returns the number of clusters
found.
 */
template <typename Distance>
int hierarchicalClustering(const Mat& features, Mat& centers, const ::cvflann::KMeansIndexParams& params,
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

/** @deprecated
*/
template <typename ELEM_TYPE, typename DIST_TYPE>
CV_DEPRECATED int hierarchicalClustering(const Mat& features, Mat& centers, const ::cvflann::KMeansIndexParams& params)
{
    printf("[WARNING] cv::flann::hierarchicalClustering<ELEM_TYPE,DIST_TYPE> is deprecated, use "
        "cv::flann::hierarchicalClustering<Distance> instead\n");

    if ( ::cvflann::flann_distance_type() == cvflann::FLANN_DIST_L2 ) {
        return hierarchicalClustering< L2<ELEM_TYPE> >(features, centers, params);
    }
    else if ( ::cvflann::flann_distance_type() == cvflann::FLANN_DIST_L1 ) {
        return hierarchicalClustering< L1<ELEM_TYPE> >(features, centers, params);
    }
    else {
        printf("[ERROR] cv::flann::hierarchicalClustering<ELEM_TYPE,DIST_TYPE> only provides backwards "
        "compatibility for the L1 and L2 distances. "
        "For other distance types you must use cv::flann::hierarchicalClustering<Distance>\n");
        CV_Assert(0);
    }
}

//! @} flann

} } // namespace cv::flann

#endif
