// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_ANNOY_HPP
#define OPENCV_ANNOY_HPP

#include "opencv2/features2d/annoy/annoylib.h"
// #include "opencv2/features2d/annoy/kissrandom.h"


namespace cv
{
namespace annoy
{

//! @addtogroup features2d_annoy
//! @{

typedef Annoy::Euclidean                            Euclidean;
typedef Annoy::Manhattan                            Manhattan;
typedef Annoy::Angular                              Angular;
typedef Annoy::Hamming                              Hamming;
typedef Annoy::DotProduct                           DotProduct;
// typedef Annoy::Kiss32Random                         Kiss32Random;
// typedef Annoy::Kiss64Random                         Kiss64Random;
typedef Annoy::AnnoyIndexSingleThreadedBuildPolicy  SingleThreaded;
// typedef Annoy::AnnoyIndexMultiThreadedBuildPolicy   MultiThreaded;

template <typename S, typename T, typename D, typename R, typename P>
class AnnoyIndex
{
public:
    AnnoyIndex(int dims);
    ~AnnoyIndex();

    // void addItem(S item, InputArray feature);
    void addItems(InputArray features);
    void build(int trees, int threads=-1);
    void save(const String& filename, bool prefault=false);
    void load(const String& filename, bool prefault=false);
    void unload();

    // void knnSearch(const std::vector<T>& query, std::vector<S>& indices, std::vector<T>& dists, int knn, int searchK=-1);
    void knnSearch(InputArray query, OutputArray indices, OutputArray dists, int knn, int searchK=-1);

    void getItemVector(S item, T* vec);
    T getDistance(S i, S j);
    S getItemNum();
    S getTreeNum();

    void buildOnDisk(const String& filename);
    void setSeed(R seed);

protected:
    int featureDim;
    Annoy::AnnoyIndex<S, T, D, R, P>* index;
};

template <typename S, typename T, typename D, typename R, typename P>
AnnoyIndex<S, T, D, R, P>::AnnoyIndex(int dims) : featureDim(dims)
{
    index = new Annoy::AnnoyIndex<S, T, D, R, P>(dims);
}

template <typename S, typename T, typename D, typename R, typename P>
AnnoyIndex<S, T, D, R, P>::~AnnoyIndex()
{
    index->unload();
    delete index;
}

template <typename S, typename T, typename D, typename R, typename P>
void AnnoyIndex<S, T, D, R, P>::addItems(InputArray features)
{
    Mat feat = features.getMat();
    int num = feat.rows;

    std::vector<T> vec(featureDim);
    for (S i = 0; i < num; ++i)
    {
        T *ptr = feat.ptr<T>(i);
        memcpy(vec.data(), ptr, featureDim*sizeof(T));
        index->add_item(i, vec.data());
    }
}

template <typename S, typename T, typename D, typename R, typename P>
void AnnoyIndex<S, T, D, R, P>::build(int trees, int threads)
{
    index->build(trees, threads);
}

template <typename S, typename T, typename D, typename R, typename P>
void AnnoyIndex<S, T, D, R, P>::save(const String& filename, bool prefault)
{
    index->save(filename, prefault);
}

template <typename S, typename T, typename D, typename R, typename P>
void AnnoyIndex<S, T, D, R, P>::load(const String& filename, bool prefault)
{
    index->load(filename, prefault);
}

// template <typename S, typename T, typename D, typename R, typename P>
// void AnnoyIndex<S, T, D, R, P>::knnSearch(const std::vector<T>& query, std::vector<S>& indices, std::vector<T>& dists, int knn, int searchK)
// {
// }

template <typename S, typename T, typename D, typename R, typename P>
void AnnoyIndex<S, T, D, R, P>::knnSearch(InputArray query, OutputArray indices, OutputArray dists, int knn, int searchK)
{
    Mat vectors = query.getMat();
    int num = vectors.rows;

    Mat mind = indices.getMat();
    Mat mdist = dists.getMat();

    std::vector<T> vec(featureDim);
    for (S i = 0; i < num; ++i)
    {
        std::vector<S> nns;
        std::vector<T> distances;

        // i-th query descriptor
        // float* data = (float*)descriptors1.data;
        T *ptr = vectors.ptr<T>(i);
        memcpy(vec.data(), ptr, featureDim*sizeof(T));
        // getting the nearest neighbours
        index->get_nns_by_vector(vec.data(), knn, searchK, &nns, &distances);

        S *indPtr = mind.ptr<S>(i);
        T *distPtr = mdist.ptr<T>(i);
    }
}

template <typename S, typename T, typename D, typename R, typename P>
void AnnoyIndex<S, T, D, R, P>::getItemVector(S item, T* vec)
{
    index->get_item(item, vec);
}

template <typename S, typename T, typename D, typename R, typename P>
T AnnoyIndex<S, T, D, R, P>::getDistance(S i, S j)
{
    return index->get_distance(i, j);
}

template <typename S, typename T, typename D, typename R, typename P>
S AnnoyIndex<S, T, D, R, P>::getItemNum()
{
    return index->get_n_items();
}

template <typename S, typename T, typename D, typename R, typename P>
S AnnoyIndex<S, T, D, R, P>::getTreeNum()
{
    return index->get_n_trees();
}

template <typename S, typename T, typename D, typename R, typename P>
void AnnoyIndex<S, T, D, R, P>::buildOnDisk(const String& filename)
{
    index->on_disk_build(filename);
}

template <typename S, typename T, typename D, typename R, typename P>
void AnnoyIndex<S, T, D, R, P>::setSeed(R seed)
{
    index->set_seed(seed);
}

//! @}
} } // namespace cv::annoy


#endif