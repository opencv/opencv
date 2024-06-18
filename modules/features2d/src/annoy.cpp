// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "precomp.hpp"
#include "annoy/annoylib.h"
#include "annoy/kissrandom.h"
#include <iostream>

namespace cv
{

template <typename DataType, typename DistanceType>
class AnnoyIndexImpl : public AnnoyIndex
{
public:
    AnnoyIndexImpl(int dim)
        : dim(dim)
        , index(makePtr<Annoy::AnnoyIndex<int, DataType, DistanceType, Annoy::Kiss32Random, Annoy::AnnoyIndexMultiThreadedBuildPolicy>>(dim))
    {}

    void addItems(InputArray _dataset) CV_OVERRIDE
    {
        Mat features = _dataset.getMat();
        if (features.cols != dim)
            CV_Error(Error::StsBadArg, "Wrong data dimension.");

        int num = features.rows;
        for (int i = 0; i < num; ++i)
        {
            index->add_item(i, features.ptr<DataType>(i));
        }
    }

    bool build(int trees) CV_OVERRIDE
    {
        return index->build(trees);
    }

    void knnSearch(InputArray _query, OutputArray _indices, OutputArray _dists, int knn, int search_k) CV_OVERRIDE
    {
        Mat query = _query.getMat(), indices, dists;
        int numQuery = query.rows;

        if (_indices.needed())
        {
            indices = _indices.getMat();
            if(!indices.isContinuous() || indices.type() != CV_32S ||
                indices.rows != numQuery || indices.cols != knn)
            {
                if(!indices.isContinuous())
                    _indices.release();
                _indices.create(numQuery, knn, CV_32S);
                indices = _indices.getMat();
            }
        }
        else
            indices.create(numQuery, knn, CV_32S);

        if(_dists.needed())
        {
            dists = _dists.getMat();
            if(!dists.isContinuous() || dists.type() != cv::DataType<DataType>::type ||
                dists.rows != numQuery || dists.cols != knn)
            {
                if(!_dists.isContinuous())
                    _dists.release();
                _dists.create(numQuery, knn, cv::DataType<DataType>::type);
                dists = _dists.getMat();
            }
        }
        else
            dists.create(numQuery, knn, cv::DataType<DataType>::type);

        auto processBatch = [&](const Range& range)
        {
            std::vector<int> nns;
            std::vector<DataType> distances;

            for (int i = range.start; i < range.end; ++i)
            {
                index->get_nns_by_vector(query.ptr<DataType>(i), knn, search_k, &nns, &distances);

                int num = nns.size();
                memcpy(indices.ptr<int>(i), nns.data(), num*sizeof(int));
                memcpy(dists.ptr<DataType>(i), distances.data(), num*sizeof(DataType));

                nns.clear();
                distances.clear();
            }
        };

        parallel_for_(Range(0, numQuery), processBatch);
    }

    bool save(const String &filename, bool prefault) CV_OVERRIDE
    {
        return index->save(filename.c_str(), prefault);
    }

    bool load(const String &filename, bool prefault) CV_OVERRIDE
    {
        return index->load(filename.c_str(), prefault);
    }

private:
    int dim;
    Ptr<Annoy::AnnoyIndex<int, DataType, DistanceType, Annoy::Kiss32Random, Annoy::AnnoyIndexMultiThreadedBuildPolicy>> index;
};

Ptr<AnnoyIndex> AnnoyIndex::create(int dim, AnnoyIndex::Distance distType)
{
    switch (distType)
    {
        case AnnoyIndex::ANNOY_DIST_EUCLIDEAN:
            return makePtr<AnnoyIndexImpl<float, Annoy::Euclidean>>(dim);
            break;
        case AnnoyIndex::ANNOY_DIST_MANHATTAN:
            return makePtr<AnnoyIndexImpl<float, Annoy::Manhattan>>(dim);
            break;
        case AnnoyIndex::ANNOY_DIST_ANGULAR:
            return makePtr<AnnoyIndexImpl<float, Annoy::Angular>>(dim);
            break;
        case AnnoyIndex::ANNOY_DIST_HAMMING:
            return makePtr<AnnoyIndexImpl<uchar, Annoy::Hamming>>(dim);
            break;
        case AnnoyIndex::ANNOY_DIST_DOTPRODUCT:
            return makePtr<AnnoyIndexImpl<float, Annoy::DotProduct>>(dim);
            break;
        default:
            CV_Error(Error::StsBadArg, "Unknown/unsupported distance type");
    }
};

}