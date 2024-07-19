// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "precomp.hpp"

#include "annoy/annoylib.h"
#include "annoy/kissrandom.h"

#include <opencv2/core/utils/logger.hpp>


namespace cv
{

template <typename DataType, typename DistanceType>
class ANNIndexImpl : public ANNIndex
{
public:
    ANNIndexImpl(int dimension) : dim(dimension)
    {
        index = makePtr<::cvannoy::AnnoyIndex<int, DataType, DistanceType, ::cvannoy::Kiss32Random, ::cvannoy::AnnoyIndexSingleThreadedBuildPolicy>>(dimension);
    }

    bool addItems(InputArray _dataset) CV_OVERRIDE
    {
        CV_Assert(!_dataset.empty());

        Mat features = _dataset.getMat();
        CV_Assert(features.cols == dim);
        CV_Assert(features.type() == cv::DataType<DataType>::type);

        int num = features.rows;
        String errorMsg;
        char* msg = const_cast<char*>(errorMsg.c_str());
        if(!index->add_item(0, features.ptr<DataType>(0), &msg))
        {
            CV_LOG_ERROR(NULL, errorMsg);
            return false;
        }
        for (int i = 1; i < num; ++i)
            index->add_item(i, features.ptr<DataType>(i));

        return true;
    }

    bool build(int trees) CV_OVERRIDE
    {
        if (trees <= 0)
            trees = -1;

        String errorMsg;
        char* msg = const_cast<char*>(errorMsg.c_str());
        if (index->build(trees, -1, &msg))
            return true;
        else
        {
            CV_LOG_ERROR(NULL, errorMsg);
            return false;
        }
    }

    void knnSearch(InputArray _query, OutputArray _indices, OutputArray _dists, int knn, int search_k) CV_OVERRIDE
    {
        CV_Assert(!_query.empty() && _query.isContinuous());
        CV_Assert(knn <= index->get_n_items());

        Mat query = _query.getMat(), indices, dists;
        CV_Assert(query.type() == cv::DataType<DataType>::type);

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

                int num = int(nns.size());
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
        String errorMsg;
        char* msg = const_cast<char*>(errorMsg.c_str());
        if (index->save(filename.c_str(), prefault, &msg))
            return true;
        else
        {
            CV_LOG_ERROR(NULL, errorMsg);
            return false;
        }
    }

    bool load(const String &filename, bool prefault) CV_OVERRIDE
    {
        String errorMsg;
        char* msg = const_cast<char*>(errorMsg.c_str());
        if (index->load(filename.c_str(), prefault, &msg))
            return true;
        else
        {
            CV_LOG_ERROR(NULL, errorMsg);
            return false;
        }
    }

    int getTreeNumber() CV_OVERRIDE
    {
        return index->get_n_trees();
    }

    int getItemNumber() CV_OVERRIDE
    {
        return index->get_n_items();
    }

    bool setOnDiskBuild(const String &filename) CV_OVERRIDE
    {
        String errorMsg;
        char* msg = const_cast<char*>(errorMsg.c_str());
        if (index->on_disk_build(filename.c_str(), &msg))
            return true;
        else
        {
            CV_LOG_ERROR(NULL, errorMsg);
            return false;
        }
    }

    void setSeed(int seed) CV_OVERRIDE
    {
        CV_Assert(seed >= 0 && seed <= 4294967295);

        index->set_seed(static_cast<uint32_t>(seed));
    }

private:
    int dim;
    Ptr<::cvannoy::AnnoyIndex<int, DataType, DistanceType, ::cvannoy::Kiss32Random, ::cvannoy::AnnoyIndexSingleThreadedBuildPolicy>> index;
};

Ptr<ANNIndex> ANNIndex::create(int dim, ANNIndex::Distance distType)
{
    switch (distType)
    {
        case ANNIndex::DIST_EUCLIDEAN:
            return makePtr<ANNIndexImpl<float, ::cvannoy::Euclidean>>(dim);
            break;
        case ANNIndex::DIST_MANHATTAN:
            return makePtr<ANNIndexImpl<float, ::cvannoy::Manhattan>>(dim);
            break;
        case ANNIndex::DIST_ANGULAR:
            return makePtr<ANNIndexImpl<float, ::cvannoy::Angular>>(dim);
            break;
        case ANNIndex::DIST_HAMMING:
            return makePtr<ANNIndexImpl<uchar, ::cvannoy::Hamming>>(dim);
            break;
        case ANNIndex::DIST_DOTPRODUCT:
            return makePtr<ANNIndexImpl<float, ::cvannoy::DotProduct>>(dim);
            break;
        default:
            CV_Error(Error::StsBadArg, "Unknown/unsupported distance type");
    }
};

}