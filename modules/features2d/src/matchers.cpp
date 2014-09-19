/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
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
//   * The name of Intel Corporation may not be used to endorse or promote products
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

#include "precomp.hpp"
#include <limits>
#include "opencl_kernels_features2d.hpp"

#if defined(HAVE_EIGEN) && EIGEN_WORLD_VERSION == 2
#include <Eigen/Array>
#endif

namespace cv
{

/////////////////////// ocl functions for BFMatcher ///////////////////////////

static void ensureSizeIsEnough(int rows, int cols, int type, UMat &m)
{
    if (m.type() == type && m.rows >= rows && m.cols >= cols)
        m = m(Rect(0, 0, cols, rows));
    else
        m.create(rows, cols, type);
}


template < int BLOCK_SIZE, int MAX_DESC_LEN >
static bool ocl_matchUnrolledCached(InputArray _query, InputArray _train,
                     const UMat &trainIdx, const UMat &distance, int distType)
{
    int depth = _query.depth();
    cv::String opts;
    opts = cv::format("-D T=%s %s -D DIST_TYPE=%d -D BLOCK_SIZE=%d -D MAX_DESC_LEN=%d",
        ocl::typeToStr(depth), depth == CV_32F ? "-D T_FLOAT" : "", distType, (int)BLOCK_SIZE, (int)MAX_DESC_LEN );
    ocl::Kernel k("BruteForceMatch_UnrollMatch", ocl::features2d::brute_force_match_oclsrc, opts);
    if(k.empty())
        return false;

    size_t globalSize[] = {(_query.size().height + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE, BLOCK_SIZE, 1};
    size_t localSize[] = {BLOCK_SIZE, BLOCK_SIZE, 1};
    const size_t smemSize = (BLOCK_SIZE * (MAX_DESC_LEN >= BLOCK_SIZE ? MAX_DESC_LEN : BLOCK_SIZE) + BLOCK_SIZE * BLOCK_SIZE) * sizeof(int);

    if(globalSize[0] != 0)
    {
        UMat query = _query.getUMat(), train = _train.getUMat();

        int idx = 0;
        idx = k.set(idx, ocl::KernelArg::PtrReadOnly(query));
        idx = k.set(idx, ocl::KernelArg::PtrReadOnly(train));
        idx = k.set(idx, ocl::KernelArg::PtrWriteOnly(trainIdx));
        idx = k.set(idx, ocl::KernelArg::PtrWriteOnly(distance));
        idx = k.set(idx, (void *)NULL, smemSize);
        idx = k.set(idx, query.rows);
        idx = k.set(idx, query.cols);
        idx = k.set(idx, train.rows);
        idx = k.set(idx, train.cols);
        idx = k.set(idx, (int)query.step);

        return k.run(2, globalSize, localSize, false);
    }
    return true;
}

template < int BLOCK_SIZE >
static bool ocl_match(InputArray _query, InputArray _train,
                     const UMat &trainIdx, const UMat &distance, int distType)
{
    int depth = _query.depth();
    cv::String opts;
    opts = cv::format("-D T=%s %s -D DIST_TYPE=%d -D BLOCK_SIZE=%d",
        ocl::typeToStr(depth), depth == CV_32F ? "-D T_FLOAT" : "", distType, (int)BLOCK_SIZE);
    ocl::Kernel k("BruteForceMatch_Match", ocl::features2d::brute_force_match_oclsrc, opts);
    if(k.empty())
        return false;

    size_t globalSize[] = {(_query.size().height + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE, BLOCK_SIZE, 1};
    size_t localSize[] = {BLOCK_SIZE, BLOCK_SIZE, 1};
    const size_t smemSize = (2 * BLOCK_SIZE * BLOCK_SIZE) * sizeof(int);

    if(globalSize[0] != 0)
    {
        UMat query = _query.getUMat(), train = _train.getUMat();

        int idx = 0;
        idx = k.set(idx, ocl::KernelArg::PtrReadOnly(query));
        idx = k.set(idx, ocl::KernelArg::PtrReadOnly(train));
        idx = k.set(idx, ocl::KernelArg::PtrWriteOnly(trainIdx));
        idx = k.set(idx, ocl::KernelArg::PtrWriteOnly(distance));
        idx = k.set(idx, (void *)NULL, smemSize);
        idx = k.set(idx, query.rows);
        idx = k.set(idx, query.cols);
        idx = k.set(idx, train.rows);
        idx = k.set(idx, train.cols);
        idx = k.set(idx, (int)query.step);

        return k.run(2, globalSize, localSize, false);
    }
    return true;
}

static bool ocl_matchDispatcher(InputArray query, InputArray train,
                     const UMat &trainIdx, const UMat &distance, int distType)
{
    int query_cols = query.size().width;
    bool is_cpu = ocl::Device::getDefault().type() == ocl::Device::TYPE_CPU;
    if (query_cols <= 64)
    {
        if(!ocl_matchUnrolledCached<16, 64>(query, train, trainIdx, distance, distType)) return false;
    }
    else if (query_cols <= 128 && !is_cpu)
    {
        if(!ocl_matchUnrolledCached<16, 128>(query, train, trainIdx,  distance, distType)) return false;
    }
    else
    {
        if(!ocl_match<16>(query, train, trainIdx, distance, distType)) return false;
    }
    return true;
}

static bool ocl_matchSingle(InputArray query, InputArray train,
        UMat &trainIdx, UMat &distance, int dstType)
{
    if (query.empty() || train.empty())
        return false;

    int query_rows = query.size().height;

    ensureSizeIsEnough(1, query_rows, CV_32S, trainIdx);
    ensureSizeIsEnough(1, query_rows, CV_32F, distance);

    return ocl_matchDispatcher(query, train, trainIdx, distance, dstType);
}

static bool ocl_matchConvert(const Mat &trainIdx, const Mat &distance, std::vector< std::vector<DMatch> > &matches)
{
    if (trainIdx.empty() || distance.empty())
        return false;

    if( (trainIdx.type() != CV_32SC1) || (distance.type() != CV_32FC1 || distance.cols != trainIdx.cols) )
        return false;

    const int nQuery = trainIdx.cols;

    matches.clear();
    matches.reserve(nQuery);

    const int *trainIdx_ptr = trainIdx.ptr<int>();
    const float *distance_ptr =  distance.ptr<float>();
    for (int queryIdx = 0; queryIdx < nQuery; ++queryIdx, ++trainIdx_ptr, ++distance_ptr)
    {
        int trainIndex = *trainIdx_ptr;

        if (trainIndex == -1)
            continue;

        float dst = *distance_ptr;

        DMatch m(queryIdx, trainIndex, 0, dst);

        std::vector<DMatch> temp;
        temp.push_back(m);
        matches.push_back(temp);
    }
    return true;
}

static bool ocl_matchDownload(const UMat &trainIdx, const UMat &distance, std::vector< std::vector<DMatch> > &matches)
{
    if (trainIdx.empty() || distance.empty())
        return false;

    Mat trainIdxCPU = trainIdx.getMat(ACCESS_READ);
    Mat distanceCPU = distance.getMat(ACCESS_READ);

    return ocl_matchConvert(trainIdxCPU, distanceCPU, matches);
}

template < int BLOCK_SIZE, int MAX_DESC_LEN >
static bool ocl_knn_matchUnrolledCached(InputArray _query, InputArray _train,
                             const UMat &trainIdx, const UMat &distance, int distType)
{
    int depth = _query.depth();
    cv::String opts;
    opts = cv::format("-D T=%s %s -D DIST_TYPE=%d -D BLOCK_SIZE=%d -D MAX_DESC_LEN=%d",
        ocl::typeToStr(depth), depth == CV_32F ? "-D T_FLOAT" : "", distType, (int)BLOCK_SIZE, (int)MAX_DESC_LEN );
    ocl::Kernel k("BruteForceMatch_knnUnrollMatch", ocl::features2d::brute_force_match_oclsrc, opts);
    if(k.empty())
        return false;

    size_t globalSize[] = {(_query.size().height + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE, BLOCK_SIZE, 1};
    size_t localSize[] = {BLOCK_SIZE, BLOCK_SIZE, 1};
    const size_t smemSize = (BLOCK_SIZE * (MAX_DESC_LEN >= BLOCK_SIZE ? MAX_DESC_LEN : BLOCK_SIZE) + BLOCK_SIZE * BLOCK_SIZE) * sizeof(int);

    if(globalSize[0] != 0)
    {
        UMat query = _query.getUMat(), train = _train.getUMat();

        int idx = 0;
        idx = k.set(idx, ocl::KernelArg::PtrReadOnly(query));
        idx = k.set(idx, ocl::KernelArg::PtrReadOnly(train));
        idx = k.set(idx, ocl::KernelArg::PtrWriteOnly(trainIdx));
        idx = k.set(idx, ocl::KernelArg::PtrWriteOnly(distance));
        idx = k.set(idx, (void *)NULL, smemSize);
        idx = k.set(idx, query.rows);
        idx = k.set(idx, query.cols);
        idx = k.set(idx, train.rows);
        idx = k.set(idx, train.cols);
        idx = k.set(idx, (int)query.step);

        return k.run(2, globalSize, localSize, false);
    }
    return true;
}

template < int BLOCK_SIZE >
static bool ocl_knn_match(InputArray _query, InputArray _train,
               const UMat &trainIdx, const UMat &distance, int distType)
{
    int depth = _query.depth();
    cv::String opts;
    opts = format("-D T=%s %s -D DIST_TYPE=%d -D BLOCK_SIZE=%d",
        ocl::typeToStr(depth), depth == CV_32F ? "-D T_FLOAT" : "", distType, (int)BLOCK_SIZE);
    ocl::Kernel k("BruteForceMatch_knnMatch", ocl::features2d::brute_force_match_oclsrc, opts);
    if(k.empty())
        return false;

    size_t globalSize[] = {(_query.size().height + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE, BLOCK_SIZE, 1};
    size_t localSize[] = {BLOCK_SIZE, BLOCK_SIZE, 1};
    const size_t smemSize = (2 * BLOCK_SIZE * BLOCK_SIZE) * sizeof(int);

    if(globalSize[0] != 0)
    {
        UMat query = _query.getUMat(), train = _train.getUMat();

        int idx = 0;
        idx = k.set(idx, ocl::KernelArg::PtrReadOnly(query));
        idx = k.set(idx, ocl::KernelArg::PtrReadOnly(train));
        idx = k.set(idx, ocl::KernelArg::PtrWriteOnly(trainIdx));
        idx = k.set(idx, ocl::KernelArg::PtrWriteOnly(distance));
        idx = k.set(idx, (void*)NULL, smemSize);
        idx = k.set(idx, query.rows);
        idx = k.set(idx, query.cols);
        idx = k.set(idx, train.rows);
        idx = k.set(idx, train.cols);
        idx = k.set(idx, (int)query.step);

        return k.run(2, globalSize, localSize, false);
    }
    return true;
}

static bool ocl_match2Dispatcher(InputArray query, InputArray train, const UMat &trainIdx, const UMat &distance, int distType)
{
    bool is_cpu = ocl::Device::getDefault().type() == ocl::Device::TYPE_CPU;
    if (query.size().width <= 64)
    {
        if(!ocl_knn_matchUnrolledCached<16, 64>(query, train, trainIdx, distance, distType))
            return false;
    }
    else if (query.size().width <= 128 && !is_cpu)
    {
        if(!ocl_knn_matchUnrolledCached<16, 128>(query, train, trainIdx, distance, distType))
            return false;
    }
    else
    {
        if(!ocl_knn_match<16>(query, train, trainIdx, distance, distType))
            return false;
    }
    return true;
}

static bool ocl_kmatchDispatcher(InputArray query, InputArray train, const UMat &trainIdx,
                                 const UMat &distance, int distType)
{
    return ocl_match2Dispatcher(query, train, trainIdx, distance, distType);
}

static bool ocl_knnMatchSingle(InputArray query, InputArray train, UMat &trainIdx,
                               UMat &distance, int dstType)
{
    if (query.empty() || train.empty())
        return false;

    const int nQuery = query.size().height;

    ensureSizeIsEnough(1, nQuery, CV_32SC2, trainIdx);
    ensureSizeIsEnough(1, nQuery, CV_32FC2, distance);

    trainIdx.setTo(Scalar::all(-1));

    return ocl_kmatchDispatcher(query, train, trainIdx, distance, dstType);
}

static bool ocl_knnMatchConvert(const Mat &trainIdx, const Mat &distance, std::vector< std::vector<DMatch> > &matches, bool compactResult)
{
    if (trainIdx.empty() || distance.empty())
        return false;

    if(trainIdx.type() != CV_32SC2 && trainIdx.type() != CV_32SC1) return false;
    if(distance.type() != CV_32FC2 && distance.type() != CV_32FC1)return false;
    if(distance.size() != trainIdx.size()) return false;
    if(!trainIdx.isContinuous() || !distance.isContinuous()) return false;

    const int nQuery = trainIdx.type() == CV_32SC2 ? trainIdx.cols : trainIdx.rows;
    const int k = trainIdx.type() == CV_32SC2 ? 2 : trainIdx.cols;

    matches.clear();
    matches.reserve(nQuery);

    const int *trainIdx_ptr = trainIdx.ptr<int>();
    const float *distance_ptr = distance.ptr<float>();

    for (int queryIdx = 0; queryIdx < nQuery; ++queryIdx)
    {
        matches.push_back(std::vector<DMatch>());
        std::vector<DMatch> &curMatches = matches.back();
        curMatches.reserve(k);

        for (int i = 0; i < k; ++i, ++trainIdx_ptr, ++distance_ptr)
        {
            int trainIndex = *trainIdx_ptr;

            if (trainIndex != -1)
            {
                float dst = *distance_ptr;

                DMatch m(queryIdx, trainIndex, 0, dst);

                curMatches.push_back(m);
            }
        }

        if (compactResult && curMatches.empty())
            matches.pop_back();
    }
    return true;
}

static bool ocl_knnMatchDownload(const UMat &trainIdx, const UMat &distance, std::vector< std::vector<DMatch> > &matches, bool compactResult)
{
    if (trainIdx.empty() || distance.empty())
        return false;

    Mat trainIdxCPU = trainIdx.getMat(ACCESS_READ);
    Mat distanceCPU = distance.getMat(ACCESS_READ);

    if (ocl_knnMatchConvert(trainIdxCPU, distanceCPU, matches, compactResult) )
        return true;
    return false;
}

template < int BLOCK_SIZE, int MAX_DESC_LEN >
static bool ocl_matchUnrolledCached(InputArray _query, InputArray _train, float maxDistance,
                  const UMat &trainIdx, const UMat &distance, const UMat &nMatches, int distType)
{
    int depth = _query.depth();
    cv::String opts;
    opts = format("-D T=%s %s -D DIST_TYPE=%d -D BLOCK_SIZE=%d -D MAX_DESC_LEN=%d",
        ocl::typeToStr(depth), depth == CV_32F ? "-D T_FLOAT" : "", distType, (int)BLOCK_SIZE, (int)MAX_DESC_LEN);
    ocl::Kernel k("BruteForceMatch_RadiusUnrollMatch", ocl::features2d::brute_force_match_oclsrc, opts);
    if(k.empty())
        return false;

    size_t globalSize[] = {(_train.size().height + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE, (_query.size().height + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE, 1};
    size_t localSize[] = {BLOCK_SIZE, BLOCK_SIZE, 1};
    const size_t smemSize = (2 * BLOCK_SIZE * BLOCK_SIZE) * sizeof(int);

    if(globalSize[0] != 0)
    {
        UMat query = _query.getUMat(), train = _train.getUMat();

        int idx = 0;
        idx = k.set(idx, ocl::KernelArg::PtrReadOnly(query));
        idx = k.set(idx, ocl::KernelArg::PtrReadOnly(train));
        idx = k.set(idx, maxDistance);
        idx = k.set(idx, ocl::KernelArg::PtrWriteOnly(trainIdx));
        idx = k.set(idx, ocl::KernelArg::PtrWriteOnly(distance));
        idx = k.set(idx, ocl::KernelArg::PtrWriteOnly(nMatches));
        idx = k.set(idx, (void*)NULL, smemSize);
        idx = k.set(idx, query.rows);
        idx = k.set(idx, query.cols);
        idx = k.set(idx, train.rows);
        idx = k.set(idx, train.cols);
        idx = k.set(idx, trainIdx.cols);
        idx = k.set(idx, (int)query.step);
        idx = k.set(idx, (int)trainIdx.step);

        return k.run(2, globalSize, localSize, false);
    }
    return true;
}

//radius_match
template < int BLOCK_SIZE >
static bool ocl_radius_match(InputArray _query, InputArray _train, float maxDistance,
                  const UMat &trainIdx, const UMat &distance, const UMat &nMatches, int distType)
{
    int depth = _query.depth();
    cv::String opts;
    opts = format("-D T=%s %s -D DIST_TYPE=%d -D BLOCK_SIZE=%d", ocl::typeToStr(depth), depth == CV_32F ? "-D T_FLOAT" : "", distType, (int)BLOCK_SIZE);
    ocl::Kernel k("BruteForceMatch_RadiusMatch", ocl::features2d::brute_force_match_oclsrc, opts);
    if(k.empty())
        return false;

    size_t globalSize[] = {(_train.size().height + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE, (_query.size().height + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE, 1};
    size_t localSize[] = {BLOCK_SIZE, BLOCK_SIZE, 1};
    const size_t smemSize = (2 * BLOCK_SIZE * BLOCK_SIZE) * sizeof(int);

    if(globalSize[0] != 0)
    {
        UMat query = _query.getUMat(), train = _train.getUMat();

        int idx = 0;
        idx = k.set(idx, ocl::KernelArg::PtrReadOnly(query));
        idx = k.set(idx, ocl::KernelArg::PtrReadOnly(train));
        idx = k.set(idx, maxDistance);
        idx = k.set(idx, ocl::KernelArg::PtrWriteOnly(trainIdx));
        idx = k.set(idx, ocl::KernelArg::PtrWriteOnly(distance));
        idx = k.set(idx, ocl::KernelArg::PtrWriteOnly(nMatches));
        idx = k.set(idx, (void*)NULL, smemSize);
        idx = k.set(idx, query.rows);
        idx = k.set(idx, query.cols);
        idx = k.set(idx, train.rows);
        idx = k.set(idx, train.cols);
        idx = k.set(idx, trainIdx.cols);
        idx = k.set(idx, (int)query.step);
        idx = k.set(idx, (int)trainIdx.step);

        return k.run(2, globalSize, localSize, false);
    }
    return true;
}

static bool ocl_rmatchDispatcher(InputArray query, InputArray train,
        UMat &trainIdx,   UMat &distance, UMat &nMatches, float maxDistance, int distType)
{
    bool is_cpu = ocl::Device::getDefault().type() == ocl::Device::TYPE_CPU;
    int query_cols = query.size().width;
    if (query_cols <= 64)
    {
        if(!ocl_matchUnrolledCached<16, 64>(query, train, maxDistance, trainIdx, distance, nMatches, distType)) return false;
    }
    else if (query_cols <= 128 && !is_cpu)
    {
        if(!ocl_matchUnrolledCached<16, 128>(query, train, maxDistance, trainIdx, distance, nMatches, distType)) return false;
    }
    else
    {
        if(!ocl_radius_match<16>(query, train, maxDistance, trainIdx, distance, nMatches, distType)) return false;
    }
    return true;
}


static bool ocl_radiusMatchSingle(InputArray query, InputArray train,
        UMat &trainIdx,   UMat &distance, UMat &nMatches, float maxDistance, int distType)
{
    if (query.empty() || train.empty())
        return false;

    const int nQuery = query.size().height;
    const int nTrain = train.size().height;

    ensureSizeIsEnough(1, nQuery, CV_32SC1, nMatches);

    if (trainIdx.empty())
    {
        ensureSizeIsEnough(nQuery, std::max((nTrain / 100), 10), CV_32SC1, trainIdx);
        ensureSizeIsEnough(nQuery, std::max((nTrain / 100), 10), CV_32FC1, distance);
    }

    nMatches.setTo(Scalar::all(0));

    return ocl_rmatchDispatcher(query, train, trainIdx, distance, nMatches, maxDistance, distType);
}

static bool ocl_radiusMatchConvert(const Mat &trainIdx, const Mat &distance, const Mat &_nMatches,
        std::vector< std::vector<DMatch> > &matches, bool compactResult)
{
    if (trainIdx.empty() || distance.empty() || _nMatches.empty())
        return false;

    if( (trainIdx.type() != CV_32SC1) ||
        (distance.type() != CV_32FC1 || distance.size() != trainIdx.size()) ||
        (_nMatches.type() != CV_32SC1 || _nMatches.cols != trainIdx.rows) )
        return false;

    const int nQuery = trainIdx.rows;

    matches.clear();
    matches.reserve(nQuery);

    const int *nMatches_ptr = _nMatches.ptr<int>();

    for (int queryIdx = 0; queryIdx < nQuery; ++queryIdx)
    {
        const int *trainIdx_ptr = trainIdx.ptr<int>(queryIdx);
        const float *distance_ptr = distance.ptr<float>(queryIdx);

        const int nMatches = std::min(nMatches_ptr[queryIdx], trainIdx.cols);

        if (nMatches == 0)
        {
            if (!compactResult)
                matches.push_back(std::vector<DMatch>());
            continue;
        }

        matches.push_back(std::vector<DMatch>(nMatches));
        std::vector<DMatch> &curMatches = matches.back();

        for (int i = 0; i < nMatches; ++i, ++trainIdx_ptr, ++distance_ptr)
        {
            int trainIndex = *trainIdx_ptr;

            float dst = *distance_ptr;

            DMatch m(queryIdx, trainIndex, 0, dst);

            curMatches[i] = m;
        }

        std::sort(curMatches.begin(), curMatches.end());
    }
    return true;
}

static bool ocl_radiusMatchDownload(const UMat &trainIdx, const UMat &distance, const UMat &nMatches,
        std::vector< std::vector<DMatch> > &matches, bool compactResult)
{
    if (trainIdx.empty() || distance.empty() || nMatches.empty())
        return false;

    Mat trainIdxCPU = trainIdx.getMat(ACCESS_READ);
    Mat distanceCPU = distance.getMat(ACCESS_READ);
    Mat nMatchesCPU = nMatches.getMat(ACCESS_READ);

    return ocl_radiusMatchConvert(trainIdxCPU, distanceCPU, nMatchesCPU, matches, compactResult);
}

/****************************************************************************************\
*                                      DescriptorMatcher                                 *
\****************************************************************************************/
DescriptorMatcher::DescriptorCollection::DescriptorCollection()
{}

DescriptorMatcher::DescriptorCollection::DescriptorCollection( const DescriptorCollection& collection )
{
    mergedDescriptors = collection.mergedDescriptors.clone();
    std::copy( collection.startIdxs.begin(), collection.startIdxs.begin(), startIdxs.begin() );
}

DescriptorMatcher::DescriptorCollection::~DescriptorCollection()
{}

void DescriptorMatcher::DescriptorCollection::set( const std::vector<Mat>& descriptors )
{
    clear();

    size_t imageCount = descriptors.size();
    CV_Assert( imageCount > 0 );

    startIdxs.resize( imageCount );

    int dim = -1;
    int type = -1;
    startIdxs[0] = 0;
    for( size_t i = 1; i < imageCount; i++ )
    {
        int s = 0;
        if( !descriptors[i-1].empty() )
        {
            dim = descriptors[i-1].cols;
            type = descriptors[i-1].type();
            s = descriptors[i-1].rows;
        }
        startIdxs[i] = startIdxs[i-1] + s;
    }
    if( imageCount == 1 )
    {
        if( descriptors[0].empty() ) return;

        dim = descriptors[0].cols;
        type = descriptors[0].type();
    }
    CV_Assert( dim > 0 );

    int count = startIdxs[imageCount-1] + descriptors[imageCount-1].rows;

    if( count > 0 )
    {
        mergedDescriptors.create( count, dim, type );
        for( size_t i = 0; i < imageCount; i++ )
        {
            if( !descriptors[i].empty() )
            {
                CV_Assert( descriptors[i].cols == dim && descriptors[i].type() == type );
                Mat m = mergedDescriptors.rowRange( startIdxs[i], startIdxs[i] + descriptors[i].rows );
                descriptors[i].copyTo(m);
            }
        }
    }
}

void DescriptorMatcher::DescriptorCollection::clear()
{
    startIdxs.clear();
    mergedDescriptors.release();
}

const Mat DescriptorMatcher::DescriptorCollection::getDescriptor( int imgIdx, int localDescIdx ) const
{
    CV_Assert( imgIdx < (int)startIdxs.size() );
    int globalIdx = startIdxs[imgIdx] + localDescIdx;
    CV_Assert( globalIdx < (int)size() );

    return getDescriptor( globalIdx );
}

const Mat& DescriptorMatcher::DescriptorCollection::getDescriptors() const
{
    return mergedDescriptors;
}

const Mat DescriptorMatcher::DescriptorCollection::getDescriptor( int globalDescIdx ) const
{
    CV_Assert( globalDescIdx < size() );
    return mergedDescriptors.row( globalDescIdx );
}

void DescriptorMatcher::DescriptorCollection::getLocalIdx( int globalDescIdx, int& imgIdx, int& localDescIdx ) const
{
    CV_Assert( (globalDescIdx>=0) && (globalDescIdx < size()) );
    std::vector<int>::const_iterator img_it = std::upper_bound(startIdxs.begin(), startIdxs.end(), globalDescIdx);
    --img_it;
    imgIdx = (int)(img_it - startIdxs.begin());
    localDescIdx = globalDescIdx - (*img_it);
}

int DescriptorMatcher::DescriptorCollection::size() const
{
    return mergedDescriptors.rows;
}

/*
 * DescriptorMatcher
 */
static void convertMatches( const std::vector<std::vector<DMatch> >& knnMatches, std::vector<DMatch>& matches )
{
    matches.clear();
    matches.reserve( knnMatches.size() );
    for( size_t i = 0; i < knnMatches.size(); i++ )
    {
        CV_Assert( knnMatches[i].size() <= 1 );
        if( !knnMatches[i].empty() )
            matches.push_back( knnMatches[i][0] );
    }
}

DescriptorMatcher::~DescriptorMatcher()
{}

void DescriptorMatcher::add( InputArrayOfArrays _descriptors )
{
    if(_descriptors.isUMatVector())
    {
        std::vector<UMat> descriptors;
        _descriptors.getUMatVector(descriptors);
        utrainDescCollection.insert( utrainDescCollection.end(), descriptors.begin(), descriptors.end() );
    }
    else if(_descriptors.isUMat())
    {
        std::vector<UMat> descriptors = std::vector<UMat>(1, _descriptors.getUMat());
        utrainDescCollection.insert( utrainDescCollection.end(), descriptors.begin(), descriptors.end() );
    }
    else if(_descriptors.isMatVector())
    {
        std::vector<Mat> descriptors;
        _descriptors.getMatVector(descriptors);
        trainDescCollection.insert( trainDescCollection.end(), descriptors.begin(), descriptors.end() );
    }
    else if(_descriptors.isMat())
    {
        std::vector<Mat> descriptors = std::vector<Mat>(1, _descriptors.getMat());
        trainDescCollection.insert( trainDescCollection.end(), descriptors.begin(), descriptors.end() );
    }
    else
        CV_Assert( _descriptors.isUMat() || _descriptors.isUMatVector() || _descriptors.isMat() || _descriptors.isMatVector() );
}

const std::vector<Mat>& DescriptorMatcher::getTrainDescriptors() const
{
    return trainDescCollection;
}

void DescriptorMatcher::clear()
{
    utrainDescCollection.clear();
    trainDescCollection.clear();
}

bool DescriptorMatcher::empty() const
{
    return trainDescCollection.empty() && utrainDescCollection.empty();
}

void DescriptorMatcher::train()
{}

void DescriptorMatcher::match( InputArray queryDescriptors, InputArray trainDescriptors,
                              std::vector<DMatch>& matches, InputArray mask ) const
{
    Ptr<DescriptorMatcher> tempMatcher = clone(true);
    tempMatcher->add(trainDescriptors);
    tempMatcher->match( queryDescriptors, matches, std::vector<Mat>(1, mask.getMat()) );
}

void DescriptorMatcher::knnMatch( InputArray queryDescriptors, InputArray trainDescriptors,
                                  std::vector<std::vector<DMatch> >& matches, int knn,
                                  InputArray mask, bool compactResult ) const
{
    Ptr<DescriptorMatcher> tempMatcher = clone(true);
    tempMatcher->add(trainDescriptors);
    tempMatcher->knnMatch( queryDescriptors, matches, knn, std::vector<Mat>(1, mask.getMat()), compactResult );
}

void DescriptorMatcher::radiusMatch( InputArray queryDescriptors, InputArray trainDescriptors,
                                     std::vector<std::vector<DMatch> >& matches, float maxDistance, InputArray mask,
                                     bool compactResult ) const
{
    Ptr<DescriptorMatcher> tempMatcher = clone(true);
    tempMatcher->add(trainDescriptors);
    tempMatcher->radiusMatch( queryDescriptors, matches, maxDistance, std::vector<Mat>(1, mask.getMat()), compactResult );
}

void DescriptorMatcher::match( InputArray queryDescriptors, std::vector<DMatch>& matches, InputArrayOfArrays masks )
{
    std::vector<std::vector<DMatch> > knnMatches;
    knnMatch( queryDescriptors, knnMatches, 1, masks, true /*compactResult*/ );
    convertMatches( knnMatches, matches );
}

void DescriptorMatcher::checkMasks( InputArrayOfArrays _masks, int queryDescriptorsCount ) const
{
    std::vector<Mat> masks;
    _masks.getMatVector(masks);

    if( isMaskSupported() && !masks.empty() )
    {
        // Check masks
        size_t imageCount = std::max(trainDescCollection.size(), utrainDescCollection.size() );
        CV_Assert( masks.size() == imageCount );
        for( size_t i = 0; i < imageCount; i++ )
        {
            if( !masks[i].empty() && (!trainDescCollection[i].empty() || !utrainDescCollection[i].empty() ) )
            {
                int rows = trainDescCollection[i].empty() ? utrainDescCollection[i].rows : trainDescCollection[i].rows;
                    CV_Assert( masks[i].rows == queryDescriptorsCount &&
                        (masks[i].cols == rows || masks[i].cols == rows) &&
                        masks[i].type() == CV_8UC1 );
            }
        }
    }
}

void DescriptorMatcher::knnMatch( InputArray queryDescriptors, std::vector<std::vector<DMatch> >& matches, int knn,
                                  InputArrayOfArrays masks, bool compactResult )
{
    if( empty() || queryDescriptors.empty() )
        return;

    CV_Assert( knn > 0 );

    checkMasks( masks, queryDescriptors.size().height );

    train();
    knnMatchImpl( queryDescriptors, matches, knn, masks, compactResult );
}

void DescriptorMatcher::radiusMatch( InputArray queryDescriptors, std::vector<std::vector<DMatch> >& matches, float maxDistance,
                                     InputArrayOfArrays masks, bool compactResult )
{
    matches.clear();
    if( empty() || queryDescriptors.empty() )
        return;

    CV_Assert( maxDistance > std::numeric_limits<float>::epsilon() );

    checkMasks( masks, queryDescriptors.size().height );

    train();
    radiusMatchImpl( queryDescriptors, matches, maxDistance, masks, compactResult );
}

void DescriptorMatcher::read( const FileNode& )
{}

void DescriptorMatcher::write( FileStorage& ) const
{}

bool DescriptorMatcher::isPossibleMatch( InputArray _mask, int queryIdx, int trainIdx )
{
    Mat mask = _mask.getMat();
    return mask.empty() || mask.at<uchar>(queryIdx, trainIdx);
}

bool DescriptorMatcher::isMaskedOut( InputArrayOfArrays _masks, int queryIdx )
{
    std::vector<Mat> masks;
    _masks.getMatVector(masks);

    size_t outCount = 0;
    for( size_t i = 0; i < masks.size(); i++ )
    {
        if( !masks[i].empty() && (countNonZero(masks[i].row(queryIdx)) == 0) )
            outCount++;
    }

    return !masks.empty() && outCount == masks.size() ;
}


////////////////////////////////////////////////////// BruteForceMatcher /////////////////////////////////////////////////

BFMatcher::BFMatcher( int _normType, bool _crossCheck )
{
    normType = _normType;
    crossCheck = _crossCheck;
}

Ptr<DescriptorMatcher> BFMatcher::clone( bool emptyTrainData ) const
{
    Ptr<BFMatcher> matcher = makePtr<BFMatcher>(normType, crossCheck);
    if( !emptyTrainData )
    {
        matcher->trainDescCollection.resize(trainDescCollection.size());
        std::transform( trainDescCollection.begin(), trainDescCollection.end(),
                        matcher->trainDescCollection.begin(), clone_op );
    }
    return matcher;
}

static bool ocl_match(InputArray query, InputArray _train, std::vector< std::vector<DMatch> > &matches, int dstType)
{
    UMat trainIdx, distance;
    if (!ocl_matchSingle(query, _train, trainIdx, distance, dstType))
        return false;
    if (!ocl_matchDownload(trainIdx, distance, matches))
        return false;
    return true;
}

static bool ocl_knnMatch(InputArray query, InputArray _train, std::vector< std::vector<DMatch> > &matches, int k, int dstType, bool compactResult)
{
    UMat trainIdx, distance;
    if (k != 2)
        return false;
    if (!ocl_knnMatchSingle(query, _train, trainIdx, distance, dstType))
        return false;
    if (!ocl_knnMatchDownload(trainIdx, distance, matches, compactResult) )
        return false;
    return true;
}

void BFMatcher::knnMatchImpl( InputArray _queryDescriptors, std::vector<std::vector<DMatch> >& matches, int knn,
                             InputArrayOfArrays _masks, bool compactResult )
{
    int trainDescType = trainDescCollection.empty() ? utrainDescCollection[0].type() : trainDescCollection[0].type();
    CV_Assert( _queryDescriptors.type() == trainDescType );

    const int IMGIDX_SHIFT = 18;
    const int IMGIDX_ONE = (1 << IMGIDX_SHIFT);

    if( _queryDescriptors.empty() || (trainDescCollection.empty() && utrainDescCollection.empty()))
    {
        matches.clear();
        return;
    }

    std::vector<Mat> masks;
    _masks.getMatVector(masks);

    if(!trainDescCollection.empty() && !utrainDescCollection.empty())
    {
        for(int i = 0; i < (int)utrainDescCollection.size(); i++)
        {
            Mat tempMat;
            utrainDescCollection[i].copyTo(tempMat);
            trainDescCollection.push_back(tempMat);
        }
        utrainDescCollection.clear();
    }

    int trainDescVectorSize = trainDescCollection.empty() ? (int)utrainDescCollection.size() : (int)trainDescCollection.size();
    Size trainDescSize = trainDescCollection.empty() ? utrainDescCollection[0].size() : trainDescCollection[0].size();
    int trainDescOffset = trainDescCollection.empty() ? (int)utrainDescCollection[0].offset : 0;

    if ( ocl::useOpenCL() && _queryDescriptors.isUMat() && _queryDescriptors.dims()<=2 && trainDescVectorSize == 1 &&
        _queryDescriptors.type() == CV_32FC1 && _queryDescriptors.offset() == 0 && trainDescOffset == 0 &&
        trainDescSize.width == _queryDescriptors.size().width && masks.size() == 1 && masks[0].total() == 0 )
    {
        if(knn == 1)
        {
            if(trainDescCollection.empty())
            {
                if(ocl_match(_queryDescriptors, utrainDescCollection[0], matches, normType))
                    return;
            }
            else
            {
                if(ocl_match(_queryDescriptors, trainDescCollection[0], matches, normType))
                    return;
            }
        }
        else
        {
            if(trainDescCollection.empty())
            {
                if(ocl_knnMatch(_queryDescriptors, utrainDescCollection[0], matches, knn, normType, compactResult) )
                    return;
            }
            else
            {
                if(ocl_knnMatch(_queryDescriptors, trainDescCollection[0], matches, knn, normType, compactResult) )
                    return;
            }
        }
    }

    Mat queryDescriptors = _queryDescriptors.getMat();
    if(trainDescCollection.empty() && !utrainDescCollection.empty())
    {
        for(int i = 0; i < (int)utrainDescCollection.size(); i++)
        {
            Mat tempMat;
            utrainDescCollection[i].copyTo(tempMat);
            trainDescCollection.push_back(tempMat);
        }
        utrainDescCollection.clear();
    }

    matches.reserve(queryDescriptors.rows);

    Mat dist, nidx;

    int iIdx, imgCount = (int)trainDescCollection.size(), update = 0;
    int dtype = normType == NORM_HAMMING || normType == NORM_HAMMING2 ||
        (normType == NORM_L1 && queryDescriptors.type() == CV_8U) ? CV_32S : CV_32F;

    CV_Assert( (int64)imgCount*IMGIDX_ONE < INT_MAX );

    for( iIdx = 0; iIdx < imgCount; iIdx++ )
    {
        CV_Assert( trainDescCollection[iIdx].rows < IMGIDX_ONE );
        batchDistance(queryDescriptors, trainDescCollection[iIdx], dist, dtype, nidx,
                      normType, knn, masks.empty() ? Mat() : masks[iIdx], update, crossCheck);
        update += IMGIDX_ONE;
    }

    if( dtype == CV_32S )
    {
        Mat temp;
        dist.convertTo(temp, CV_32F);
        dist = temp;
    }

    for( int qIdx = 0; qIdx < queryDescriptors.rows; qIdx++ )
    {
        const float* distptr = dist.ptr<float>(qIdx);
        const int* nidxptr = nidx.ptr<int>(qIdx);

        matches.push_back( std::vector<DMatch>() );
        std::vector<DMatch>& mq = matches.back();
        mq.reserve(knn);

        for( int k = 0; k < nidx.cols; k++ )
        {
            if( nidxptr[k] < 0 )
                break;
            mq.push_back( DMatch(qIdx, nidxptr[k] & (IMGIDX_ONE - 1),
                          nidxptr[k] >> IMGIDX_SHIFT, distptr[k]) );
        }

        if( mq.empty() && compactResult )
            matches.pop_back();
    }
}

static bool ocl_radiusMatch(InputArray query, InputArray _train, std::vector< std::vector<DMatch> > &matches,
        float maxDistance, int dstType, bool compactResult)
{
    UMat trainIdx, distance, nMatches;
    if (!ocl_radiusMatchSingle(query, _train, trainIdx, distance, nMatches, maxDistance, dstType))
        return false;
    if (!ocl_radiusMatchDownload(trainIdx, distance, nMatches, matches, compactResult))
        return false;
    return true;
}

void BFMatcher::radiusMatchImpl( InputArray _queryDescriptors, std::vector<std::vector<DMatch> >& matches,
                                float maxDistance, InputArrayOfArrays _masks, bool compactResult )
{
    int trainDescType = trainDescCollection.empty() ? utrainDescCollection[0].type() : trainDescCollection[0].type();
    CV_Assert( _queryDescriptors.type() == trainDescType );

    if( _queryDescriptors.empty() || (trainDescCollection.empty() && utrainDescCollection.empty()))
    {
        matches.clear();
        return;
    }

    std::vector<Mat> masks;
    _masks.getMatVector(masks);

    if(!trainDescCollection.empty() && !utrainDescCollection.empty())
    {
        for(int i = 0; i < (int)utrainDescCollection.size(); i++)
        {
            Mat tempMat;
            utrainDescCollection[i].copyTo(tempMat);
            trainDescCollection.push_back(tempMat);
        }
        utrainDescCollection.clear();
    }

    int trainDescVectorSize = trainDescCollection.empty() ? (int)utrainDescCollection.size() : (int)trainDescCollection.size();
    Size trainDescSize = trainDescCollection.empty() ? utrainDescCollection[0].size() : trainDescCollection[0].size();
    int trainDescOffset = trainDescCollection.empty() ? (int)utrainDescCollection[0].offset : 0;

    if ( ocl::useOpenCL() && _queryDescriptors.isUMat() && _queryDescriptors.dims()<=2 && trainDescVectorSize == 1 &&
        _queryDescriptors.type() == CV_32FC1 && _queryDescriptors.offset() == 0 && trainDescOffset == 0 &&
        trainDescSize.width == _queryDescriptors.size().width && masks.size() == 1 && masks[0].total() == 0 )
    {
        if (trainDescCollection.empty())
        {
            if(ocl_radiusMatch(_queryDescriptors, utrainDescCollection[0], matches, maxDistance, normType, compactResult) )
                return;
        }
        else
        {
            if (ocl_radiusMatch(_queryDescriptors, trainDescCollection[0], matches, maxDistance, normType, compactResult) )
                return;
        }
    }

    Mat queryDescriptors = _queryDescriptors.getMat();
    if(trainDescCollection.empty() && !utrainDescCollection.empty())
    {
        for(int i = 0; i < (int)utrainDescCollection.size(); i++)
        {
            Mat tempMat;
            utrainDescCollection[i].copyTo(tempMat);
            trainDescCollection.push_back(tempMat);
        }
        utrainDescCollection.clear();
    }

    matches.resize(queryDescriptors.rows);
    Mat dist, distf;

    int iIdx, imgCount = (int)trainDescCollection.size();
    int dtype = normType == NORM_HAMMING ||
        (normType == NORM_L1 && queryDescriptors.type() == CV_8U) ? CV_32S : CV_32F;

    for( iIdx = 0; iIdx < imgCount; iIdx++ )
    {
        batchDistance(queryDescriptors, trainDescCollection[iIdx], dist, dtype, noArray(),
                      normType, 0, masks.empty() ? Mat() : masks[iIdx], 0, false);
        if( dtype == CV_32S )
            dist.convertTo(distf, CV_32F);
        else
            distf = dist;

        for( int qIdx = 0; qIdx < queryDescriptors.rows; qIdx++ )
        {
            const float* distptr = distf.ptr<float>(qIdx);

            std::vector<DMatch>& mq = matches[qIdx];
            for( int k = 0; k < distf.cols; k++ )
            {
                if( distptr[k] <= maxDistance )
                    mq.push_back( DMatch(qIdx, k, iIdx, distptr[k]) );
            }
        }
    }

    int qIdx0 = 0;
    for( int qIdx = 0; qIdx < queryDescriptors.rows; qIdx++ )
    {
        if( matches[qIdx].empty() && compactResult )
            continue;

        if( qIdx0 < qIdx )
            std::swap(matches[qIdx], matches[qIdx0]);

        std::sort( matches[qIdx0].begin(), matches[qIdx0].end() );
        qIdx0++;
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////

/*
 * Factory function for DescriptorMatcher creating
 */
Ptr<DescriptorMatcher> DescriptorMatcher::create( const String& descriptorMatcherType )
{
    Ptr<DescriptorMatcher> dm;
    if( !descriptorMatcherType.compare( "FlannBased" ) )
    {
        dm = makePtr<FlannBasedMatcher>();
    }
    else if( !descriptorMatcherType.compare( "BruteForce" ) ) // L2
    {
        dm = makePtr<BFMatcher>(int(NORM_L2)); // anonymous enums can't be template parameters
    }
    else if( !descriptorMatcherType.compare( "BruteForce-SL2" ) ) // Squared L2
    {
        dm = makePtr<BFMatcher>(int(NORM_L2SQR));
    }
    else if( !descriptorMatcherType.compare( "BruteForce-L1" ) )
    {
        dm = makePtr<BFMatcher>(int(NORM_L1));
    }
    else if( !descriptorMatcherType.compare("BruteForce-Hamming") ||
             !descriptorMatcherType.compare("BruteForce-HammingLUT") )
    {
        dm = makePtr<BFMatcher>(int(NORM_HAMMING));
    }
    else if( !descriptorMatcherType.compare("BruteForce-Hamming(2)") )
    {
        dm = makePtr<BFMatcher>(int(NORM_HAMMING2));
    }
    else
        CV_Error( Error::StsBadArg, "Unknown matcher name" );

    return dm;
}


/*
 * Flann based matcher
 */
FlannBasedMatcher::FlannBasedMatcher( const Ptr<flann::IndexParams>& _indexParams, const Ptr<flann::SearchParams>& _searchParams )
    : indexParams(_indexParams), searchParams(_searchParams), addedDescCount(0)
{
    CV_Assert( _indexParams );
    CV_Assert( _searchParams );
}

void FlannBasedMatcher::add( InputArrayOfArrays _descriptors )
{
    DescriptorMatcher::add( _descriptors );
    std::vector<UMat> descriptors;
    _descriptors.getUMatVector(descriptors);

    for( size_t i = 0; i < descriptors.size(); i++ )
    {
        addedDescCount += descriptors[i].rows;
    }
}

void FlannBasedMatcher::clear()
{
    DescriptorMatcher::clear();

    mergedDescriptors.clear();
    flannIndex.release();

    addedDescCount = 0;
}

void FlannBasedMatcher::train()
{
    if( !flannIndex || mergedDescriptors.size() < addedDescCount )
    {
        // FIXIT: Workaround for 'utrainDescCollection' issue (PR #2142)
        if (!utrainDescCollection.empty())
        {
            CV_Assert(trainDescCollection.size() == 0);
            for (size_t i = 0; i < utrainDescCollection.size(); ++i)
                trainDescCollection.push_back(utrainDescCollection[i].getMat(ACCESS_READ));
        }
        mergedDescriptors.set( trainDescCollection );
        flannIndex = makePtr<flann::Index>( mergedDescriptors.getDescriptors(), *indexParams );
    }
}

void FlannBasedMatcher::read( const FileNode& fn)
{
     if (!indexParams)
         indexParams = makePtr<flann::IndexParams>();

     FileNode ip = fn["indexParams"];
     CV_Assert(ip.type() == FileNode::SEQ);

     for(int i = 0; i < (int)ip.size(); ++i)
     {
        CV_Assert(ip[i].type() == FileNode::MAP);
        String _name =  (String)ip[i]["name"];
        int type =  (int)ip[i]["type"];

        switch(type)
        {
        case CV_8U:
        case CV_8S:
        case CV_16U:
        case CV_16S:
        case CV_32S:
            indexParams->setInt(_name, (int) ip[i]["value"]);
            break;
        case CV_32F:
            indexParams->setFloat(_name, (float) ip[i]["value"]);
            break;
        case CV_64F:
            indexParams->setDouble(_name, (double) ip[i]["value"]);
            break;
        case CV_USRTYPE1:
            indexParams->setString(_name, (String) ip[i]["value"]);
            break;
        case CV_MAKETYPE(CV_USRTYPE1,2):
            indexParams->setBool(_name, (int) ip[i]["value"] != 0);
            break;
        case CV_MAKETYPE(CV_USRTYPE1,3):
            indexParams->setAlgorithm((int) ip[i]["value"]);
            break;
        };
     }

     if (!searchParams)
         searchParams = makePtr<flann::SearchParams>();

     FileNode sp = fn["searchParams"];
     CV_Assert(sp.type() == FileNode::SEQ);

     for(int i = 0; i < (int)sp.size(); ++i)
     {
        CV_Assert(sp[i].type() == FileNode::MAP);
        String _name =  (String)sp[i]["name"];
        int type =  (int)sp[i]["type"];

        switch(type)
        {
        case CV_8U:
        case CV_8S:
        case CV_16U:
        case CV_16S:
        case CV_32S:
            searchParams->setInt(_name, (int) sp[i]["value"]);
            break;
        case CV_32F:
            searchParams->setFloat(_name, (float) ip[i]["value"]);
            break;
        case CV_64F:
            searchParams->setDouble(_name, (double) ip[i]["value"]);
            break;
        case CV_USRTYPE1:
            searchParams->setString(_name, (String) ip[i]["value"]);
            break;
        case CV_MAKETYPE(CV_USRTYPE1,2):
            searchParams->setBool(_name, (int) ip[i]["value"] != 0);
            break;
        case CV_MAKETYPE(CV_USRTYPE1,3):
            searchParams->setAlgorithm((int) ip[i]["value"]);
            break;
        };
     }

    flannIndex.release();
}

void FlannBasedMatcher::write( FileStorage& fs) const
{
     fs << "indexParams" << "[";

     if (indexParams)
     {
         std::vector<String> names;
         std::vector<int> types;
         std::vector<String> strValues;
         std::vector<double> numValues;

         indexParams->getAll(names, types, strValues, numValues);

         for(size_t i = 0; i < names.size(); ++i)
         {
             fs << "{" << "name" << names[i] << "type" << types[i] << "value";
             switch(types[i])
             {
             case CV_8U:
                 fs << (uchar)numValues[i];
                 break;
             case CV_8S:
                 fs << (char)numValues[i];
                 break;
             case CV_16U:
                 fs << (ushort)numValues[i];
                 break;
             case CV_16S:
                 fs << (short)numValues[i];
                 break;
             case CV_32S:
             case CV_MAKETYPE(CV_USRTYPE1,2):
             case CV_MAKETYPE(CV_USRTYPE1,3):
                 fs << (int)numValues[i];
                 break;
             case CV_32F:
                 fs << (float)numValues[i];
                 break;
             case CV_64F:
                 fs << (double)numValues[i];
                 break;
             case CV_USRTYPE1:
                 fs << strValues[i];
                 break;
             default:
                 fs << (double)numValues[i];
                 fs << "typename" << strValues[i];
                 break;
             }
             fs << "}";
         }
     }

     fs << "]" << "searchParams" << "[";

     if (searchParams)
     {
         std::vector<String> names;
         std::vector<int> types;
         std::vector<String> strValues;
         std::vector<double> numValues;

         searchParams->getAll(names, types, strValues, numValues);

         for(size_t i = 0; i < names.size(); ++i)
         {
             fs << "{" << "name" << names[i] << "type" << types[i] << "value";
             switch(types[i])
             {
             case CV_8U:
                 fs << (uchar)numValues[i];
                 break;
             case CV_8S:
                 fs << (char)numValues[i];
                 break;
             case CV_16U:
                 fs << (ushort)numValues[i];
                 break;
             case CV_16S:
                 fs << (short)numValues[i];
                 break;
             case CV_32S:
             case CV_MAKETYPE(CV_USRTYPE1,2):
             case CV_MAKETYPE(CV_USRTYPE1,3):
                 fs << (int)numValues[i];
                 break;
             case CV_32F:
                 fs << (float)numValues[i];
                 break;
             case CV_64F:
                 fs << (double)numValues[i];
                 break;
             case CV_USRTYPE1:
                 fs << strValues[i];
                 break;
             default:
                 fs << (double)numValues[i];
                 fs << "typename" << strValues[i];
                 break;
             }
             fs << "}";
         }
     }
     fs << "]";
}

bool FlannBasedMatcher::isMaskSupported() const
{
    return false;
}

Ptr<DescriptorMatcher> FlannBasedMatcher::clone( bool emptyTrainData ) const
{
    Ptr<FlannBasedMatcher> matcher = makePtr<FlannBasedMatcher>(indexParams, searchParams);
    if( !emptyTrainData )
    {
        CV_Error( Error::StsNotImplemented, "deep clone functionality is not implemented, because "
                  "Flann::Index has not copy constructor or clone method ");
        //matcher->flannIndex;
        matcher->addedDescCount = addedDescCount;
        matcher->mergedDescriptors = DescriptorCollection( mergedDescriptors );
        std::transform( trainDescCollection.begin(), trainDescCollection.end(),
                        matcher->trainDescCollection.begin(), clone_op );
    }
    return matcher;
}

void FlannBasedMatcher::convertToDMatches( const DescriptorCollection& collection, const Mat& indices, const Mat& dists,
                                           std::vector<std::vector<DMatch> >& matches )
{
    matches.resize( indices.rows );
    for( int i = 0; i < indices.rows; i++ )
    {
        for( int j = 0; j < indices.cols; j++ )
        {
            int idx = indices.at<int>(i, j);
            if( idx >= 0 )
            {
                int imgIdx, trainIdx;
                collection.getLocalIdx( idx, imgIdx, trainIdx );
                float dist = 0;
                if (dists.type() == CV_32S)
                    dist = static_cast<float>( dists.at<int>(i,j) );
                else
                    dist = std::sqrt(dists.at<float>(i,j));
                matches[i].push_back( DMatch( i, trainIdx, imgIdx, dist ) );
            }
        }
    }
}

void FlannBasedMatcher::knnMatchImpl( InputArray _queryDescriptors, std::vector<std::vector<DMatch> >& matches, int knn,
                                     InputArrayOfArrays /*masks*/, bool /*compactResult*/ )
{
    Mat queryDescriptors = _queryDescriptors.getMat();
    Mat indices( queryDescriptors.rows, knn, CV_32SC1 );
    Mat dists( queryDescriptors.rows, knn, CV_32FC1);
    flannIndex->knnSearch( queryDescriptors, indices, dists, knn, *searchParams );

    convertToDMatches( mergedDescriptors, indices, dists, matches );
}

void FlannBasedMatcher::radiusMatchImpl( InputArray _queryDescriptors, std::vector<std::vector<DMatch> >& matches, float maxDistance,
                                         InputArrayOfArrays /*masks*/, bool /*compactResult*/ )
{
    Mat queryDescriptors = _queryDescriptors.getMat();
    const int count = mergedDescriptors.size(); // TODO do count as param?
    Mat indices( queryDescriptors.rows, count, CV_32SC1, Scalar::all(-1) );
    Mat dists( queryDescriptors.rows, count, CV_32FC1, Scalar::all(-1) );
    for( int qIdx = 0; qIdx < queryDescriptors.rows; qIdx++ )
    {
        Mat queryDescriptorsRow = queryDescriptors.row(qIdx);
        Mat indicesRow = indices.row(qIdx);
        Mat distsRow = dists.row(qIdx);
        flannIndex->radiusSearch( queryDescriptorsRow, indicesRow, distsRow, maxDistance*maxDistance, count, *searchParams );
    }

    convertToDMatches( mergedDescriptors, indices, dists, matches );
}

}
