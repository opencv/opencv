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
//     and/or other GpuMaterials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or bpied warranties, including, but not limited to, the bpied
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

using namespace cv;
using namespace cv::gpu;
using namespace std;

#if !defined (HAVE_CUDA)

cv::gpu::BruteForceMatcher_GPU_base::BruteForceMatcher_GPU_base(DistType) { throw_nogpu(); }
void cv::gpu::BruteForceMatcher_GPU_base::add(const vector<GpuMat>&) { throw_nogpu(); }
const vector<GpuMat>& cv::gpu::BruteForceMatcher_GPU_base::getTrainDescriptors() const { throw_nogpu(); return trainDescCollection; }
void cv::gpu::BruteForceMatcher_GPU_base::clear() { throw_nogpu(); }
bool cv::gpu::BruteForceMatcher_GPU_base::empty() const { throw_nogpu(); return true; }
bool cv::gpu::BruteForceMatcher_GPU_base::isMaskSupported() const { throw_nogpu(); return true; }
void cv::gpu::BruteForceMatcher_GPU_base::matchSingle(const GpuMat&, const GpuMat&, GpuMat&, GpuMat&, const GpuMat&, Stream&) { throw_nogpu(); }
void cv::gpu::BruteForceMatcher_GPU_base::matchDownload(const GpuMat&, const GpuMat&, vector<DMatch>&) { throw_nogpu(); }
void cv::gpu::BruteForceMatcher_GPU_base::matchConvert(const Mat&, const Mat&, std::vector<DMatch>&) { throw_nogpu(); }
void cv::gpu::BruteForceMatcher_GPU_base::match(const GpuMat&, const GpuMat&, vector<DMatch>&, const GpuMat&) { throw_nogpu(); }
void cv::gpu::BruteForceMatcher_GPU_base::makeGpuCollection(GpuMat&, GpuMat&, const vector<GpuMat>&) { throw_nogpu(); }
void cv::gpu::BruteForceMatcher_GPU_base::matchCollection(const GpuMat&, const GpuMat&, GpuMat&, GpuMat&, GpuMat&, const GpuMat&, Stream&) { throw_nogpu(); }
void cv::gpu::BruteForceMatcher_GPU_base::matchDownload(const GpuMat&, const GpuMat&, const GpuMat&, std::vector<DMatch>&) { throw_nogpu(); }
void cv::gpu::BruteForceMatcher_GPU_base::matchConvert(const Mat&, const Mat&, const Mat&, std::vector<DMatch>&) { throw_nogpu(); }
void cv::gpu::BruteForceMatcher_GPU_base::match(const GpuMat&, std::vector<DMatch>&, const std::vector<GpuMat>&) { throw_nogpu(); }
void cv::gpu::BruteForceMatcher_GPU_base::knnMatch(const GpuMat&, const GpuMat&, GpuMat&, GpuMat&, GpuMat&, int, const GpuMat&, Stream&) { throw_nogpu(); }
void cv::gpu::BruteForceMatcher_GPU_base::knnMatchDownload(const GpuMat&, const GpuMat&, std::vector< std::vector<DMatch> >&, bool) { throw_nogpu(); }
void cv::gpu::BruteForceMatcher_GPU_base::knnMatchConvert(const Mat&, const Mat&, std::vector< std::vector<DMatch> >&, bool) { throw_nogpu(); }
void cv::gpu::BruteForceMatcher_GPU_base::knnMatch(const GpuMat&, const GpuMat&, std::vector< std::vector<DMatch> >&, int, const GpuMat&, bool) { throw_nogpu(); }
void cv::gpu::BruteForceMatcher_GPU_base::knnMatch(const GpuMat&, std::vector< std::vector<DMatch> >&, int, const std::vector<GpuMat>&, bool) { throw_nogpu(); }
void cv::gpu::BruteForceMatcher_GPU_base::radiusMatch(const GpuMat&, const GpuMat&, GpuMat&, GpuMat&, GpuMat&, float, const GpuMat&, Stream&) { throw_nogpu(); }
void cv::gpu::BruteForceMatcher_GPU_base::radiusMatchDownload(const GpuMat&, const GpuMat&, const GpuMat&, std::vector< std::vector<DMatch> >&, bool) { throw_nogpu(); }
void cv::gpu::BruteForceMatcher_GPU_base::radiusMatchConvert(const Mat&, const Mat&, const Mat&, std::vector< std::vector<DMatch> >&, bool) { throw_nogpu(); }
void cv::gpu::BruteForceMatcher_GPU_base::radiusMatch(const GpuMat&, const GpuMat&, std::vector< std::vector<DMatch> >&, float, const GpuMat&, bool) { throw_nogpu(); }
void cv::gpu::BruteForceMatcher_GPU_base::radiusMatch(const GpuMat&, std::vector< std::vector<DMatch> >&, float, const std::vector<GpuMat>&, bool) { throw_nogpu(); }

#else /* !defined (HAVE_CUDA) */

namespace cv { namespace gpu { namespace bfmatcher
{
    template <typename T> void matchSingleL1_gpu(const DevMem2D& query, const DevMem2D& train, const DevMem2D& mask, 
        const DevMem2D& trainIdx, const DevMem2D& distance, 
        int cc, cudaStream_t stream);
    template <typename T> void matchSingleL2_gpu(const DevMem2D& query, const DevMem2D& train, const DevMem2D& mask, 
        const DevMem2D& trainIdx, const DevMem2D& distance, 
        int cc, cudaStream_t stream);
    template <typename T> void matchSingleHamming_gpu(const DevMem2D& query, const DevMem2D& train, const DevMem2D& mask, 
        const DevMem2D& trainIdx, const DevMem2D& distance, 
        int cc, cudaStream_t stream);

    template <typename T> void matchCollectionL1_gpu(const DevMem2D& query, const DevMem2D& trainCollection, const DevMem2D_<PtrStep>& maskCollection, 
        const DevMem2D& trainIdx, const DevMem2D& imgIdx, const DevMem2D& distance, 
        int cc, cudaStream_t stream);
    template <typename T> void matchCollectionL2_gpu(const DevMem2D& query, const DevMem2D& trainCollection, const DevMem2D_<PtrStep>& maskCollection, 
        const DevMem2D& trainIdx, const DevMem2D& imgIdx, const DevMem2D& distance, 
        int cc, cudaStream_t stream);
    template <typename T> void matchCollectionHamming_gpu(const DevMem2D& query, const DevMem2D& trainCollection, const DevMem2D_<PtrStep>& maskCollection, 
        const DevMem2D& trainIdx, const DevMem2D& imgIdx, const DevMem2D& distance,
        int cc, cudaStream_t stream);

    template <typename T> void knnMatchL1_gpu(const DevMem2D& query, const DevMem2D& train, int k, const DevMem2D& mask, 
        const DevMem2D& trainIdx, const DevMem2D& distance, const DevMem2D& allDist, 
        int cc, cudaStream_t stream);
    template <typename T> void knnMatchL2_gpu(const DevMem2D& query, const DevMem2D& train, int k, const DevMem2D& mask, 
        const DevMem2D& trainIdx, const DevMem2D& distance, const DevMem2D& allDist, 
        int cc, cudaStream_t stream);
    template <typename T> void knnMatchHamming_gpu(const DevMem2D& query, const DevMem2D& train, int k, const DevMem2D& mask, 
        const DevMem2D& trainIdx, const DevMem2D& distance, const DevMem2D& allDist, 
        int cc, cudaStream_t stream);

    template <typename T> void radiusMatchL1_gpu(const DevMem2D& query, const DevMem2D& train, float maxDistance, const DevMem2D& mask, 
        const DevMem2D& trainIdx, const DevMem2D& nMatches, const DevMem2D& distance, 
        cudaStream_t stream);
    template <typename T> void radiusMatchL2_gpu(const DevMem2D& query, const DevMem2D& train, float maxDistance, const DevMem2D& mask, 
        const DevMem2D& trainIdx, const DevMem2D& nMatches, const DevMem2D& distance, 
        cudaStream_t stream);
    template <typename T> void radiusMatchHamming_gpu(const DevMem2D& query, const DevMem2D& train, float maxDistance, const DevMem2D& mask, 
        const DevMem2D& trainIdx, const DevMem2D& nMatches, const DevMem2D& distance, 
        cudaStream_t stream);
}}}

namespace
{
    struct ImgIdxSetter
    {
        explicit inline ImgIdxSetter(int imgIdx_) : imgIdx(imgIdx_) {}
        inline void operator()(DMatch& m) const {m.imgIdx = imgIdx;}
        int imgIdx;
    };
}

cv::gpu::BruteForceMatcher_GPU_base::BruteForceMatcher_GPU_base(DistType distType_) : distType(distType_)
{
}

////////////////////////////////////////////////////////////////////
// Train collection

void cv::gpu::BruteForceMatcher_GPU_base::add(const vector<GpuMat>& descCollection)
{
    trainDescCollection.insert(trainDescCollection.end(), descCollection.begin(), descCollection.end());
}

const vector<GpuMat>& cv::gpu::BruteForceMatcher_GPU_base::getTrainDescriptors() const
{
    return trainDescCollection;
}

void cv::gpu::BruteForceMatcher_GPU_base::clear()
{
    trainDescCollection.clear();
}

bool cv::gpu::BruteForceMatcher_GPU_base::empty() const
{
    return trainDescCollection.empty();
}

bool cv::gpu::BruteForceMatcher_GPU_base::isMaskSupported() const
{
    return true;
}

////////////////////////////////////////////////////////////////////
// Match

void cv::gpu::BruteForceMatcher_GPU_base::matchSingle(const GpuMat& queryDescs, const GpuMat& trainDescs,
    GpuMat& trainIdx, GpuMat& distance, const GpuMat& mask, Stream& stream)
{
    if (queryDescs.empty() || trainDescs.empty())
        return;

    using namespace cv::gpu::bfmatcher;

    typedef void (*match_caller_t)(const DevMem2D& query, const DevMem2D& train, const DevMem2D& mask, 
        const DevMem2D& trainIdx, const DevMem2D& distance, 
        int cc, cudaStream_t stream);

    static const match_caller_t match_callers[3][8] =
    {
        {
            matchSingleL1_gpu<unsigned char>, matchSingleL1_gpu<signed char>, 
            matchSingleL1_gpu<unsigned short>, matchSingleL1_gpu<short>, 
            matchSingleL1_gpu<int>, matchSingleL1_gpu<float>, 0, 0
        },
        {
            matchSingleL2_gpu<unsigned char>, matchSingleL2_gpu<signed char>, 
            matchSingleL2_gpu<unsigned short>, matchSingleL2_gpu<short>, 
            matchSingleL2_gpu<int>, matchSingleL2_gpu<float>, 0, 0
        },
        {
            matchSingleHamming_gpu<unsigned char>, matchSingleHamming_gpu<signed char>, 
            matchSingleHamming_gpu<unsigned short>, matchSingleHamming_gpu<short>, 
            matchSingleHamming_gpu<int>, 0, 0, 0
        }
    };

    CV_Assert(queryDescs.channels() == 1 && queryDescs.depth() < CV_64F);
    CV_Assert(trainDescs.cols == queryDescs.cols && trainDescs.type() == queryDescs.type());

    const int nQuery = queryDescs.rows;

    ensureSizeIsEnough(1, nQuery, CV_32S, trainIdx);
    ensureSizeIsEnough(1, nQuery, CV_32F, distance);

    match_caller_t func = match_callers[distType][queryDescs.depth()];
    CV_Assert(func != 0);

    DeviceInfo info;
    int cc = info.majorVersion() * 10 + info.minorVersion();

    func(queryDescs, trainDescs, mask, trainIdx, distance, cc, StreamAccessor::getStream(stream));
}

void cv::gpu::BruteForceMatcher_GPU_base::matchDownload(const GpuMat& trainIdx, const GpuMat& distance, vector<DMatch>& matches)
{
    if (trainIdx.empty() || distance.empty())
        return;

    Mat trainIdxCPU = trainIdx;
    Mat distanceCPU = distance;

    matchConvert(trainIdxCPU, distanceCPU, matches);
}

void cv::gpu::BruteForceMatcher_GPU_base::matchConvert(const Mat& trainIdx, const Mat& distance, std::vector<DMatch>& matches)
{
    if (trainIdx.empty() || distance.empty())
        return;

    CV_Assert(trainIdx.type() == CV_32SC1 && trainIdx.isContinuous());
    CV_Assert(distance.type() == CV_32FC1 && distance.isContinuous() && distance.cols == trainIdx.cols);

    const int nQuery = trainIdx.cols;

    matches.clear();
    matches.reserve(nQuery);

    const int* trainIdx_ptr = trainIdx.ptr<int>();
    const float* distance_ptr =  distance.ptr<float>();
    for (int queryIdx = 0; queryIdx < nQuery; ++queryIdx, ++trainIdx_ptr, ++distance_ptr)
    {
        int trainIdx = *trainIdx_ptr;
        if (trainIdx == -1)
            continue;

        float distance = *distance_ptr;

        DMatch m(queryIdx, trainIdx, 0, distance);

        matches.push_back(m);
    }
}

void cv::gpu::BruteForceMatcher_GPU_base::match(const GpuMat& queryDescs, const GpuMat& trainDescs,
    vector<DMatch>& matches, const GpuMat& mask)
{
    GpuMat trainIdx, distance;
    matchSingle(queryDescs, trainDescs, trainIdx, distance, mask);
    matchDownload(trainIdx, distance, matches);
}

void cv::gpu::BruteForceMatcher_GPU_base::makeGpuCollection(GpuMat& trainCollection, GpuMat& maskCollection,
    const vector<GpuMat>& masks)
{
    if (empty())
        return;

    if (masks.empty())
    {
        Mat trainCollectionCPU(1, static_cast<int>(trainDescCollection.size()), CV_8UC(sizeof(DevMem2D)));

        for (size_t i = 0; i < trainDescCollection.size(); ++i)
        {
            const GpuMat& trainDescs = trainDescCollection[i];

            trainCollectionCPU.ptr<DevMem2D>(0)[i] = trainDescs;
        }

        trainCollection.upload(trainCollectionCPU);
    }
    else
    {
        CV_Assert(masks.size() == trainDescCollection.size());

        Mat trainCollectionCPU(1, static_cast<int>(trainDescCollection.size()), CV_8UC(sizeof(DevMem2D)));
        Mat maskCollectionCPU(1, static_cast<int>(trainDescCollection.size()), CV_8UC(sizeof(PtrStep)));

        for (size_t i = 0; i < trainDescCollection.size(); ++i)
        {
            const GpuMat& trainDescs = trainDescCollection[i];
            const GpuMat& mask = masks[i];

            CV_Assert(mask.empty() || (mask.type() == CV_8UC1 && mask.cols == trainDescs.rows));

            trainCollectionCPU.ptr<DevMem2D>(0)[i] = trainDescs;

            maskCollectionCPU.ptr<PtrStep>(0)[i] = mask;
        }

        trainCollection.upload(trainCollectionCPU);
        maskCollection.upload(maskCollectionCPU);
    }
}

void cv::gpu::BruteForceMatcher_GPU_base::matchCollection(const GpuMat& queryDescs, const GpuMat& trainCollection,
    GpuMat& trainIdx, GpuMat& imgIdx, GpuMat& distance, const GpuMat& maskCollection, Stream& stream)
{
    if (queryDescs.empty() || trainCollection.empty())
        return;

    using namespace cv::gpu::bfmatcher;

    typedef void (*match_caller_t)(const DevMem2D& query, const DevMem2D& trainCollection, const DevMem2D_<PtrStep>& maskCollection, 
        const DevMem2D& trainIdx, const DevMem2D& imgIdx, const DevMem2D& distance, 
        int cc, cudaStream_t stream);

    static const match_caller_t match_callers[3][8] =
    {
        {
            matchCollectionL1_gpu<unsigned char>, matchCollectionL1_gpu<signed char>,
            matchCollectionL1_gpu<unsigned short>, matchCollectionL1_gpu<short>,
            matchCollectionL1_gpu<int>, matchCollectionL1_gpu<float>, 0, 0
        },
        {
            matchCollectionL2_gpu<unsigned char>, matchCollectionL2_gpu<signed char>,
            matchCollectionL2_gpu<unsigned short>, matchCollectionL2_gpu<short>,
            matchCollectionL2_gpu<int>, matchCollectionL2_gpu<float>, 0, 0
        },
        {
            matchCollectionHamming_gpu<unsigned char>, matchCollectionHamming_gpu<signed char>,
            matchCollectionHamming_gpu<unsigned short>, matchCollectionHamming_gpu<short>,
            matchCollectionHamming_gpu<int>, 0, 0, 0
        }
    };

    CV_Assert(queryDescs.channels() == 1 && queryDescs.depth() < CV_64F);

    const int nQuery = queryDescs.rows;

    ensureSizeIsEnough(1, nQuery, CV_32S, trainIdx);
    ensureSizeIsEnough(1, nQuery, CV_32S, imgIdx);
    ensureSizeIsEnough(1, nQuery, CV_32F, distance);

    match_caller_t func = match_callers[distType][queryDescs.depth()];
    CV_Assert(func != 0);

    DeviceInfo info;
    int cc = info.majorVersion() * 10 + info.minorVersion();

    func(queryDescs, trainCollection, maskCollection, trainIdx, imgIdx, distance, cc, StreamAccessor::getStream(stream));
}

void cv::gpu::BruteForceMatcher_GPU_base::matchDownload(const GpuMat& trainIdx, const GpuMat& imgIdx, const GpuMat& distance, vector<DMatch>& matches)
{
    if (trainIdx.empty() || imgIdx.empty() || distance.empty())
        return;

    Mat trainIdxCPU = trainIdx;
    Mat imgIdxCPU = imgIdx;
    Mat distanceCPU = distance;

    matchConvert(trainIdxCPU, imgIdxCPU, distanceCPU, matches);
}

void cv::gpu::BruteForceMatcher_GPU_base::matchConvert(const Mat& trainIdx, const Mat& imgIdx, const Mat& distance, std::vector<DMatch>& matches)
{
    if (trainIdx.empty() || imgIdx.empty() || distance.empty())
        return;

    CV_Assert(trainIdx.type() == CV_32SC1 && trainIdx.isContinuous());
    CV_Assert(imgIdx.type() == CV_32SC1 && imgIdx.isContinuous() && imgIdx.cols == trainIdx.cols);
    CV_Assert(distance.type() == CV_32FC1 && distance.isContinuous() && imgIdx.cols == trainIdx.cols);

    const int nQuery = trainIdx.cols;

    matches.clear();
    matches.reserve(nQuery);

    const int* trainIdx_ptr = trainIdx.ptr<int>();
    const int* imgIdx_ptr = imgIdx.ptr<int>();
    const float* distance_ptr =  distance.ptr<float>();
    for (int queryIdx = 0; queryIdx < nQuery; ++queryIdx, ++trainIdx_ptr, ++imgIdx_ptr, ++distance_ptr)
    {
        int trainIdx = *trainIdx_ptr;
        if (trainIdx == -1)
            continue;

        int imgIdx = *imgIdx_ptr;

        float distance = *distance_ptr;

        DMatch m(queryIdx, trainIdx, imgIdx, distance);

        matches.push_back(m);
    }
}

void cv::gpu::BruteForceMatcher_GPU_base::match(const GpuMat& queryDescs, vector<DMatch>& matches, const vector<GpuMat>& masks)
{
    GpuMat trainCollection;
    GpuMat maskCollection;

    makeGpuCollection(trainCollection, maskCollection, masks);

    GpuMat trainIdx, imgIdx, distance;

    matchCollection(queryDescs, trainCollection, trainIdx, imgIdx, distance, maskCollection);
    matchDownload(trainIdx, imgIdx, distance, matches);
}

////////////////////////////////////////////////////////////////////
// KnnMatch

void cv::gpu::BruteForceMatcher_GPU_base::knnMatch(const GpuMat& queryDescs, const GpuMat& trainDescs,
    GpuMat& trainIdx, GpuMat& distance, GpuMat& allDist, int k, const GpuMat& mask, Stream& stream)
{
    if (queryDescs.empty() || trainDescs.empty())
        return;

    using namespace cv::gpu::bfmatcher;

    typedef void (*match_caller_t)(const DevMem2D& query, const DevMem2D& train, int k, const DevMem2D& mask, 
        const DevMem2D& trainIdx, const DevMem2D& distance, const DevMem2D& allDist, 
        int cc, cudaStream_t stream);

    static const match_caller_t match_callers[3][8] =
    {
        {
            knnMatchL1_gpu<unsigned char>, knnMatchL1_gpu<signed char>, knnMatchL1_gpu<unsigned short>,
            knnMatchL1_gpu<short>, knnMatchL1_gpu<int>, knnMatchL1_gpu<float>, 0, 0
        },
        {
            knnMatchL2_gpu<unsigned char>, knnMatchL2_gpu<signed char>, knnMatchL2_gpu<unsigned short>,
            knnMatchL2_gpu<short>, knnMatchL2_gpu<int>, knnMatchL2_gpu<float>, 0, 0
        },
        {
            knnMatchHamming_gpu<unsigned char>, knnMatchHamming_gpu<signed char>, knnMatchHamming_gpu<unsigned short>,
            knnMatchHamming_gpu<short>, knnMatchHamming_gpu<int>, 0, 0, 0
        }
    };

    CV_Assert(queryDescs.channels() == 1 && queryDescs.depth() < CV_64F);
    CV_Assert(trainDescs.type() == queryDescs.type() && trainDescs.cols == queryDescs.cols);

    const int nQuery = queryDescs.rows;
    const int nTrain = trainDescs.rows;

    if (k == 2)
    {
        ensureSizeIsEnough(1, nQuery, CV_32SC2, trainIdx);
        ensureSizeIsEnough(1, nQuery, CV_32FC2, distance);
    }
    else
    {
        ensureSizeIsEnough(nQuery, k, CV_32S, trainIdx);
        ensureSizeIsEnough(nQuery, k, CV_32F, distance);
        ensureSizeIsEnough(nQuery, nTrain, CV_32FC1, allDist);
    }

    if (stream)
    {
        stream.enqueueMemSet(trainIdx, Scalar::all(-1));
        if (k != 2)
            stream.enqueueMemSet(allDist, Scalar::all(numeric_limits<float>::max()));
    }
    else
    {
        trainIdx.setTo(Scalar::all(-1));
        if (k != 2)
            allDist.setTo(Scalar::all(numeric_limits<float>::max()));
    }

    match_caller_t func = match_callers[distType][queryDescs.depth()];
    CV_Assert(func != 0);
    
    DeviceInfo info;
    int cc = info.majorVersion() * 10 + info.minorVersion();

    func(queryDescs, trainDescs, k, mask, trainIdx, distance, allDist, cc, StreamAccessor::getStream(stream));
}

void cv::gpu::BruteForceMatcher_GPU_base::knnMatchDownload(const GpuMat& trainIdx, const GpuMat& distance,
    vector< vector<DMatch> >& matches, bool compactResult)
{
    if (trainIdx.empty() || distance.empty())
        return;

    Mat trainIdxCPU = trainIdx;
    Mat distanceCPU = distance;

    knnMatchConvert(trainIdxCPU, distanceCPU, matches, compactResult);
}

void cv::gpu::BruteForceMatcher_GPU_base::knnMatchConvert(const Mat& trainIdx, const Mat& distance, 
                                                          std::vector< std::vector<DMatch> >& matches, bool compactResult)
{
    if (trainIdx.empty() || distance.empty())
        return;

    CV_Assert(trainIdx.type() == CV_32SC2 || trainIdx.type() == CV_32SC1);
    CV_Assert(distance.type() == CV_32FC2 || distance.type() == CV_32FC1);
    CV_Assert(distance.size() == trainIdx.size());
    CV_Assert(trainIdx.isContinuous() && distance.isContinuous());

    const int nQuery = trainIdx.type() == CV_32SC2 ? trainIdx.cols : trainIdx.rows;
    const int k = trainIdx.type() == CV_32SC2 ? 2 :trainIdx.cols;

    matches.clear();
    matches.reserve(nQuery);
    
    const int* trainIdx_ptr = trainIdx.ptr<int>();
    const float* distance_ptr = distance.ptr<float>();

    for (int queryIdx = 0; queryIdx < nQuery; ++queryIdx)
    {
        matches.push_back(vector<DMatch>());
        vector<DMatch>& curMatches = matches.back();
        curMatches.reserve(k);

        for (int i = 0; i < k; ++i, ++trainIdx_ptr, ++distance_ptr)
        {
            int trainIdx = *trainIdx_ptr;

            if (trainIdx != -1)
            {
                float distance = *distance_ptr;

                DMatch m(queryIdx, trainIdx, 0, distance);

                curMatches.push_back(m);
            }
        }

        if (compactResult && curMatches.empty())
            matches.pop_back();
    }
}

void cv::gpu::BruteForceMatcher_GPU_base::knnMatch(const GpuMat& queryDescs, const GpuMat& trainDescs,
    vector< vector<DMatch> >& matches, int k, const GpuMat& mask, bool compactResult)
{
    GpuMat trainIdx, distance, allDist;
    knnMatch(queryDescs, trainDescs, trainIdx, distance, allDist, k, mask);
    knnMatchDownload(trainIdx, distance, matches, compactResult);
}

void cv::gpu::BruteForceMatcher_GPU_base::knnMatch(const GpuMat& queryDescs,
    vector< vector<DMatch> >& matches, int knn, const vector<GpuMat>& masks, bool compactResult)
{
    if (queryDescs.empty() || empty())
        return;

    vector< vector<DMatch> > curMatches;
    vector<DMatch> temp;
    temp.reserve(2 * knn);

    matches.resize(queryDescs.rows);
    for_each(matches.begin(), matches.end(), bind2nd(mem_fun_ref(&vector<DMatch>::reserve), knn));

    for (size_t imgIdx = 0; imgIdx < trainDescCollection.size(); ++imgIdx)
    {
        knnMatch(queryDescs, trainDescCollection[imgIdx], curMatches, knn,
            masks.empty() ? GpuMat() : masks[imgIdx]);

        for (int queryIdx = 0; queryIdx < queryDescs.rows; ++queryIdx)
        {
            vector<DMatch>& localMatch = curMatches[queryIdx];
            vector<DMatch>& globalMatch = matches[queryIdx];

            for_each(localMatch.begin(), localMatch.end(), ImgIdxSetter(static_cast<int>(imgIdx)));

            temp.clear();
            merge(globalMatch.begin(), globalMatch.end(), localMatch.begin(), localMatch.end(), back_inserter(temp));

            globalMatch.clear();
            const size_t count = std::min((size_t)knn, temp.size());
            copy(temp.begin(), temp.begin() + count, back_inserter(globalMatch));
        }
    }

    if (compactResult)
    {
        vector< vector<DMatch> >::iterator new_end = remove_if(matches.begin(), matches.end(),
            mem_fun_ref(&vector<DMatch>::empty));
        matches.erase(new_end, matches.end());
    }
}

////////////////////////////////////////////////////////////////////
// RadiusMatch

void cv::gpu::BruteForceMatcher_GPU_base::radiusMatch(const GpuMat& queryDescs, const GpuMat& trainDescs,
    GpuMat& trainIdx, GpuMat& nMatches, GpuMat& distance, float maxDistance, const GpuMat& mask, Stream& stream)
{
    if (queryDescs.empty() || trainDescs.empty())
        return;

    using namespace cv::gpu::bfmatcher;

    typedef void (*radiusMatch_caller_t)(const DevMem2D& query, const DevMem2D& train, float maxDistance, const DevMem2D& mask, 
        const DevMem2D& trainIdx, const DevMem2D& nMatches, const DevMem2D& distance, 
        cudaStream_t stream);

    static const radiusMatch_caller_t radiusMatch_callers[3][8] =
    {
        {
            radiusMatchL1_gpu<unsigned char>, radiusMatchL1_gpu<signed char>, radiusMatchL1_gpu<unsigned short>,
            radiusMatchL1_gpu<short>, radiusMatchL1_gpu<int>, radiusMatchL1_gpu<float>, 0, 0
        },
        {
            radiusMatchL2_gpu<unsigned char>, radiusMatchL2_gpu<signed char>, radiusMatchL2_gpu<unsigned short>,
            radiusMatchL2_gpu<short>, radiusMatchL2_gpu<int>, radiusMatchL2_gpu<float>, 0, 0
        },
        {
            radiusMatchHamming_gpu<unsigned char>, radiusMatchHamming_gpu<signed char>, radiusMatchHamming_gpu<unsigned short>,
            radiusMatchHamming_gpu<short>, radiusMatchHamming_gpu<int>, 0, 0, 0
        }
    };

    CV_Assert(DeviceInfo().supports(GLOBAL_ATOMICS));

    const int nQuery = queryDescs.rows;
    const int nTrain = trainDescs.rows;

    CV_Assert(queryDescs.channels() == 1 && queryDescs.depth() < CV_64F);
    CV_Assert(trainDescs.type() == queryDescs.type() && trainDescs.cols == queryDescs.cols);
    CV_Assert(trainIdx.empty() || (trainIdx.rows == nQuery && trainIdx.size() == distance.size()));

    ensureSizeIsEnough(1, nQuery, CV_32SC1, nMatches);
    if (trainIdx.empty())
    {
        ensureSizeIsEnough(nQuery, nTrain, CV_32SC1, trainIdx);
        ensureSizeIsEnough(nQuery, nTrain, CV_32FC1, distance);
    }

    if (stream)
        stream.enqueueMemSet(nMatches, Scalar::all(0));
    else
        nMatches.setTo(Scalar::all(0));

    radiusMatch_caller_t func = radiusMatch_callers[distType][queryDescs.depth()];
    CV_Assert(func != 0);

    func(queryDescs, trainDescs, maxDistance, mask, trainIdx, nMatches, distance, StreamAccessor::getStream(stream));
}

void cv::gpu::BruteForceMatcher_GPU_base::radiusMatchDownload(const GpuMat& trainIdx, const GpuMat& nMatches,
    const GpuMat& distance, std::vector< std::vector<DMatch> >& matches, bool compactResult)
{
    if (trainIdx.empty() || nMatches.empty() || distance.empty())
        return;

    Mat trainIdxCPU = trainIdx;
    Mat nMatchesCPU = nMatches;
    Mat distanceCPU = distance;

    radiusMatchConvert(trainIdxCPU, nMatchesCPU, distanceCPU, matches, compactResult);
}

void cv::gpu::BruteForceMatcher_GPU_base::radiusMatchConvert(const Mat& trainIdx, const Mat& nMatches, const Mat& distance,
                std::vector< std::vector<DMatch> >& matches, bool compactResult)
{
    if (trainIdx.empty() || nMatches.empty() || distance.empty())
        return;

    CV_Assert(trainIdx.type() == CV_32SC1);
    CV_Assert(nMatches.type() == CV_32SC1 && nMatches.isContinuous() && nMatches.cols >= trainIdx.rows);
    CV_Assert(distance.type() == CV_32FC1 && distance.size() == trainIdx.size());

    const int nQuery = trainIdx.rows;

    matches.clear();
    matches.reserve(nQuery);

    const unsigned int* nMatches_ptr = nMatches.ptr<unsigned int>();
    for (int queryIdx = 0; queryIdx < nQuery; ++queryIdx)
    {
        const int* trainIdx_ptr = trainIdx.ptr<int>(queryIdx);
        const float* distance_ptr = distance.ptr<float>(queryIdx);

        const int nMatches = std::min(static_cast<int>(nMatches_ptr[queryIdx]), trainIdx.cols);

        if (nMatches == 0)
        {
            if (!compactResult)
                matches.push_back(vector<DMatch>());
            continue;
        }

        matches.push_back(vector<DMatch>());
        vector<DMatch>& curMatches = matches.back();
        curMatches.reserve(nMatches);

        for (int i = 0; i < nMatches; ++i, ++trainIdx_ptr, ++distance_ptr)
        {
            int trainIdx = *trainIdx_ptr;

            float distance = *distance_ptr;

            DMatch m(queryIdx, trainIdx, 0, distance);

            curMatches.push_back(m);
        }
        sort(curMatches.begin(), curMatches.end());
    }
}

void cv::gpu::BruteForceMatcher_GPU_base::radiusMatch(const GpuMat& queryDescs, const GpuMat& trainDescs,
    vector< vector<DMatch> >& matches, float maxDistance, const GpuMat& mask, bool compactResult)
{
    GpuMat trainIdx, nMatches, distance;
    radiusMatch(queryDescs, trainDescs, trainIdx, nMatches, distance, maxDistance, mask);
    radiusMatchDownload(trainIdx, nMatches, distance, matches, compactResult);
}

void cv::gpu::BruteForceMatcher_GPU_base::radiusMatch(const GpuMat& queryDescs, vector< vector<DMatch> >& matches,
    float maxDistance, const vector<GpuMat>& masks, bool compactResult)
{
    if (queryDescs.empty() || empty())
        return;

    matches.resize(queryDescs.rows);

    vector< vector<DMatch> > curMatches;

    for (size_t imgIdx = 0; imgIdx < trainDescCollection.size(); ++imgIdx)
    {
        radiusMatch(queryDescs, trainDescCollection[imgIdx], curMatches, maxDistance,
            masks.empty() ? GpuMat() : masks[imgIdx]);

        for (int queryIdx = 0; queryIdx < queryDescs.rows; ++queryIdx)
        {
            vector<DMatch>& localMatch = curMatches[queryIdx];
            vector<DMatch>& globalMatch = matches[queryIdx];

            for_each(localMatch.begin(), localMatch.end(), ImgIdxSetter(static_cast<int>(imgIdx)));

            const size_t oldSize = globalMatch.size();

            copy(localMatch.begin(), localMatch.end(), back_inserter(globalMatch));
            inplace_merge(globalMatch.begin(), globalMatch.begin() + oldSize, globalMatch.end());
        }
    }

    if (compactResult)
    {
        vector< vector<DMatch> >::iterator new_end = remove_if(matches.begin(), matches.end(),
            mem_fun_ref(&vector<DMatch>::empty));
        matches.erase(new_end, matches.end());
    }
}

#endif /* !defined (HAVE_CUDA) */
