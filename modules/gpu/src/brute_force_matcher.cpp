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
void cv::gpu::BruteForceMatcher_GPU_base::matchConvert(const Mat&, const Mat&, vector<DMatch>&) { throw_nogpu(); }
void cv::gpu::BruteForceMatcher_GPU_base::match(const GpuMat&, const GpuMat&, vector<DMatch>&, const GpuMat&) { throw_nogpu(); }
void cv::gpu::BruteForceMatcher_GPU_base::makeGpuCollection(GpuMat&, GpuMat&, const vector<GpuMat>&) { throw_nogpu(); }
void cv::gpu::BruteForceMatcher_GPU_base::matchCollection(const GpuMat&, const GpuMat&, GpuMat&, GpuMat&, GpuMat&, const GpuMat&, Stream&) { throw_nogpu(); }
void cv::gpu::BruteForceMatcher_GPU_base::matchDownload(const GpuMat&, const GpuMat&, const GpuMat&, vector<DMatch>&) { throw_nogpu(); }
void cv::gpu::BruteForceMatcher_GPU_base::matchConvert(const Mat&, const Mat&, const Mat&, vector<DMatch>&) { throw_nogpu(); }
void cv::gpu::BruteForceMatcher_GPU_base::match(const GpuMat&, vector<DMatch>&, const vector<GpuMat>&) { throw_nogpu(); }
void cv::gpu::BruteForceMatcher_GPU_base::knnMatchSingle(const GpuMat&, const GpuMat&, GpuMat&, GpuMat&, GpuMat&, int, const GpuMat&, Stream&) { throw_nogpu(); }
void cv::gpu::BruteForceMatcher_GPU_base::knnMatchDownload(const GpuMat&, const GpuMat&, vector< vector<DMatch> >&, bool) { throw_nogpu(); }
void cv::gpu::BruteForceMatcher_GPU_base::knnMatchConvert(const Mat&, const Mat&, vector< vector<DMatch> >&, bool) { throw_nogpu(); }
void cv::gpu::BruteForceMatcher_GPU_base::knnMatch(const GpuMat&, const GpuMat&, vector< vector<DMatch> >&, int, const GpuMat&, bool) { throw_nogpu(); }
void cv::gpu::BruteForceMatcher_GPU_base::knnMatch2Collection(const GpuMat&, const GpuMat&, GpuMat&, GpuMat&, GpuMat&, const GpuMat&, Stream&) { throw_nogpu(); }
void cv::gpu::BruteForceMatcher_GPU_base::knnMatch2Download(const GpuMat&, const GpuMat&, const GpuMat&, vector< vector<DMatch> >&, bool) { throw_nogpu(); }
void cv::gpu::BruteForceMatcher_GPU_base::knnMatch2Convert(const Mat&, const Mat&, const Mat&, vector< vector<DMatch> >&, bool) { throw_nogpu(); }
void cv::gpu::BruteForceMatcher_GPU_base::knnMatch(const GpuMat&, vector< vector<DMatch> >&, int, const vector<GpuMat>&, bool) { throw_nogpu(); }
void cv::gpu::BruteForceMatcher_GPU_base::radiusMatchSingle(const GpuMat&, const GpuMat&, GpuMat&, GpuMat&, GpuMat&, float, const GpuMat&, Stream&) { throw_nogpu(); }
void cv::gpu::BruteForceMatcher_GPU_base::radiusMatchDownload(const GpuMat&, const GpuMat&, const GpuMat&, vector< vector<DMatch> >&, bool) { throw_nogpu(); }
void cv::gpu::BruteForceMatcher_GPU_base::radiusMatchConvert(const Mat&, const Mat&, const Mat&, vector< vector<DMatch> >&, bool) { throw_nogpu(); }
void cv::gpu::BruteForceMatcher_GPU_base::radiusMatch(const GpuMat&, const GpuMat&, vector< vector<DMatch> >&, float, const GpuMat&, bool) { throw_nogpu(); }
void cv::gpu::BruteForceMatcher_GPU_base::radiusMatchCollection(const GpuMat&, GpuMat&, GpuMat&, GpuMat&, GpuMat&, float, const vector<GpuMat>&, Stream&) { throw_nogpu(); }
void cv::gpu::BruteForceMatcher_GPU_base::radiusMatchDownload(const GpuMat&, const GpuMat&, const GpuMat&, const GpuMat&, vector< vector<DMatch> >&, bool) { throw_nogpu(); }
void cv::gpu::BruteForceMatcher_GPU_base::radiusMatchConvert(const Mat&, const Mat&, const Mat&, const Mat&, vector< vector<DMatch> >&, bool) { throw_nogpu(); }
void cv::gpu::BruteForceMatcher_GPU_base::radiusMatch(const GpuMat&, vector< vector<DMatch> >&, float, const vector<GpuMat>&, bool) { throw_nogpu(); }

#else /* !defined (HAVE_CUDA) */

namespace cv { namespace gpu { namespace device 
{
    namespace bf_match
    {
        template <typename T> void matchL1_gpu(const DevMem2Db& query, const DevMem2Db& train, const DevMem2Db& mask, 
            const DevMem2Di& trainIdx, const DevMem2Df& distance, 
            int cc, cudaStream_t stream);
        template <typename T> void matchL2_gpu(const DevMem2Db& query, const DevMem2Db& train, const DevMem2Db& mask, 
            const DevMem2Di& trainIdx, const DevMem2Df& distance, 
            int cc, cudaStream_t stream);
        template <typename T> void matchHamming_gpu(const DevMem2Db& query, const DevMem2Db& train, const DevMem2Db& mask, 
            const DevMem2Di& trainIdx, const DevMem2Df& distance, 
            int cc, cudaStream_t stream);

        template <typename T> void matchL1_gpu(const DevMem2Db& query, const DevMem2Db& trains, const DevMem2D_<PtrStepb>& masks, 
            const DevMem2Di& trainIdx, const DevMem2Di& imgIdx, const DevMem2Df& distance, 
            int cc, cudaStream_t stream);
        template <typename T> void matchL2_gpu(const DevMem2Db& query, const DevMem2Db& trains, const DevMem2D_<PtrStepb>& masks, 
            const DevMem2Di& trainIdx, const DevMem2Di& imgIdx, const DevMem2Df& distance, 
            int cc, cudaStream_t stream);
        template <typename T> void matchHamming_gpu(const DevMem2Db& query, const DevMem2Db& trains, const DevMem2D_<PtrStepb>& masks, 
            const DevMem2Di& trainIdx, const DevMem2Di& imgIdx, const DevMem2Df& distance,
            int cc, cudaStream_t stream);
    }

    namespace bf_knnmatch
    {
        template <typename T> void matchL1_gpu(const DevMem2Db& query, const DevMem2Db& train, int k, const DevMem2Db& mask, 
            const DevMem2Db& trainIdx, const DevMem2Db& distance, const DevMem2Df& allDist, 
            int cc, cudaStream_t stream);
        template <typename T> void matchL2_gpu(const DevMem2Db& query, const DevMem2Db& train, int k, const DevMem2Db& mask, 
            const DevMem2Db& trainIdx, const DevMem2Db& distance, const DevMem2Df& allDist, 
            int cc, cudaStream_t stream);
        template <typename T> void matchHamming_gpu(const DevMem2Db& query, const DevMem2Db& train, int k, const DevMem2Db& mask, 
            const DevMem2Db& trainIdx, const DevMem2Db& distance, const DevMem2Df& allDist, 
            int cc, cudaStream_t stream);

        template <typename T> void match2L1_gpu(const DevMem2Db& query, const DevMem2Db& trains, const DevMem2D_<PtrStepb>& masks, 
            const DevMem2Db& trainIdx, const DevMem2Db& imgIdx, const DevMem2Db& distance, 
            int cc, cudaStream_t stream);
        template <typename T> void match2L2_gpu(const DevMem2Db& query, const DevMem2Db& trains, const DevMem2D_<PtrStepb>& masks, 
            const DevMem2Db& trainIdx, const DevMem2Db& imgIdx, const DevMem2Db& distance, 
            int cc, cudaStream_t stream);
        template <typename T> void match2Hamming_gpu(const DevMem2Db& query, const DevMem2Db& trains, const DevMem2D_<PtrStepb>& masks, 
            const DevMem2Db& trainIdx, const DevMem2Db& imgIdx, const DevMem2Db& distance, 
            int cc, cudaStream_t stream);
    }

    namespace bf_radius_match 
    {
        template <typename T> void matchL1_gpu(const DevMem2Db& query, const DevMem2Db& train, float maxDistance, const DevMem2Db& mask, 
            const DevMem2Di& trainIdx, const DevMem2Df& distance, const DevMem2D_<unsigned int>& nMatches, 
            int cc, cudaStream_t stream);
        template <typename T> void matchL2_gpu(const DevMem2Db& query, const DevMem2Db& train, float maxDistance, const DevMem2Db& mask, 
            const DevMem2Di& trainIdx, const DevMem2Df& distance, const DevMem2D_<unsigned int>& nMatches, 
            int cc, cudaStream_t stream);
        template <typename T> void matchHamming_gpu(const DevMem2Db& query, const DevMem2Db& train, float maxDistance, const DevMem2Db& mask, 
            const DevMem2Di& trainIdx, const DevMem2Df& distance, const DevMem2D_<unsigned int>& nMatches, 
            int cc, cudaStream_t stream);

        template <typename T> void matchL1_gpu(const DevMem2Db& query, const DevMem2Db* trains, int n, float maxDistance, const DevMem2Db* masks, 
            const DevMem2Di& trainIdx, const DevMem2Di& imgIdx, const DevMem2Df& distance, const DevMem2D_<unsigned int>& nMatches, 
            int cc, cudaStream_t stream);

        template <typename T> void matchL2_gpu(const DevMem2Db& query, const DevMem2Db* trains, int n, float maxDistance, const DevMem2Db* masks, 
            const DevMem2Di& trainIdx, const DevMem2Di& imgIdx, const DevMem2Df& distance, const DevMem2D_<unsigned int>& nMatches, 
            int cc, cudaStream_t stream);

        template <typename T> void matchHamming_gpu(const DevMem2Db& query, const DevMem2Db* trains, int n, float maxDistance, const DevMem2Db* masks, 
            const DevMem2Di& trainIdx, const DevMem2Di& imgIdx, const DevMem2Df& distance, const DevMem2D_<unsigned int>& nMatches, 
            int cc, cudaStream_t stream);
    }
}}}

////////////////////////////////////////////////////////////////////
// Train collection

cv::gpu::BruteForceMatcher_GPU_base::BruteForceMatcher_GPU_base(DistType distType_) : distType(distType_)
{
}

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

void cv::gpu::BruteForceMatcher_GPU_base::matchSingle(const GpuMat& query, const GpuMat& train,
    GpuMat& trainIdx, GpuMat& distance, 
    const GpuMat& mask, Stream& stream)
{
    if (query.empty() || train.empty())
        return;

    using namespace ::cv::gpu::device::bf_match;

    typedef void (*caller_t)(const DevMem2Db& query, const DevMem2Db& train, const DevMem2Db& mask, 
                             const DevMem2Di& trainIdx, const DevMem2Df& distance,
                             int cc, cudaStream_t stream);

    static const caller_t callers[3][6] =
    {
        {
            matchL1_gpu<unsigned char>, 0/*matchL1_gpu<signed char>*/, 
            matchL1_gpu<unsigned short>, matchL1_gpu<short>, 
            matchL1_gpu<int>, matchL1_gpu<float>
        },
        {
            0/*matchL2_gpu<unsigned char>*/, 0/*matchL2_gpu<signed char>*/, 
            0/*matchL2_gpu<unsigned short>*/, 0/*matchL2_gpu<short>*/, 
            0/*matchL2_gpu<int>*/, matchL2_gpu<float>
        },
        {
            matchHamming_gpu<unsigned char>, 0/*matchHamming_gpu<signed char>*/, 
            matchHamming_gpu<unsigned short>, 0/*matchHamming_gpu<short>*/, 
            matchHamming_gpu<int>, 0/*matchHamming_gpu<float>*/
        }
    };

    CV_Assert(query.channels() == 1 && query.depth() < CV_64F);
    CV_Assert(train.cols == query.cols && train.type() == query.type());

    const int nQuery = query.rows;

    ensureSizeIsEnough(1, nQuery, CV_32S, trainIdx);
    ensureSizeIsEnough(1, nQuery, CV_32F, distance);

    caller_t func = callers[distType][query.depth()];
    CV_Assert(func != 0);

    DeviceInfo info;
    int cc = info.majorVersion() * 10 + info.minorVersion();

    func(query, train, mask, trainIdx, distance, cc, StreamAccessor::getStream(stream));
}

void cv::gpu::BruteForceMatcher_GPU_base::matchDownload(const GpuMat& trainIdx, const GpuMat& distance, vector<DMatch>& matches)
{
    if (trainIdx.empty() || distance.empty())
        return;

    Mat trainIdxCPU(trainIdx);
    Mat distanceCPU(distance);

    matchConvert(trainIdxCPU, distanceCPU, matches);
}

void cv::gpu::BruteForceMatcher_GPU_base::matchConvert(const Mat& trainIdx, const Mat& distance, vector<DMatch>& matches)
{
    if (trainIdx.empty() || distance.empty())
        return;

    CV_Assert(trainIdx.type() == CV_32SC1);
    CV_Assert(distance.type() == CV_32FC1 && distance.cols == trainIdx.cols);

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

void cv::gpu::BruteForceMatcher_GPU_base::match(const GpuMat& query, const GpuMat& train,
    vector<DMatch>& matches, const GpuMat& mask)
{
    GpuMat trainIdx, distance;
    matchSingle(query, train, trainIdx, distance, mask);
    matchDownload(trainIdx, distance, matches);
}

void cv::gpu::BruteForceMatcher_GPU_base::makeGpuCollection(GpuMat& trainCollection, GpuMat& maskCollection,
    const vector<GpuMat>& masks)
{
    if (empty())
        return;

    if (masks.empty())
    {
        Mat trainCollectionCPU(1, static_cast<int>(trainDescCollection.size()), CV_8UC(sizeof(DevMem2Db)));

        DevMem2Db* trainCollectionCPU_ptr = trainCollectionCPU.ptr<DevMem2Db>();

        for (size_t i = 0, size = trainDescCollection.size(); i < size; ++i, ++trainCollectionCPU_ptr)
            *trainCollectionCPU_ptr = trainDescCollection[i];

        trainCollection.upload(trainCollectionCPU);
        maskCollection.release();
    }
    else
    {
        CV_Assert(masks.size() == trainDescCollection.size());

        Mat trainCollectionCPU(1, static_cast<int>(trainDescCollection.size()), CV_8UC(sizeof(DevMem2Db)));
        Mat maskCollectionCPU(1, static_cast<int>(trainDescCollection.size()), CV_8UC(sizeof(PtrStepb)));

        DevMem2Db* trainCollectionCPU_ptr = trainCollectionCPU.ptr<DevMem2Db>();
        PtrStepb* maskCollectionCPU_ptr = maskCollectionCPU.ptr<PtrStepb>();

        for (size_t i = 0, size = trainDescCollection.size(); i < size; ++i, ++trainCollectionCPU_ptr, ++maskCollectionCPU_ptr)
        {
            const GpuMat& train = trainDescCollection[i];
            const GpuMat& mask = masks[i];

            CV_Assert(mask.empty() || (mask.type() == CV_8UC1 && mask.cols == train.rows));

            *trainCollectionCPU_ptr = train;
            *maskCollectionCPU_ptr = mask;
        }

        trainCollection.upload(trainCollectionCPU);
        maskCollection.upload(maskCollectionCPU);
    }
}

void cv::gpu::BruteForceMatcher_GPU_base::matchCollection(const GpuMat& query, const GpuMat& trainCollection,
    GpuMat& trainIdx, GpuMat& imgIdx, GpuMat& distance, 
    const GpuMat& masks, Stream& stream)
{
    if (query.empty() || trainCollection.empty())
        return;

    using namespace ::cv::gpu::device::bf_match;

    typedef void (*caller_t)(const DevMem2Db& query, const DevMem2Db& trains, const DevMem2D_<PtrStepb>& masks, 
                             const DevMem2Di& trainIdx, const DevMem2Di& imgIdx, const DevMem2Df& distance, 
                             int cc, cudaStream_t stream);

    static const caller_t callers[3][6] =
    {
        {
            matchL1_gpu<unsigned char>, 0/*matchL1_gpu<signed char>*/,
            matchL1_gpu<unsigned short>, matchL1_gpu<short>,
            matchL1_gpu<int>, matchL1_gpu<float>
        },
        {
            0/*matchL2_gpu<unsigned char>*/, 0/*matchL2_gpu<signed char>*/,
            0/*matchL2_gpu<unsigned short>*/, 0/*matchL2_gpu<short>*/,
            0/*matchL2_gpu<int>*/, matchL2_gpu<float>
        },
        {
            matchHamming_gpu<unsigned char>, 0/*matchHamming_gpu<signed char>*/,
            matchHamming_gpu<unsigned short>, 0/*matchHamming_gpu<short>*/,
            matchHamming_gpu<int>, 0/*matchHamming_gpu<float>*/
        }
    };

    CV_Assert(query.channels() == 1 && query.depth() < CV_64F);

    const int nQuery = query.rows;

    ensureSizeIsEnough(1, nQuery, CV_32S, trainIdx);
    ensureSizeIsEnough(1, nQuery, CV_32S, imgIdx);
    ensureSizeIsEnough(1, nQuery, CV_32F, distance);

    caller_t func = callers[distType][query.depth()];
    CV_Assert(func != 0);

    DeviceInfo info;
    int cc = info.majorVersion() * 10 + info.minorVersion();

    func(query, trainCollection, masks, trainIdx, imgIdx, distance, cc, StreamAccessor::getStream(stream));
}

void cv::gpu::BruteForceMatcher_GPU_base::matchDownload(const GpuMat& trainIdx, const GpuMat& imgIdx, const GpuMat& distance, vector<DMatch>& matches)
{
    if (trainIdx.empty() || imgIdx.empty() || distance.empty())
        return;

    Mat trainIdxCPU(trainIdx);
    Mat imgIdxCPU(imgIdx);
    Mat distanceCPU(distance);

    matchConvert(trainIdxCPU, imgIdxCPU, distanceCPU, matches);
}

void cv::gpu::BruteForceMatcher_GPU_base::matchConvert(const Mat& trainIdx, const Mat& imgIdx, const Mat& distance, std::vector<DMatch>& matches)
{
    if (trainIdx.empty() || imgIdx.empty() || distance.empty())
        return;

    CV_Assert(trainIdx.type() == CV_32SC1);
    CV_Assert(imgIdx.type() == CV_32SC1 && imgIdx.cols == trainIdx.cols);
    CV_Assert(distance.type() == CV_32FC1 && distance.cols == trainIdx.cols);

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

void cv::gpu::BruteForceMatcher_GPU_base::match(const GpuMat& query, vector<DMatch>& matches, const vector<GpuMat>& masks)
{
    GpuMat trainCollection;
    GpuMat maskCollection;

    makeGpuCollection(trainCollection, maskCollection, masks);

    GpuMat trainIdx, imgIdx, distance;

    matchCollection(query, trainCollection, trainIdx, imgIdx, distance, maskCollection);
    matchDownload(trainIdx, imgIdx, distance, matches);
}

////////////////////////////////////////////////////////////////////
// KnnMatch

void cv::gpu::BruteForceMatcher_GPU_base::knnMatchSingle(const GpuMat& query, const GpuMat& train,
    GpuMat& trainIdx, GpuMat& distance, GpuMat& allDist, int k,
    const GpuMat& mask, Stream& stream)
{
    if (query.empty() || train.empty())
        return;

    using namespace ::cv::gpu::device::bf_knnmatch;

    typedef void (*caller_t)(const DevMem2Db& query, const DevMem2Db& train, int k, const DevMem2Db& mask, 
                             const DevMem2Db& trainIdx, const DevMem2Db& distance, const DevMem2Df& allDist, 
                             int cc, cudaStream_t stream);

    static const caller_t callers[3][6] =
    {
        {
            matchL1_gpu<unsigned char>, 0/*matchL1_gpu<signed char>*/, 
            matchL1_gpu<unsigned short>, matchL1_gpu<short>, 
            matchL1_gpu<int>, matchL1_gpu<float>
        },
        {
            0/*matchL2_gpu<unsigned char>*/, 0/*matchL2_gpu<signed char>*/, 
            0/*matchL2_gpu<unsigned short>*/, 0/*matchL2_gpu<short>*/, 
            0/*matchL2_gpu<int>*/, matchL2_gpu<float>
        },
        {
            matchHamming_gpu<unsigned char>, 0/*matchHamming_gpu<signed char>*/, 
            matchHamming_gpu<unsigned short>, 0/*matchHamming_gpu<short>*/, 
            matchHamming_gpu<int>, 0/*matchHamming_gpu<float>*/
        }
    };

    CV_Assert(query.channels() == 1 && query.depth() < CV_64F);
    CV_Assert(train.type() == query.type() && train.cols == query.cols);

    const int nQuery = query.rows;
    const int nTrain = train.rows;

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
        stream.enqueueMemSet(trainIdx, Scalar::all(-1));
    else
        trainIdx.setTo(Scalar::all(-1));

    caller_t func = callers[distType][query.depth()];
    CV_Assert(func != 0);
    
    DeviceInfo info;
    int cc = info.majorVersion() * 10 + info.minorVersion();

    func(query, train, k, mask, trainIdx, distance, allDist, cc, StreamAccessor::getStream(stream));
}

void cv::gpu::BruteForceMatcher_GPU_base::knnMatchDownload(const GpuMat& trainIdx, const GpuMat& distance,
    vector< vector<DMatch> >& matches, bool compactResult)
{
    if (trainIdx.empty() || distance.empty())
        return;

    Mat trainIdxCPU(trainIdx);
    Mat distanceCPU(distance);

    knnMatchConvert(trainIdxCPU, distanceCPU, matches, compactResult);
}

void cv::gpu::BruteForceMatcher_GPU_base::knnMatchConvert(const Mat& trainIdx, const Mat& distance, 
    vector< vector<DMatch> >& matches, bool compactResult)
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

void cv::gpu::BruteForceMatcher_GPU_base::knnMatch(const GpuMat& query, const GpuMat& train,
    vector< vector<DMatch> >& matches, int k, const GpuMat& mask, bool compactResult)
{
    GpuMat trainIdx, distance, allDist;
    knnMatchSingle(query, train, trainIdx, distance, allDist, k, mask);
    knnMatchDownload(trainIdx, distance, matches, compactResult);
}

void cv::gpu::BruteForceMatcher_GPU_base::knnMatch2Collection(const GpuMat& query, const GpuMat& trainCollection,
    GpuMat& trainIdx, GpuMat& imgIdx, GpuMat& distance,
    const GpuMat& maskCollection, Stream& stream)
{
    if (query.empty() || trainCollection.empty())
        return;

    using namespace ::cv::gpu::device::bf_knnmatch;

    typedef void (*caller_t)(const DevMem2Db& query, const DevMem2Db& trains, const DevMem2D_<PtrStepb>& masks, 
                             const DevMem2Db& trainIdx, const DevMem2Db& imgIdx, const DevMem2Db& distance, 
                             int cc, cudaStream_t stream);

    static const caller_t callers[3][6] =
    {
        {
            match2L1_gpu<unsigned char>, 0/*match2L1_gpu<signed char>*/, 
            match2L1_gpu<unsigned short>, match2L1_gpu<short>, 
            match2L1_gpu<int>, match2L1_gpu<float>
        },
        {
            0/*match2L2_gpu<unsigned char>*/, 0/*match2L2_gpu<signed char>*/, 
            0/*match2L2_gpu<unsigned short>*/, 0/*match2L2_gpu<short>*/, 
            0/*match2L2_gpu<int>*/, match2L2_gpu<float>
        },
        {
            match2Hamming_gpu<unsigned char>, 0/*match2Hamming_gpu<signed char>*/, 
            match2Hamming_gpu<unsigned short>, 0/*match2Hamming_gpu<short>*/, 
            match2Hamming_gpu<int>, 0/*match2Hamming_gpu<float>*/
        }
    };

    CV_Assert(query.channels() == 1 && query.depth() < CV_64F);

    const int nQuery = query.rows;

    ensureSizeIsEnough(1, nQuery, CV_32SC2, trainIdx);
    ensureSizeIsEnough(1, nQuery, CV_32SC2, imgIdx);
    ensureSizeIsEnough(1, nQuery, CV_32FC2, distance);

    if (stream)
        stream.enqueueMemSet(trainIdx, Scalar::all(-1));
    else
        trainIdx.setTo(Scalar::all(-1));

    caller_t func = callers[distType][query.depth()];
    CV_Assert(func != 0);
    
    DeviceInfo info;
    int cc = info.majorVersion() * 10 + info.minorVersion();

    func(query, trainCollection, maskCollection, trainIdx, imgIdx, distance, cc, StreamAccessor::getStream(stream));
}

void cv::gpu::BruteForceMatcher_GPU_base::knnMatch2Download(const GpuMat& trainIdx, const GpuMat& imgIdx, const GpuMat& distance,
    vector< vector<DMatch> >& matches, bool compactResult)
{
    if (trainIdx.empty() || imgIdx.empty() || distance.empty())
        return;

    Mat trainIdxCPU(trainIdx);
    Mat imgIdxCPU(imgIdx);
    Mat distanceCPU(distance);

    knnMatch2Convert(trainIdxCPU, imgIdxCPU, distanceCPU, matches, compactResult);
}

void cv::gpu::BruteForceMatcher_GPU_base::knnMatch2Convert(const Mat& trainIdx, const Mat& imgIdx, const Mat& distance,
    vector< vector<DMatch> >& matches, bool compactResult)
{
    if (trainIdx.empty() || imgIdx.empty() || distance.empty())
        return;

    CV_Assert(trainIdx.type() == CV_32SC2);
    CV_Assert(imgIdx.type() == CV_32SC2 && imgIdx.cols == trainIdx.cols);
    CV_Assert(distance.type() == CV_32FC2 && distance.cols == trainIdx.cols);

    const int nQuery = trainIdx.cols;

    matches.clear();
    matches.reserve(nQuery);
    
    const int* trainIdx_ptr = trainIdx.ptr<int>();
    const int* imgIdx_ptr = imgIdx.ptr<int>();
    const float* distance_ptr = distance.ptr<float>();

    for (int queryIdx = 0; queryIdx < nQuery; ++queryIdx)
    {
        matches.push_back(vector<DMatch>());
        vector<DMatch>& curMatches = matches.back();
        curMatches.reserve(2);

        for (int i = 0; i < 2; ++i, ++trainIdx_ptr, ++imgIdx_ptr, ++distance_ptr)
        {
            int trainIdx = *trainIdx_ptr;

            if (trainIdx != -1)
            {
                int imgIdx = *imgIdx_ptr;

                float distance = *distance_ptr;

                DMatch m(queryIdx, trainIdx, imgIdx, distance);

                curMatches.push_back(m);
            }
        }

        if (compactResult && curMatches.empty())
            matches.pop_back();
    }
}

namespace
{
    struct ImgIdxSetter
    {
        explicit inline ImgIdxSetter(int imgIdx_) : imgIdx(imgIdx_) {}
        inline void operator()(DMatch& m) const {m.imgIdx = imgIdx;}
        int imgIdx;
    };
}

void cv::gpu::BruteForceMatcher_GPU_base::knnMatch(const GpuMat& query, vector< vector<DMatch> >& matches, int k,
    const vector<GpuMat>& masks, bool compactResult)
{
    if (k == 2)
    {
        GpuMat trainCollection;
        GpuMat maskCollection;

        makeGpuCollection(trainCollection, maskCollection, masks);

        GpuMat trainIdx, imgIdx, distance;

        knnMatch2Collection(query, trainCollection, trainIdx, imgIdx, distance, maskCollection);
        knnMatch2Download(trainIdx, imgIdx, distance, matches);
    }
    else
    {
        if (query.empty() || empty())
            return;

        vector< vector<DMatch> > curMatches;
        vector<DMatch> temp;
        temp.reserve(2 * k);

        matches.resize(query.rows);
        for_each(matches.begin(), matches.end(), bind2nd(mem_fun_ref(&vector<DMatch>::reserve), k));

        for (size_t imgIdx = 0, size = trainDescCollection.size(); imgIdx < size; ++imgIdx)
        {
            knnMatch(query, trainDescCollection[imgIdx], curMatches, k, masks.empty() ? GpuMat() : masks[imgIdx]);

            for (int queryIdx = 0; queryIdx < query.rows; ++queryIdx)
            {
                vector<DMatch>& localMatch = curMatches[queryIdx];
                vector<DMatch>& globalMatch = matches[queryIdx];

                for_each(localMatch.begin(), localMatch.end(), ImgIdxSetter(static_cast<int>(imgIdx)));

                temp.clear();
                merge(globalMatch.begin(), globalMatch.end(), localMatch.begin(), localMatch.end(), back_inserter(temp));

                globalMatch.clear();
                const size_t count = std::min((size_t)k, temp.size());
                copy(temp.begin(), temp.begin() + count, back_inserter(globalMatch));
            }
        }

        if (compactResult)
        {
            vector< vector<DMatch> >::iterator new_end = remove_if(matches.begin(), matches.end(), mem_fun_ref(&vector<DMatch>::empty));
            matches.erase(new_end, matches.end());
        }
    }
}

////////////////////////////////////////////////////////////////////
// RadiusMatch

void cv::gpu::BruteForceMatcher_GPU_base::radiusMatchSingle(const GpuMat& query, const GpuMat& train,
    GpuMat& trainIdx, GpuMat& distance, GpuMat& nMatches, float maxDistance, 
    const GpuMat& mask, Stream& stream)
{
    if (query.empty() || train.empty())
        return;

    using namespace ::cv::gpu::device::bf_radius_match;

    typedef void (*caller_t)(const DevMem2Db& query, const DevMem2Db& train, float maxDistance, const DevMem2Db& mask, 
                             const DevMem2Di& trainIdx, const DevMem2Df& distance, const DevMem2D_<unsigned int>& nMatches, 
                             int cc, cudaStream_t stream);

    static const caller_t callers[3][6] =
    {
        {
            matchL1_gpu<unsigned char>, 0/*matchL1_gpu<signed char>*/, 
            matchL1_gpu<unsigned short>, matchL1_gpu<short>, 
            matchL1_gpu<int>, matchL1_gpu<float>
        },
        {
            0/*matchL2_gpu<unsigned char>*/, 0/*matchL2_gpu<signed char>*/, 
            0/*matchL2_gpu<unsigned short>*/, 0/*matchL2_gpu<short>*/, 
            0/*matchL2_gpu<int>*/, matchL2_gpu<float>
        },
        {
            matchHamming_gpu<unsigned char>, 0/*matchHamming_gpu<signed char>*/, 
            matchHamming_gpu<unsigned short>, 0/*matchHamming_gpu<short>*/, 
            matchHamming_gpu<int>, 0/*matchHamming_gpu<float>*/
        }
    };

    DeviceInfo info;
    int cc = info.majorVersion() * 10 + info.minorVersion();

    CV_Assert(TargetArchs::builtWith(GLOBAL_ATOMICS) && info.supports(GLOBAL_ATOMICS));

    const int nQuery = query.rows;
    const int nTrain = train.rows;

    CV_Assert(query.channels() == 1 && query.depth() < CV_64F);
    CV_Assert(train.type() == query.type() && train.cols == query.cols);
    CV_Assert(trainIdx.empty() || (trainIdx.rows == nQuery && trainIdx.size() == distance.size()));

    ensureSizeIsEnough(1, nQuery, CV_32SC1, nMatches);
    if (trainIdx.empty())
    {
        ensureSizeIsEnough(nQuery, std::max((nTrain / 100), 10), CV_32SC1, trainIdx);
        ensureSizeIsEnough(nQuery, std::max((nTrain / 100), 10), CV_32FC1, distance);
    }
    
    if (stream)
        stream.enqueueMemSet(nMatches, Scalar::all(0));
    else
        nMatches.setTo(Scalar::all(0));

    caller_t func = callers[distType][query.depth()];
    CV_Assert(func != 0);    

    func(query, train, maxDistance, mask, trainIdx, distance, nMatches, cc, StreamAccessor::getStream(stream));
}

void cv::gpu::BruteForceMatcher_GPU_base::radiusMatchDownload(const GpuMat& trainIdx, const GpuMat& distance, const GpuMat& nMatches, 
    vector< vector<DMatch> >& matches, bool compactResult)
{
    if (trainIdx.empty() || distance.empty() || nMatches.empty())
        return;

    Mat trainIdxCPU(trainIdx);
    Mat distanceCPU(distance);
    Mat nMatchesCPU(nMatches);

    radiusMatchConvert(trainIdxCPU, distanceCPU, nMatchesCPU, matches, compactResult);
}

void cv::gpu::BruteForceMatcher_GPU_base::radiusMatchConvert(const Mat& trainIdx, const Mat& distance, const Mat& nMatches,
    vector< vector<DMatch> >& matches, bool compactResult)
{
    if (trainIdx.empty() || distance.empty() || nMatches.empty())
        return;

    CV_Assert(trainIdx.type() == CV_32SC1);
    CV_Assert(distance.type() == CV_32FC1 && distance.size() == trainIdx.size());
    CV_Assert(nMatches.type() == CV_32SC1 && nMatches.cols == trainIdx.rows);

    const int nQuery = trainIdx.rows;

    matches.clear();
    matches.reserve(nQuery);

    const int* nMatches_ptr = nMatches.ptr<int>();

    for (int queryIdx = 0; queryIdx < nQuery; ++queryIdx)
    {
        const int* trainIdx_ptr = trainIdx.ptr<int>(queryIdx);
        const float* distance_ptr = distance.ptr<float>(queryIdx);

        const int nMatches = std::min(nMatches_ptr[queryIdx], trainIdx.cols);

        if (nMatches == 0)
        {
            if (!compactResult)
                matches.push_back(vector<DMatch>());
            continue;
        }

        matches.push_back(vector<DMatch>(nMatches));
        vector<DMatch>& curMatches = matches.back();

        for (int i = 0; i < nMatches; ++i, ++trainIdx_ptr, ++distance_ptr)
        {
            int trainIdx = *trainIdx_ptr;

            float distance = *distance_ptr;

            DMatch m(queryIdx, trainIdx, 0, distance);

            curMatches[i] = m;
        }

        sort(curMatches.begin(), curMatches.end());
    }
}

void cv::gpu::BruteForceMatcher_GPU_base::radiusMatch(const GpuMat& query, const GpuMat& train,
    vector< vector<DMatch> >& matches, float maxDistance, const GpuMat& mask, bool compactResult)
{
    GpuMat trainIdx, distance, nMatches;
    radiusMatchSingle(query, train, trainIdx, distance, nMatches, maxDistance, mask);
    radiusMatchDownload(trainIdx, distance, nMatches, matches, compactResult);
}

void cv::gpu::BruteForceMatcher_GPU_base::radiusMatchCollection(const GpuMat& query, GpuMat& trainIdx, GpuMat& imgIdx, GpuMat& distance, GpuMat& nMatches, 
    float maxDistance, const vector<GpuMat>& masks, Stream& stream)
{
    if (query.empty() || empty())
        return;

    using namespace ::cv::gpu::device::bf_radius_match;

    typedef void (*caller_t)(const DevMem2Db& query, const DevMem2Db* trains, int n, float maxDistance, const DevMem2Db* masks, 
                             const DevMem2Di& trainIdx, const DevMem2Di& imgIdx, const DevMem2Df& distance, const DevMem2D_<unsigned int>& nMatches, 
                             int cc, cudaStream_t stream);

    static const caller_t callers[3][6] =
    {
        {
            matchL1_gpu<unsigned char>, 0/*matchL1_gpu<signed char>*/, 
            matchL1_gpu<unsigned short>, matchL1_gpu<short>, 
            matchL1_gpu<int>, matchL1_gpu<float>
        },
        {
            0/*matchL2_gpu<unsigned char>*/, 0/*matchL2_gpu<signed char>*/, 
            0/*matchL2_gpu<unsigned short>*/, 0/*matchL2_gpu<short>*/, 
            0/*matchL2_gpu<int>*/, matchL2_gpu<float>
        },
        {
            matchHamming_gpu<unsigned char>, 0/*matchHamming_gpu<signed char>*/, 
            matchHamming_gpu<unsigned short>, 0/*matchHamming_gpu<short>*/, 
            matchHamming_gpu<int>, 0/*matchHamming_gpu<float>*/
        }
    };

    DeviceInfo info;
    int cc = info.majorVersion() * 10 + info.minorVersion();

    CV_Assert(TargetArchs::builtWith(GLOBAL_ATOMICS) && info.supports(GLOBAL_ATOMICS));

    const int nQuery = query.rows;

    CV_Assert(query.channels() == 1 && query.depth() < CV_64F);
    CV_Assert(trainIdx.empty() || (trainIdx.rows == nQuery && trainIdx.size() == distance.size() && trainIdx.size() == imgIdx.size()));

    ensureSizeIsEnough(1, nQuery, CV_32SC1, nMatches);
    if (trainIdx.empty())
    {
        ensureSizeIsEnough(nQuery, std::max((nQuery / 100), 10), CV_32SC1, trainIdx);
        ensureSizeIsEnough(nQuery, std::max((nQuery / 100), 10), CV_32SC1, imgIdx);
        ensureSizeIsEnough(nQuery, std::max((nQuery / 100), 10), CV_32FC1, distance);
    }
    
    if (stream)
        stream.enqueueMemSet(nMatches, Scalar::all(0));
    else
        nMatches.setTo(Scalar::all(0));

    caller_t func = callers[distType][query.depth()];
    CV_Assert(func != 0);

    vector<DevMem2Db> trains_(trainDescCollection.begin(), trainDescCollection.end());
    vector<DevMem2Db> masks_(masks.begin(), masks.end());

    func(query, &trains_[0], static_cast<int>(trains_.size()), maxDistance, masks_.size() == 0 ? 0 : &masks_[0], 
        trainIdx, imgIdx, distance, nMatches, cc, StreamAccessor::getStream(stream));
}

void cv::gpu::BruteForceMatcher_GPU_base::radiusMatchDownload(const GpuMat& trainIdx, const GpuMat& imgIdx, const GpuMat& distance, const GpuMat& nMatches,
    vector< vector<DMatch> >& matches, bool compactResult)
{
    if (trainIdx.empty() || imgIdx.empty() || distance.empty() || nMatches.empty())
        return;

    Mat trainIdxCPU(trainIdx);
    Mat imgIdxCPU(imgIdx);
    Mat distanceCPU(distance);
    Mat nMatchesCPU(nMatches);

    radiusMatchConvert(trainIdxCPU, imgIdxCPU, distanceCPU, nMatchesCPU, matches, compactResult);
}

void cv::gpu::BruteForceMatcher_GPU_base::radiusMatchConvert(const Mat& trainIdx, const Mat& imgIdx, const Mat& distance, const Mat& nMatches,
    vector< vector<DMatch> >& matches, bool compactResult)
{
    if (trainIdx.empty() || imgIdx.empty() || distance.empty() || nMatches.empty())
        return;

    CV_Assert(trainIdx.type() == CV_32SC1);
    CV_Assert(imgIdx.type() == CV_32SC1 && imgIdx.size() == trainIdx.size());
    CV_Assert(distance.type() == CV_32FC1 && distance.size() == trainIdx.size());
    CV_Assert(nMatches.type() == CV_32SC1 && nMatches.cols == trainIdx.rows);

    const int nQuery = trainIdx.rows;

    matches.clear();
    matches.reserve(nQuery);

    const int* nMatches_ptr = nMatches.ptr<int>();

    for (int queryIdx = 0; queryIdx < nQuery; ++queryIdx)
    {
        const int* trainIdx_ptr = trainIdx.ptr<int>(queryIdx);
        const int* imgIdx_ptr = imgIdx.ptr<int>(queryIdx);
        const float* distance_ptr = distance.ptr<float>(queryIdx);

        const int nMatches = std::min(nMatches_ptr[queryIdx], trainIdx.cols);

        if (nMatches == 0)
        {
            if (!compactResult)
                matches.push_back(vector<DMatch>());
            continue;
        }

        matches.push_back(vector<DMatch>());
        vector<DMatch>& curMatches = matches.back();
        curMatches.reserve(nMatches);

        for (int i = 0; i < nMatches; ++i, ++trainIdx_ptr, ++imgIdx_ptr, ++distance_ptr)
        {
            int trainIdx = *trainIdx_ptr;
            int imgIdx = *imgIdx_ptr;
            float distance = *distance_ptr;

            DMatch m(queryIdx, trainIdx, imgIdx, distance);

            curMatches.push_back(m);
        }

        sort(curMatches.begin(), curMatches.end());
    }
}

void cv::gpu::BruteForceMatcher_GPU_base::radiusMatch(const GpuMat& query, vector< vector<DMatch> >& matches,
    float maxDistance, const vector<GpuMat>& masks, bool compactResult)
{
    GpuMat trainIdx, imgIdx, distance, nMatches;
    radiusMatchCollection(query, trainIdx, imgIdx, distance, nMatches, maxDistance, masks);
    radiusMatchDownload(trainIdx, imgIdx, distance, nMatches, matches, compactResult);
}

#endif /* !defined (HAVE_CUDA) */
