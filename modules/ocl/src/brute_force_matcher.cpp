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
// Copyright (C) 2010-2012, Multicoreware, Inc., all rights reserved.
// Copyright (C) 2010-2012, Advanced Micro Devices, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// @Authors
//    Nathan, liujun@multicorewareinc.com
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other oclMaterials provided with the distribution.
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

#include "precomp.hpp"
#include <functional>
#include <iterator>
#include <vector>
using namespace cv;
using namespace cv::ocl;
using namespace std;

#if !defined (HAVE_OPENCL)
cv::ocl::BruteForceMatcher_OCL_base::BruteForceMatcher_OCL_base(DistType)
{
    throw_nogpu();
}
void cv::ocl::BruteForceMatcher_OCL_base::add(const vector<oclMat> &)
{
    throw_nogpu();
}
const vector<oclMat> &cv::ocl::BruteForceMatcher_OCL_base::getTrainDescriptors() const
{
    throw_nogpu();
    return trainDescCollection;
}
void cv::ocl::BruteForceMatcher_OCL_base::clear()
{
    throw_nogpu();
}
bool cv::ocl::BruteForceMatcher_OCL_base::empty() const
{
    throw_nogpu();
    return true;
}
bool cv::ocl::BruteForceMatcher_OCL_base::isMaskSupported() const
{
    throw_nogpu();
    return true;
}
void cv::ocl::BruteForceMatcher_OCL_base::matchSingle(const oclMat &, const oclMat &, oclMat &, oclMat &, const oclMat &)
{
    throw_nogpu();
}
void cv::ocl::BruteForceMatcher_OCL_base::matchDownload(const oclMat &, const oclMat &, vector<DMatch> &)
{
    throw_nogpu();
}
void cv::ocl::BruteForceMatcher_OCL_base::matchConvert(const Mat &, const Mat &, vector<DMatch> &)
{
    throw_nogpu();
}
void cv::ocl::BruteForceMatcher_OCL_base::match(const oclMat &, const oclMat &, vector<DMatch> &, const oclMat &)
{
    throw_nogpu();
}
void cv::ocl::BruteForceMatcher_OCL_base::makeGpuCollection(oclMat &, oclMat &, const vector<oclMat> &)
{
    throw_nogpu();
}
void cv::ocl::BruteForceMatcher_OCL_base::matchCollection(const oclMat &, const oclMat &, oclMat &, oclMat &, oclMat &, const oclMat &)
{
    throw_nogpu();
}
void cv::ocl::BruteForceMatcher_OCL_base::matchDownload(const oclMat &, const oclMat &, const oclMat &, vector<DMatch> &)
{
    throw_nogpu();
}
void cv::ocl::BruteForceMatcher_OCL_base::matchConvert(const Mat &, const Mat &, const Mat &, vector<DMatch> &)
{
    throw_nogpu();
}
void cv::ocl::BruteForceMatcher_OCL_base::match(const oclMat &, vector<DMatch> &, const vector<oclMat> &)
{
    throw_nogpu();
}
void cv::ocl::BruteForceMatcher_OCL_base::knnMatchSingle(const oclMat &, const oclMat &, oclMat &, oclMat &, oclMat &, int, const oclMat &)
{
    throw_nogpu();
}
void cv::ocl::BruteForceMatcher_OCL_base::knnMatchDownload(const oclMat &, const oclMat &, vector< vector<DMatch> > &, bool)
{
    throw_nogpu();
}
void cv::ocl::BruteForceMatcher_OCL_base::knnMatchConvert(const Mat &, const Mat &, vector< vector<DMatch> > &, bool)
{
    throw_nogpu();
}
void cv::ocl::BruteForceMatcher_OCL_base::knnMatch(const oclMat &, const oclMat &, vector< vector<DMatch> > &, int, const oclMat &, bool)
{
    throw_nogpu();
}
void cv::ocl::BruteForceMatcher_OCL_base::knnMatch2Collection(const oclMat &, const oclMat &, oclMat &, oclMat &, oclMat &, const oclMat &)
{
    throw_nogpu();
}
void cv::ocl::BruteForceMatcher_OCL_base::knnMatch2Download(const oclMat &, const oclMat &, const oclMat &, vector< vector<DMatch> > &, bool)
{
    throw_nogpu();
}
void cv::ocl::BruteForceMatcher_OCL_base::knnMatch2Convert(const Mat &, const Mat &, const Mat &, vector< vector<DMatch> > &, bool)
{
    throw_nogpu();
}
void cv::ocl::BruteForceMatcher_OCL_base::knnMatch(const oclMat &, vector< vector<DMatch> > &, int, const vector<oclMat> &, bool)
{
    throw_nogpu();
}
void cv::ocl::BruteForceMatcher_OCL_base::radiusMatchSingle(const oclMat &, const oclMat &, oclMat &, oclMat &, oclMat &, float, const oclMat &)
{
    throw_nogpu();
}
void cv::ocl::BruteForceMatcher_OCL_base::radiusMatchDownload(const oclMat &, const oclMat &, const oclMat &, vector< vector<DMatch> > &, bool)
{
    throw_nogpu();
}
void cv::ocl::BruteForceMatcher_OCL_base::radiusMatchConvert(const Mat &, const Mat &, const Mat &, vector< vector<DMatch> > &, bool)
{
    throw_nogpu();
}
void cv::ocl::BruteForceMatcher_OCL_base::radiusMatch(const oclMat &, const oclMat &, vector< vector<DMatch> > &, float, const oclMat &, bool)
{
    throw_nogpu();
}
void cv::ocl::BruteForceMatcher_OCL_base::radiusMatchCollection(const oclMat &, oclMat &, oclMat &, oclMat &, oclMat &, float, const vector<oclMat> &)
{
    throw_nogpu();
}
void cv::ocl::BruteForceMatcher_OCL_base::radiusMatchDownload(const oclMat &, const oclMat &, const oclMat &, const oclMat &, vector< vector<DMatch> > &, bool)
{
    throw_nogpu();
}
void cv::ocl::BruteForceMatcher_OCL_base::radiusMatchConvert(const Mat &, const Mat &, const Mat &, const Mat &, vector< vector<DMatch> > &, bool)
{
    throw_nogpu();
}
void cv::ocl::BruteForceMatcher_OCL_base::radiusMatch(const oclMat &, vector< vector<DMatch> > &, float, const vector<oclMat> &, bool)
{
    throw_nogpu();
}
#else /* !defined (HAVE_OPENCL) */

using namespace std;
namespace cv
{
    namespace ocl
    {
        ////////////////////////////////////OpenCL kernel strings//////////////////////////
        extern const char *brute_force_match;
    }
}

template < int BLOCK_SIZE, int MAX_DESC_LEN,  typename T/*, typename Mask*/ >
void matchUnrolledCached(const oclMat &query, const oclMat &train, const oclMat &mask,
                         const oclMat &trainIdx, const oclMat &distance, int distType)
{
    cv::ocl::Context *ctx = query.clCxt;
    size_t globalSize[] = {(query.rows + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE, BLOCK_SIZE, 1};
    size_t localSize[] = {BLOCK_SIZE, BLOCK_SIZE, 1};
    const size_t smemSize = (BLOCK_SIZE * (MAX_DESC_LEN >= 2 * BLOCK_SIZE ? MAX_DESC_LEN : 2 * BLOCK_SIZE) + BLOCK_SIZE * BLOCK_SIZE) * sizeof(int);
    int block_size = BLOCK_SIZE;
    int m_size = MAX_DESC_LEN;
    vector< pair<size_t, const void *> > args;

    if(globalSize[0] != 0)
    {
        args.push_back( make_pair( sizeof(cl_mem), (void *)&query.data ));
        args.push_back( make_pair( sizeof(cl_mem), (void *)&train.data ));
        args.push_back( make_pair( sizeof(cl_mem), (void *)&mask.data ));
        args.push_back( make_pair( sizeof(cl_mem), (void *)&trainIdx.data ));
        args.push_back( make_pair( sizeof(cl_mem), (void *)&distance.data ));
        args.push_back( make_pair( smemSize, (void *)NULL));
        args.push_back( make_pair( sizeof(cl_int), (void *)&block_size ));
        args.push_back( make_pair( sizeof(cl_int), (void *)&m_size ));
        args.push_back( make_pair( sizeof(cl_int), (void *)&query.rows ));
        args.push_back( make_pair( sizeof(cl_int), (void *)&query.cols ));
        args.push_back( make_pair( sizeof(cl_int), (void *)&train.rows ));
        args.push_back( make_pair( sizeof(cl_int), (void *)&train.cols ));
        args.push_back( make_pair( sizeof(cl_int), (void *)&query.step ));
        args.push_back( make_pair( sizeof(cl_int), (void *)&distType ));

        std::string kernelName = "BruteForceMatch_UnrollMatch";

        openCLExecuteKernel(ctx, &brute_force_match, kernelName, globalSize, localSize, args, -1, -1);
    }
}

template < int BLOCK_SIZE, int MAX_DESC_LEN,  typename T/*, typename Mask*/ >
void matchUnrolledCached(const oclMat /*query*/, const oclMat * /*trains*/, int /*n*/, const oclMat /*mask*/,
                         const oclMat &/*bestTrainIdx*/, const oclMat & /*bestImgIdx*/, const oclMat & /*bestDistance*/, int /*distType*/)
{
}

template < int BLOCK_SIZE,  typename T/*, typename Mask*/ >
void match(const oclMat &query, const oclMat &train, const oclMat &mask,
           const oclMat &trainIdx, const oclMat &distance, int distType)
{
    cv::ocl::Context *ctx = query.clCxt;
    size_t globalSize[] = {(query.rows + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE, BLOCK_SIZE, 1};
    size_t localSize[] = {BLOCK_SIZE, BLOCK_SIZE, 1};
    const size_t smemSize = (2 * BLOCK_SIZE * BLOCK_SIZE) * sizeof(int);
    int block_size = BLOCK_SIZE;
    vector< pair<size_t, const void *> > args;

    if(globalSize[0] != 0)
    {
        args.push_back( make_pair( sizeof(cl_mem), (void *)&query.data ));
        args.push_back( make_pair( sizeof(cl_mem), (void *)&train.data ));
        args.push_back( make_pair( sizeof(cl_mem), (void *)&mask.data ));
        args.push_back( make_pair( sizeof(cl_mem), (void *)&trainIdx.data ));
        args.push_back( make_pair( sizeof(cl_mem), (void *)&distance.data ));
        args.push_back( make_pair( smemSize, (void *)NULL));
        args.push_back( make_pair( sizeof(cl_int), (void *)&block_size ));
        args.push_back( make_pair( sizeof(cl_int), (void *)&query.rows ));
        args.push_back( make_pair( sizeof(cl_int), (void *)&query.cols ));
        args.push_back( make_pair( sizeof(cl_int), (void *)&train.rows ));
        args.push_back( make_pair( sizeof(cl_int), (void *)&train.cols ));
        args.push_back( make_pair( sizeof(cl_int), (void *)&query.step ));
        args.push_back( make_pair( sizeof(cl_int), (void *)&distType ));

        std::string kernelName = "BruteForceMatch_Match";

        openCLExecuteKernel(ctx, &brute_force_match, kernelName, globalSize, localSize, args, -1, -1);
    }
}

template < int BLOCK_SIZE,  typename T/*, typename Mask*/ >
void match(const oclMat /*query*/, const oclMat * /*trains*/, int /*n*/, const oclMat /*mask*/,
           const oclMat &/*bestTrainIdx*/, const oclMat & /*bestImgIdx*/, const oclMat & /*bestDistance*/, int /*distType*/)
{
}

//radius_matchUnrolledCached
template < int BLOCK_SIZE, int MAX_DESC_LEN,  typename T/*, typename Mask*/ >
void matchUnrolledCached(const oclMat &query, const oclMat &train, float maxDistance, const oclMat &mask,
                         const oclMat &trainIdx, const oclMat &distance, const oclMat &nMatches, int distType)
{
    cv::ocl::Context *ctx = query.clCxt;
    size_t globalSize[] = {(train.rows + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE, (query.rows + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE, 1};
    size_t localSize[] = {BLOCK_SIZE, BLOCK_SIZE, 1};
    const size_t smemSize = (2 * BLOCK_SIZE * BLOCK_SIZE) * sizeof(int);
    int block_size = BLOCK_SIZE;
    int m_size = MAX_DESC_LEN;
    vector< pair<size_t, const void *> > args;

    if(globalSize[0] != 0)
    {
        args.push_back( make_pair( sizeof(cl_mem), (void *)&query.data ));
        args.push_back( make_pair( sizeof(cl_mem), (void *)&train.data ));
        args.push_back( make_pair( sizeof(cl_float), (void *)&maxDistance ));
        args.push_back( make_pair( sizeof(cl_mem), (void *)&mask.data ));
        args.push_back( make_pair( sizeof(cl_mem), (void *)&trainIdx.data ));
        args.push_back( make_pair( sizeof(cl_mem), (void *)&distance.data ));
        args.push_back( make_pair( sizeof(cl_mem), (void *)&nMatches.data ));
        args.push_back( make_pair( smemSize, (void *)NULL));
        args.push_back( make_pair( sizeof(cl_int), (void *)&block_size ));
        args.push_back( make_pair( sizeof(cl_int), (void *)&m_size ));
        args.push_back( make_pair( sizeof(cl_int), (void *)&query.rows ));
        args.push_back( make_pair( sizeof(cl_int), (void *)&query.cols ));
        args.push_back( make_pair( sizeof(cl_int), (void *)&train.rows ));
        args.push_back( make_pair( sizeof(cl_int), (void *)&train.cols ));
        args.push_back( make_pair( sizeof(cl_int), (void *)&trainIdx.cols ));
        args.push_back( make_pair( sizeof(cl_int), (void *)&query.step ));
        args.push_back( make_pair( sizeof(cl_int), (void *)&trainIdx.step ));
        args.push_back( make_pair( sizeof(cl_int), (void *)&distType ));

        std::string kernelName = "BruteForceMatch_RadiusUnrollMatch";

        openCLExecuteKernel(ctx, &brute_force_match, kernelName, globalSize, localSize, args, -1, -1);
    }
}

//radius_match
template < int BLOCK_SIZE, typename T/*, typename Mask*/ >
void radius_match(const oclMat &query, const oclMat &train, float maxDistance, const oclMat &mask,
                  const oclMat &trainIdx, const oclMat &distance, const oclMat &nMatches, int distType)
{
    cv::ocl::Context *ctx = query.clCxt;
    size_t globalSize[] = {(train.rows + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE, (query.rows + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE, 1};
    size_t localSize[] = {BLOCK_SIZE, BLOCK_SIZE, 1};
    const size_t smemSize = (2 * BLOCK_SIZE * BLOCK_SIZE) * sizeof(int);
    int block_size = BLOCK_SIZE;
    vector< pair<size_t, const void *> > args;

    if(globalSize[0] != 0)
    {
        args.push_back( make_pair( sizeof(cl_mem), (void *)&query.data ));
        args.push_back( make_pair( sizeof(cl_mem), (void *)&train.data ));
        args.push_back( make_pair( sizeof(cl_float), (void *)&maxDistance ));
        args.push_back( make_pair( sizeof(cl_mem), (void *)&mask.data ));
        args.push_back( make_pair( sizeof(cl_mem), (void *)&trainIdx.data ));
        args.push_back( make_pair( sizeof(cl_mem), (void *)&distance.data ));
        args.push_back( make_pair( sizeof(cl_mem), (void *)&nMatches.data ));
        args.push_back( make_pair( smemSize, (void *)NULL));
        args.push_back( make_pair( sizeof(cl_int), (void *)&block_size ));
        args.push_back( make_pair( sizeof(cl_int), (void *)&query.rows ));
        args.push_back( make_pair( sizeof(cl_int), (void *)&query.cols ));
        args.push_back( make_pair( sizeof(cl_int), (void *)&train.rows ));
        args.push_back( make_pair( sizeof(cl_int), (void *)&train.cols ));
        args.push_back( make_pair( sizeof(cl_int), (void *)&trainIdx.cols ));
        args.push_back( make_pair( sizeof(cl_int), (void *)&query.step ));
        args.push_back( make_pair( sizeof(cl_int), (void *)&trainIdx.step ));
        args.push_back( make_pair( sizeof(cl_int), (void *)&distType ));

        std::string kernelName = "BruteForceMatch_RadiusMatch";

        openCLExecuteKernel(ctx, &brute_force_match, kernelName, globalSize, localSize, args, -1, -1);
        //float *dis = (float *)clEnqueueMapBuffer(ctx->impl->clCmdQueue, (cl_mem)distance.data, CL_TRUE, CL_MAP_READ, 0, 8, 0, NULL, NULL, NULL);
        //printf("%f, %f\n", dis[0], dis[1]);
    }
}

// with mask
template < typename T/*, typename Mask*/ >
void matchDispatcher(const oclMat &query, const oclMat &train, const oclMat &mask,
                     const oclMat &trainIdx, const oclMat &distance, int distType)
{
    if (query.cols <= 64)
    {
        matchUnrolledCached<16, 64, T>(query, train, mask, trainIdx, distance, distType);
    }
    else if (query.cols <= 128)
    {
        matchUnrolledCached<16, 128, T>(query, train, mask, trainIdx,  distance, distType);
    }
    /*else if (query.cols <= 256)
    {
        matchUnrolled<16, 256, Dist>(query, train, mask, trainIdx, distance, stream);
    }
    else if (query.cols <= 512)
    {
        matchUnrolled<16, 512, Dist>(query, train, mask, trainIdx, distance, stream);
    }
    else if (query.cols <= 1024)
    {
        matchUnrolled<16, 1024, Dist>(query, train, mask, trainIdx, distance, stream);
    }*/
    else
    {
        match<16, T>(query, train, mask, trainIdx, distance, distType);
    }
}

// without mask
template < typename T/*, typename Mask*/ >
void matchDispatcher(const oclMat &query, const oclMat &train, const oclMat &trainIdx, const oclMat &distance, int distType)
{
    oclMat mask;
    if (query.cols <= 64)
    {
        matchUnrolledCached<16, 64, T>(query, train, mask, trainIdx, distance, distType);
    }
    else if (query.cols <= 128)
    {
        matchUnrolledCached<16, 128, T>(query, train, mask, trainIdx,  distance, distType);
    }
    /*else if (query.cols <= 256)
    {
        matchUnrolled<16, 256, Dist>(query, trains, n, mask, trainIdx, imgIdx, distance);
    }
    else if (query.cols <= 512)
    {
        matchUnrolled<16, 512, Dist>(query, trains, n, mask, trainIdx, imgIdx, distance);
    }
    else if (query.cols <= 1024)
    {
        matchUnrolled<16, 1024, Dist>(query, trains, n, mask, trainIdx, imgIdx, distance);
    }*/
    else
    {
        match<16, T>(query, train, mask, trainIdx, distance, distType);
    }
}

template < typename T/*, typename Mask*/ >
void matchDispatcher(const oclMat &query, const oclMat *trains, int n, const oclMat &mask,
                     const oclMat &trainIdx, const oclMat &imgIdx, const oclMat &distance, int distType)
{
    if (query.cols <= 64)
    {
        matchUnrolledCached<16, 64, T>(query, trains, n, mask, trainIdx, imgIdx, distance, distType);
    }
    else if (query.cols <= 128)
    {
        matchUnrolledCached<16, 128, T>(query, trains, n, mask, trainIdx, imgIdx, distance, distType);
    }
    /*else if (query.cols <= 256)
    {
        matchUnrolled<16, 256, Dist>(query, trains, n, mask, trainIdx, imgIdx, distance, stream);
    }
    else if (query.cols <= 512)
    {
        matchUnrolled<16, 512, Dist>(query, trains, n, mask, trainIdx, imgIdx, distance, stream);
    }
    else if (query.cols <= 1024)
    {
        matchUnrolled<16, 1024, Dist>(query, trains, n, mask, trainIdx, imgIdx, distance, stream);
    }*/
    else
    {
        match<16, T>(query, trains, n, mask, trainIdx, imgIdx, distance, distType);
    }
}

template < typename T/*, typename Mask*/ >
void matchDispatcher(const oclMat &query, const oclMat *trains, int n, const oclMat &trainIdx,
                     const oclMat &imgIdx, const oclMat &distance, int distType)
{
    oclMat mask;
    if (query.cols <= 64)
    {
        matchUnrolledCached<16, 64, T>(query, trains, n, mask, trainIdx, imgIdx, distance, distType);
    }
    else if (query.cols <= 128)
    {
        matchUnrolledCached<16, 128, T>(query, trains, n, mask, trainIdx, imgIdx, distance, distType);
    }
    /*else if (query.cols <= 256)
    {
        matchUnrolled<16, 256, Dist>(query, trains, n, mask, trainIdx, imgIdx, distance, stream);
    }
    else if (query.cols <= 512)
    {
        matchUnrolled<16, 512, Dist>(query, trains, n, mask, trainIdx, imgIdx, distance, stream);
    }
    else if (query.cols <= 1024)
    {
        matchUnrolled<16, 1024, Dist>(query, trains, n, mask, trainIdx, imgIdx, distance, stream);
    }*/
    else
    {
        match<16, T>(query, trains, n, mask, trainIdx, imgIdx, distance, distType);
    }
}

//radius matchDispatcher
// with mask
template < typename T/*, typename Mask*/ >
void matchDispatcher(const oclMat &query, const oclMat &train, float maxDistance, const oclMat &mask,
                     const oclMat &trainIdx, const oclMat &distance, const oclMat &nMatches, int distType)
{
    if (query.cols <= 64)
    {
        matchUnrolledCached<16, 64, T>(query, train, maxDistance, mask, trainIdx, distance, nMatches, distType);
    }
    else if (query.cols <= 128)
    {
        matchUnrolledCached<16, 128, T>(query, train, maxDistance, mask, trainIdx, distance, nMatches, distType);
    }
    /*else if (query.cols <= 256)
    {
        matchUnrolled<16, 256, Dist>(query, train, maxDistance, mask, trainIdx, distance, nMatches, stream);
    }
    else if (query.cols <= 512)
    {
        matchUnrolled<16, 512, Dist>(query, train, maxDistance, mask, trainIdx, distance, nMatches, stream);
    }
    else if (query.cols <= 1024)
    {
        matchUnrolled<16, 1024, Dist>(query, train, maxDistance, mask, trainIdx, distance, nMatches, stream);
    }*/
    else
    {
        radius_match<16, T>(query, train, maxDistance, mask, trainIdx, distance, nMatches, distType);
    }
}

// without mask
template < typename T/*, typename Mask*/ >
void matchDispatcher(const oclMat &query, const oclMat &train, float maxDistance, const oclMat &trainIdx,
                     const oclMat &distance, const oclMat &nMatches, int distType)
{
    oclMat mask;
    if (query.cols <= 64)
    {
        matchUnrolledCached<16, 64, T>(query, train, maxDistance, mask, trainIdx, distance, nMatches, distType);
    }
    else if (query.cols <= 128)
    {
        matchUnrolledCached<16, 128, T>(query, train, maxDistance, mask, trainIdx, distance, nMatches, distType);
    }
    /*else if (query.cols <= 256)
    {
        matchUnrolled<16, 256, Dist>(query, train, maxDistance, mask, trainIdx, distance, nMatches, stream);
    }
    else if (query.cols <= 512)
    {
        matchUnrolled<16, 512, Dist>(query, train, maxDistance, mask, trainIdx, distance, nMatches, stream);
    }
    else if (query.cols <= 1024)
    {
        matchUnrolled<16, 1024, Dist>(query, train, maxDistance, mask, trainIdx, distance, nMatches, stream);
    }*/
    else
    {
        radius_match<16, T>(query, train, maxDistance, mask, trainIdx, distance, nMatches, distType);
    }
}

template < typename T/*, typename Mask*/ >
void matchDispatcher(const oclMat &query, const oclMat &train, int n, float maxDistance, const oclMat &mask,
                     const oclMat &trainIdx, const oclMat &distance, const oclMat &nMatches, int distType)
{
    if (query.cols <= 64)
    {
        matchUnrolledCached<16, 64, T>(query, train, n, maxDistance, mask, trainIdx, distance, nMatches, distType);
    }
    else if (query.cols <= 128)
    {
        matchUnrolledCached<16, 128, T>(query, train, n, maxDistance, mask, trainIdx, distance, nMatches, distType);
    }
    /*else if (query.cols <= 256)
    {
        matchUnrolled<16, 256, Dist>(query, trains, n, maxDistance, masks, trainIdx, imgIdx, distance, nMatches, stream);
    }
    else if (query.cols <= 512)
    {
        matchUnrolled<16, 512, Dist>(query, trains, n, maxDistance, masks, trainIdx, imgIdx, distance, nMatches, stream);
    }
    else if (query.cols <= 1024)
    {
        matchUnrolled<16, 1024, Dist>(query, trains, n, maxDistance, masks, trainIdx, imgIdx, distance, nMatches, stream);
    }*/
    else
    {
        match<16, T>(query, train, n, maxDistance, mask, trainIdx, distance, nMatches, distType);
    }
}

// without mask
template < typename T/*, typename Mask*/ >
void matchDispatcher(const oclMat &query, const oclMat &train, int n, float maxDistance, const oclMat &trainIdx,
                     const oclMat &distance, const oclMat &nMatches, int distType)
{
    oclMat mask;
    if (query.cols <= 64)
    {
        matchUnrolledCached<16, 64, T>(query, train, n, maxDistance, mask, trainIdx, distance, nMatches, distType);
    }
    else if (query.cols <= 128)
    {
        matchUnrolledCached<16, 128, T>(query, train, n, maxDistance, mask, trainIdx, distance, nMatches, distType);
    }
    /*else if (query.cols <= 256)
    {
        matchUnrolled<16, 256, Dist>(query, trains, n, maxDistance, masks, trainIdx, imgIdx, distance, nMatches, stream);
    }
    else if (query.cols <= 512)
    {
        matchUnrolled<16, 512, Dist>(query, trains, n, maxDistance, masks, trainIdx, imgIdx, distance, nMatches, stream);
    }
    else if (query.cols <= 1024)
    {
        matchUnrolled<16, 1024, Dist>(query, trains, n, maxDistance, masks, trainIdx, imgIdx, distance, nMatches, stream);
    }*/
    else
    {
        match<16, T>(query, train, n, maxDistance, mask, trainIdx, distance, nMatches, distType);
    }
}

//knn match Dispatcher
template < int BLOCK_SIZE, int MAX_DESC_LEN,  typename T/*, typename Mask*/ >
void knn_matchUnrolledCached(const oclMat &query, const oclMat &train, const oclMat &mask,
                             const oclMat &trainIdx, const oclMat &distance, int distType)
{
    cv::ocl::Context *ctx = query.clCxt;
    size_t globalSize[] = {(query.rows + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE, BLOCK_SIZE, 1};
    size_t localSize[] = {BLOCK_SIZE, BLOCK_SIZE, 1};
    const size_t smemSize = (BLOCK_SIZE * (MAX_DESC_LEN >= BLOCK_SIZE ? MAX_DESC_LEN : BLOCK_SIZE) + BLOCK_SIZE * BLOCK_SIZE) * sizeof(int);
    int block_size = BLOCK_SIZE;
    int m_size = MAX_DESC_LEN;
    vector< pair<size_t, const void *> > args;

    if(globalSize[0] != 0)
    {
        args.push_back( make_pair( sizeof(cl_mem), (void *)&query.data ));
        args.push_back( make_pair( sizeof(cl_mem), (void *)&train.data ));
        args.push_back( make_pair( sizeof(cl_mem), (void *)&mask.data ));
        args.push_back( make_pair( sizeof(cl_mem), (void *)&trainIdx.data ));
        args.push_back( make_pair( sizeof(cl_mem), (void *)&distance.data ));
        args.push_back( make_pair( smemSize, (void *)NULL));
        args.push_back( make_pair( sizeof(cl_int), (void *)&block_size ));
        args.push_back( make_pair( sizeof(cl_int), (void *)&m_size ));
        args.push_back( make_pair( sizeof(cl_int), (void *)&query.rows ));
        args.push_back( make_pair( sizeof(cl_int), (void *)&query.cols ));
        args.push_back( make_pair( sizeof(cl_int), (void *)&train.rows ));
        args.push_back( make_pair( sizeof(cl_int), (void *)&train.cols ));
        args.push_back( make_pair( sizeof(cl_int), (void *)&query.step ));
        args.push_back( make_pair( sizeof(cl_int), (void *)&distType ));

        std::string kernelName = "BruteForceMatch_knnUnrollMatch";

        openCLExecuteKernel(ctx, &brute_force_match, kernelName, globalSize, localSize, args, -1, -1);
    }
}

template < int BLOCK_SIZE,  typename T/*, typename Mask*/ >
void knn_match(const oclMat &query, const oclMat &train, const oclMat &mask,
               const oclMat &trainIdx, const oclMat &distance, int distType)
{
    cv::ocl::Context *ctx = query.clCxt;
    size_t globalSize[] = {(query.rows + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE, BLOCK_SIZE, 1};
    size_t localSize[] = {BLOCK_SIZE, BLOCK_SIZE, 1};
    const size_t smemSize = (2 * BLOCK_SIZE * BLOCK_SIZE) * sizeof(int);
    int block_size = BLOCK_SIZE;
    vector< pair<size_t, const void *> > args;

    if(globalSize[0] != 0)
    {
        args.push_back( make_pair( sizeof(cl_mem), (void *)&query.data ));
        args.push_back( make_pair( sizeof(cl_mem), (void *)&train.data ));
        args.push_back( make_pair( sizeof(cl_mem), (void *)&mask.data ));
        args.push_back( make_pair( sizeof(cl_mem), (void *)&trainIdx.data ));
        args.push_back( make_pair( sizeof(cl_mem), (void *)&distance.data ));
        args.push_back( make_pair( smemSize, (void *)NULL));
        args.push_back( make_pair( sizeof(cl_int), (void *)&block_size ));
        args.push_back( make_pair( sizeof(cl_int), (void *)&query.rows ));
        args.push_back( make_pair( sizeof(cl_int), (void *)&query.cols ));
        args.push_back( make_pair( sizeof(cl_int), (void *)&train.rows ));
        args.push_back( make_pair( sizeof(cl_int), (void *)&train.cols ));
        args.push_back( make_pair( sizeof(cl_int), (void *)&query.step ));
        args.push_back( make_pair( sizeof(cl_int), (void *)&distType ));

        std::string kernelName = "BruteForceMatch_knnMatch";

        openCLExecuteKernel(ctx, &brute_force_match, kernelName, globalSize, localSize, args, -1, -1);
    }
}

template < int BLOCK_SIZE, int MAX_DESC_LEN, typename T/*, typename Mask*/ >
void calcDistanceUnrolled(const oclMat &query, const oclMat &train, const oclMat &mask, const oclMat &allDist, int distType)
{
    cv::ocl::Context *ctx = query.clCxt;
    size_t globalSize[] = {(query.rows + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE, BLOCK_SIZE, 1};
    size_t localSize[] = {BLOCK_SIZE, BLOCK_SIZE, 1};
    const size_t smemSize = (2 * BLOCK_SIZE * BLOCK_SIZE) * sizeof(int);
    int block_size = BLOCK_SIZE;
    int m_size = MAX_DESC_LEN;
    vector< pair<size_t, const void *> > args;

    if(globalSize[0] != 0)
    {
        args.push_back( make_pair( sizeof(cl_mem), (void *)&query.data ));
        args.push_back( make_pair( sizeof(cl_mem), (void *)&train.data ));
        args.push_back( make_pair( sizeof(cl_mem), (void *)&mask.data ));
        args.push_back( make_pair( sizeof(cl_mem), (void *)&allDist.data ));
        args.push_back( make_pair( smemSize, (void *)NULL));
        args.push_back( make_pair( sizeof(cl_int), (void *)&block_size ));
        args.push_back( make_pair( sizeof(cl_int), (void *)&m_size ));
        args.push_back( make_pair( sizeof(cl_int), (void *)&query.rows ));
        args.push_back( make_pair( sizeof(cl_int), (void *)&query.cols ));
        args.push_back( make_pair( sizeof(cl_int), (void *)&train.rows ));
        args.push_back( make_pair( sizeof(cl_int), (void *)&train.cols ));
        args.push_back( make_pair( sizeof(cl_int), (void *)&query.step ));
        args.push_back( make_pair( sizeof(cl_int), (void *)&distType ));

        std::string kernelName = "BruteForceMatch_calcDistanceUnrolled";

        openCLExecuteKernel(ctx, &brute_force_match, kernelName, globalSize, localSize, args, -1, -1);
    }
}

template < int BLOCK_SIZE, typename T/*, typename Mask*/ >
void calcDistance(const oclMat &query, const oclMat &train, const oclMat &mask, const oclMat &allDist, int distType)
{
    cv::ocl::Context *ctx = query.clCxt;
    size_t globalSize[] = {(query.rows + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE, BLOCK_SIZE, 1};
    size_t localSize[] = {BLOCK_SIZE, BLOCK_SIZE, 1};
    const size_t smemSize = (2 * BLOCK_SIZE * BLOCK_SIZE) * sizeof(int);
    int block_size = BLOCK_SIZE;
    vector< pair<size_t, const void *> > args;

    if(globalSize[0] != 0)
    {
        args.push_back( make_pair( sizeof(cl_mem), (void *)&query.data ));
        args.push_back( make_pair( sizeof(cl_mem), (void *)&train.data ));
        args.push_back( make_pair( sizeof(cl_mem), (void *)&mask.data ));
        args.push_back( make_pair( sizeof(cl_mem), (void *)&allDist.data ));
        args.push_back( make_pair( smemSize, (void *)NULL));
        args.push_back( make_pair( sizeof(cl_int), (void *)&block_size ));
        args.push_back( make_pair( sizeof(cl_int), (void *)&query.rows ));
        args.push_back( make_pair( sizeof(cl_int), (void *)&query.cols ));
        args.push_back( make_pair( sizeof(cl_int), (void *)&train.rows ));
        args.push_back( make_pair( sizeof(cl_int), (void *)&train.cols ));
        args.push_back( make_pair( sizeof(cl_int), (void *)&query.step ));
        args.push_back( make_pair( sizeof(cl_int), (void *)&distType ));

        std::string kernelName = "BruteForceMatch_calcDistance";

        openCLExecuteKernel(ctx, &brute_force_match, kernelName, globalSize, localSize, args, -1, -1);
    }
}

///////////////////////////////////////////////////////////////////////////////
// Calc Distance dispatcher
template < typename T/*, typename Mask*/ >
void calcDistanceDispatcher(const oclMat &query, const oclMat &train, const oclMat &mask,
                            const oclMat &allDist, int distType)
{
    if (query.cols <= 64)
    {
        calcDistanceUnrolled<16, 64, T>(query, train, mask, allDist, distType);
    }
    else if (query.cols <= 128)
    {
        calcDistanceUnrolled<16, 128, T>(query, train, mask, allDist, distType);
    }
    /*else if (query.cols <= 256)
    {
        calcDistanceUnrolled<16, 256, Dist>(query, train, mask, allDist, stream);
    }
    else if (query.cols <= 512)
    {
        calcDistanceUnrolled<16, 512, Dist>(query, train, mask, allDist, stream);
    }
    else if (query.cols <= 1024)
    {
        calcDistanceUnrolled<16, 1024, Dist>(query, train, mask, allDist, stream);
    }*/
    else
    {
        calcDistance<16, T>(query, train, mask, allDist, distType);
    }
}

template < typename T/*, typename Mask*/ >
void match2Dispatcher(const oclMat &query, const oclMat &train, const oclMat &mask,
                      const oclMat &trainIdx, const oclMat &distance, int distType)
{
    if (query.cols <= 64)
    {
        knn_matchUnrolledCached<16, 64, T>(query, train, mask, trainIdx, distance, distType);
    }
    else if (query.cols <= 128)
    {
        knn_matchUnrolledCached<16, 128, T>(query, train, mask, trainIdx, distance, distType);
    }
    /*else if (query.cols <= 256)
    {
        matchUnrolled<16, 256, Dist>(query, train, mask, static_cast< DevMem2D_<int2> >(trainIdx), static_cast< DevMem2D_<float2> > (distance), stream);
    }
    else if (query.cols <= 512)
    {
        matchUnrolled<16, 512, Dist>(query, train, mask, static_cast< DevMem2D_<int2> >(trainIdx), static_cast< DevMem2D_<float2> > (distance), stream);
    }
    else if (query.cols <= 1024)
    {
        matchUnrolled<16, 1024, Dist>(query, train, mask, static_cast< DevMem2D_<int2> >(trainIdx), static_cast< DevMem2D_<float2> > (distance), stream);
    }*/
    else
    {
        knn_match<16, T>(query, train, mask, trainIdx, distance, distType);
    }
}

template <int BLOCK_SIZE>
void findKnnMatch(int k, const oclMat &trainIdx, const oclMat &distance, const oclMat &allDist, int /*distType*/)
{
    cv::ocl::Context *ctx = trainIdx.clCxt;
    size_t globalSize[] = {trainIdx.rows * BLOCK_SIZE, 1, 1};
    size_t localSize[] = {BLOCK_SIZE, 1, 1};
    int block_size = BLOCK_SIZE;
    std::string kernelName = "BruteForceMatch_findBestMatch";

    for (int i = 0; i < k; ++i)
    {
        vector< pair<size_t, const void *> > args;

        args.push_back( make_pair( sizeof(cl_mem), (void *)&allDist.data ));
        args.push_back( make_pair( sizeof(cl_mem), (void *)&trainIdx.data ));
        args.push_back( make_pair( sizeof(cl_mem), (void *)&distance.data ));
        args.push_back( make_pair( sizeof(cl_mem), (void *)&i));
        args.push_back( make_pair( sizeof(cl_int), (void *)&block_size ));
        //args.push_back( make_pair( sizeof(cl_int), (void *)&train.rows ));
        //args.push_back( make_pair( sizeof(cl_int), (void *)&train.cols ));
        //args.push_back( make_pair( sizeof(cl_int), (void *)&query.step ));

        openCLExecuteKernel(ctx, &brute_force_match, kernelName, globalSize, localSize, args, -1, -1);
    }
}

static void findKnnMatchDispatcher(int k, const oclMat &trainIdx, const oclMat &distance, const oclMat &allDist, int distType)
{
    findKnnMatch<256>(k, trainIdx, distance, allDist, distType);
}

//with mask
template < typename T/*, typename Mask*/ >
void kmatchDispatcher(const oclMat &query, const oclMat &train, int k, const oclMat &mask,
                      const oclMat &trainIdx, const oclMat &distance, const oclMat &allDist, int distType)
{
    if (k == 2)
    {
        match2Dispatcher<T>(query, train, mask, trainIdx, distance, distType);
    }
    else
    {
        calcDistanceDispatcher<T>(query, train, mask, allDist, distType);
        findKnnMatchDispatcher(k, trainIdx, distance, allDist, distType);
    }
}

//without mask
template < typename T/*, typename Mask*/ >
void kmatchDispatcher(const oclMat &query, const oclMat &train, int k,
                      const oclMat &trainIdx, const oclMat &distance, const oclMat &allDist, int distType)
{
    oclMat mask;
    if (k == 2)
    {
        match2Dispatcher<T>(query, train, mask, trainIdx, distance, distType);
    }
    else
    {
        calcDistanceDispatcher<T>(query, train, mask, allDist, distType);
        findKnnMatchDispatcher(k, trainIdx, distance, allDist, distType);
    }
}



template <typename T>
void ocl_matchL1_gpu(const oclMat &query, const oclMat &train, const oclMat &mask,
                     const oclMat &trainIdx, const oclMat &distance)
{
    int distType = 0;
    if (mask.data)
    {
        matchDispatcher<T>(query, train, mask, trainIdx, distance, distType);
    }
    else
    {
        matchDispatcher< T >(query, train, trainIdx, distance, distType);
    }
}

template <typename T>
void ocl_matchL1_gpu(const oclMat &query, const oclMat &trains, const oclMat &masks,
                     const oclMat &trainIdx, const oclMat &imgIdx, const oclMat &distance)
{
    int distType = 0;

    if (masks.data)
    {
        matchDispatcher<T>(query, (const oclMat *)trains.ptr(), trains.cols, masks, trainIdx, imgIdx, distance, distType);
    }
    else
    {
        matchDispatcher<T>(query, (const oclMat *)trains.ptr(), trains.cols, trainIdx, imgIdx, distance, distType);
    }
}

template <typename T>
void ocl_matchL2_gpu(const oclMat &query, const oclMat &train, const oclMat &mask,
                     const oclMat &trainIdx, const oclMat &distance)
{
    int distType = 1;
    if (mask.data)
    {
        matchDispatcher<T>(query, train, mask, trainIdx, distance, distType);
    }
    else
    {
        matchDispatcher<T >(query, train, trainIdx, distance, distType);
    }
}

template <typename T>
void ocl_matchL2_gpu(const oclMat &query, const oclMat &trains, const oclMat &masks,
                     const oclMat &trainIdx, const oclMat &imgIdx, const oclMat &distance)
{
    int distType = 1;
    if (masks.data)
    {
        matchDispatcher<T>(query, (const oclMat *)trains.ptr(), trains.cols, masks, trainIdx, imgIdx, distance, distType);
    }
    else
    {
        matchDispatcher<T>(query, (const oclMat *)trains.ptr(), trains.cols, trainIdx, imgIdx, distance, distType);
    }
}

template <typename T>
void ocl_matchHamming_gpu(const oclMat &query, const oclMat &train, const oclMat &mask,
                          const oclMat &trainIdx, const oclMat &distance)
{
    int distType = 2;
    if (mask.data)
    {
        matchDispatcher<T>(query, train, mask, trainIdx, distance, distType);
    }
    else
    {
        matchDispatcher< T >(query, train, trainIdx, distance, distType);
    }
}

template <typename T>
void ocl_matchHamming_gpu(const oclMat &query, const oclMat &trains, const oclMat &masks,
                          const oclMat &trainIdx, const oclMat &imgIdx, const oclMat &distance)
{
    int distType = 2;
    if (masks.data)
    {
        matchDispatcher<T>(query, (const oclMat *)trains.ptr(), trains.cols, masks, trainIdx, imgIdx, distance, distType);
    }
    else
    {
        matchDispatcher<T>(query, (const oclMat *)trains.ptr(), trains.cols, trainIdx, imgIdx, distance, distType);
    }
}

// knn caller
template <typename T>
void ocl_matchL1_gpu(const oclMat &query, const oclMat &train, int k, const oclMat &mask,
                     const oclMat &trainIdx, const oclMat &distance, const oclMat &allDist)
{
    int distType = 0;

    if (mask.data)
        kmatchDispatcher<T>(query, train, k, mask, trainIdx, distance, allDist, distType);
    else
        kmatchDispatcher<T>(query, train, k, trainIdx, distance, allDist, distType);
}

template <typename T>
void ocl_matchL2_gpu(const oclMat &query, const oclMat &train, int k, const oclMat &mask,
                     const oclMat &trainIdx, const oclMat &distance, const oclMat &allDist)
{
    int distType = 1;

    if (mask.data)
        kmatchDispatcher<T>(query, train, k, mask, trainIdx, distance, allDist, distType);
    else
        kmatchDispatcher<T>(query, train, k, trainIdx, distance, allDist, distType);
}

template <typename T>
void ocl_matchHamming_gpu(const oclMat &query, const oclMat &train, int k, const oclMat &mask,
                          const oclMat &trainIdx, const oclMat &distance, const oclMat &allDist)
{
    int distType = 2;

    if (mask.data)
        kmatchDispatcher<T>(query, train, k, mask, trainIdx, distance, allDist, distType);
    else
        kmatchDispatcher<T>(query, train, k,  trainIdx, distance, allDist, distType);
}

//radius caller
template <typename T>
void ocl_matchL1_gpu(const oclMat &query, const oclMat &train, float maxDistance, const oclMat &mask,
                     const oclMat &trainIdx, const oclMat &distance, const oclMat &nMatches)
{
    int distType = 0;

    if (mask.data)
        matchDispatcher<T>(query, train, maxDistance, mask, trainIdx, distance, nMatches, distType);
    else
        matchDispatcher<T>(query, train, maxDistance, trainIdx, distance, nMatches, distType);
}

template <typename T>
void ocl_matchL2_gpu(const oclMat &query, const oclMat &train, float maxDistance, const oclMat &mask,
                     const oclMat &trainIdx, const oclMat &distance, const oclMat &nMatches)
{
    int distType = 1;

    if (mask.data)
        matchDispatcher<T>(query, train, maxDistance, mask, trainIdx, distance, nMatches, distType);
    else
        matchDispatcher<T>(query, train, maxDistance, trainIdx, distance, nMatches, distType);
}

template <typename T>
void ocl_matchHamming_gpu(const oclMat &query, const oclMat &train, float maxDistance, const oclMat &mask,
                          const oclMat &trainIdx, const oclMat &distance, const oclMat &nMatches)
{
    int distType = 2;

    if (mask.data)
        matchDispatcher<T>(query, train, maxDistance, mask, trainIdx, distance,  nMatches, distType);
    else
        matchDispatcher<T>(query, train, maxDistance, trainIdx, distance, nMatches, distType);
}

cv::ocl::BruteForceMatcher_OCL_base::BruteForceMatcher_OCL_base(DistType distType_) : distType(distType_)
{
}

void cv::ocl::BruteForceMatcher_OCL_base::add(const vector<oclMat> &descCollection)
{
    trainDescCollection.insert(trainDescCollection.end(), descCollection.begin(), descCollection.end());
}

const vector<oclMat> &cv::ocl::BruteForceMatcher_OCL_base::getTrainDescriptors() const
{
    return trainDescCollection;
}

void cv::ocl::BruteForceMatcher_OCL_base::clear()
{
    trainDescCollection.clear();
}

bool cv::ocl::BruteForceMatcher_OCL_base::empty() const
{
    return trainDescCollection.empty();
}

bool cv::ocl::BruteForceMatcher_OCL_base::isMaskSupported() const
{
    return true;
}

void cv::ocl::BruteForceMatcher_OCL_base::matchSingle(const oclMat &query, const oclMat &train,
        oclMat &trainIdx, oclMat &distance, const oclMat &mask)
{
    if (query.empty() || train.empty())
        return;

    typedef void (*caller_t)(const oclMat & query, const oclMat & train, const oclMat & mask,
                             const oclMat & trainIdx, const oclMat & distance);

    static const caller_t callers[3][6] =
    {
        {
            ocl_matchL1_gpu<unsigned char>, 0/*ocl_matchL1_gpu<signed char>*/,
            ocl_matchL1_gpu<unsigned short>, ocl_matchL1_gpu<short>,
            ocl_matchL1_gpu<int>, ocl_matchL1_gpu<float>
        },
        {
            0/*ocl_matchL2_gpu<unsigned char>*/, 0/*ocl_matchL2_gpu<signed char>*/,
            0/*ocl_matchL2_gpu<unsigned short>*/, 0/*ocl_matchL2_gpu<short>*/,
            0/*ocl_matchL2_gpu<int>*/, ocl_matchL2_gpu<float>
        },
        {
            ocl_matchHamming_gpu<unsigned char>, 0/*ocl_matchHamming_gpu<signed char>*/,
            ocl_matchHamming_gpu<unsigned short>, 0/*ocl_matchHamming_gpu<short>*/,
            ocl_matchHamming_gpu<int>, 0/*ocl_matchHamming_gpu<float>*/
        }
    };

    CV_Assert(query.channels() == 1 && query.depth() < CV_64F);
    CV_Assert(train.cols == query.cols && train.type() == query.type());

    const int nQuery = query.rows;
    trainIdx.create(1, nQuery, CV_32S);
    distance.create(1, nQuery, CV_32F);

    caller_t func = callers[distType][query.depth()];
    func(query, train, mask, trainIdx, distance);
}

void cv::ocl::BruteForceMatcher_OCL_base::matchDownload(const oclMat &trainIdx, const oclMat &distance, vector<DMatch> &matches)
{
    if (trainIdx.empty() || distance.empty())
        return;

    Mat trainIdxCPU(trainIdx);
    Mat distanceCPU(distance);

    matchConvert(trainIdxCPU, distanceCPU, matches);
}

void cv::ocl::BruteForceMatcher_OCL_base::matchConvert(const Mat &trainIdx, const Mat &distance, vector<DMatch> &matches)
{
    if (trainIdx.empty() || distance.empty())
        return;

    CV_Assert(trainIdx.type() == CV_32SC1);
    CV_Assert(distance.type() == CV_32FC1 && distance.cols == trainIdx.cols);

    const int nQuery = trainIdx.cols;

    matches.clear();
    matches.reserve(nQuery);

    const int *trainIdx_ptr = trainIdx.ptr<int>();
    const float *distance_ptr =  distance.ptr<float>();
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

void cv::ocl::BruteForceMatcher_OCL_base::match(const oclMat &query, const oclMat &train, vector<DMatch> &matches, const oclMat &mask)
{
    oclMat trainIdx, distance;
    matchSingle(query, train, trainIdx, distance, mask);
    matchDownload(trainIdx, distance, matches);
}

void cv::ocl::BruteForceMatcher_OCL_base::makeGpuCollection(oclMat &trainCollection, oclMat &maskCollection, const vector<oclMat> &masks)
{

    if (empty())
        return;

    if (masks.empty())
    {
        Mat trainCollectionCPU(1, static_cast<int>(trainDescCollection.size()), CV_8UC(sizeof(oclMat)));

        oclMat *trainCollectionCPU_ptr = trainCollectionCPU.ptr<oclMat>();

        for (size_t i = 0, size = trainDescCollection.size(); i < size; ++i, ++trainCollectionCPU_ptr)
            *trainCollectionCPU_ptr = trainDescCollection[i];

        trainCollection.upload(trainCollectionCPU);
        maskCollection.release();
    }
    else
    {
        CV_Assert(masks.size() == trainDescCollection.size());

        Mat trainCollectionCPU(1, static_cast<int>(trainDescCollection.size()), CV_8UC(sizeof(oclMat)));
        Mat maskCollectionCPU(1, static_cast<int>(trainDescCollection.size()), CV_8UC(sizeof(oclMat)));

        oclMat *trainCollectionCPU_ptr = trainCollectionCPU.ptr<oclMat>();
        oclMat *maskCollectionCPU_ptr = maskCollectionCPU.ptr<oclMat>();

        for (size_t i = 0, size = trainDescCollection.size(); i < size; ++i, ++trainCollectionCPU_ptr, ++maskCollectionCPU_ptr)
        {
            const oclMat &train = trainDescCollection[i];
            const oclMat &mask = masks[i];

            CV_Assert(mask.empty() || (mask.type() == CV_8UC1 && mask.cols == train.rows));

            *trainCollectionCPU_ptr = train;
            *maskCollectionCPU_ptr = mask;
        }

        trainCollection.upload(trainCollectionCPU);
        maskCollection.upload(maskCollectionCPU);
    }
}

void cv::ocl::BruteForceMatcher_OCL_base::matchCollection(const oclMat &query, const oclMat &trainCollection, oclMat &trainIdx,
        oclMat &imgIdx, oclMat &distance, const oclMat &masks)
{
    if (query.empty() || trainCollection.empty())
        return;

    typedef void (*caller_t)(const oclMat & query, const oclMat & trains, const oclMat & masks,
                             const oclMat & trainIdx, const oclMat & imgIdx, const oclMat & distance);

    static const caller_t callers[3][6] =
    {
        {
            ocl_matchL1_gpu<unsigned char>, 0/*matchL1_gpu<signed char>*/,
            ocl_matchL1_gpu<unsigned short>, ocl_matchL1_gpu<short>,
            ocl_matchL1_gpu<int>, ocl_matchL1_gpu<float>
        },
        {
            0/*matchL2_gpu<unsigned char>*/, 0/*matchL2_gpu<signed char>*/,
            0/*matchL2_gpu<unsigned short>*/, 0/*matchL2_gpu<short>*/,
            0/*matchL2_gpu<int>*/, ocl_matchL2_gpu<float>
        },
        {
            ocl_matchHamming_gpu<unsigned char>, 0/*matchHamming_gpu<signed char>*/,
            ocl_matchHamming_gpu<unsigned short>, 0/*matchHamming_gpu<short>*/,
            ocl_matchHamming_gpu<int>, 0/*matchHamming_gpu<float>*/
        }
    };

    CV_Assert(query.channels() == 1 && query.depth() < CV_64F);

    const int nQuery = query.rows;

    trainIdx.create(1, nQuery, CV_32S);
    imgIdx.create(1, nQuery, CV_32S);
    distance.create(1, nQuery, CV_32F);

    caller_t func = callers[distType][query.depth()];
    CV_Assert(func != 0);

    func(query, trainCollection, masks, trainIdx, imgIdx, distance);
}

void cv::ocl::BruteForceMatcher_OCL_base::matchDownload(const oclMat &trainIdx, const oclMat &imgIdx, const oclMat &distance, vector<DMatch> &matches)
{
    if (trainIdx.empty() || imgIdx.empty() || distance.empty())
        return;

    Mat trainIdxCPU(trainIdx);
    Mat imgIdxCPU(imgIdx);
    Mat distanceCPU(distance);

    matchConvert(trainIdxCPU, imgIdxCPU, distanceCPU, matches);
}

void cv::ocl::BruteForceMatcher_OCL_base::matchConvert(const Mat &trainIdx, const Mat &imgIdx, const Mat &distance, vector<DMatch> &matches)
{
    if (trainIdx.empty() || imgIdx.empty() || distance.empty())
        return;

    CV_Assert(trainIdx.type() == CV_32SC1);
    CV_Assert(imgIdx.type() == CV_32SC1 && imgIdx.cols == trainIdx.cols);
    CV_Assert(distance.type() == CV_32FC1 && distance.cols == trainIdx.cols);

    const int nQuery = trainIdx.cols;

    matches.clear();
    matches.reserve(nQuery);

    const int *trainIdx_ptr = trainIdx.ptr<int>();
    const int *imgIdx_ptr = imgIdx.ptr<int>();
    const float *distance_ptr =  distance.ptr<float>();
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

void cv::ocl::BruteForceMatcher_OCL_base::match(const oclMat &query, vector<DMatch> &matches, const vector<oclMat> &masks)
{
    oclMat trainCollection;
    oclMat maskCollection;

    makeGpuCollection(trainCollection, maskCollection, masks);

    oclMat trainIdx, imgIdx, distance;

    matchCollection(query, trainCollection, trainIdx, imgIdx, distance, maskCollection);
    matchDownload(trainIdx, imgIdx, distance, matches);
}

// knn match
void cv::ocl::BruteForceMatcher_OCL_base::knnMatchSingle(const oclMat &query, const oclMat &train, oclMat &trainIdx,
        oclMat &distance, oclMat &allDist, int k, const oclMat &mask)
{
    if (query.empty() || train.empty())
        return;

    typedef void (*caller_t)(const oclMat & query, const oclMat & train, int k, const oclMat & mask,
                             const oclMat & trainIdx, const oclMat & distance, const oclMat & allDist);

    static const caller_t callers[3][6] =
    {
        {
            ocl_matchL1_gpu<unsigned char>, 0/*ocl_matchL1_gpu<signed char>*/,
            ocl_matchL1_gpu<unsigned short>, ocl_matchL1_gpu<short>,
            ocl_matchL1_gpu<int>, ocl_matchL1_gpu<float>
        },
        {
            0/*ocl_matchL2_gpu<unsigned char>*/, 0/*ocl_matchL2_gpu<signed char>*/,
            0/*ocl_matchL2_gpu<unsigned short>*/, 0/*ocl_matchL2_gpu<short>*/,
            0/*ocl_matchL2_gpu<int>*/, ocl_matchL2_gpu<float>
        },
        {
            ocl_matchHamming_gpu<unsigned char>, 0/*ocl_matchHamming_gpu<signed char>*/,
            ocl_matchHamming_gpu<unsigned short>, 0/*ocl_matchHamming_gpu<short>*/,
            ocl_matchHamming_gpu<int>, 0/*ocl_matchHamming_gpu<float>*/
        }
    };

    CV_Assert(query.channels() == 1 && query.depth() < CV_64F);
    CV_Assert(train.type() == query.type() && train.cols == query.cols);

    const int nQuery = query.rows;
    const int nTrain = train.rows;

    if (k == 2)
    {
        trainIdx.create(1, nQuery, CV_32SC2);
        distance.create(1, nQuery, CV_32FC2);
    }
    else
    {
        trainIdx.create(nQuery, k, CV_32S);
        distance.create(nQuery, k, CV_32F);
        allDist.create(nQuery, nTrain, CV_32FC1);
    }

    trainIdx.setTo(Scalar::all(-1));

    caller_t func = callers[distType][query.depth()];
    CV_Assert(func != 0);

    func(query, train, k, mask, trainIdx, distance, allDist);
}

void cv::ocl::BruteForceMatcher_OCL_base::knnMatchDownload(const oclMat &trainIdx, const oclMat &distance, vector< vector<DMatch> > &matches, bool compactResult)
{
    if (trainIdx.empty() || distance.empty())
        return;

    Mat trainIdxCPU(trainIdx);
    Mat distanceCPU(distance);

    knnMatchConvert(trainIdxCPU, distanceCPU, matches, compactResult);
}

void cv::ocl::BruteForceMatcher_OCL_base::knnMatchConvert(const Mat &trainIdx, const Mat &distance, vector< vector<DMatch> > &matches, bool compactResult)
{
    if (trainIdx.empty() || distance.empty())
        return;

    CV_Assert(trainIdx.type() == CV_32SC2 || trainIdx.type() == CV_32SC1);
    CV_Assert(distance.type() == CV_32FC2 || distance.type() == CV_32FC1);
    CV_Assert(distance.size() == trainIdx.size());
    CV_Assert(trainIdx.isContinuous() && distance.isContinuous());

    const int nQuery = trainIdx.type() == CV_32SC2 ? trainIdx.cols : trainIdx.rows;
    const int k = trainIdx.type() == CV_32SC2 ? 2 : trainIdx.cols;

    matches.clear();
    matches.reserve(nQuery);

    const int *trainIdx_ptr = trainIdx.ptr<int>();
    const float *distance_ptr = distance.ptr<float>();

    for (int queryIdx = 0; queryIdx < nQuery; ++queryIdx)
    {
        matches.push_back(vector<DMatch>());
        vector<DMatch> &curMatches = matches.back();
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

void cv::ocl::BruteForceMatcher_OCL_base::knnMatch(const oclMat &query, const oclMat &train, vector< vector<DMatch> > &matches
        , int k, const oclMat &mask, bool compactResult)
{
    oclMat trainIdx, distance, allDist;
    knnMatchSingle(query, train, trainIdx, distance, allDist, k, mask);
    knnMatchDownload(trainIdx, distance, matches, compactResult);
}

void cv::ocl::BruteForceMatcher_OCL_base::knnMatch2Collection(const oclMat &query, const oclMat &trainCollection,
        oclMat &trainIdx, oclMat &imgIdx, oclMat &distance, const oclMat &/*maskCollection*/)
{
    if (query.empty() || trainCollection.empty())
        return;

    typedef void (*caller_t)(const oclMat & query, const oclMat & trains, const oclMat & masks,
                             const oclMat & trainIdx, const oclMat & imgIdx, const oclMat & distance);
#if 0
    static const caller_t callers[3][6] =
    {
        {
            ocl_match2L1_gpu<unsigned char>, 0/*match2L1_gpu<signed char>*/,
            ocl_match2L1_gpu<unsigned short>, ocl_match2L1_gpu<short>,
            ocl_match2L1_gpu<int>, ocl_match2L1_gpu<float>
        },
        {
            0/*match2L2_gpu<unsigned char>*/, 0/*match2L2_gpu<signed char>*/,
            0/*match2L2_gpu<unsigned short>*/, 0/*match2L2_gpu<short>*/,
            0/*match2L2_gpu<int>*/, ocl_match2L2_gpu<float>
        },
        {
            ocl_match2Hamming_gpu<unsigned char>, 0/*match2Hamming_gpu<signed char>*/,
            ocl_match2Hamming_gpu<unsigned short>, 0/*match2Hamming_gpu<short>*/,
            ocl_match2Hamming_gpu<int>, 0/*match2Hamming_gpu<float>*/
        }
    };
#endif
    CV_Assert(query.channels() == 1 && query.depth() < CV_64F);

    const int nQuery = query.rows;

    trainIdx.create(1, nQuery, CV_32SC2);
    imgIdx.create(1, nQuery, CV_32SC2);
    distance.create(1, nQuery, CV_32SC2);

    trainIdx.setTo(Scalar::all(-1));

    //caller_t func = callers[distType][query.depth()];
    //CV_Assert(func != 0);

    //func(query, trainCollection, maskCollection, trainIdx, imgIdx, distance, cc, StreamAccessor::getStream(stream));
}

void cv::ocl::BruteForceMatcher_OCL_base::knnMatch2Download(const oclMat &trainIdx, const oclMat &imgIdx,
        const oclMat &distance, vector< vector<DMatch> > &matches, bool compactResult)
{
    if (trainIdx.empty() || imgIdx.empty() || distance.empty())
        return;

    Mat trainIdxCPU(trainIdx);
    Mat imgIdxCPU(imgIdx);
    Mat distanceCPU(distance);

    knnMatch2Convert(trainIdxCPU, imgIdxCPU, distanceCPU, matches, compactResult);
}

void cv::ocl::BruteForceMatcher_OCL_base::knnMatch2Convert(const Mat &trainIdx, const Mat &imgIdx, const Mat &distance,
        vector< vector<DMatch> > &matches, bool compactResult)
{
    if (trainIdx.empty() || imgIdx.empty() || distance.empty())
        return;

    CV_Assert(trainIdx.type() == CV_32SC2);
    CV_Assert(imgIdx.type() == CV_32SC2 && imgIdx.cols == trainIdx.cols);
    CV_Assert(distance.type() == CV_32FC2 && distance.cols == trainIdx.cols);

    const int nQuery = trainIdx.cols;

    matches.clear();
    matches.reserve(nQuery);

    const int *trainIdx_ptr = trainIdx.ptr<int>();
    const int *imgIdx_ptr = imgIdx.ptr<int>();
    const float *distance_ptr = distance.ptr<float>();

    for (int queryIdx = 0; queryIdx < nQuery; ++queryIdx)
    {
        matches.push_back(vector<DMatch>());
        vector<DMatch> &curMatches = matches.back();
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
        inline void operator()(DMatch &m) const
        {
            m.imgIdx = imgIdx;
        }
        int imgIdx;
    };
}

void cv::ocl::BruteForceMatcher_OCL_base::knnMatch(const oclMat &query, vector< vector<DMatch> > &matches, int k,
        const vector<oclMat> &masks, bool compactResult)
{


    if (k == 2)
    {
        oclMat trainCollection;
        oclMat maskCollection;

        makeGpuCollection(trainCollection, maskCollection, masks);

        oclMat trainIdx, imgIdx, distance;

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
            knnMatch(query, trainDescCollection[imgIdx], curMatches, k, masks.empty() ? oclMat() : masks[imgIdx]);

            for (int queryIdx = 0; queryIdx < query.rows; ++queryIdx)
            {
                vector<DMatch> &localMatch = curMatches[queryIdx];
                vector<DMatch> &globalMatch = matches[queryIdx];

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

// radiusMatchSingle
void cv::ocl::BruteForceMatcher_OCL_base::radiusMatchSingle(const oclMat &query, const oclMat &train,
        oclMat &trainIdx,	oclMat &distance, oclMat &nMatches, float maxDistance, const oclMat &mask)
{
    if (query.empty() || train.empty())
        return;

    typedef void (*caller_t)(const oclMat & query, const oclMat & train, float maxDistance, const oclMat & mask,
                             const oclMat & trainIdx, const oclMat & distance, const oclMat & nMatches);

    //#if 0
    static const caller_t callers[3][6] =
    {
        {
            ocl_matchL1_gpu<unsigned char>, 0/*ocl_matchL1_gpu<signed char>*/,
            ocl_matchL1_gpu<unsigned short>, ocl_matchL1_gpu<short>,
            ocl_matchL1_gpu<int>, ocl_matchL1_gpu<float>
        },
        {
            0/*ocl_matchL2_gpu<unsigned char>*/, 0/*ocl_matchL2_gpu<signed char>*/,
            0/*ocl_matchL2_gpu<unsigned short>*/, 0/*ocl_matchL2_gpu<short>*/,
            0/*ocl_matchL2_gpu<int>*/, ocl_matchL2_gpu<float>
        },
        {
            ocl_matchHamming_gpu<unsigned char>, 0/*ocl_matchHamming_gpu<signed char>*/,
            ocl_matchHamming_gpu<unsigned short>, 0/*ocl_matchHamming_gpu<short>*/,
            ocl_matchHamming_gpu<int>, 0/*ocl_matchHamming_gpu<float>*/
        }
    };
    //#endif

    const int nQuery = query.rows;
    const int nTrain = train.rows;

    CV_Assert(query.channels() == 1 && query.depth() < CV_64F);
    CV_Assert(train.type() == query.type() && train.cols == query.cols);
    CV_Assert(trainIdx.empty() || (trainIdx.rows == nQuery && trainIdx.size() == distance.size()));

    nMatches.create(1, nQuery, CV_32SC1);
    if (trainIdx.empty())
    {
        trainIdx.create(nQuery, std::max((nTrain / 100), 10), CV_32SC1);
        distance.create(nQuery, std::max((nTrain / 100), 10), CV_32FC1);
    }

    nMatches.setTo(Scalar::all(0));

    caller_t func = callers[distType][query.depth()];
    //CV_Assert(func != 0);
    //func(query, train, maxDistance, mask, trainIdx, distance, nMatches, cc, StreamAccessor::getStream(stream));
    func(query, train, maxDistance, mask, trainIdx, distance, nMatches);
}

void cv::ocl::BruteForceMatcher_OCL_base::radiusMatchDownload(const oclMat &trainIdx, const oclMat &distance, const oclMat &nMatches,
        vector< vector<DMatch> > &matches, bool compactResult)
{
    if (trainIdx.empty() || distance.empty() || nMatches.empty())
        return;

    Mat trainIdxCPU(trainIdx);
    Mat distanceCPU(distance);
    Mat nMatchesCPU(nMatches);

    radiusMatchConvert(trainIdxCPU, distanceCPU, nMatchesCPU, matches, compactResult);
}

void cv::ocl::BruteForceMatcher_OCL_base::radiusMatchConvert(const Mat &trainIdx, const Mat &distance, const Mat &nMatches,
        vector< vector<DMatch> > &matches, bool compactResult)
{
    if (trainIdx.empty() || distance.empty() || nMatches.empty())
        return;

    CV_Assert(trainIdx.type() == CV_32SC1);
    CV_Assert(distance.type() == CV_32FC1 && distance.size() == trainIdx.size());
    CV_Assert(nMatches.type() == CV_32SC1 && nMatches.cols == trainIdx.rows);

    const int nQuery = trainIdx.rows;

    matches.clear();
    matches.reserve(nQuery);

    const int *nMatches_ptr = nMatches.ptr<int>();

    for (int queryIdx = 0; queryIdx < nQuery; ++queryIdx)
    {
        const int *trainIdx_ptr = trainIdx.ptr<int>(queryIdx);
        const float *distance_ptr = distance.ptr<float>(queryIdx);

        const int nMatches = std::min(nMatches_ptr[queryIdx], trainIdx.cols);

        if (nMatches == 0)
        {
            if (!compactResult)
                matches.push_back(vector<DMatch>());
            continue;
        }

        matches.push_back(vector<DMatch>(nMatches));
        vector<DMatch> &curMatches = matches.back();

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

void cv::ocl::BruteForceMatcher_OCL_base::radiusMatch(const oclMat &query, const oclMat &train, vector< vector<DMatch> > &matches,
        float maxDistance, const oclMat &mask, bool compactResult)
{
    oclMat trainIdx, distance, nMatches;
    radiusMatchSingle(query, train, trainIdx, distance, nMatches, maxDistance, mask);
    radiusMatchDownload(trainIdx, distance, nMatches, matches, compactResult);
}

void cv::ocl::BruteForceMatcher_OCL_base::radiusMatchCollection(const oclMat &query, oclMat &trainIdx, oclMat &imgIdx, oclMat &distance,
        oclMat &nMatches, float /*maxDistance*/, const vector<oclMat> &masks)
{
    if (query.empty() || empty())
        return;

    typedef void (*caller_t)(const oclMat & query, const oclMat * trains, int n, float maxDistance, const oclMat * masks,
                             const oclMat & trainIdx, const oclMat & imgIdx, const oclMat & distance, const oclMat & nMatches);
#if 0
    static const caller_t callers[3][6] =
    {
        {
            ocl_matchL1_gpu<unsigned char>, 0/*matchL1_gpu<signed char>*/,
            ocl_matchL1_gpu<unsigned short>, matchL1_gpu<short>,
            ocl_matchL1_gpu<int>, matchL1_gpu<float>
        },
        {
            0/*matchL2_gpu<unsigned char>*/, 0/*matchL2_gpu<signed char>*/,
            0/*matchL2_gpu<unsigned short>*/, 0/*matchL2_gpu<short>*/,
            0/*matchL2_gpu<int>*/, ocl_matchL2_gpu<float>
        },
        {
            ocl_matchHamming_gpu<unsigned char>, 0/*matchHamming_gpu<signed char>*/,
            ocl_matchHamming_gpu<unsigned short>, 0/*matchHamming_gpu<short>*/,
            ocl_matchHamming_gpu<int>, 0/*matchHamming_gpu<float>*/
        }
    };
#endif
    const int nQuery = query.rows;

    CV_Assert(query.channels() == 1 && query.depth() < CV_64F);
    CV_Assert(trainIdx.empty() || (trainIdx.rows == nQuery && trainIdx.size() == distance.size() && trainIdx.size() == imgIdx.size()));

    nMatches.create(1, nQuery, CV_32SC1);
    if (trainIdx.empty())
    {
        trainIdx.create(nQuery, std::max((nQuery / 100), 10), CV_32SC1);
        imgIdx.create(nQuery, std::max((nQuery / 100), 10), CV_32SC1);
        distance.create(nQuery, std::max((nQuery / 100), 10), CV_32FC1);
    }

    nMatches.setTo(Scalar::all(0));

    //caller_t func = callers[distType][query.depth()];
    //CV_Assert(func != 0);

    vector<oclMat> trains_(trainDescCollection.begin(), trainDescCollection.end());
    vector<oclMat> masks_(masks.begin(), masks.end());

    /*  func(query, &trains_[0], static_cast<int>(trains_.size()), maxDistance, masks_.size() == 0 ? 0 : &masks_[0],
          trainIdx, imgIdx, distance, nMatches));*/
}

void cv::ocl::BruteForceMatcher_OCL_base::radiusMatchDownload(const oclMat &trainIdx, const oclMat &imgIdx, const oclMat &distance,
        const oclMat &nMatches, vector< vector<DMatch> > &matches, bool compactResult)
{
    if (trainIdx.empty() || imgIdx.empty() || distance.empty() || nMatches.empty())
        return;

    Mat trainIdxCPU(trainIdx);
    Mat imgIdxCPU(imgIdx);
    Mat distanceCPU(distance);
    Mat nMatchesCPU(nMatches);

    radiusMatchConvert(trainIdxCPU, imgIdxCPU, distanceCPU, nMatchesCPU, matches, compactResult);
}

void cv::ocl::BruteForceMatcher_OCL_base::radiusMatchConvert(const Mat &trainIdx, const Mat &imgIdx, const Mat &distance, const Mat &nMatches,
        vector< vector<DMatch> > &matches, bool compactResult)
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

    const int *nMatches_ptr = nMatches.ptr<int>();

    for (int queryIdx = 0; queryIdx < nQuery; ++queryIdx)
    {
        const int *trainIdx_ptr = trainIdx.ptr<int>(queryIdx);
        const int *imgIdx_ptr = imgIdx.ptr<int>(queryIdx);
        const float *distance_ptr = distance.ptr<float>(queryIdx);

        const int nMatches = std::min(nMatches_ptr[queryIdx], trainIdx.cols);

        if (nMatches == 0)
        {
            if (!compactResult)
                matches.push_back(vector<DMatch>());
            continue;
        }

        matches.push_back(vector<DMatch>());
        vector<DMatch> &curMatches = matches.back();
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

void cv::ocl::BruteForceMatcher_OCL_base::radiusMatch(const oclMat &query, vector< vector<DMatch> > &matches, float maxDistance,
        const vector<oclMat> &masks, bool compactResult)
{
    oclMat trainIdx, imgIdx, distance, nMatches;
    radiusMatchCollection(query, trainIdx, imgIdx, distance, nMatches, maxDistance, masks);
    radiusMatchDownload(trainIdx, imgIdx, distance, nMatches, matches, compactResult);
}

#endif


