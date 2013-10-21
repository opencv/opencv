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
//     Xiaopeng Fu, fuxiaopeng2222@163.com
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
// This software is provided by the copyright holders and contributors as is and
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
#include "opencl_kernels.hpp"

using namespace cv;
using namespace cv::ocl;

static void generateRandomCenter(const std::vector<Vec2f>& box, float* center, RNG& rng)
{
    size_t j, dims = box.size();
    float margin = 1.f/dims;
    for( j = 0; j < dims; j++ )
        center[j] = ((float)rng*(1.f+margin*2.f)-margin)*(box[j][1] - box[j][0]) + box[j][0];
}

// This class is copied from matrix.cpp in core module.
class KMeansPPDistanceComputer : public ParallelLoopBody
{
public:
    KMeansPPDistanceComputer( float *_tdist2,
                              const float *_data,
                              const float *_dist,
                              int _dims,
                              size_t _step,
                              size_t _stepci )
        : tdist2(_tdist2),
          data(_data),
          dist(_dist),
          dims(_dims),
          step(_step),
          stepci(_stepci) { }

    void operator()( const cv::Range& range ) const
    {
        const int begin = range.start;
        const int end = range.end;

        for ( int i = begin; i<end; i++ )
        {
            tdist2[i] = std::min(normL2Sqr_(data + step*i, data + stepci, dims), dist[i]);
        }
    }

private:
    KMeansPPDistanceComputer& operator=(const KMeansPPDistanceComputer&); // to quiet MSVC

    float *tdist2;
    const float *data;
    const float *dist;
    const int dims;
    const size_t step;
    const size_t stepci;
};
/*
k-means center initialization using the following algorithm:
Arthur & Vassilvitskii (2007) k-means++: The Advantages of Careful Seeding
*/
static void generateCentersPP(const Mat& _data, Mat& _out_centers,
                              int K, RNG& rng, int trials)
{
    int i, j, k, dims = _data.cols, N = _data.rows;
    const float* data = (float*)_data.data;
    size_t step = _data.step/sizeof(data[0]);
    std::vector<int> _centers(K);
    int* centers = &_centers[0];
    std::vector<float> _dist(N*3);
    float* dist = &_dist[0], *tdist = dist + N, *tdist2 = tdist + N;
    double sum0 = 0;

    centers[0] = (unsigned)rng % N;

    for( i = 0; i < N; i++ )
    {
        dist[i] = normL2Sqr_(data + step*i, data + step*centers[0], dims);
        sum0 += dist[i];
    }

    for( k = 1; k < K; k++ )
    {
        double bestSum = DBL_MAX;
        int bestCenter = -1;

        for( j = 0; j < trials; j++ )
        {
            double p = (double)rng*sum0, s = 0;
            for( i = 0; i < N-1; i++ )
                if( (p -= dist[i]) <= 0 )
                    break;
            int ci = i;

            parallel_for_(Range(0, N),
                          KMeansPPDistanceComputer(tdist2, data, dist, dims, step, step*ci));
            for( i = 0; i < N; i++ )
            {
                s += tdist2[i];
            }

            if( s < bestSum )
            {
                bestSum = s;
                bestCenter = ci;
                std::swap(tdist, tdist2);
            }
        }
        centers[k] = bestCenter;
        sum0 = bestSum;
        std::swap(dist, tdist);
    }

    for( k = 0; k < K; k++ )
    {
        const float* src = data + step*centers[k];
        float* dst = _out_centers.ptr<float>(k);
        for( j = 0; j < dims; j++ )
            dst[j] = src[j];
    }
}

void cv::ocl::distanceToCenters(oclMat &dists, oclMat &labels, const oclMat &src, const oclMat &centers)
{
    //if(src.clCxt -> impl -> double_support == 0 && src.type() == CV_64F)
    //{
    //    CV_Error(Error::OpenCLDoubleNotSupported, "Selected device doesn't support double");
    //    return;
    //}

    Context  *clCxt = src.clCxt;
    int labels_step = (int)(labels.step/labels.elemSize());
    String kernelname = "distanceToCenters";
    int threadNum = src.rows > 256 ? 256 : src.rows;
    size_t localThreads[3]  = {1, threadNum, 1};
    size_t globalThreads[3] = {1, src.rows, 1};

    std::vector<std::pair<size_t, const void *> > args;
    args.push_back(std::make_pair(sizeof(cl_int), (void *)&labels_step));
    args.push_back(std::make_pair(sizeof(cl_int), (void *)&centers.rows));
    args.push_back(std::make_pair(sizeof(cl_mem), (void *)&src.data));
    args.push_back(std::make_pair(sizeof(cl_mem), (void *)&labels.data));
    args.push_back(std::make_pair(sizeof(cl_int), (void *)&centers.cols));
    args.push_back(std::make_pair(sizeof(cl_int), (void *)&src.rows));
    args.push_back(std::make_pair(sizeof(cl_mem), (void *)&centers.data));
    args.push_back(std::make_pair(sizeof(cl_mem), (void*)&dists.data));

    openCLExecuteKernel(clCxt, &kmeans_kernel, kernelname, globalThreads, localThreads, args, -1, -1, NULL);
}
///////////////////////////////////k - means /////////////////////////////////////////////////////////
double cv::ocl::kmeans(const oclMat &_src, int K, oclMat &_bestLabels,
                       TermCriteria criteria, int attempts, int flags, oclMat &_centers)
{
    const int SPP_TRIALS = 3;
    bool isrow = _src.rows == 1 && _src.oclchannels() > 1;
    int N = !isrow ? _src.rows : _src.cols;
    int dims = (!isrow ? _src.cols : 1) * _src.oclchannels();
    int type = _src.depth();

    attempts = std::max(attempts, 1);
    CV_Assert(type == CV_32F && K > 0 );
    CV_Assert( N >= K );

    Mat _labels;
    if( flags & KMEANS_USE_INITIAL_LABELS )
    {
        CV_Assert( (_bestLabels.cols == 1 || _bestLabels.rows == 1) &&
                   _bestLabels.cols * _bestLabels.rows == N &&
                   _bestLabels.type() == CV_32S );
        _bestLabels.download(_labels);
    }
    else
    {
        if( !((_bestLabels.cols == 1 || _bestLabels.rows == 1) &&
                _bestLabels.cols * _bestLabels.rows == N &&
                _bestLabels.type() == CV_32S &&
                _bestLabels.isContinuous()))
            _bestLabels.create(N, 1, CV_32S);
        _labels.create(_bestLabels.size(), _bestLabels.type());
    }
    int* labels = _labels.ptr<int>();

    Mat data;
    _src.download(data);
    Mat centers(K, dims, type), old_centers(K, dims, type), temp(1, dims, type);
    std::vector<int> counters(K);
    std::vector<Vec2f> _box(dims);
    Vec2f* box = &_box[0];
    double best_compactness = DBL_MAX, compactness = 0;
    RNG& rng = theRNG();
    int a, iter, i, j, k;

    if( criteria.type & TermCriteria::EPS )
        criteria.epsilon = std::max(criteria.epsilon, 0.);
    else
        criteria.epsilon = FLT_EPSILON;
    criteria.epsilon *= criteria.epsilon;

    if( criteria.type & TermCriteria::COUNT )
        criteria.maxCount = std::min(std::max(criteria.maxCount, 2), 100);
    else
        criteria.maxCount = 100;

    if( K == 1 )
    {
        attempts = 1;
        criteria.maxCount = 2;
    }

    const float* sample = data.ptr<float>();
    for( j = 0; j < dims; j++ )
        box[j] = Vec2f(sample[j], sample[j]);

    for( i = 1; i < N; i++ )
    {
        sample = data.ptr<float>(i);
        for( j = 0; j < dims; j++ )
        {
            float v = sample[j];
            box[j][0] = std::min(box[j][0], v);
            box[j][1] = std::max(box[j][1], v);
        }
    }

    for( a = 0; a < attempts; a++ )
    {
        double max_center_shift = DBL_MAX;
        for( iter = 0;; )
        {
            swap(centers, old_centers);

            if( iter == 0 && (a > 0 || !(flags & KMEANS_USE_INITIAL_LABELS)) )
            {
                if( flags & KMEANS_PP_CENTERS )
                    generateCentersPP(data, centers, K, rng, SPP_TRIALS);
                else
                {
                    for( k = 0; k < K; k++ )
                        generateRandomCenter(_box, centers.ptr<float>(k), rng);
                }
            }
            else
            {
                if( iter == 0 && a == 0 && (flags & KMEANS_USE_INITIAL_LABELS) )
                {
                    for( i = 0; i < N; i++ )
                        CV_Assert( (unsigned)labels[i] < (unsigned)K );
                }

                // compute centers
                centers = Scalar(0);
                for( k = 0; k < K; k++ )
                    counters[k] = 0;

                for( i = 0; i < N; i++ )
                {
                    sample = data.ptr<float>(i);
                    k = labels[i];
                    float* center = centers.ptr<float>(k);
                    j=0;
#if CV_ENABLE_UNROLLED
                    for(; j <= dims - 4; j += 4 )
                    {
                        float t0 = center[j] + sample[j];
                        float t1 = center[j+1] + sample[j+1];

                        center[j] = t0;
                        center[j+1] = t1;

                        t0 = center[j+2] + sample[j+2];
                        t1 = center[j+3] + sample[j+3];

                        center[j+2] = t0;
                        center[j+3] = t1;
                    }
#endif
                    for( ; j < dims; j++ )
                        center[j] += sample[j];
                    counters[k]++;
                }

                if( iter > 0 )
                    max_center_shift = 0;

                for( k = 0; k < K; k++ )
                {
                    if( counters[k] != 0 )
                        continue;

                    // if some cluster appeared to be empty then:
                    //   1. find the biggest cluster
                    //   2. find the farthest from the center point in the biggest cluster
                    //   3. exclude the farthest point from the biggest cluster and form a new 1-point cluster.
                    int max_k = 0;
                    for( int k1 = 1; k1 < K; k1++ )
                    {
                        if( counters[max_k] < counters[k1] )
                            max_k = k1;
                    }

                    double max_dist = 0;
                    int farthest_i = -1;
                    float* new_center =  centers.ptr<float>(k);
                    float* old_center =  centers.ptr<float>(max_k);
                    float* _old_center = temp.ptr<float>(); // normalized
                    float scale = 1.f/counters[max_k];
                    for( j = 0; j < dims; j++ )
                        _old_center[j] = old_center[j]*scale;

                    for( i = 0; i < N; i++ )
                    {
                        if( labels[i] != max_k )
                            continue;
                        sample = data.ptr<float>(i);
                        double dist = normL2Sqr_(sample, _old_center, dims);

                        if( max_dist <= dist )
                        {
                            max_dist = dist;
                            farthest_i = i;
                        }
                    }

                    counters[max_k]--;
                    counters[k]++;
                    labels[farthest_i] = k;
                    sample = data.ptr<float>(farthest_i);

                    for( j = 0; j < dims; j++ )
                    {
                        old_center[j] -= sample[j];
                        new_center[j] += sample[j];
                    }
                }

                for( k = 0; k < K; k++ )
                {
                    float* center = centers.ptr<float>(k);
                    CV_Assert( counters[k] != 0 );

                    float scale = 1.f/counters[k];
                    for( j = 0; j < dims; j++ )
                        center[j] *= scale;

                    if( iter > 0 )
                    {
                        double dist = 0;
                        const float* old_center = old_centers.ptr<float>(k);
                        for( j = 0; j < dims; j++ )
                        {
                            double t = center[j] - old_center[j];
                            dist += t*t;
                        }
                        max_center_shift = std::max(max_center_shift, dist);
                    }
                }
            }

            if( ++iter == MAX(criteria.maxCount, 2) || max_center_shift <= criteria.epsilon )
                break;

            // assign labels
            oclMat _dists(1, N, CV_64F);

            _bestLabels.upload(_labels);
            _centers.upload(centers);
            distanceToCenters(_dists, _bestLabels, _src, _centers);

            Mat dists;
            _dists.download(dists);
            _bestLabels.download(_labels);

            double* dist = dists.ptr<double>(0);
            compactness = 0;
            for( i = 0; i < N; i++ )
            {
                compactness += dist[i];
            }
        }

        if( compactness < best_compactness )
        {
            best_compactness = compactness;
        }
    }

    return best_compactness;
}
