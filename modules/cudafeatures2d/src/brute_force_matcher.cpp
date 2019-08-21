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

#include "precomp.hpp"

using namespace cv;
using namespace cv::cuda;

#if !defined (HAVE_CUDA) || defined (CUDA_DISABLER)

Ptr<cv::cuda::DescriptorMatcher> cv::cuda::DescriptorMatcher::createBFMatcher(int) { throw_no_cuda(); return Ptr<cv::cuda::DescriptorMatcher>(); }

#else /* !defined (HAVE_CUDA) */

namespace cv { namespace cuda { namespace device
{
    namespace bf_match
    {
        template <typename T> void matchL1_gpu(const PtrStepSzb& query, const PtrStepSzb& train, const PtrStepSzb& mask,
            const PtrStepSzi& trainIdx, const PtrStepSzf& distance,
            cudaStream_t stream);
        template <typename T> void matchL2_gpu(const PtrStepSzb& query, const PtrStepSzb& train, const PtrStepSzb& mask,
            const PtrStepSzi& trainIdx, const PtrStepSzf& distance,
            cudaStream_t stream);
        template <typename T> void matchHamming_gpu(const PtrStepSzb& query, const PtrStepSzb& train, const PtrStepSzb& mask,
            const PtrStepSzi& trainIdx, const PtrStepSzf& distance,
            cudaStream_t stream);

        template <typename T> void matchL1_gpu(const PtrStepSzb& query, const PtrStepSzb& trains, const PtrStepSz<PtrStepb>& masks,
            const PtrStepSzi& trainIdx, const PtrStepSzi& imgIdx, const PtrStepSzf& distance,
            cudaStream_t stream);
        template <typename T> void matchL2_gpu(const PtrStepSzb& query, const PtrStepSzb& trains, const PtrStepSz<PtrStepb>& masks,
            const PtrStepSzi& trainIdx, const PtrStepSzi& imgIdx, const PtrStepSzf& distance,
            cudaStream_t stream);
        template <typename T> void matchHamming_gpu(const PtrStepSzb& query, const PtrStepSzb& trains, const PtrStepSz<PtrStepb>& masks,
            const PtrStepSzi& trainIdx, const PtrStepSzi& imgIdx, const PtrStepSzf& distance,
            cudaStream_t stream);
    }

    namespace bf_knnmatch
    {
        template <typename T> void matchL1_gpu(const PtrStepSzb& query, const PtrStepSzb& train, int k, const PtrStepSzb& mask,
            const PtrStepSzb& trainIdx, const PtrStepSzb& distance, const PtrStepSzf& allDist,
            cudaStream_t stream);
        template <typename T> void matchL2_gpu(const PtrStepSzb& query, const PtrStepSzb& train, int k, const PtrStepSzb& mask,
            const PtrStepSzb& trainIdx, const PtrStepSzb& distance, const PtrStepSzf& allDist,
            cudaStream_t stream);
        template <typename T> void matchHamming_gpu(const PtrStepSzb& query, const PtrStepSzb& train, int k, const PtrStepSzb& mask,
            const PtrStepSzb& trainIdx, const PtrStepSzb& distance, const PtrStepSzf& allDist,
            cudaStream_t stream);

        template <typename T> void match2L1_gpu(const PtrStepSzb& query, const PtrStepSzb& trains, const PtrStepSz<PtrStepb>& masks,
            const PtrStepSzb& trainIdx, const PtrStepSzb& imgIdx, const PtrStepSzb& distance,
            cudaStream_t stream);
        template <typename T> void match2L2_gpu(const PtrStepSzb& query, const PtrStepSzb& trains, const PtrStepSz<PtrStepb>& masks,
            const PtrStepSzb& trainIdx, const PtrStepSzb& imgIdx, const PtrStepSzb& distance,
            cudaStream_t stream);
        template <typename T> void match2Hamming_gpu(const PtrStepSzb& query, const PtrStepSzb& trains, const PtrStepSz<PtrStepb>& masks,
            const PtrStepSzb& trainIdx, const PtrStepSzb& imgIdx, const PtrStepSzb& distance,
            cudaStream_t stream);
    }

    namespace bf_radius_match
    {
        template <typename T> void matchL1_gpu(const PtrStepSzb& query, const PtrStepSzb& train, float maxDistance, const PtrStepSzb& mask,
            const PtrStepSzi& trainIdx, const PtrStepSzf& distance, const PtrStepSz<unsigned int>& nMatches,
            cudaStream_t stream);
        template <typename T> void matchL2_gpu(const PtrStepSzb& query, const PtrStepSzb& train, float maxDistance, const PtrStepSzb& mask,
            const PtrStepSzi& trainIdx, const PtrStepSzf& distance, const PtrStepSz<unsigned int>& nMatches,
            cudaStream_t stream);
        template <typename T> void matchHamming_gpu(const PtrStepSzb& query, const PtrStepSzb& train, float maxDistance, const PtrStepSzb& mask,
            const PtrStepSzi& trainIdx, const PtrStepSzf& distance, const PtrStepSz<unsigned int>& nMatches,
            cudaStream_t stream);

        template <typename T> void matchL1_gpu(const PtrStepSzb& query, const PtrStepSzb* trains, int n, float maxDistance, const PtrStepSzb* masks,
            const PtrStepSzi& trainIdx, const PtrStepSzi& imgIdx, const PtrStepSzf& distance, const PtrStepSz<unsigned int>& nMatches,
            cudaStream_t stream);

        template <typename T> void matchL2_gpu(const PtrStepSzb& query, const PtrStepSzb* trains, int n, float maxDistance, const PtrStepSzb* masks,
            const PtrStepSzi& trainIdx, const PtrStepSzi& imgIdx, const PtrStepSzf& distance, const PtrStepSz<unsigned int>& nMatches,
            cudaStream_t stream);

        template <typename T> void matchHamming_gpu(const PtrStepSzb& query, const PtrStepSzb* trains, int n, float maxDistance, const PtrStepSzb* masks,
            const PtrStepSzi& trainIdx, const PtrStepSzi& imgIdx, const PtrStepSzf& distance, const PtrStepSz<unsigned int>& nMatches,
            cudaStream_t stream);
    }
}}}

namespace
{
    static void makeGpuCollection(const std::vector<GpuMat>& trainDescCollection,
                                  const std::vector<GpuMat>& masks,
                                  GpuMat& trainCollection,
                                  GpuMat& maskCollection)
    {
        if (trainDescCollection.empty())
            return;

        if (masks.empty())
        {
            Mat trainCollectionCPU(1, static_cast<int>(trainDescCollection.size()), CV_8UC(sizeof(PtrStepSzb)));

            PtrStepSzb* trainCollectionCPU_ptr = trainCollectionCPU.ptr<PtrStepSzb>();

            for (size_t i = 0, size = trainDescCollection.size(); i < size; ++i, ++trainCollectionCPU_ptr)
                *trainCollectionCPU_ptr = trainDescCollection[i];

            trainCollection.upload(trainCollectionCPU);
            maskCollection.release();
        }
        else
        {
            CV_Assert( masks.size() == trainDescCollection.size() );

            Mat trainCollectionCPU(1, static_cast<int>(trainDescCollection.size()), CV_8UC(sizeof(PtrStepSzb)));
            Mat maskCollectionCPU(1, static_cast<int>(trainDescCollection.size()), CV_8UC(sizeof(PtrStepb)));

            PtrStepSzb* trainCollectionCPU_ptr = trainCollectionCPU.ptr<PtrStepSzb>();
            PtrStepb* maskCollectionCPU_ptr = maskCollectionCPU.ptr<PtrStepb>();

            for (size_t i = 0, size = trainDescCollection.size(); i < size; ++i, ++trainCollectionCPU_ptr, ++maskCollectionCPU_ptr)
            {
                const GpuMat& train = trainDescCollection[i];
                const GpuMat& mask = masks[i];

                CV_Assert( mask.empty() || (mask.type() == CV_8UC1 && mask.cols == train.rows) );

                *trainCollectionCPU_ptr = train;
                *maskCollectionCPU_ptr = mask;
            }

            trainCollection.upload(trainCollectionCPU);
            maskCollection.upload(maskCollectionCPU);
        }
    }

    class BFMatcher_Impl : public cv::cuda::DescriptorMatcher
    {
    public:
        explicit BFMatcher_Impl(int norm) : norm_(norm)
        {
            CV_Assert( norm == NORM_L1 || norm == NORM_L2 || norm == NORM_HAMMING );
        }

        virtual bool isMaskSupported() const { return true; }

        virtual void add(const std::vector<GpuMat>& descriptors)
        {
            trainDescCollection_.insert(trainDescCollection_.end(), descriptors.begin(), descriptors.end());
        }

        virtual const std::vector<GpuMat>& getTrainDescriptors() const
        {
            return trainDescCollection_;
        }

        virtual void clear()
        {
            trainDescCollection_.clear();
        }

        virtual bool empty() const
        {
            return trainDescCollection_.empty();
        }

        virtual void train()
        {
        }

        virtual void match(InputArray queryDescriptors, InputArray trainDescriptors,
                           std::vector<DMatch>& matches,
                           InputArray mask = noArray());

        virtual void match(InputArray queryDescriptors,
                           std::vector<DMatch>& matches,
                           const std::vector<GpuMat>& masks = std::vector<GpuMat>());

        virtual void matchAsync(InputArray queryDescriptors, InputArray trainDescriptors,
                                OutputArray matches,
                                InputArray mask = noArray(),
                                Stream& stream = Stream::Null());

        virtual void matchAsync(InputArray queryDescriptors,
                                OutputArray matches,
                                const std::vector<GpuMat>& masks = std::vector<GpuMat>(),
                                Stream& stream = Stream::Null());

        virtual void matchConvert(InputArray gpu_matches,
                                  std::vector<DMatch>& matches);

        virtual void knnMatch(InputArray queryDescriptors, InputArray trainDescriptors,
                              std::vector<std::vector<DMatch> >& matches,
                              int k,
                              InputArray mask = noArray(),
                              bool compactResult = false);

        virtual void knnMatch(InputArray queryDescriptors,
                              std::vector<std::vector<DMatch> >& matches,
                              int k,
                              const std::vector<GpuMat>& masks = std::vector<GpuMat>(),
                              bool compactResult = false);

        virtual void knnMatchAsync(InputArray queryDescriptors, InputArray trainDescriptors,
                                   OutputArray matches,
                                   int k,
                                   InputArray mask = noArray(),
                                   Stream& stream = Stream::Null());

        virtual void knnMatchAsync(InputArray queryDescriptors,
                                   OutputArray matches,
                                   int k,
                                   const std::vector<GpuMat>& masks = std::vector<GpuMat>(),
                                   Stream& stream = Stream::Null());

        virtual void knnMatchConvert(InputArray gpu_matches,
                                     std::vector< std::vector<DMatch> >& matches,
                                     bool compactResult = false);

        virtual void radiusMatch(InputArray queryDescriptors, InputArray trainDescriptors,
                                 std::vector<std::vector<DMatch> >& matches,
                                 float maxDistance,
                                 InputArray mask = noArray(),
                                 bool compactResult = false);

        virtual void radiusMatch(InputArray queryDescriptors,
                                 std::vector<std::vector<DMatch> >& matches,
                                 float maxDistance,
                                 const std::vector<GpuMat>& masks = std::vector<GpuMat>(),
                                 bool compactResult = false);

        virtual void radiusMatchAsync(InputArray queryDescriptors, InputArray trainDescriptors,
                                      OutputArray matches,
                                      float maxDistance,
                                      InputArray mask = noArray(),
                                      Stream& stream = Stream::Null());

        virtual void radiusMatchAsync(InputArray queryDescriptors,
                                      OutputArray matches,
                                      float maxDistance,
                                      const std::vector<GpuMat>& masks = std::vector<GpuMat>(),
                                      Stream& stream = Stream::Null());

        virtual void radiusMatchConvert(InputArray gpu_matches,
                                        std::vector< std::vector<DMatch> >& matches,
                                        bool compactResult = false);

    private:
        int norm_;
        std::vector<GpuMat> trainDescCollection_;
    };

    //
    // 1 to 1 match
    //

    void BFMatcher_Impl::match(InputArray _queryDescriptors, InputArray _trainDescriptors,
                               std::vector<DMatch>& matches,
                               InputArray _mask)
    {
        GpuMat d_matches;
        matchAsync(_queryDescriptors, _trainDescriptors, d_matches, _mask);
        matchConvert(d_matches, matches);
    }

    void BFMatcher_Impl::match(InputArray _queryDescriptors,
                               std::vector<DMatch>& matches,
                               const std::vector<GpuMat>& masks)
    {
        GpuMat d_matches;
        matchAsync(_queryDescriptors, d_matches, masks);
        matchConvert(d_matches, matches);
    }

    void BFMatcher_Impl::matchAsync(InputArray _queryDescriptors, InputArray _trainDescriptors,
                                    OutputArray _matches,
                                    InputArray _mask,
                                    Stream& stream)
    {
        using namespace cv::cuda::device::bf_match;

        const GpuMat query = _queryDescriptors.getGpuMat();
        const GpuMat train = _trainDescriptors.getGpuMat();
        const GpuMat mask = _mask.getGpuMat();

        if (query.empty() || train.empty())
        {
            _matches.release();
            return;
        }

        CV_Assert( query.channels() == 1 && query.depth() < CV_64F );
        CV_Assert( train.cols == query.cols && train.type() == query.type() );
        CV_Assert( mask.empty() || (mask.type() == CV_8UC1 && mask.rows == query.rows && mask.cols == train.rows) );

        typedef void (*caller_t)(const PtrStepSzb& query, const PtrStepSzb& train, const PtrStepSzb& mask,
                                 const PtrStepSzi& trainIdx, const PtrStepSzf& distance,
                                 cudaStream_t stream);

        static const caller_t callersL1[] =
        {
            matchL1_gpu<unsigned char>, 0/*matchL1_gpu<signed char>*/,
            matchL1_gpu<unsigned short>, matchL1_gpu<short>,
            matchL1_gpu<int>, matchL1_gpu<float>
        };
        static const caller_t callersL2[] =
        {
            0/*matchL2_gpu<unsigned char>*/, 0/*matchL2_gpu<signed char>*/,
            0/*matchL2_gpu<unsigned short>*/, 0/*matchL2_gpu<short>*/,
            0/*matchL2_gpu<int>*/, matchL2_gpu<float>
        };
        static const caller_t callersHamming[] =
        {
            matchHamming_gpu<unsigned char>, 0/*matchHamming_gpu<signed char>*/,
            matchHamming_gpu<unsigned short>, 0/*matchHamming_gpu<short>*/,
            matchHamming_gpu<int>, 0/*matchHamming_gpu<float>*/
        };

        const caller_t* callers = norm_ == NORM_L1 ? callersL1 : norm_ == NORM_L2 ? callersL2 : callersHamming;

        const caller_t func = callers[query.depth()];
        if (func == 0)
        {
            CV_Error(Error::StsUnsupportedFormat, "unsupported combination of query.depth() and norm");
        }

        const int nQuery = query.rows;

        _matches.create(2, nQuery, CV_32SC1);
        GpuMat matches = _matches.getGpuMat();

        GpuMat trainIdx(1, nQuery, CV_32SC1, matches.ptr(0));
        GpuMat distance(1, nQuery, CV_32FC1, matches.ptr(1));

        func(query, train, mask, trainIdx, distance, StreamAccessor::getStream(stream));
    }

    void BFMatcher_Impl::matchAsync(InputArray _queryDescriptors,
                                    OutputArray _matches,
                                    const std::vector<GpuMat>& masks,
                                    Stream& stream)
    {
        using namespace cv::cuda::device::bf_match;

        const GpuMat query = _queryDescriptors.getGpuMat();

        if (query.empty() || trainDescCollection_.empty())
        {
            _matches.release();
            return;
        }

        CV_Assert( query.channels() == 1 && query.depth() < CV_64F );

        GpuMat trainCollection, maskCollection;
        makeGpuCollection(trainDescCollection_, masks, trainCollection, maskCollection);

        typedef void (*caller_t)(const PtrStepSzb& query, const PtrStepSzb& trains, const PtrStepSz<PtrStepb>& masks,
                                 const PtrStepSzi& trainIdx, const PtrStepSzi& imgIdx, const PtrStepSzf& distance,
                                 cudaStream_t stream);

        static const caller_t callersL1[] =
        {
            matchL1_gpu<unsigned char>, 0/*matchL1_gpu<signed char>*/,
            matchL1_gpu<unsigned short>, matchL1_gpu<short>,
            matchL1_gpu<int>, matchL1_gpu<float>
        };
        static const caller_t callersL2[] =
        {
            0/*matchL2_gpu<unsigned char>*/, 0/*matchL2_gpu<signed char>*/,
            0/*matchL2_gpu<unsigned short>*/, 0/*matchL2_gpu<short>*/,
            0/*matchL2_gpu<int>*/, matchL2_gpu<float>
        };
        static const caller_t callersHamming[] =
        {
            matchHamming_gpu<unsigned char>, 0/*matchHamming_gpu<signed char>*/,
            matchHamming_gpu<unsigned short>, 0/*matchHamming_gpu<short>*/,
            matchHamming_gpu<int>, 0/*matchHamming_gpu<float>*/
        };

        const caller_t* callers = norm_ == NORM_L1 ? callersL1 : norm_ == NORM_L2 ? callersL2 : callersHamming;

        const caller_t func = callers[query.depth()];
        if (func == 0)
        {
            CV_Error(Error::StsUnsupportedFormat, "unsupported combination of query.depth() and norm");
        }

        const int nQuery = query.rows;

        _matches.create(3, nQuery, CV_32SC1);
        GpuMat matches = _matches.getGpuMat();

        GpuMat trainIdx(1, nQuery, CV_32SC1, matches.ptr(0));
        GpuMat imgIdx(1, nQuery, CV_32SC1, matches.ptr(1));
        GpuMat distance(1, nQuery, CV_32FC1, matches.ptr(2));

        func(query, trainCollection, maskCollection, trainIdx, imgIdx, distance, StreamAccessor::getStream(stream));
    }

    void BFMatcher_Impl::matchConvert(InputArray _gpu_matches,
                                      std::vector<DMatch>& matches)
    {
        Mat gpu_matches;
        if (_gpu_matches.kind() == _InputArray::CUDA_GPU_MAT)
        {
            _gpu_matches.getGpuMat().download(gpu_matches);
        }
        else
        {
            gpu_matches = _gpu_matches.getMat();
        }

        if (gpu_matches.empty())
        {
            matches.clear();
            return;
        }

        CV_Assert( (gpu_matches.type() == CV_32SC1) && (gpu_matches.rows == 2 || gpu_matches.rows == 3) );

        const int nQuery = gpu_matches.cols;

        matches.clear();
        matches.reserve(nQuery);

        const int* trainIdxPtr = NULL;
        const int* imgIdxPtr = NULL;
        const float* distancePtr = NULL;

        if (gpu_matches.rows == 2)
        {
            trainIdxPtr = gpu_matches.ptr<int>(0);
            distancePtr =  gpu_matches.ptr<float>(1);
        }
        else
        {
            trainIdxPtr = gpu_matches.ptr<int>(0);
            imgIdxPtr =  gpu_matches.ptr<int>(1);
            distancePtr =  gpu_matches.ptr<float>(2);
        }

        for (int queryIdx = 0; queryIdx < nQuery; ++queryIdx)
        {
            const int trainIdx = trainIdxPtr[queryIdx];
            if (trainIdx == -1)
                continue;

            const int imgIdx = imgIdxPtr ? imgIdxPtr[queryIdx] : 0;
            const float distance = distancePtr[queryIdx];

            DMatch m(queryIdx, trainIdx, imgIdx, distance);

            matches.push_back(m);
        }
    }

    //
    // knn match
    //

    void BFMatcher_Impl::knnMatch(InputArray _queryDescriptors, InputArray _trainDescriptors,
                                  std::vector<std::vector<DMatch> >& matches,
                                  int k,
                                  InputArray _mask,
                                  bool compactResult)
    {
        GpuMat d_matches;
        knnMatchAsync(_queryDescriptors, _trainDescriptors, d_matches, k, _mask);
        knnMatchConvert(d_matches, matches, compactResult);
    }

    void BFMatcher_Impl::knnMatch(InputArray _queryDescriptors,
                                  std::vector<std::vector<DMatch> >& matches,
                                  int k,
                                  const std::vector<GpuMat>& masks,
                                  bool compactResult)
    {
        if (k == 2)
        {
            GpuMat d_matches;
            knnMatchAsync(_queryDescriptors, d_matches, k, masks);
            knnMatchConvert(d_matches, matches, compactResult);
        }
        else
        {
            const GpuMat query = _queryDescriptors.getGpuMat();

            if (query.empty() || trainDescCollection_.empty())
            {
                matches.clear();
                return;
            }

            CV_Assert( query.channels() == 1 && query.depth() < CV_64F );

            std::vector< std::vector<DMatch> > curMatches;
            std::vector<DMatch> temp;
            temp.reserve(2 * k);

            matches.resize(query.rows);
            for (size_t i = 0; i < matches.size(); ++i)
                matches[i].reserve(k);

            for (size_t imgIdx = 0; imgIdx < trainDescCollection_.size(); ++imgIdx)
            {
                knnMatch(query, trainDescCollection_[imgIdx], curMatches, k, masks.empty() ? GpuMat() : masks[imgIdx]);

                for (int queryIdx = 0; queryIdx < query.rows; ++queryIdx)
                {
                    std::vector<DMatch>& localMatch = curMatches[queryIdx];
                    std::vector<DMatch>& globalMatch = matches[queryIdx];

                    for (size_t i = 0; i < localMatch.size(); ++i)
                        localMatch[i].imgIdx = imgIdx;

                    temp.clear();
                    std::merge(globalMatch.begin(), globalMatch.end(), localMatch.begin(), localMatch.end(), std::back_inserter(temp));

                    globalMatch.clear();
                    const size_t count = std::min(static_cast<size_t>(k), temp.size());
                    std::copy(temp.begin(), temp.begin() + count, std::back_inserter(globalMatch));
                }
            }

            if (compactResult)
            {
#ifdef CV_CXX11
                std::vector< std::vector<DMatch> >::iterator new_end = std::remove_if(matches.begin(), matches.end(),
                    [](const std::vector<DMatch>& e)->bool { return e.empty(); });
#else
                std::vector< std::vector<DMatch> >::iterator new_end = std::remove_if(matches.begin(), matches.end(), std::mem_fun_ref(&std::vector<DMatch>::empty));
#endif
                matches.erase(new_end, matches.end());
            }
        }
    }

    void BFMatcher_Impl::knnMatchAsync(InputArray _queryDescriptors, InputArray _trainDescriptors,
                                       OutputArray _matches,
                                       int k,
                                       InputArray _mask,
                                       Stream& stream)
    {
        using namespace cv::cuda::device::bf_knnmatch;

        const GpuMat query = _queryDescriptors.getGpuMat();
        const GpuMat train = _trainDescriptors.getGpuMat();
        const GpuMat mask = _mask.getGpuMat();

        if (query.empty() || train.empty())
        {
            _matches.release();
            return;
        }

        CV_Assert( query.channels() == 1 && query.depth() < CV_64F );
        CV_Assert( train.cols == query.cols && train.type() == query.type() );
        CV_Assert( mask.empty() || (mask.type() == CV_8UC1 && mask.rows == query.rows && mask.cols == train.rows) );

        typedef void (*caller_t)(const PtrStepSzb& query, const PtrStepSzb& train, int k, const PtrStepSzb& mask,
                                 const PtrStepSzb& trainIdx, const PtrStepSzb& distance, const PtrStepSzf& allDist,
                                 cudaStream_t stream);

        static const caller_t callersL1[] =
        {
            matchL1_gpu<unsigned char>, 0/*matchL1_gpu<signed char>*/,
            matchL1_gpu<unsigned short>, matchL1_gpu<short>,
            matchL1_gpu<int>, matchL1_gpu<float>
        };
        static const caller_t callersL2[] =
        {
            0/*matchL2_gpu<unsigned char>*/, 0/*matchL2_gpu<signed char>*/,
            0/*matchL2_gpu<unsigned short>*/, 0/*matchL2_gpu<short>*/,
            0/*matchL2_gpu<int>*/, matchL2_gpu<float>
        };
        static const caller_t callersHamming[] =
        {
            matchHamming_gpu<unsigned char>, 0/*matchHamming_gpu<signed char>*/,
            matchHamming_gpu<unsigned short>, 0/*matchHamming_gpu<short>*/,
            matchHamming_gpu<int>, 0/*matchHamming_gpu<float>*/
        };

        const caller_t* callers = norm_ == NORM_L1 ? callersL1 : norm_ == NORM_L2 ? callersL2 : callersHamming;

        const caller_t func = callers[query.depth()];
        if (func == 0)
        {
            CV_Error(Error::StsUnsupportedFormat, "unsupported combination of query.depth() and norm");
        }

        const int nQuery = query.rows;
        const int nTrain = train.rows;

        GpuMat trainIdx, distance, allDist;
        if (k == 2)
        {
            _matches.create(2, nQuery, CV_32SC2);
            GpuMat matches = _matches.getGpuMat();

            trainIdx = GpuMat(1, nQuery, CV_32SC2, matches.ptr(0));
            distance = GpuMat(1, nQuery, CV_32FC2, matches.ptr(1));
        }
        else
        {
            _matches.create(2 * nQuery, k, CV_32SC1);
            GpuMat matches = _matches.getGpuMat();

            trainIdx = GpuMat(nQuery, k, CV_32SC1, matches.ptr(0), matches.step);
            distance = GpuMat(nQuery, k, CV_32FC1, matches.ptr(nQuery), matches.step);

            BufferPool pool(stream);
            allDist = pool.getBuffer(nQuery, nTrain, CV_32FC1);
        }

        trainIdx.setTo(Scalar::all(-1), stream);

        func(query, train, k, mask, trainIdx, distance, allDist, StreamAccessor::getStream(stream));
    }

    void BFMatcher_Impl::knnMatchAsync(InputArray _queryDescriptors,
                                       OutputArray _matches,
                                       int k,
                                       const std::vector<GpuMat>& masks,
                                       Stream& stream)
    {
        using namespace cv::cuda::device::bf_knnmatch;

        if (k != 2)
        {
            CV_Error(Error::StsNotImplemented, "only k=2 mode is supported for now");
        }

        const GpuMat query = _queryDescriptors.getGpuMat();

        if (query.empty() || trainDescCollection_.empty())
        {
            _matches.release();
            return;
        }

        CV_Assert( query.channels() == 1 && query.depth() < CV_64F );

        GpuMat trainCollection, maskCollection;
        makeGpuCollection(trainDescCollection_, masks, trainCollection, maskCollection);

        typedef void (*caller_t)(const PtrStepSzb& query, const PtrStepSzb& trains, const PtrStepSz<PtrStepb>& masks,
                                 const PtrStepSzb& trainIdx, const PtrStepSzb& imgIdx, const PtrStepSzb& distance,
                                 cudaStream_t stream);

        static const caller_t callersL1[] =
        {
            match2L1_gpu<unsigned char>, 0/*match2L1_gpu<signed char>*/,
            match2L1_gpu<unsigned short>, match2L1_gpu<short>,
            match2L1_gpu<int>, match2L1_gpu<float>
        };
        static const caller_t callersL2[] =
        {
            0/*match2L2_gpu<unsigned char>*/, 0/*match2L2_gpu<signed char>*/,
            0/*match2L2_gpu<unsigned short>*/, 0/*match2L2_gpu<short>*/,
            0/*match2L2_gpu<int>*/, match2L2_gpu<float>
        };
        static const caller_t callersHamming[] =
        {
            match2Hamming_gpu<unsigned char>, 0/*match2Hamming_gpu<signed char>*/,
            match2Hamming_gpu<unsigned short>, 0/*match2Hamming_gpu<short>*/,
            match2Hamming_gpu<int>, 0/*match2Hamming_gpu<float>*/
        };

        const caller_t* callers = norm_ == NORM_L1 ? callersL1 : norm_ == NORM_L2 ? callersL2 : callersHamming;

        const caller_t func = callers[query.depth()];
        if (func == 0)
        {
            CV_Error(Error::StsUnsupportedFormat, "unsupported combination of query.depth() and norm");
        }

        const int nQuery = query.rows;

        _matches.create(3, nQuery, CV_32SC2);
        GpuMat matches = _matches.getGpuMat();

        GpuMat trainIdx(1, nQuery, CV_32SC2, matches.ptr(0));
        GpuMat imgIdx(1, nQuery, CV_32SC2, matches.ptr(1));
        GpuMat distance(1, nQuery, CV_32FC2, matches.ptr(2));

        trainIdx.setTo(Scalar::all(-1), stream);

        func(query, trainCollection, maskCollection, trainIdx, imgIdx, distance, StreamAccessor::getStream(stream));
    }

    void BFMatcher_Impl::knnMatchConvert(InputArray _gpu_matches,
                                         std::vector< std::vector<DMatch> >& matches,
                                         bool compactResult)
    {
        Mat gpu_matches;
        if (_gpu_matches.kind() == _InputArray::CUDA_GPU_MAT)
        {
            _gpu_matches.getGpuMat().download(gpu_matches);
        }
        else
        {
            gpu_matches = _gpu_matches.getMat();
        }

        if (gpu_matches.empty())
        {
            matches.clear();
            return;
        }

        CV_Assert( ((gpu_matches.type() == CV_32SC2) && (gpu_matches.rows == 2 || gpu_matches.rows == 3)) ||
                   (gpu_matches.type() == CV_32SC1) );

        int nQuery = -1, k = -1;

        const int* trainIdxPtr = NULL;
        const int* imgIdxPtr = NULL;
        const float* distancePtr = NULL;

        if (gpu_matches.type() == CV_32SC2)
        {
            nQuery = gpu_matches.cols;
            k = 2;

            if (gpu_matches.rows == 2)
            {
                trainIdxPtr = gpu_matches.ptr<int>(0);
                distancePtr =  gpu_matches.ptr<float>(1);
            }
            else
            {
                trainIdxPtr = gpu_matches.ptr<int>(0);
                imgIdxPtr =  gpu_matches.ptr<int>(1);
                distancePtr =  gpu_matches.ptr<float>(2);
            }
        }
        else
        {
            nQuery = gpu_matches.rows / 2;
            k = gpu_matches.cols;

            trainIdxPtr = gpu_matches.ptr<int>(0);
            distancePtr =  gpu_matches.ptr<float>(nQuery);
        }

        matches.clear();
        matches.reserve(nQuery);

        for (int queryIdx = 0; queryIdx < nQuery; ++queryIdx)
        {
            matches.push_back(std::vector<DMatch>());
            std::vector<DMatch>& curMatches = matches.back();
            curMatches.reserve(k);

            for (int i = 0; i < k; ++i)
            {
                const int trainIdx = *trainIdxPtr;
                if (trainIdx == -1)
                    continue;

                const int imgIdx = imgIdxPtr ? *imgIdxPtr : 0;
                const float distance = *distancePtr;

                DMatch m(queryIdx, trainIdx, imgIdx, distance);

                curMatches.push_back(m);

                ++trainIdxPtr;
                ++distancePtr;
                if (imgIdxPtr)
                    ++imgIdxPtr;
            }

            if (compactResult && curMatches.empty())
            {
                matches.pop_back();
            }
        }
    }

    //
    // radius match
    //

    void BFMatcher_Impl::radiusMatch(InputArray _queryDescriptors, InputArray _trainDescriptors,
                                     std::vector<std::vector<DMatch> >& matches,
                                     float maxDistance,
                                     InputArray _mask,
                                     bool compactResult)
    {
        GpuMat d_matches;
        radiusMatchAsync(_queryDescriptors, _trainDescriptors, d_matches, maxDistance, _mask);
        radiusMatchConvert(d_matches, matches, compactResult);
    }

    void BFMatcher_Impl::radiusMatch(InputArray _queryDescriptors,
                                     std::vector<std::vector<DMatch> >& matches,
                                     float maxDistance,
                                     const std::vector<GpuMat>& masks,
                                     bool compactResult)
    {
        GpuMat d_matches;
        radiusMatchAsync(_queryDescriptors, d_matches, maxDistance, masks);
        radiusMatchConvert(d_matches, matches, compactResult);
    }

    void BFMatcher_Impl::radiusMatchAsync(InputArray _queryDescriptors, InputArray _trainDescriptors,
                                          OutputArray _matches,
                                          float maxDistance,
                                          InputArray _mask,
                                          Stream& stream)
    {
        using namespace cv::cuda::device::bf_radius_match;

        const GpuMat query = _queryDescriptors.getGpuMat();
        const GpuMat train = _trainDescriptors.getGpuMat();
        const GpuMat mask = _mask.getGpuMat();

        if (query.empty() || train.empty())
        {
            _matches.release();
            return;
        }

        CV_Assert( query.channels() == 1 && query.depth() < CV_64F );
        CV_Assert( train.cols == query.cols && train.type() == query.type() );
        CV_Assert( mask.empty() || (mask.type() == CV_8UC1 && mask.rows == query.rows && mask.cols == train.rows) );

        typedef void (*caller_t)(const PtrStepSzb& query, const PtrStepSzb& train, float maxDistance, const PtrStepSzb& mask,
                                 const PtrStepSzi& trainIdx, const PtrStepSzf& distance, const PtrStepSz<unsigned int>& nMatches,
                                 cudaStream_t stream);

        static const caller_t callersL1[] =
        {
            matchL1_gpu<unsigned char>, 0/*matchL1_gpu<signed char>*/,
            matchL1_gpu<unsigned short>, matchL1_gpu<short>,
            matchL1_gpu<int>, matchL1_gpu<float>
        };
        static const caller_t callersL2[] =
        {
            0/*matchL2_gpu<unsigned char>*/, 0/*matchL2_gpu<signed char>*/,
            0/*matchL2_gpu<unsigned short>*/, 0/*matchL2_gpu<short>*/,
            0/*matchL2_gpu<int>*/, matchL2_gpu<float>
        };
        static const caller_t callersHamming[] =
        {
            matchHamming_gpu<unsigned char>, 0/*matchHamming_gpu<signed char>*/,
            matchHamming_gpu<unsigned short>, 0/*matchHamming_gpu<short>*/,
            matchHamming_gpu<int>, 0/*matchHamming_gpu<float>*/
        };

        const caller_t* callers = norm_ == NORM_L1 ? callersL1 : norm_ == NORM_L2 ? callersL2 : callersHamming;

        const caller_t func = callers[query.depth()];
        if (func == 0)
        {
            CV_Error(Error::StsUnsupportedFormat, "unsupported combination of query.depth() and norm");
        }

        const int nQuery = query.rows;
        const int nTrain = train.rows;

        const int cols = std::max((nTrain / 100), nQuery);

        _matches.create(2 * nQuery + 1, cols, CV_32SC1);
        GpuMat matches = _matches.getGpuMat();

        GpuMat trainIdx(nQuery, cols, CV_32SC1, matches.ptr(0), matches.step);
        GpuMat distance(nQuery, cols, CV_32FC1, matches.ptr(nQuery), matches.step);
        GpuMat nMatches(1, nQuery, CV_32SC1, matches.ptr(2 * nQuery));

        nMatches.setTo(Scalar::all(0), stream);

        func(query, train, maxDistance, mask, trainIdx, distance, nMatches, StreamAccessor::getStream(stream));
    }

    void BFMatcher_Impl::radiusMatchAsync(InputArray _queryDescriptors,
                                          OutputArray _matches,
                                          float maxDistance,
                                          const std::vector<GpuMat>& masks,
                                          Stream& stream)
    {
        using namespace cv::cuda::device::bf_radius_match;

        const GpuMat query = _queryDescriptors.getGpuMat();

        if (query.empty() || trainDescCollection_.empty())
        {
            _matches.release();
            return;
        }

        CV_Assert( query.channels() == 1 && query.depth() < CV_64F );

        GpuMat trainCollection, maskCollection;
        makeGpuCollection(trainDescCollection_, masks, trainCollection, maskCollection);

        typedef void (*caller_t)(const PtrStepSzb& query, const PtrStepSzb* trains, int n, float maxDistance, const PtrStepSzb* masks,
                                 const PtrStepSzi& trainIdx, const PtrStepSzi& imgIdx, const PtrStepSzf& distance, const PtrStepSz<unsigned int>& nMatches,
                                 cudaStream_t stream);

        static const caller_t callersL1[] =
        {
            matchL1_gpu<unsigned char>, 0/*matchL1_gpu<signed char>*/,
            matchL1_gpu<unsigned short>, matchL1_gpu<short>,
            matchL1_gpu<int>, matchL1_gpu<float>
        };
        static const caller_t callersL2[] =
        {
            0/*matchL2_gpu<unsigned char>*/, 0/*matchL2_gpu<signed char>*/,
            0/*matchL2_gpu<unsigned short>*/, 0/*matchL2_gpu<short>*/,
            0/*matchL2_gpu<int>*/, matchL2_gpu<float>
        };
        static const caller_t callersHamming[] =
        {
            matchHamming_gpu<unsigned char>, 0/*matchHamming_gpu<signed char>*/,
            matchHamming_gpu<unsigned short>, 0/*matchHamming_gpu<short>*/,
            matchHamming_gpu<int>, 0/*matchHamming_gpu<float>*/
        };

        const caller_t* callers = norm_ == NORM_L1 ? callersL1 : norm_ == NORM_L2 ? callersL2 : callersHamming;

        const caller_t func = callers[query.depth()];
        if (func == 0)
        {
            CV_Error(Error::StsUnsupportedFormat, "unsupported combination of query.depth() and norm");
        }

        const int nQuery = query.rows;

        _matches.create(3 * nQuery + 1, nQuery, CV_32FC1);
        GpuMat matches = _matches.getGpuMat();

        GpuMat trainIdx(nQuery, nQuery, CV_32SC1, matches.ptr(0), matches.step);
        GpuMat imgIdx(nQuery, nQuery, CV_32SC1, matches.ptr(nQuery), matches.step);
        GpuMat distance(nQuery, nQuery, CV_32FC1, matches.ptr(2 * nQuery), matches.step);
        GpuMat nMatches(1, nQuery, CV_32SC1, matches.ptr(3 * nQuery));

        nMatches.setTo(Scalar::all(0), stream);

        std::vector<PtrStepSzb> trains_(trainDescCollection_.begin(), trainDescCollection_.end());
        std::vector<PtrStepSzb> masks_(masks.begin(), masks.end());

        func(query, &trains_[0], static_cast<int>(trains_.size()), maxDistance, masks_.size() == 0 ? 0 : &masks_[0],
            trainIdx, imgIdx, distance, nMatches, StreamAccessor::getStream(stream));
    }

    void BFMatcher_Impl::radiusMatchConvert(InputArray _gpu_matches,
                                            std::vector< std::vector<DMatch> >& matches,
                                            bool compactResult)
    {
        Mat gpu_matches;
        if (_gpu_matches.kind() == _InputArray::CUDA_GPU_MAT)
        {
            _gpu_matches.getGpuMat().download(gpu_matches);
        }
        else
        {
            gpu_matches = _gpu_matches.getMat();
        }

        if (gpu_matches.empty())
        {
            matches.clear();
            return;
        }

        CV_Assert( gpu_matches.type() == CV_32SC1 || gpu_matches.type() == CV_32FC1 );

        int nQuery = -1;

        const int* trainIdxPtr = NULL;
        const int* imgIdxPtr = NULL;
        const float* distancePtr = NULL;
        const int* nMatchesPtr = NULL;

        if (gpu_matches.type() == CV_32SC1)
        {
            nQuery = (gpu_matches.rows - 1) / 2;

            trainIdxPtr = gpu_matches.ptr<int>(0);
            distancePtr =  gpu_matches.ptr<float>(nQuery);
            nMatchesPtr = gpu_matches.ptr<int>(2 * nQuery);
        }
        else
        {
            nQuery = (gpu_matches.rows - 1) / 3;

            trainIdxPtr = gpu_matches.ptr<int>(0);
            imgIdxPtr = gpu_matches.ptr<int>(nQuery);
            distancePtr =  gpu_matches.ptr<float>(2 * nQuery);
            nMatchesPtr = gpu_matches.ptr<int>(3 * nQuery);
        }

        matches.clear();
        matches.reserve(nQuery);

        for (int queryIdx = 0; queryIdx < nQuery; ++queryIdx)
        {
            const int nMatched = std::min(nMatchesPtr[queryIdx], gpu_matches.cols);

            if (nMatched == 0)
            {
                if (!compactResult)
                {
                    matches.push_back(std::vector<DMatch>());
                }
            }
            else
            {
                matches.push_back(std::vector<DMatch>(nMatched));
                std::vector<DMatch>& curMatches = matches.back();

                for (int i = 0; i < nMatched; ++i)
                {
                    const int trainIdx = trainIdxPtr[i];

                    const int imgIdx = imgIdxPtr ? imgIdxPtr[i] : 0;
                    const float distance = distancePtr[i];

                    DMatch m(queryIdx, trainIdx, imgIdx, distance);

                    curMatches[i] = m;
                }

                std::sort(curMatches.begin(), curMatches.end());
            }

            trainIdxPtr += gpu_matches.cols;
            distancePtr += gpu_matches.cols;
            if (imgIdxPtr)
                imgIdxPtr += gpu_matches.cols;
        }
    }
}

Ptr<cv::cuda::DescriptorMatcher> cv::cuda::DescriptorMatcher::createBFMatcher(int norm)
{
    return makePtr<BFMatcher_Impl>(norm);
}

#endif /* !defined (HAVE_CUDA) */
