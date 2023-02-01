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

Ptr<cv::cuda::SparsePyrLKOpticalFlow> cv::cuda::SparsePyrLKOpticalFlow::create(Size, int, int, bool) { throw_no_cuda(); return Ptr<SparsePyrLKOpticalFlow>(); }

Ptr<cv::cuda::DensePyrLKOpticalFlow> cv::cuda::DensePyrLKOpticalFlow::create(Size, int, int, bool) { throw_no_cuda(); return Ptr<DensePyrLKOpticalFlow>(); }

#else /* !defined (HAVE_CUDA) */

namespace pyrlk
{
    void loadConstants(int* winSize, int iters, cudaStream_t stream);
    void loadWinSize(int* winSize, int* halfWinSize, cudaStream_t stream);
    void loadIters(int* iters, cudaStream_t stream);
    template<typename T, int cn> struct pyrLK_caller
    {
        static void sparse(PtrStepSz<typename device::TypeVec<T, cn>::vec_type> I, PtrStepSz<typename device::TypeVec<T, cn>::vec_type> J, const float2* prevPts, float2* nextPts, uchar* status, float* err, int ptcount,
            int level, dim3 block, dim3 patch, cudaStream_t stream);

        static void dense(PtrStepSzf I, PtrStepSzf J, PtrStepSzf u, PtrStepSzf v, PtrStepSzf prevU, PtrStepSzf prevV,
            PtrStepSzf err, int2 winSize, cudaStream_t stream);
    };

    template<typename T, int cn> void dispatcher(GpuMat I, GpuMat J, const float2* prevPts, float2* nextPts, uchar* status, float* err, int ptcount,
        int level, dim3 block, dim3 patch, cudaStream_t stream)
    {
        pyrLK_caller<T, cn>::sparse(I, J, prevPts, nextPts, status, err, ptcount, level, block, patch, stream);
    }
}

namespace
{
    class PyrLKOpticalFlowBase
    {
    public:
        PyrLKOpticalFlowBase(Size winSize, int maxLevel, int iters, bool useInitialFlow);

        void sparse(const GpuMat& prevImg, const GpuMat& nextImg, const GpuMat& prevPts, GpuMat& nextPts,
            GpuMat& status, GpuMat* err, Stream& stream);

        void sparse(std::vector<GpuMat>& prevPyr, std::vector<GpuMat>& nextPyr, const GpuMat& prevPts, GpuMat& nextPts,
            GpuMat& status, GpuMat* err, Stream& stream);

        void dense(const GpuMat& prevImg, const GpuMat& nextImg, GpuMat& u, GpuMat& v, Stream& stream);

    protected:
        int winSize_[2];
        int halfWinSize_[2];
        int maxLevel_;
        int iters_;
        bool useInitialFlow_;
        void buildImagePyramid(const GpuMat& prevImg, std::vector<GpuMat>& prevPyr, const GpuMat& nextImg, std::vector<GpuMat>& nextPyr, Stream stream);
    private:
        friend class SparsePyrLKOpticalFlowImpl;
        std::vector<GpuMat> prevPyr_;
        std::vector<GpuMat> nextPyr_;
    };

    PyrLKOpticalFlowBase::PyrLKOpticalFlowBase(Size winSize, int maxLevel, int iters, bool useInitialFlow) :
        maxLevel_(maxLevel), iters_(iters), useInitialFlow_(useInitialFlow)
    {
        winSize_[0] = winSize.width;
        winSize_[1] = winSize.height;
        halfWinSize_[0] = (winSize.width - 1) / 2;
        halfWinSize_[1] = (winSize.height - 1) / 2;
        pyrlk::loadWinSize(winSize_, halfWinSize_, 0);
        pyrlk::loadIters(&iters_, 0);
    }

    void calcPatchSize(Size winSize, dim3& block, dim3& patch)
    {
        if (winSize.width > 32 && winSize.width > 2 * winSize.height)
        {
            block.x = deviceSupports(FEATURE_SET_COMPUTE_12) ? 32 : 16;
            block.y = 8;
        }
        else
        {
            block.x = 16;
            block.y = deviceSupports(FEATURE_SET_COMPUTE_12) ? 16 : 8;
        }

        patch.x = (winSize.width  + block.x - 1) / block.x;
        patch.y = (winSize.height + block.y - 1) / block.y;

        block.z = patch.z = 1;
    }

    void PyrLKOpticalFlowBase::buildImagePyramid(const GpuMat& prevImg, std::vector<GpuMat>& prevPyr, const GpuMat& nextImg, std::vector<GpuMat>& nextPyr, Stream stream)
    {
        prevPyr.resize(maxLevel_ + 1);
        nextPyr.resize(maxLevel_ + 1);

        int cn = prevImg.channels();

        CV_Assert(cn == 1 || cn == 3 || cn == 4);

        prevPyr[0] = prevImg;
        nextPyr[0] = nextImg;

        for (int level = 1; level <= maxLevel_; ++level)
        {
            cuda::pyrDown(prevPyr[level - 1], prevPyr[level], stream);
            cuda::pyrDown(nextPyr[level - 1], nextPyr[level], stream);
        }
    }
    void PyrLKOpticalFlowBase::sparse(std::vector<GpuMat>& prevPyr, std::vector<GpuMat>& nextPyr, const GpuMat& prevPts, GpuMat& nextPts,
        GpuMat& status, GpuMat* err, Stream& stream)
    {
        CV_Assert(prevPyr.size() && nextPyr.size() && "Pyramid needs to at least contain the original matrix as the first element");
        CV_Assert(prevPyr[0].size() == nextPyr[0].size());
        CV_Assert(prevPts.rows == 1 && prevPts.type() == CV_32FC2);
        CV_Assert(maxLevel_ >= 0);
        CV_Assert(winSize_[0] > 2 && winSize_[1] > 2);
        if (useInitialFlow_)
            CV_Assert(nextPts.size() == prevPts.size() && nextPts.type() == prevPts.type());
        else
            ensureSizeIsEnough(1, prevPts.cols, prevPts.type(), nextPts);

        GpuMat temp1 = (useInitialFlow_ ? nextPts : prevPts).reshape(1);
        GpuMat temp2 = nextPts.reshape(1);
        cuda::multiply(temp1, Scalar::all(1.0 / (1 << maxLevel_) / 2.0), temp2, 1, -1, stream);


        ensureSizeIsEnough(1, prevPts.cols, CV_8UC1, status);
        status.setTo(Scalar::all(1), stream);

        if (err)
            ensureSizeIsEnough(1, prevPts.cols, CV_32FC1, *err);

        if (prevPyr.size() != size_t(maxLevel_ + 1) || nextPyr.size() != size_t(maxLevel_ + 1))
        {
            buildImagePyramid(prevPyr[0], prevPyr, nextPyr[0], nextPyr, stream);
        }

        dim3 block, patch;
        calcPatchSize(Size(winSize_[0], winSize_[1]), block, patch);
        CV_Assert(patch.x > 0 && patch.x < 6 && patch.y > 0 && patch.y < 6);
        cudaStream_t stream_ = StreamAccessor::getStream(stream);
        pyrlk::loadWinSize(winSize_, halfWinSize_, stream_);
        pyrlk::loadIters(&iters_, stream_);

        const int cn = prevPyr[0].channels();
        const int type = prevPyr[0].depth();

        typedef void(*func_t)(GpuMat I, GpuMat J, const float2* prevPts, float2* nextPts, uchar* status, float* err, int ptcount,
            int level, dim3 block, dim3 patch, cudaStream_t stream);

        // Current int datatype is disabled due to pyrDown not implementing it
        // while ushort does work, it has significantly worse performance, and thus doesn't pass accuracy tests.
        static const func_t funcs[6][4] =
        {
          {   pyrlk::dispatcher<uchar, 1>     , /*pyrlk::dispatcher<uchar, 2>*/ 0,   pyrlk::dispatcher<uchar, 3>      ,   pyrlk::dispatcher<uchar, 4>    },
          { /*pyrlk::dispatcher<char, 1>*/   0, /*pyrlk::dispatcher<char, 2>*/  0, /*pyrlk::dispatcher<char, 3>*/  0  , /*pyrlk::dispatcher<char, 4>*/ 0 },
          {   pyrlk::dispatcher<ushort, 1>    , /*pyrlk::dispatcher<ushort, 2>*/0,   pyrlk::dispatcher<ushort, 3>     ,   pyrlk::dispatcher<ushort, 4>   },
          { /*pyrlk::dispatcher<short, 1>*/  0, /*pyrlk::dispatcher<short, 2>*/ 0, /*pyrlk::dispatcher<short, 3>*/ 0  , /*pyrlk::dispatcher<short, 4>*/0 },
          {   pyrlk::dispatcher<int, 1>       , /*pyrlk::dispatcher<int, 2>*/   0,   pyrlk::dispatcher<int, 3>        ,   pyrlk::dispatcher<int, 4>      },
          {   pyrlk::dispatcher<float, 1>     , /*pyrlk::dispatcher<float, 2>*/ 0,   pyrlk::dispatcher<float, 3>      ,   pyrlk::dispatcher<float, 4>    }
        };

        func_t func = funcs[type][cn-1];
        CV_Assert(func != NULL && "Datatype not implemented");
        for (int level = maxLevel_; level >= 0; level--)
        {
            func(prevPyr[level], nextPyr[level],
                prevPts.ptr<float2>(), nextPts.ptr<float2>(),
                status.ptr(), level == 0 && err ? err->ptr<float>() : 0,
                prevPts.cols, level, block, patch,
                stream_);
        }
    }

    void PyrLKOpticalFlowBase::sparse(const GpuMat& prevImg, const GpuMat& nextImg, const GpuMat& prevPts, GpuMat& nextPts, GpuMat& status, GpuMat* err, Stream& stream)
    {
        if (prevPts.empty())
        {
            nextPts.release();
            status.release();
            if (err) err->release();
            return;
        }
        CV_Assert( prevImg.channels() == 1 || prevImg.channels() == 3 || prevImg.channels() == 4 );
        CV_Assert( prevImg.size() == nextImg.size() && prevImg.type() == nextImg.type() );

        // build the image pyramids.
        buildImagePyramid(prevImg, prevPyr_, nextImg, nextPyr_, stream);

        sparse(prevPyr_, nextPyr_, prevPts, nextPts, status, err, stream);

    }

    void PyrLKOpticalFlowBase::dense(const GpuMat& prevImg, const GpuMat& nextImg, GpuMat& u, GpuMat& v, Stream& stream)
    {
        CV_Assert( prevImg.type() == CV_8UC1 );
        CV_Assert( prevImg.size() == nextImg.size() && prevImg.type() == nextImg.type() );
        CV_Assert( maxLevel_ >= 0 );
        CV_Assert( winSize_[0] > 2 && winSize_[1] > 2 );

        // build the image pyramids.

        prevPyr_.resize(maxLevel_ + 1);
        nextPyr_.resize(maxLevel_ + 1);

        //prevPyr_[0] = prevImg;

        prevImg.convertTo(prevPyr_[0], CV_32F, stream);
        nextImg.convertTo(nextPyr_[0], CV_32F, stream);

        for (int level = 1; level <= maxLevel_; ++level)
        {
            cuda::pyrDown(prevPyr_[level - 1], prevPyr_[level], stream);
            cuda::pyrDown(nextPyr_[level - 1], nextPyr_[level], stream);
        }

        BufferPool pool(stream);

        GpuMat uPyr[] = {
            pool.getBuffer(prevImg.size(), CV_32FC1),
            pool.getBuffer(prevImg.size(), CV_32FC1),
        };
        GpuMat vPyr[] = {
            pool.getBuffer(prevImg.size(), CV_32FC1),
            pool.getBuffer(prevImg.size(), CV_32FC1),
        };

        uPyr[0].setTo(Scalar::all(0), stream);
        vPyr[0].setTo(Scalar::all(0), stream);
        uPyr[1].setTo(Scalar::all(0), stream);
        vPyr[1].setTo(Scalar::all(0), stream);
        cudaStream_t stream_ = StreamAccessor::getStream(stream);
        pyrlk::loadWinSize(winSize_, halfWinSize_, stream_);
        pyrlk::loadIters(&iters_, stream_);
        int2 winSize2i = make_int2(winSize_[0], winSize_[1]);
        //pyrlk::loadConstants(winSize2i, iters_, StreamAccessor::getStream(stream));

        int idx = 0;

        for (int level = maxLevel_; level >= 0; level--)
        {
            int idx2 = (idx + 1) & 1;

            pyrlk::pyrLK_caller<float,1>::dense(prevPyr_[level], nextPyr_[level],
                         uPyr[idx], vPyr[idx], uPyr[idx2], vPyr[idx2],
                         PtrStepSzf(), winSize2i,
                         stream_);

            if (level > 0)
                idx = idx2;
        }

        uPyr[idx].copyTo(u, stream);
        vPyr[idx].copyTo(v, stream);
    }

    class SparsePyrLKOpticalFlowImpl : public cv::cuda::SparsePyrLKOpticalFlow, private PyrLKOpticalFlowBase
    {
    public:
        SparsePyrLKOpticalFlowImpl(Size winSize, int maxLevel, int iters, bool useInitialFlow) :
            PyrLKOpticalFlowBase(winSize, maxLevel, iters, useInitialFlow)
        {
        }

        virtual Size getWinSize() const { return cv::Size(winSize_[0], winSize_[1]); }
        virtual void setWinSize(Size winSize) {
            winSize_[0] = winSize.width;
            winSize_[1] = winSize.height;
            halfWinSize_[0] = (winSize.width - 1) / 2;
            halfWinSize_[1] = (winSize.height -1) / 2;
        }

        virtual int getMaxLevel() const { return maxLevel_; }
        virtual void setMaxLevel(int maxLevel) { maxLevel_ = maxLevel; }

        virtual int getNumIters() const { return iters_; }
        virtual void setNumIters(int iters) { iters_ = iters; }

        virtual bool getUseInitialFlow() const { return useInitialFlow_; }
        virtual void setUseInitialFlow(bool useInitialFlow) { useInitialFlow_ = useInitialFlow; }

        virtual void calc(InputArray _prevImg, InputArray _nextImg,
                          InputArray _prevPts, InputOutputArray _nextPts,
                          OutputArray _status,
                          OutputArray _err,
                          Stream& stream)
        {
            const GpuMat prevPts = _prevPts.getGpuMat();
            GpuMat& nextPts = _nextPts.getGpuMatRef();
            GpuMat& status = _status.getGpuMatRef();
            GpuMat* err = _err.needed() ? &(_err.getGpuMatRef()) : NULL;
            if (_prevImg.kind() == _InputArray::STD_VECTOR_CUDA_GPU_MAT && _nextImg.kind() == _InputArray::STD_VECTOR_CUDA_GPU_MAT)
            {
                std::vector<GpuMat> prevPyr, nextPyr;
                _prevImg.getGpuMatVector(prevPyr);
                _nextImg.getGpuMatVector(nextPyr);
                sparse(prevPyr, nextPyr, prevPts, nextPts, status, err, stream);
            }
            else
            {
                const GpuMat prevImg = _prevImg.getGpuMat();
                const GpuMat nextImg = _nextImg.getGpuMat();
                sparse(prevImg, nextImg, prevPts, nextPts, status, err, stream);
            }
        }

        virtual String getDefaultName() const { return "SparseOpticalFlow.SparsePyrLKOpticalFlow"; }
    };

    class DensePyrLKOpticalFlowImpl : public DensePyrLKOpticalFlow, private PyrLKOpticalFlowBase
    {
    public:
        DensePyrLKOpticalFlowImpl(Size winSize, int maxLevel, int iters, bool useInitialFlow) :
            PyrLKOpticalFlowBase(winSize, maxLevel, iters, useInitialFlow)
        {
        }

        virtual Size getWinSize() const { return cv::Size(winSize_[0], winSize_[1]); }
        virtual void setWinSize(Size winSize) {
            winSize_[0] = winSize.width;
            winSize_[1] = winSize.height;
            halfWinSize_[0] = (winSize.width - 1) / 2;
            halfWinSize_[1] = (winSize.height -1) / 2;
        }

        virtual int getMaxLevel() const { return maxLevel_; }
        virtual void setMaxLevel(int maxLevel) { maxLevel_ = maxLevel; }

        virtual int getNumIters() const { return iters_; }
        virtual void setNumIters(int iters) { iters_ = iters; }

        virtual bool getUseInitialFlow() const { return useInitialFlow_; }
        virtual void setUseInitialFlow(bool useInitialFlow) { useInitialFlow_ = useInitialFlow; }

        virtual void calc(InputArray _prevImg, InputArray _nextImg, InputOutputArray _flow, Stream& stream)
        {
            const GpuMat prevImg = _prevImg.getGpuMat();
            const GpuMat nextImg = _nextImg.getGpuMat();

            BufferPool pool(stream);
            GpuMat u = pool.getBuffer(prevImg.size(), CV_32FC1);
            GpuMat v = pool.getBuffer(prevImg.size(), CV_32FC1);

            dense(prevImg, nextImg, u, v, stream);

            GpuMat flows[] = {u, v};
            cuda::merge(flows, 2, _flow, stream);
        }

        virtual String getDefaultName() const { return "DenseOpticalFlow.DensePyrLKOpticalFlow"; }
    };
}

Ptr<cv::cuda::SparsePyrLKOpticalFlow> cv::cuda::SparsePyrLKOpticalFlow::create(Size winSize, int maxLevel, int iters, bool useInitialFlow)
{
    return makePtr<SparsePyrLKOpticalFlowImpl>(winSize, maxLevel, iters, useInitialFlow);
}

Ptr<cv::cuda::DensePyrLKOpticalFlow> cv::cuda::DensePyrLKOpticalFlow::create(Size winSize, int maxLevel, int iters, bool useInitialFlow)
{
    return makePtr<DensePyrLKOpticalFlowImpl>(winSize, maxLevel, iters, useInitialFlow);
}

#endif /* !defined (HAVE_CUDA) */
