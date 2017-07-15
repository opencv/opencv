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
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
// Copyright (C) 2017, Intel Corporation, all rights reserved.
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

#include "../precomp.hpp"
#include "layers_common.hpp"
#include "op_halide.hpp"

namespace cv
{
namespace dnn
{

class ConcatLayerImpl : public ConcatLayer
{
public:
    ConcatLayerImpl(const LayerParams& params)
    {
        setParamsFrom(params);
        axis = params.get<int>("axis", 1);
    }

    virtual bool getMemoryShapes(const std::vector<MatShape> &inputs,
                                 const int requiredOutputs,
                                 std::vector<MatShape> &outputs,
                                 std::vector<MatShape> &internals) const
    {
        CV_Assert(inputs.size() > 0);
        outputs.clear();
        outputs.push_back(inputs[0]);
        int cAxis = clamp(axis, inputs[0]);

        int axisSum = 0;
        for (size_t i = 0; i < inputs.size(); i++)
        {
            MatShape curShape = inputs[i];

            CV_Assert(curShape.size() == outputs.back().size());
            for (int curAxis = 0; curAxis < outputs.back().size(); curAxis++)
            {
                if (curAxis != cAxis && outputs.back()[curAxis] != curShape[curAxis])
                    CV_Error(Error::StsBadSize, "Inconsitent shape for ConcatLayer");
            }

            axisSum += curShape[cAxis];
        }

        outputs.back()[cAxis] = axisSum;

        return false;
    }

    virtual bool supportBackend(int backendId)
    {
        return backendId == DNN_BACKEND_DEFAULT ||
               backendId == DNN_BACKEND_HALIDE && haveHalide() && axis == 1;  // By channels
    }

    class ChannelConcatInvoker : public ParallelLoopBody
    {
    public:
        std::vector<Mat*>* inputs;
        Mat* output;
        int nstripes;
        std::vector<const float*> chptrs;

        static void run(std::vector<Mat*>& inputs, Mat& output, int nstripes)
        {
            ChannelConcatInvoker cc;
            cc.inputs = &inputs;
            cc.output = &output;
            cc.nstripes = nstripes;

            size_t i, ninputs = inputs.size();
            int nchannels = 0, batchsz = output.size[0];
            for( i = 0; i < ninputs; i++ )
            {
                Mat& inp = *inputs[i];
                CV_Assert( inp.isContinuous() && inp.type() == CV_32F &&
                           inp.dims == 4 && inp.size[0] == output.size[0] &&
                           inp.size[2] == output.size[2] &&
                           inp.size[3] == output.size[3] );
                nchannels += inp.size[1];
            }
            CV_Assert( nchannels == output.size[1] );
            CV_Assert( output.isContinuous() && output.type() == CV_32F );

            cc.chptrs.resize(nchannels*batchsz);

            int ofs = 0;
            for( i = 0; i < ninputs; i++)
            {
                Mat& inp = *inputs[i];
                for( int j = 0; j < batchsz; j++ )
                    for( int k = 0; k < inp.size[1]; k++ )
                    {
                        const float* ptr = inp.ptr<float>(j, k);
                        cc.chptrs[ofs + j*nchannels + k] = ptr;
                    }
                ofs += inp.size[1];
            }

            parallel_for_(Range(0, nstripes), cc, nstripes);
        }

        ChannelConcatInvoker() {}

        void operator()(const Range& r) const
        {
            size_t planeSize = (size_t)output->size[2]*output->size[3];
            size_t nch = chptrs.size();
            size_t total = nch*planeSize;
            size_t stripeSize = (total + nstripes - 1)/nstripes;
            size_t stripeStart = r.start*stripeSize;
            size_t stripeEnd = std::min(total, r.end*stripeSize);
            const float** ptrs = (const float**)&chptrs[0];
            float* outptr = output->ptr<float>();
            size_t blockSize0 = 1 << 16;

            for( size_t ofs0 = stripeStart; ofs0 < stripeEnd; )
            {
                size_t ch = ofs0/planeSize;
                size_t ofs = ofs0 - ch*planeSize;
                size_t blockSize = std::min(blockSize0, planeSize - ofs);
                memcpy(outptr + ofs0, ptrs[ch] + ofs, blockSize*sizeof(outptr[0]));
                ofs0 += blockSize;
            }
        }
    };

    void forward(std::vector<Mat*> &inputs, std::vector<Mat> &outputs, std::vector<Mat> &internals)
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        int cAxis = clamp(axis, inputs[0]->dims);
        Mat& outMat = outputs[0];

        if( cAxis == 1 && outMat.dims == 4 )
        {
            int nstripes = getNumThreads();
            ChannelConcatInvoker::run(inputs, outMat, nstripes);
        }
        else
        {
            std::vector<Range> ranges(outputs[0].dims, Range::all());

            ranges[cAxis].start = 0;
            for (size_t i = 0; i < inputs.size(); i++)
            {
                ranges[cAxis].end = ranges[cAxis].start + inputs[i]->size[cAxis];
                inputs[i]->copyTo(outMat(&ranges[0]));
                ranges[cAxis].start = ranges[cAxis].end;
            }
        }
    }

    virtual Ptr<BackendNode> initHalide(const std::vector<Ptr<BackendWrapper> > &input)
    {
#ifdef HAVE_HALIDE
        std::vector<Halide::Buffer<> > inputBuffers = halideBuffers(input);

        Halide::Var x("x"), y("y"), c("c"), n("n");
        Halide::Func top = (name.empty() ? Halide::Func() : Halide::Func(name));
        int offset = inputBuffers[0].channels();
        Halide::Expr topExpr = select(c < offset,
                                      inputBuffers[0](x, y, c, n),
                                      inputBuffers[1](x, y, c - offset, n));
        for (int i = 2; i < input.size(); ++i)
        {
            offset += inputBuffers[i - 1].channels();
            topExpr = select(c < offset, topExpr,
                             inputBuffers[i](x, y, c - offset, n));
        }
        top(x, y, c, n) = topExpr;
        return Ptr<BackendNode>(new HalideBackendNode(top));
#endif  // HAVE_HALIDE
        return Ptr<BackendNode>();
    }
};

Ptr<ConcatLayer> ConcatLayer::create(const LayerParams& params)
{
    return Ptr<ConcatLayer>(new ConcatLayerImpl(params));
}

}
}
