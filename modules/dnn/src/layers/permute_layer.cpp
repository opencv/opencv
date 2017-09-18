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
#include <float.h>
#include <algorithm>

namespace cv
{
namespace dnn
{
class PermuteLayerImpl : public PermuteLayer
{
public:
    void checkCurrentOrder(int currentOrder)
    {
        if(currentOrder < 0 || currentOrder > 3)
        {
            CV_Error(
                     Error::StsBadArg,
                     "Orders of dimensions in Permute layer parameter"
                     "must be in [0...3] interval");
        }

        if(std::find(_order.begin(), _order.end(), currentOrder) != _order.end())
        {
            CV_Error(Error::StsBadArg,
                     "Permute layer parameter contains duplicated orders.");
        }
    }

    void checkNeedForPermutation()
    {
        _needsPermute = false;
        for (size_t i = 0; i < _numAxes; ++i)
        {
            if (_order[i] != i)
            {
                _needsPermute = true;
                break;
            }
        }
    }

    PermuteLayerImpl(const LayerParams &params)
        : _count(0), _needsPermute(false), _numAxes(0)
    {
        if (!params.has("order"))
        {
            return;
        }

        DictValue paramOrder = params.get("order");
        if(paramOrder.size() > 4)
        {
            CV_Error(
                     Error::StsBadArg,
                     "Too many (> 4) orders of dimensions in Permute layer");
        }

        _numAxes = paramOrder.size();

        for (size_t i = 0; i < _numAxes; i++)
        {
            int currentOrder = paramOrder.get<int>(i);
            checkCurrentOrder(currentOrder);
            _order.push_back(currentOrder);
        }

        setParamsFrom(params);
        checkNeedForPermutation();
    }

    bool getMemoryShapes(const std::vector<MatShape> &inputs,
                         const int requiredOutputs,
                         std::vector<MatShape> &outputs,
                         std::vector<MatShape> &internals) const
    {
        if(!_needsPermute)
            return true;

        CV_Assert(inputs.size() > 0);
        CV_Assert((int)_numAxes == inputs[0].size());

        MatShape shapeBefore = inputs[0], shapeAfter;
        for (size_t i = 0; i < _numAxes; i++)
        {
            shapeAfter.push_back(shapeBefore[_order[i]]);
        }

        outputs.clear();

        for (size_t i = 0; i < inputs.size(); i++)
        {
            CV_Assert(inputs[i].size() == 4);
            CV_Assert(inputs[i][2] == shapeBefore[2] && inputs[i][3] == shapeBefore[3]);
            CV_Assert(total(inputs[i]) == total(shapeAfter));
            outputs.push_back(shapeAfter);
        }

        return false;
    }

    void computeStrides(const MatShape &shapeBefore, const MatShape &shapeAfter)
    {
        _oldStride.resize(_numAxes);
        _newStride.resize(_numAxes);

        _oldStride[_numAxes - 1] = 1;
        _newStride[_numAxes - 1] = 1;

        for(int i = _numAxes - 2; i >= 0; i--)
        {
            _oldStride[i] = _oldStride[i + 1] * shapeBefore[i + 1];
            _newStride[i] = _newStride[i + 1] * shapeAfter[i + 1];
        }

        _count = _oldStride[0] * shapeBefore[0];
    }

    void finalize(const std::vector<Mat*> &inputs, std::vector<Mat> &outputs)
    {
        if(!_needsPermute)
        {
            return;
        }

        CV_Assert(inputs.size() > 0);
        const Mat& inp0 = *inputs[0];
        CV_Assert((int)_numAxes == inp0.dims);

        computeStrides(shape(*inputs[0]), shape(outputs[0]));
    }

    class PermuteInvoker : public ParallelLoopBody
    {
    public:
        const Mat* inp;
        Mat* out;
        const std::vector<size_t>* order;
        int nstripes;

        static void run(const Mat& inp, Mat& out, const std::vector<size_t>& order, int nstripes)
        {
            PermuteInvoker p;
            p.inp = &inp;
            p.out = &out;
            p.order = &order;
            p.nstripes = nstripes;

            CV_Assert( out.size[0] == inp.size[order[0]] &&
                      out.size[1] == inp.size[order[1]] &&
                      out.size[2] == inp.size[order[2]] &&
                      out.size[3] == inp.size[order[3]]);

            parallel_for_(Range(0, nstripes), p, nstripes);
        }

        PermuteInvoker() : inp(0), out(0), order(0), nstripes(0) {}

        void operator()(const Range& r) const
        {
            int n0 = out->size[0], n1 = out->size[1], n2 = out->size[2], n3 = out->size[3];

            size_t orows = (size_t)n0*n1*n2;
            size_t stripeSize = (orows + nstripes - 1)/nstripes;
            size_t stripeStart = r.start*stripeSize;
            size_t stripeEnd = std::min(r.end*stripeSize, orows);

            const size_t esz = sizeof(float);
            size_t ostep0 = out->step[0]/esz, ostep1 = out->step[1]/esz, ostep2 = out->step[2]/esz;
            const size_t* ord = &order->at(0);
            size_t istep0 = inp->step[ord[0]]/esz, istep1 = inp->step[ord[1]]/esz,
            istep2 = inp->step[ord[2]]/esz, istep3 = inp->step[ord[3]]/esz;

            size_t val = stripeStart;
            int i2 = (int)(val % n2);
            val /= n2;
            int i1 = (int)(val % n1);
            int i0 = (int)(val / n1);

            const float* inptr_orig = inp->ptr<float>();
            float* outptr_orig = out->ptr<float>();

            for( size_t ofs = stripeStart; ofs < stripeEnd; ofs++ )
            {
                const float* inptr = inptr_orig + i0*istep0 + i1*istep1 + i2*istep2;
                float* outptr = outptr_orig + i0*ostep0 + i1*ostep1 + i2*ostep2;

                for( int i3 = 0; i3 < n3; i3++ )
                    outptr[i3] = inptr[i3*istep3];

                if( ++i2 >= n2 )
                {
                    i2 = 0;
                    if( ++i1 >= n1 )
                    {
                        i1 = 0;
                        if( ++i0 >= n0 )
                            break;
                    }
                }
            }
        }
    };

    void forward(std::vector<Mat*> &inputs, std::vector<Mat> &outputs, std::vector<Mat> &internals)
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        size_t k, ninputs = inputs.size();
        if(!_needsPermute)
        {
            for (k = 0; k < ninputs; k++)
                outputs[k] = *inputs[k];
        }
        else
        {
            size_t i, j, count = _count, numAxes = _numAxes;
            const size_t* newStride = &_newStride[0];
            const size_t* oldStride = &_oldStride[0];
            const size_t* order = &_order[0];

            for (k = 0; k < ninputs; k++)
            {
                const Mat& inp = *inputs[k];
                Mat& out = outputs[k];

                CV_Assert(inp.dims == numAxes && inp.size == inputs[0]->size);
                CV_Assert(out.dims == numAxes && out.size == outputs[0].size);

                CV_Assert(inp.isContinuous() && out.isContinuous());
                CV_Assert(inp.type() == CV_32F && out.type() == CV_32F);

                if( numAxes == 4 )
                {
                    int nstripes = getNumThreads();
                    PermuteInvoker::run(inp, out, _order, nstripes);
                }
                else
                {
                    const float *srcData = inp.ptr<float>();
                    float *dstData = out.ptr<float>();

                    for (i = 0; i < count; ++i)
                    {
                        size_t oldPosition = 0;
                        size_t newPosition = i;

                        for (j = 0; j < numAxes; ++j)
                        {
                            oldPosition += (newPosition / newStride[j]) * oldStride[order[j]];
                            newPosition %= newStride[j];
                        }
                        dstData[i] = srcData[oldPosition];
                    }
                }
            }
        }
    }

    size_t _count;
    std::vector<size_t> _order;

    std::vector<int> _oldDimensionSize;
    std::vector<int> _newDimensionSize;

    std::vector<size_t> _oldStride;
    std::vector<size_t> _newStride;
    bool _needsPermute;

    size_t _numAxes;
};

Ptr<PermuteLayer> PermuteLayer::create(const LayerParams &params)
{
    return Ptr<PermuteLayer>(new PermuteLayerImpl(params));
}

}
}
