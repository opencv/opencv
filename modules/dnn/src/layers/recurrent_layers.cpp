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
#include <iostream>
#include <iterator>
#include <cmath>
#include <opencv2/dnn/shape_utils.hpp>

namespace cv
{
namespace dnn
{

template<typename Dtype>
static void tanh(const Mat &src, Mat &dst)
{
    MatConstIterator_<Dtype> itSrc = src.begin<Dtype>();
    MatIterator_<Dtype> itDst = dst.begin<Dtype>();

    for (; itSrc != src.end<Dtype>(); itSrc++, itDst++)
        *itDst = std::tanh(*itSrc);
}

//TODO: make utils method
static void tanh(const Mat &src, Mat &dst)
{
    dst.create(src.dims, (const int*)src.size, src.type());

    if (src.type() == CV_32F)
        tanh<float>(src, dst);
    else if (src.type() == CV_64F)
        tanh<double>(src, dst);
    else
        CV_Error(Error::StsUnsupportedFormat, "Function supports only floating point types");
}

static void sigmoid(const Mat &src, Mat &dst)
{
    cv::exp(-src, dst);
    cv::pow(1 + dst, -1, dst);
}

class LSTMLayerImpl : public LSTMLayer
{
    int numTimeStamps, numSamples;
    bool allocated;

    MatShape outTailShape;                 //shape of single output sample
    MatShape outTsShape;    //shape of N output samples

    bool useTimestampDim;
    bool produceCellOutput;
    float forgetBias, cellClip;
    bool useCellClip, usePeephole;

public:

    LSTMLayerImpl(const LayerParams& params)
        : numTimeStamps(0), numSamples(0)
    {
        setParamsFrom(params);

        if (!blobs.empty())
        {
            CV_Assert(blobs.size() >= 3);

            blobs[2] = blobs[2].reshape(1, 1);

            const Mat& Wh = blobs[0];
            const Mat& Wx = blobs[1];
            const Mat& bias = blobs[2];
            CV_Assert(Wh.dims == 2 && Wx.dims == 2);
            CV_Assert(Wh.rows == Wx.rows);
            CV_Assert(Wh.rows == 4*Wh.cols);
            CV_Assert(Wh.rows == (int)bias.total());
            CV_Assert(Wh.type() == Wx.type() && Wx.type() == bias.type());

            // Peephole weights.
            if (blobs.size() > 3)
            {
                CV_Assert(blobs.size() == 6);
                for (int i = 3; i < 6; ++i)
                {
                    CV_Assert(blobs[i].rows == Wh.cols && blobs[i].cols == Wh.cols);
                    CV_Assert(blobs[i].type() == bias.type());
                }
            }
        }
        useTimestampDim = params.get<bool>("use_timestamp_dim", true);
        produceCellOutput = params.get<bool>("produce_cell_output", false);
        forgetBias = params.get<float>("forget_bias", 0.0f);
        cellClip = params.get<float>("cell_clip", 0.0f);
        useCellClip = params.get<bool>("use_cell_clip", false);
        usePeephole = params.get<bool>("use_peephole", false);

        allocated = false;
        outTailShape.clear();
    }

    void setUseTimstampsDim(bool use)
    {
        CV_Assert(!allocated);
        useTimestampDim = use;
    }

    void setProduceCellOutput(bool produce)
    {
        CV_Assert(!allocated);
        produceCellOutput = produce;
    }

    void setOutShape(const MatShape &outTailShape_)
    {
        CV_Assert(!allocated || total(outTailShape) == total(outTailShape_));
        outTailShape = outTailShape_;
    }

    void setWeights(const Mat &Wh, const Mat &Wx, const Mat &bias)
    {
        CV_Assert(Wh.dims == 2 && Wx.dims == 2);
        CV_Assert(Wh.rows == Wx.rows);
        CV_Assert(Wh.rows == 4*Wh.cols);
        CV_Assert(Wh.rows == (int)bias.total());
        CV_Assert(Wh.type() == Wx.type() && Wx.type() == bias.type());

        blobs.resize(3);
        blobs[0] = Mat(Wh.clone());
        blobs[1] = Mat(Wx.clone());
        blobs[2] = Mat(bias.clone()).reshape(1, 1);
    }

    bool getMemoryShapes(const std::vector<MatShape> &inputs,
                         const int requiredOutputs,
                         std::vector<MatShape> &outputs,
                         std::vector<MatShape> &internals) const
    {
        CV_Assert(!usePeephole && blobs.size() == 3 || usePeephole && blobs.size() == 6);
        CV_Assert(inputs.size() == 1);
        const MatShape& inp0 = inputs[0];

        const Mat &Wh = blobs[0], &Wx = blobs[1];
        int _numOut = Wh.size[1];
        int _numInp = Wx.size[1];
        MatShape outTailShape_(outTailShape), outResShape;

        if (!outTailShape_.empty())
            CV_Assert(total(outTailShape_) == _numOut);
        else
            outTailShape_.assign(1, _numOut);

        int _numTimeStamps, _numSamples;
        if (useTimestampDim)
        {
            CV_Assert(inp0.size() >= 2 && total(inp0, 2) == _numInp);
            _numTimeStamps = inp0[0];
            _numSamples = inp0[1];
            outResShape.push_back(_numTimeStamps);
        }
        else
        {
            CV_Assert(inp0.size() >= 2 && total(inp0, 1) == _numInp);
            _numTimeStamps = 1;
            _numSamples = inp0[0];
        }

        outResShape.push_back(_numSamples);
        outResShape.insert(outResShape.end(), outTailShape_.begin(), outTailShape_.end());

        size_t noutputs = produceCellOutput ? 2 : 1;
        outputs.assign(noutputs, outResShape);

        internals.assign(1, shape(_numSamples, _numOut)); // hInternal
        internals.push_back(shape(_numSamples, _numOut)); // cInternal
        internals.push_back(shape(_numSamples, 1)); // dummyOnes
        internals.push_back(shape(_numSamples, 4*_numOut)); // gates

        return false;
    }

    void finalize(const std::vector<Mat*> &input, std::vector<Mat> &output)
    {
        CV_Assert(!usePeephole && blobs.size() == 3 || usePeephole && blobs.size() == 6);
        CV_Assert(input.size() == 1);
        const Mat& inp0 = *input[0];

        Mat &Wh = blobs[0], &Wx = blobs[1];
        int numOut = Wh.size[1];
        int numInp = Wx.size[1];

        if (!outTailShape.empty())
            CV_Assert(total(outTailShape) == numOut);
        else
            outTailShape.assign(1, numOut);

        if (useTimestampDim)
        {
            CV_Assert(inp0.dims >= 2 && (int)inp0.total(2) == numInp);
            numTimeStamps = inp0.size[0];
            numSamples = inp0.size[1];
        }
        else
        {
            CV_Assert(inp0.dims >= 2 && (int)inp0.total(1) == numInp);
            numTimeStamps = 1;
            numSamples = inp0.size[0];
        }

        outTsShape.clear();
        outTsShape.push_back(numSamples);
        outTsShape.insert(outTsShape.end(), outTailShape.begin(), outTailShape.end());

        allocated = true;
    }

    void forward(std::vector<Mat*> &input, std::vector<Mat> &output, std::vector<Mat> &internals)
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        const Mat &Wh = blobs[0];
        const Mat &Wx = blobs[1];
        const Mat &bias = blobs[2];

        int numOut = Wh.size[1];

        Mat hInternal = internals[0], cInternal = internals[1],
                dummyOnes = internals[2], gates = internals[3];
        hInternal.setTo(0.);
        cInternal.setTo(0.);
        dummyOnes.setTo(1.);

        int numSamplesTotal = numTimeStamps*numSamples;
        Mat xTs = input[0]->reshape(1, numSamplesTotal);

        Mat hOutTs = output[0].reshape(1, numSamplesTotal);
        Mat cOutTs = produceCellOutput ? output[1].reshape(1, numSamplesTotal) : Mat();

        for (int ts = 0; ts < numTimeStamps; ts++)
        {
            Range curRowRange(ts*numSamples, (ts + 1)*numSamples);
            Mat xCurr = xTs.rowRange(curRowRange);

            gemm(xCurr, Wx, 1, gates, 0, gates, GEMM_2_T);      // Wx * x_t
            gemm(hInternal, Wh, 1, gates, 1, gates, GEMM_2_T);  //+Wh * h_{t-1}
            gemm(dummyOnes, bias, 1, gates, 1, gates);          //+b

            Mat gateI = gates.colRange(0*numOut, 1*numOut);
            Mat gateF = gates.colRange(1*numOut, 2*numOut);
            Mat gateO = gates.colRange(2*numOut, 3*numOut);
            Mat gateG = gates.colRange(3*numOut, 4*numOut);

            if (forgetBias)
                add(gateF, forgetBias, gateF);

            if (usePeephole)
            {
                Mat gatesIF = gates.colRange(0, 2*numOut);
                gemm(cInternal, blobs[3], 1, gateI, 1, gateI);
                gemm(cInternal, blobs[4], 1, gateF, 1, gateF);
                sigmoid(gatesIF, gatesIF);
            }
            else
            {
                Mat gatesIFO = gates.colRange(0, 3*numOut);
                sigmoid(gatesIFO, gatesIFO);
            }

            tanh(gateG, gateG);

            //compute c_t
            multiply(gateF, cInternal, gateF);  // f_t (*) c_{t-1}
            multiply(gateI, gateG, gateI);      // i_t (*) g_t
            add(gateF, gateI, cInternal);       // c_t = f_t (*) c_{t-1} + i_t (*) g_t

            if (useCellClip)
            {
                min(cInternal, cellClip, cInternal);
                max(cInternal, -cellClip, cInternal);
            }
            if (usePeephole)
            {
                gemm(cInternal, blobs[5], 1, gateO, 1, gateO);
                sigmoid(gateO, gateO);
            }

            //compute h_t
            tanh(cInternal, hInternal);
            multiply(gateO, hInternal, hInternal);

            //save results in output blobs
            hInternal.copyTo(hOutTs.rowRange(curRowRange));
            if (produceCellOutput)
                cInternal.copyTo(cOutTs.rowRange(curRowRange));
        }
    }
};

Ptr<LSTMLayer> LSTMLayer::create(const LayerParams& params)
{
    return Ptr<LSTMLayer>(new LSTMLayerImpl(params));
}

int LSTMLayer::inputNameToIndex(String inputName)
{
    if (inputName.toLowerCase() == "x")
        return 0;
    return -1;
}

int LSTMLayer::outputNameToIndex(String outputName)
{
    if (outputName.toLowerCase() == "h")
        return 0;
    else if (outputName.toLowerCase() == "c")
        return 1;
    return -1;
}


class RNNLayerImpl : public RNNLayer
{
    int numX, numH, numO;
    int numSamples, numTimestamps, numSamplesTotal;
    int dtype;
    Mat Whh, Wxh, bh;
    Mat Who, bo;
    bool produceH;

public:

    RNNLayerImpl(const LayerParams& params)
        : numX(0), numH(0), numO(0), numSamples(0), numTimestamps(0), numSamplesTotal(0), dtype(0)
    {
        setParamsFrom(params);
        type = "RNN";
        produceH = false;
    }

    void setProduceHiddenOutput(bool produce = false)
    {
        produceH = produce;
    }

    void setWeights(const Mat &W_xh, const Mat &b_h, const Mat &W_hh, const Mat &W_ho, const Mat &b_o)
    {
        CV_Assert(W_hh.dims == 2 && W_xh.dims == 2);
        CV_Assert(W_hh.size[0] == W_xh.size[0] && W_hh.size[0] == W_hh.size[1] && (int)b_h.total() == W_xh.size[0]);
        CV_Assert(W_ho.size[0] == (int)b_o.total());
        CV_Assert(W_ho.size[1] == W_hh.size[1]);

        blobs.resize(5);
        blobs[0] = Mat(W_xh.clone());
        blobs[1] = Mat(b_h.clone());
        blobs[2] = Mat(W_hh.clone());
        blobs[3] = Mat(W_ho.clone());
        blobs[4] = Mat(b_o.clone());
    }

    bool getMemoryShapes(const std::vector<MatShape> &inputs,
                         const int requiredOutputs,
                         std::vector<MatShape> &outputs,
                         std::vector<MatShape> &internals) const
    {
        CV_Assert(inputs.size() >= 1 && inputs.size() <= 2);

        Mat Who_ = blobs[3];
        Mat Wxh_ = blobs[0];

        int numTimestamps_ = inputs[0][0];
        int numSamples_ = inputs[0][1];

        int numO_ = Who_.rows;
        int numH_ = Wxh_.rows;

        outputs.clear();
        int dims[] = {numTimestamps_, numSamples_, numO_};
        outputs.push_back(shape(dims, 3));
        dims[2] = numH_;
        if (produceH)
            outputs.push_back(shape(dims, 3));

        internals.assign(2, shape(numSamples_, numH_));
        internals.push_back(shape(numSamples_, 1));

        return false;
    }

    void finalize(const std::vector<Mat*> &input, std::vector<Mat> &output)
    {
        CV_Assert(input.size() >= 1 && input.size() <= 2);

        Wxh = blobs[0];
        bh  = blobs[1];
        Whh = blobs[2];
        Who = blobs[3];
        bo  = blobs[4];

        numH = Wxh.rows;
        numX = Wxh.cols;
        numO = Who.rows;

        const Mat& inp0 = *input[0];

        CV_Assert(inp0.dims >= 2);
        CV_Assert(inp0.total(2) == numX);
        dtype = CV_32F;
        CV_Assert(inp0.type() == dtype);
        numTimestamps = inp0.size[0];
        numSamples = inp0.size[1];
        numSamplesTotal = numTimestamps * numSamples;

        bh = bh.reshape(1, 1); //is 1 x numH Mat
        bo = bo.reshape(1, 1); //is 1 x numO Mat
    }

    void reshapeOutput(std::vector<Mat> &output)
    {
        output.resize(produceH ? 2 : 1);
        int sz0[] = { numTimestamps, numSamples, numO };
        output[0].create(3, sz0, dtype);
        if (produceH)
        {
            int sz1[] = { numTimestamps, numSamples, numH };
            output[1].create(3, sz1, dtype);
        }
    }

    void forward(std::vector<Mat*> &input, std::vector<Mat> &output, std::vector<Mat> &internals)
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        Mat xTs = input[0]->reshape(1, numSamplesTotal);
        Mat oTs = output[0].reshape(1, numSamplesTotal);
        Mat hTs = produceH ? output[1].reshape(1, numSamplesTotal) : Mat();
        Mat hCurr = internals[0];
        Mat hPrev = internals[1];
        Mat dummyBiasOnes = internals[2];

        hPrev.setTo(0.);
        dummyBiasOnes.setTo(1.);

        for (int ts = 0; ts < numTimestamps; ts++)
        {
            Range curRowRange = Range(ts * numSamples, (ts + 1) * numSamples);
            Mat xCurr = xTs.rowRange(curRowRange);

            gemm(hPrev, Whh, 1, hCurr, 0, hCurr, GEMM_2_T); // W_{hh} * h_{prev}
            gemm(xCurr, Wxh, 1, hCurr, 1, hCurr, GEMM_2_T); //+W_{xh} * x_{curr}
            gemm(dummyBiasOnes, bh, 1, hCurr, 1, hCurr);    //+bh
            tanh(hCurr, hPrev);

            Mat oCurr = oTs.rowRange(curRowRange);
            gemm(hPrev, Who, 1, oCurr, 0, oCurr, GEMM_2_T); // W_{ho} * h_{prev}
            gemm(dummyBiasOnes, bo, 1, oCurr, 1, oCurr);    //+b_o
            tanh(oCurr, oCurr);

            if (produceH)
                hPrev.copyTo(hTs.rowRange(curRowRange));
        }
    }
};

CV_EXPORTS_W Ptr<RNNLayer> RNNLayer::create(const LayerParams& params)
{
    return Ptr<RNNLayer>(new RNNLayerImpl(params));
}

}
}
