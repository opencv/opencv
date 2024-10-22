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
#include <cmath>
#include <opencv2/dnn/shape_utils.hpp>
#include "layers_common.hpp"
#include "../net_impl.hpp"

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

typedef void (*ActivationFunction)(const Mat &src, Mat &dst);
static ActivationFunction get_activation_function(const String& activation) {
    if (activation == "Tanh"){
        return tanh;
    }
    else if (activation == "Sigmoid"){
        return sigmoid;
    }
    else
    {
        CV_Error(Error::StsNotImplemented,
                 cv::format("Activation function [%s] for layer LSTM  is not supported", activation.c_str()));
    }
}


class LSTM2LayerImpl CV_FINAL : public LSTM2Layer
{
    int numTimeStamps, numSamples, numHidden;
    MatShape outTailShape;  //shape of single output sample
    enum layout_t : int {
        SEQ_BATCH_HID = 0,
        BATCH_SEQ_HID = 1
    };

    bool useTimestampDim;
    bool produceCellOutput;
    bool reverse;   // If true, go in negative direction along the time axis
    bool bidirectional;  // If true, produces both forward and reversed directions along time axis
    float forgetBias, cellClip;
    layout_t layout;  // If layout == BATCH_SEQ_HID, uses batch_size x seq_length x num_hidden for input and output
                      // else uses seq_length x batch_size x num_hidden

    ActivationFunction f_activation;
    ActivationFunction g_activation;
    ActivationFunction h_activation;
    bool isDefaultActivations{true};

    public:
        LSTM2LayerImpl(const LayerParams& params)
        {
            produceCellOutput = params.get<bool>("produce_cell_output", false);
            bidirectional = params.get<bool>("bidirectional", false);
            layout = (layout_t) params.get<int>("layout", SEQ_BATCH_HID);
            reverse = params.get<bool>("reverse", false);
            useTimestampDim = params.get<bool>("use_timestamp_dim", true);
            numHidden = params.get<int>("hidden_size", 1);
            forgetBias = params.get<float>("forget_bias", 0.0f);
            cellClip = params.get<float>("cell_clip", 0.0f);


            // read activations
            DictValue activations = params.get<DictValue>("activations", DictValue(String()));
            if (activations.size() == 1) // if activations wasn't specified use default
            {
                f_activation = sigmoid;
                g_activation = tanh;
                h_activation = tanh;
                isDefaultActivations = true;
            } else {
                CV_Assert(activations.size() == 3);
                f_activation = get_activation_function(activations.getStringValue(0));
                g_activation = get_activation_function(activations.getStringValue(1));
                h_activation = get_activation_function(activations.getStringValue(2));
                isDefaultActivations = activations.getStringValue(0) == "Sigmoid"
                                    && activations.getStringValue(1) == "Tanh"
                                    && activations.getStringValue(2) == "Tanh";
            }



            outTailShape.clear();
        }

        bool getMemoryShapes(const std::vector<MatShape> &inputs,
                         const int requiredOutputs,
                         std::vector<MatShape> &outputs,
                         std::vector<MatShape> &internals) const CV_OVERRIDE
        {
            std::cout << "================" << std::endl;
            std::cout << "getMemoryShapes" << std::endl;
            Net::Impl* netimpl_ = getNetImpl(this);
            CV_Assert(netimpl_);

            size_t ninputs = this->inputs.size();
            size_t noutputs = this->outputs.size();
            std::cout << "ninputs: " << ninputs << std::endl;
            std::cout << "noutputs: " << noutputs << std::endl;

            // for (int i = 0; i < ninputs; i++)
            // {
            //     std::cout << "input " << netimpl_->argName(this->inputs[i]) << " is constant:" << netimpl_->isConstArg(this->inputs[i]) << " shape: " << inputs[i] << std::endl;
            // }

            const MatShape& inp0 = inputs[0];
            const MatShape& Wx = inputs[1];
            const MatShape& Wh = inputs[2];
            const MatShape& bias = inputs[3];
            const MatShape& seq_len = inputs[4];
            const MatShape& initial_h = inputs[5];
            const MatShape& initial_c = inputs[6];

            std::cout << "inp0: " << inp0 << " is constant: " << netimpl_->isConstArg(this->inputs[0]) << std::endl;
            std::cout << "Wx: " << Wx << " is constant: " << netimpl_->isConstArg(this->inputs[1]) << std::endl;
            std::cout << "Wh: " << Wh << " is constant: " << netimpl_->isConstArg(this->inputs[2]) << std::endl;
            std::cout << "bias: " << bias << " is constant: " << netimpl_->isConstArg(this->inputs[3]) << std::endl;
            std::cout << "seq_len: " << seq_len << " is constant: " << netimpl_->isConstArg(this->inputs[4]) << std::endl;
            std::cout << "initial_h: " << initial_h << " is constant: " << netimpl_->isConstArg(this->inputs[5]) << std::endl;
            std::cout << "initial_c: " << initial_c << " is constant: " << netimpl_->isConstArg(this->inputs[6]) << std::endl;
            std::cout << "output size: " << outputs.size() << std::endl;

            int _numOut = Wh[2];
            int _numInp = Wx[2];
            MatShape outTailShape_(outTailShape), outResShape;

            if (!outTailShape_.empty())
                CV_Assert(total(outTailShape_) == _numOut);
            else
                outTailShape_.assign(1, _numOut);


            int _numSamples;
            if (useTimestampDim)
            {
                CV_Assert(inp0.size() >= 2 && total(inp0, 2) == _numInp);
                if (layout == SEQ_BATCH_HID) {
                    _numSamples = inp0[1];
                    outResShape.push_back(inp0[0]);
                } else {
                    _numSamples = inp0[0];
                    outResShape.push_back(inp0[1]);
                }
            }
            else
            {
                CV_Assert(inp0.size() >= 2 && total(inp0, 1) == _numInp);
                _numSamples = inp0[0];
            }

            outResShape.push_back(_numSamples);
            outResShape.insert(outResShape.end(), outTailShape_.begin(), outTailShape_.end());
            outResShape.back() *= (1 + static_cast<int>(bidirectional));

            outputs.assign(1, outResShape);
            if (produceCellOutput)
            {
                // the producer is ONNX, so CellState is different
                // if (!originalBlobs.empty())
                // {
                    int shp[] = {(1 + static_cast<int>(bidirectional)), _numSamples, numHidden};
                    MatShape newShape(shp, shp + sizeof(shp)/sizeof(shp[0]));
                    outputs.push_back(newShape);
                    outputs.push_back(newShape);
                // }
                // else
                // {
                    // outputs.push_back(outResShape);
                // }
            }

            // TEMPORARY DUMMY OUTPUT
            // for (int i = noutputs - outputs.size(); i <= noutputs; i++)
            //     outputs.push_back(outResShape);

            // lets print output names
            // for (int i = 0; i < outputs.size(); i++) {
            //     std::cout << "output[" << i << "] name: " << netimpl_->argName(this->outputs[i]) << std::endl;
            // }


            internals.assign(1, shape(_numSamples, _numOut)); // hInternal
            internals.push_back(shape(_numSamples, _numOut)); // cInternal
            internals.push_back(shape(_numSamples, 1)); // dummyOnes
            internals.push_back(shape(_numSamples, 4*_numOut)); // gates

            for (int i = 0; i < outputs .size(); i++) {
                std::cout << "output[" << i << "] shape: " << outputs[i] << std::endl;
            }
            for (int i = 0; i < internals.size(); i++) {
                std::cout << "internal[" << i << "] shape: " << internals[i] << std::endl;
            }


            std::cout << "================" << std::endl;
            std::cout << "getMemoryShapes done" << std::endl;
            return false;
        }

        void getTypes(const std::vector<MatType>& inputs,
                      const int requiredOutputs,
                      const int requiredInternals,
                      std::vector<MatType>& outputs,
                      std::vector<MatType>& internals) const CV_OVERRIDE
        {
            std::cout << "getTypes" << std::endl;
            outputs.assign(requiredOutputs, inputs[0]);
            internals.assign(4, inputs[0]);
            // CV_Assert(requiredInternals == 0);
            // internals.clear();
            std::cout << "getTypes done" << std::endl;
        }

        void forward(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr, OutputArrayOfArrays internals_arr) CV_OVERRIDE
        {

            std::cout << "================" << std::endl;
            std::cout << "forward" << std::endl;
            std::vector<Mat> input, output, internals;
            inputs_arr.getMatVector(input);
            outputs_arr.getMatVector(output);
            internals_arr.getMatVector(internals);

            for (const auto& inp : input)
                std::cout << "input shape: " << inp.size << std::endl;

            for (const auto& intern : internals)
                std::cout << "internal shape: " << intern.size << std::endl;

            // extract inputs and weights matrices
            // Mat &Wx = input[1];
            // Mat &Wh = input[2];
            // Mat &bias = input[3];
            // Mat &seq_len = input[4];
            // Mat &h_0 = input[5];
            // Mat &c_0 = input[6];


            // std::cout << "Wx shape: " << Wx.size << std::endl;
            // std::cout << "Wh shape: " << Wh.size << std::endl;
            // std::cout << "bias shape: " << bias.size << std::endl;
            // std::cout << "h_0 shape: " << h_0.size << std::endl;
            // std::cout << "c_0 shape: " << c_0.size << std::endl;

            std::cout << "transforming blobs" << std::endl;
            // transform weights matrices similar to old LSTM parser
            std::vector<Mat> blobs_ = {input[1], input[2], input[3], input[5], input[6]};
            // std::vector<Mat> blobs_ = {Wx, Wh, bias, h_0, c_0};
            transformBlobs(blobs_);

            Mat &Wx = blobs_[0];
            Mat &Wh = blobs_[1];
            Mat &bias = blobs_[2];
            Mat &h_0 = blobs_[3];
            Mat &c_0 = blobs_[4];

            std::cout << "Wx sum: " << cv::sum(Wx)[0] << std::endl;
            std::cout << "Wh sum: " << cv::sum(Wh)[0] << std::endl;
            std::cout << "bias sum: " << cv::sum(bias)[0] << std::endl;
            std::cout << "blobs transformed" << std::endl;
            std::cout << "blobs size: " << blobs_.size() << std::endl;

            for (const auto& blob : blobs_)
                std::cout << "blob shape: " << blob.size << std::endl;

            int numOut = Wh.size[1];
            int numInp = Wx.size[1];

            Mat cOut = produceCellOutput ? output[0].clone() : Mat();
            Mat hInternal = internals[0], cInternal = internals[1],
                    dummyOnes = internals[2], gates = internals[3];
            h_0.copyTo(hInternal);
            c_0.copyTo(cInternal);
            dummyOnes.setTo(1.);
            std::cout << "input[0] shape: " << input[0].size << std::endl;

            if (useTimestampDim)
            {
                CV_Assert(input[0].dims >= 2 && (int)input[0].total(2) == numInp);
                if (layout == SEQ_BATCH_HID){
                    numTimeStamps = input[0].size[0];
                    numSamples = input[0].size[1];
                }else{
                    numTimeStamps = input[0].size[1];
                    numSamples = input[0].size[0];
                }
            }
            else
            {
                CV_Assert(input[0].dims >= 2 && (int)input[0].total(1) == numInp);
                numTimeStamps = 1;
                numSamples = input[0].size[0];
            }

            int numSamplesTotal = numTimeStamps*numSamples;
            Mat xTs = input[0].reshape(1, numSamplesTotal);

            std::cout << "input[0] sum: " << cv::sum(xTs) << std::endl;
            std::cout << "hInternal sum: " << cv::sum(hInternal) << std::endl;
            std::cout << "cInternal sum: " << cv::sum(cInternal) << std::endl;
            std::cout << "dummyOnes sum: " << cv::sum(dummyOnes) << std::endl;
            std::cout << "gates sum: " << cv::sum(gates) << std::endl;

            Mat hOutTs = output[0].reshape(1, numSamplesTotal);
            Mat cOutTs;
            if (produceCellOutput)
            {
                cOutTs = cOut.reshape(1, numSamplesTotal);
                // cOutTs = cOutTs.colRange(i * cOutTs.cols / numDirs, (i + 1) * cOutTs.cols / numDirs);
            }
            std::cout << "numTimeStamps: " << numTimeStamps << std::endl;
            std::cout << "numSamples: " << numSamples << std::endl;
            for (int ts = 0; ts < numTimeStamps; ts++)
            {
                Range curRowRange(ts*numSamples, (ts + 1)*numSamples);
                Mat xCurr = xTs.rowRange(curRowRange);
                std::cout << "xCurr shape: " << xCurr.size << std::endl;
                std::cout << "Wx shape: " << Wx.size << std::endl;
                std::cout << "gates shape: " << gates.size << std::endl;
                gemm(xCurr, Wx, 1, gates, 0, gates, GEMM_2_T);      // Wx * x_t
                std::cout << "gates sum: " << cv::sum(gates) << std::endl;

                std::cout << "dummyOnes shape: " << dummyOnes.size << std::endl;
                std::cout << "bias shape: " << bias.size << std::endl;
                gemm(dummyOnes, bias, 1, gates, 1, gates);          //+b
                std::cout << "gates sum: " << cv::sum(gates) << std::endl;

                std::cout << "hInternal shape: " << hInternal.size << std::endl;
                std::cout << "Wh shape: " << Wh.size << std::endl;
                gemm(hInternal, Wh, 1, gates, 1, gates, GEMM_2_T);  //+Wh * h_{t-1}
                std::cout << "gates sum: " << cv::sum(gates) << std::endl;
                std::cout << "numOut: " << numOut << std::endl;
                Mat gateI = gates.colRange(0*numOut, 1*numOut);
                Mat gateF = gates.colRange(1*numOut, 2*numOut);
                Mat gateO = gates.colRange(2*numOut, 3*numOut);
                Mat gateG = gates.colRange(3*numOut, 4*numOut);

                if (forgetBias)
                    add(gateF, forgetBias, gateF);

                // if (usePeephole)
                // {
                //     Mat gatesIF = gates.colRange(0, 2*numOut);
                //     gemm(cInternal, pI, 1, gateI, 1, gateI);
                //     gemm(cInternal, pF, 1, gateF, 1, gateF);
                //     f_activation(gatesIF, gatesIF);
                // }
                // else
                // {
                    Mat gatesIFO = gates.colRange(0, 3*numOut);
                    f_activation(gatesIFO, gatesIFO);
                // }

                g_activation(gateG, gateG);


                //compute c_t
                multiply(gateF, cInternal, gateF);  // f_t (*) c_{t-1}
                multiply(gateI, gateG, gateI);      // i_t (*) g_t
                add(gateF, gateI, cInternal);       // c_t = f_t (*) c_{t-1} + i_t (*) g_t

                //compute h_t
                h_activation(cInternal, hInternal);
                multiply(gateO, hInternal, hInternal);

                //save results in output blobs
                hInternal.copyTo(hOutTs.rowRange(curRowRange));
                std::cout << " ===> " << std::endl;
                std::cout << "produceCellOutput: " << produceCellOutput << std::endl;
                std::cout << "cInternal shape: " << cInternal.size << std::endl;
                std::cout << "cOutTs shape: " << cOutTs.size << std::endl;

                std::cout << "hInternal shape: " << hInternal.size << std::endl;
                std::cout << "hOutTs shape: " << hOutTs.size << std::endl;

                std::cout << "cInternal sum: " << cv::sum(cInternal) << std::endl;
                std::cout << "cOutTs sum: " << cv::sum(cOutTs) << std::endl;
                std::cout << "hInternal sum: " << cv::sum(hInternal) << std::endl;
                std::cout << "hOutTs sum: " << cv::sum(hOutTs) << std::endl;

                if (produceCellOutput)
                    cInternal.copyTo(cOutTs.rowRange(curRowRange));
                std::cout << "cOutTs sum: " << cv::sum(cOutTs) << std::endl;
            }





            std::cout << "output[0] shape: " << output[0].size << std::endl;
            std::cout << "output[0] sum: " << cv::sum(output[0]) << std::endl;
            std::cout << "output[1] shape: " << output[1].size << std::endl;
            std::cout << "output[1] sum: " << cv::sum(output[1]) << std::endl;
            std::cout << "forward done" << std::endl;
            std::cout << "================" << std::endl;
        }

        void transformBlobs(std::vector<Mat>& blobs)
        {
            std::cout << "transformBlobs" << std::endl;
            std::cout << "blobs[0] shape: " << blobs.size() << std::endl;
            Mat &Wx = blobs[0];
            Mat &Wh = blobs[1];
            Mat &b = blobs[2];

            std::vector<Mat> cudaWorkaround;
            cudaWorkaround.push_back(Wx.clone());
            cudaWorkaround.push_back(Wh.clone());
            cudaWorkaround.push_back(b.clone());

            const int numHidden = Wh.size[2];

            Mat h0, c0;
            // check weather input is dynamic or not: hx, cx are given by user.
            // Resahpe if only they are given
            if (!blobs[3].empty()){
                h0 = blobs[3];
                h0 = h0.reshape(1, h0.size[0] * h0.size[1]);
            }
            if (!blobs[4].empty()){
                c0 = blobs[4];
                c0 = c0.reshape(1, c0.size[0] * c0.size[1]);
            }

            b = b.reshape(1, b.size[0]);
            Mat bx = b.colRange(0, b.cols / 2);
            Mat bh = b.colRange(b.cols / 2, b.cols);
            b = bx + bh;

            auto toIFOC = [] (Mat& in) {
                int first = in.size[0];
                int rest = in.total() / first / 4;
                // every weight blob contains weights for Input, Output, Forget and Cell gates
                Mat m = in.reshape(1, {first, 4, rest});
                Mat outputGate = m.col(1);
                Mat forgetGate = m.col(2);
                std::swap_ranges(outputGate.begin<float>(), outputGate.end<float>(), forgetGate.begin<float>());
            };

            toIFOC(Wx);
            toIFOC(Wh);
            toIFOC(b);

            std::cout << "Wx shape: " << Wx.size << std::endl;
            std::cout << "Wh shape: " << Wh.size << std::endl;

            Wx = Wx.reshape(1, Wx.size[0] * Wx.size[1]);
            Wh = Wh.reshape(1, Wh.size[0] * Wh.size[1]);

            std::cout << "Wx shape: " << Wx.size << std::endl;
            std::cout << "Wh shape: " << Wh.size << std::endl;
            std::cout << "b shape: " << b.size << std::endl;

            blobs[0] = Wx;
            blobs[1] = Wh;
            blobs[2] = b.reshape(1, 1);
            std::cout << "blobs[0] shape: " << blobs[0].size << std::endl;
            std::cout << "blobs[1] shape: " << blobs[1].size << std::endl;
            std::cout << "blobs[2] shape: " << blobs[2].size << std::endl;

            if (!blobs[3].empty()){
                blobs[3] = h0;
            }
            if (!blobs[4].empty()){
                blobs[4] = c0;
            }

            if (blobs.size() == 5) {
                // so that future patch removing copies can leave all indexing as is
                // blobs.insert(blobs.begin(), cudaWorkaround.begin(), cudaWorkaround.end());
                return;
            }

            Mat P = blobs[5];
            blobs[5] = P.colRange(0, numHidden);
            blobs[5] = blobs[5].clone().reshape(1, blobs[5].total());  // Single column.
            blobs[5] = Mat::diag(blobs[5]);

            blobs.push_back(P.colRange(numHidden, 2 * numHidden));
            blobs[6] = blobs[6].clone().reshape(1, blobs[6].total());  // Single column.
            blobs[6] = Mat::diag(blobs[6]);

            blobs.push_back(P.colRange(2 * numHidden, 3 * numHidden));
            blobs[7] = blobs[7].clone().reshape(1, blobs[7].total());  // Single column.
            blobs[7] = Mat::diag(blobs[7]);

            // so that future patch removing copies can leave all indexing as is
            blobs.insert(blobs.begin(), cudaWorkaround.begin(), cudaWorkaround.end());
        }
};

Ptr<LSTM2Layer> LSTM2Layer::create(const LayerParams& params)
{
    return Ptr<LSTM2Layer>(new LSTM2LayerImpl(params));
};

}}
