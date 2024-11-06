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
    int seqLenth, batchSize, numHidden;
    MatShape outTailShape;  //shape of single output sample
    enum layout_t : int {
        SEQ_BATCH_HID = 0,
        BATCH_SEQ_HID = 1
    };

    bool blobsInitializers{false};
    bool useTimestampDim;
    bool produceCellOutput, produceOutputYh;
    bool useCellClip, usePeephole;
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
            numHidden = params.get<int>("hidden_size", 1);
            layout = (layout_t) params.get<int>("layout", SEQ_BATCH_HID);

            produceCellOutput = params.get<bool>("produce_cell_output", false);
            produceOutputYh = params.get<bool>("produce_output_yh", false);
            bidirectional = params.get<bool>("bidirectional", false);
            reverse = params.get<bool>("reverse", false);
            useTimestampDim = params.get<bool>("use_timestamp_dim", true);
            usePeephole = params.get<bool>("use_peephole", false);
            useCellClip = params.get<bool>("use_cell_clip", false);

            forgetBias = params.get<float>("forget_bias", 0.0f);
            cellClip = params.get<float>("cell_clip", 0.0f);

            CV_Assert(!reverse || !bidirectional);

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

        void weightsConstants(){
            Net::Impl* netimpl_ = getNetImpl(this);
            CV_Assert(netimpl_);
            blobsInitializers = netimpl_->isConstArg(this->inputs[1]) && netimpl_->isConstArg(this->inputs[2]) && netimpl_->isConstArg(this->inputs[3]);
        }

        bool getMemoryShapes(const std::vector<MatShape> &inputs,
                         const int requiredOutputs,
                         std::vector<MatShape> &outputs,
                         std::vector<MatShape> &internals) const CV_OVERRIDE
        {
            const MatShape& inp0 = inputs[0];
            const MatShape& Wx = inputs[1];
            const MatShape& Wh = inputs[2];

            int _hidSize = Wh[2];
            int _inpSize = Wx[2];
            MatShape outTailShape_(outTailShape), outResShape;

            if (!outTailShape_.empty())
                CV_Assert(total(outTailShape_) == _hidSize);
            else
                outTailShape_.assign(1, _hidSize);

            // compute output shape of y
            // figure out batch size
            int _batchSize;
            int _seqLen;
            if (useTimestampDim)
            {
                CV_Assert(inp0.size() >= 2 && total(inp0, 2) == _inpSize);
                if (layout == SEQ_BATCH_HID) {
                    _batchSize = inp0[1];
                    _seqLen = inp0[0];
                } else {
                    _batchSize = inp0[0];
                    _seqLen = inp0[1];
                }
                outResShape.push_back(_seqLen);
            }
            else
            {
                CV_Assert(inp0.size() >= 2 && total(inp0, 1) == _inpSize);
                _batchSize = inp0[0];
            }

            outResShape.push_back(1 + static_cast<int>(bidirectional));
            outResShape.push_back(_batchSize);
            outResShape.push_back(_hidSize);
            outputs.assign(1, outResShape);

            // compute output shape of yc
            if (produceCellOutput)
            {
                    int shp[] = {(1 + static_cast<int>(bidirectional)), _batchSize, numHidden};
                    MatShape newShape(shp, shp + sizeof(shp)/sizeof(shp[0]));
                    outputs.push_back(newShape);
            }

            // compute output shape of yh
            if (produceOutputYh)
            {
                int shp[] = {1 + static_cast<int>(bidirectional), _batchSize, numHidden};
                MatShape newShape(shp, shp + sizeof(shp)/sizeof(shp[0]));
                outputs.push_back(newShape);
            }

            // internal shapes need during forward pass
            internals.assign(1, shape(_batchSize, _hidSize)); // hInternal
            internals.push_back(shape(_batchSize, _hidSize)); // cInternal
            internals.push_back(shape(_batchSize, 1)); // dummyOnes
            internals.push_back(shape(_batchSize, 4*_hidSize)); // gates


            return false;
        }

        void getTypes(const std::vector<MatType>& inputs,
                      const int requiredOutputs,
                      const int requiredInternals,
                      std::vector<MatType>& outputs,
                      std::vector<MatType>& internals) const CV_OVERRIDE
        {
            outputs.assign(requiredOutputs, inputs[0]);
            internals.assign(4, inputs[0]);
        }


        void forward(InputArrayOfArrays inputs_arr,
                     OutputArrayOfArrays outputs_arr,
                     OutputArrayOfArrays internals_arr) CV_OVERRIDE
        {

            std::vector<Mat> input, output, internals;

            inputs_arr.getMatVector(input);
            outputs_arr.getMatVector(output);
            internals_arr.getMatVector(internals);


            // set outputs to 0
            for (auto& out : output)
                out.setTo(0);

            // transform weights matrices similar to old LSTM parser
            std::vector<Mat> blobs_ = {input[1], input[2], input[3], input[5], input[6]};
            if (usePeephole){
                blobs_.push_back(input[7]);
            }


            transformBlobs(blobs_);


            //TODO: Maybe remove this complitely
            // weightsConstants(); //TODO: Call inside the constructor??
            // const bool needYcTransform = blobsInitializers ? true : false; // if the producer is onnx

            const int numDirs = 1 + static_cast<int>(bidirectional);

            Mat cOut = produceCellOutput ? output[0].clone() : Mat();
            Mat hOut = produceOutputYh ? output[0].clone() : Mat();
            std::vector<Mat> tempOutputs;
            for (int i = 0; i < numDirs; i++)
            {

                // get weights
                Mat Wx = blobs_[0].clone();
                Mat Wh = blobs_[1].clone();
                Mat bias = blobs_[2].clone();
                Mat h_0 = blobs_[3].clone();
                Mat c_0 = blobs_[4].clone();



                // get hidden and input sizes
                int hidSize = Wh.size[1];
                int inpSize = Wx.size[1];

                Wh = Wh.rowRange(i * Wh.rows / numDirs, (i + 1) * Wh.rows / numDirs);
                Wx = Wx.rowRange(i * Wx.rows / numDirs, (i + 1) * Wx.rows / numDirs);
                bias = bias.colRange(i * bias.cols / numDirs, (i + 1) * bias.cols / numDirs);
                h_0 = h_0.rowRange(i * h_0.rows / numDirs, (i + 1) * h_0.rows / numDirs);
                c_0 = c_0.rowRange(i * c_0.rows / numDirs, (i + 1) * c_0.rows / numDirs);

                Mat pI, pF, pO;
                if (usePeephole)
                {
                    pI = blobs_[5];
                    pF = blobs_[6];
                    pO = blobs_[7];

                    pI = pI.rowRange(i * pI.rows / numDirs, (i + 1) * pI.rows / numDirs);
                    pI = pI.colRange(i * pI.cols / numDirs, (i + 1) * pI.cols / numDirs);

                    pF = pF.rowRange(i * pF.rows / numDirs, (i + 1) * pF.rows / numDirs);
                    pF = pF.colRange(i * pF.cols / numDirs, (i + 1) * pF.cols / numDirs);

                    pO = pO.rowRange(i * pO.rows / numDirs, (i + 1) * pO.rows / numDirs);
                    pO = pO.colRange(i * pO.cols / numDirs, (i + 1) * pO.cols / numDirs);
                }


                Mat hInternal = internals[0],
                    cInternal = internals[1],
                    dummyOnes = internals[2],
                    gates = internals[3];

                h_0.copyTo(hInternal);
                c_0.copyTo(cInternal);
                dummyOnes.setTo(1.);
                gates.setTo(0.);

                // determine seqLen and batchSize
                // TODO: could be moved out of the loop
                if (useTimestampDim)
                {
                    CV_Assert(input[0].dims >= 2 && (int)input[0].total(2) == inpSize);
                    if (layout == SEQ_BATCH_HID){
                        seqLenth = input[0].size[0];
                        batchSize = input[0].size[1];
                    }else{
                        seqLenth = input[0].size[1];
                        batchSize = input[0].size[0];
                    }
                } else {
                    CV_Assert(input[0].dims >= 2 && (int)input[0].total(1) == inpSize);
                    seqLenth = 1;
                    batchSize = input[0].size[0];
                }

                int batchSizeTotal = seqLenth*batchSize;
                Mat xTs = input[0].reshape(1, batchSizeTotal);

                Mat hOutTs = Mat::zeros(seqLenth * batchSize, hidSize, output[0].type());
                Mat cOutTs;
                if (produceCellOutput)
                {
                    cOutTs = cOut.reshape(1, batchSizeTotal);
                    cOutTs = cOutTs.colRange(i * cOutTs.cols / numDirs, (i + 1) * cOutTs.cols / numDirs);
                }

                int tsStart, tsEnd, tsInc;
                if (reverse || i == 1) {
                    tsStart = seqLenth - 1;
                    tsEnd = -1;
                    tsInc = -1;
                }
                else {
                    tsStart = 0;
                    tsEnd = seqLenth;
                    tsInc = 1;
                }

                for (int ts = tsStart; ts != tsEnd; ts += tsInc)
                {
                    Range curRowRange(ts*batchSize, (ts + 1)*batchSize);
                    Mat xCurr = xTs.rowRange(curRowRange);


                    gemm(xCurr, Wx, 1, gates, 0, gates, GEMM_2_T);      // Wx * x_t

                    gemm(dummyOnes, bias, 1, gates, 1, gates);          //+b

                    gemm(hInternal, Wh, 1, gates, 1, gates, GEMM_2_T);  //+Wh * h_{t-1}

                    Mat gateI = gates.colRange(0*hidSize, 1*hidSize);
                    Mat gateF = gates.colRange(1*hidSize, 2*hidSize);
                    Mat gateO = gates.colRange(2*hidSize, 3*hidSize);
                    Mat gateG = gates.colRange(3*hidSize, 4*hidSize);


                    if (forgetBias){
                        add(gateF, forgetBias, gateF);
                    }

                    if (usePeephole)
                    {
                        Mat gatesIF = gates.colRange(0, 2*hidSize);
                        gemm(cInternal, pI, 1, gateI, 1, gateI);
                        gemm(cInternal, pF, 1, gateF, 1, gateF);
                        f_activation(gatesIF, gatesIF);
                    }
                    else
                    {
                        Mat gatesIFO = gates.colRange(0, 3*hidSize);
                        f_activation(gatesIFO, gatesIFO);
                    }

                    g_activation(gateG, gateG);



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
                        gemm(cInternal, pO, 1, gateO, 1, gateO);
                        f_activation(gateO, gateO);
                    }

                    //compute h_t
                    h_activation(cInternal, hInternal);
                    multiply(gateO, hInternal, hInternal);


                    //save results in output blobs
                    hInternal.copyTo(hOutTs.rowRange(curRowRange));



                    if (produceCellOutput)
                        cInternal.copyTo(cOutTs.rowRange(curRowRange));

                }

                // collect hOutTs for each direction
                tempOutputs.push_back(hOutTs.clone());

            }

            if (numDirs == 2){
                hconcat(tempOutputs[0], tempOutputs[1], output[0]);
            }
            else{
                output[0] = tempOutputs[0];
            }


            int shp[] = {seqLenth, batchSize, numDirs * numHidden};
            output[0] = output[0].reshape(1, sizeof(shp)/sizeof(shp[0]), shp);


            int shp1[] = {seqLenth, batchSize, numDirs, numHidden};
            output[0] = output[0].reshape(1, sizeof(shp1)/sizeof(shp1[0]), shp1);


            Mat tmp = output[0].clone();
            cv::transposeND(tmp, {0, 2, 1, 3}, output[0]);


            if (produceOutputYh){
                fixCellStateYh(output[0], output[1], numDirs);
            }

            if (produceCellOutput){
                fixCellStateYc(cOut, numDirs);
                cOut.copyTo(output[2]);
            }


            // Make sure changes are written back to outputs_arr
            outputs_arr.assign(output);
        }

        void fixCellStateYh(Mat& scr, Mat& dst, int numDirs)
        {
            // TODO: implement
            if (numDirs == 1){
                // take a slice of output[0]
                Mat hOut = scr.rowRange(scr.size[0] - 1, scr.size[0]);

                // reshape 1x1xBxH -> 1xBxH
                int shp[] = {1, batchSize, numHidden};
                hOut = hOut.reshape(1, sizeof(shp)/sizeof(shp[0]), shp);
                hOut.copyTo(dst);

            } else {
                // there is issue here.
                // Slice: SxDxBxH -> last sequence, first direction
                Range ranges1[] = {cv::Range(scr.size[0] - 1, scr.size[0]), cv::Range(0, 1), cv::Range::all(), cv::Range::all()};
                Mat part1 = scr(ranges1);


                // Slice: SxDxBxH -> first sequence, last direction
                Range ranges2[] = {cv::Range(0, 1), cv::Range(scr.size[1] - 1, scr.size[1]), cv::Range::all(), cv::Range::all()};
                Mat part2 = scr(ranges2);


                int shp[] = {1, part1.size[2] * part1.size[3]};
                part1 = part1.reshape(1, sizeof(shp)/sizeof(shp[0]), shp);
                part2 = part2.reshape(1, sizeof(shp)/sizeof(shp[0]), shp);

                vconcat(part1, part2, dst);

                int finalShape[] = {2, batchSize, numHidden};
                dst = dst.reshape(1, sizeof(finalShape)/sizeof(finalShape[0]), finalShape);

            }
        }


        void fixCellStateYc(Mat& cOut, int numDirs)
        {
            // seq, batch, dirs, hidden
            int shp[] = {0, batchSize, numDirs, numHidden};
            cOut = cOut.reshape(1, sizeof(shp)/sizeof(shp[0]), shp);

            // permute to {0, 2, 1, 3};
            cv::Mat newCellState;
            // transpose to match batch first output
            if (layout == BATCH_SEQ_HID){
                cv::transposeND(cOut, {2, 0, 1, 3}, newCellState);
            }
            else{
                cv::transposeND(cOut, {0, 2, 1, 3}, newCellState);
            }
            cOut = newCellState;

            if (numDirs == 1)
            {
                // Slice: Yh = Y[-1, :, :, :]
                Range ranges[] = {cv::Range(cOut.size[0] - 1, cOut.size[0]), cv::Range::all(), cv::Range::all(), cv::Range::all()};
                cOut = cOut(ranges);
                // Reshape: 1x1xBxH -> 1xBxH
                int shp[] = {1, batchSize, numHidden};
                cOut = cOut.reshape(1, sizeof(shp)/sizeof(shp[0]), shp);
            }
            else
            {
                // Slice: SxDxBxH -> last sequence, first direction
                Range ranges1[] = {cv::Range(cOut.size[0] - 1, cOut.size[0]), cv::Range(0, 1), cv::Range::all(), cv::Range::all()};
                Mat part1 = cOut(ranges1);

                // Slice: SxDxBxH -> first sequence, last direction
                Range ranges2[] = {cv::Range(0, 1), cv::Range(cOut.size[1] - 1, cOut.size[1]), cv::Range::all(), cv::Range::all()};
                Mat part2 = cOut(ranges2);

                int shp[] = {1, part1.size[2] * part1.size[3]};
                part1 = part1.reshape(1, sizeof(shp)/sizeof(shp[0]), shp);
                part2 = part2.reshape(1, sizeof(shp)/sizeof(shp[0]), shp);

                vconcat(part1, part2, cOut);

                // Reshape: 1x2xBxH -> 2xBxH
                int finalShape[] = {2, batchSize, numHidden};
                cOut = cOut.reshape(1, sizeof(finalShape)/sizeof(finalShape[0]), finalShape);
            }
        }

        void transformBlobs(std::vector<Mat>& blobs)
        {
            Mat &Wx = blobs[0];
            Mat &Wh = blobs[1];
            Mat &b = blobs[2];

            // std::vector<Mat> cudaWorkaround;
            // cudaWorkaround.push_back(Wx.clone());
            // cudaWorkaround.push_back(Wh.clone());
            // cudaWorkaround.push_back(b.clone());

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


            Wx = Wx.reshape(1, Wx.size[0] * Wx.size[1]);
            Wh = Wh.reshape(1, Wh.size[0] * Wh.size[1]);


            blobs[0] = Wx;
            blobs[1] = Wh;
            blobs[2] = b.reshape(1, 1);

            if (!blobs[3].empty()){
                blobs[3] = h0;
            }
            if (!blobs[4].empty()){
                blobs[4] = c0;
            }

            if (blobs.size() == 5) {
                // TODO: fix
                // so that future patch removing copies can leave all indexing as is
                // blobs.insert(blobs.begin(), cudaWorkaround.begin(), cudaWorkaround.end());
                return;
            }

            // TODO: fix this. Currently blobs.size = 5 only
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
            // blobs.insert(blobs.begin(), cudaWorkaround.begin(), cudaWorkaround.end());
            return;
        }
};

Ptr<LSTM2Layer> LSTM2Layer::create(const LayerParams& params)
{
    return Ptr<LSTM2Layer>(new LSTM2LayerImpl(params));
};

}}
