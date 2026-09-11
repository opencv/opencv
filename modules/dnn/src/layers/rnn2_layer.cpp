// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "../precomp.hpp"
#include "layers_common.hpp"
#include <opencv2/dnn/shape_utils.hpp>

namespace cv {
namespace dnn {

// ONNX RNN operator: Ht = f(Xt * Wi^T + Ht-1 * Ri^T + Wbi + Rbi)
// Spec: https://onnx.ai/onnx/operators/onnx__RNN.html
// Supported opsets: 7-22 (opset 1 output_sequence is not handled)
// W is [D, H, I], R is [D, H, H], B is [D, 2H] (Wb then Rb).

enum RNNActivation { RNN_TANH = 0, RNN_RELU, RNN_SIGMOID };

static RNNActivation parseRNNActivation(const String& name)
{
    if (name == "Tanh")    return RNN_TANH;
    if (name == "Relu")    return RNN_RELU;
    if (name == "Sigmoid") return RNN_SIGMOID;
    CV_Error(Error::StsNotImplemented,
             cv::format("Activation function [%s] is not supported by RNN", name.c_str()));
}

static void applyRNNActivation(Mat& m, RNNActivation kind, float clip)
{
    CV_Assert(m.type() == CV_32F);
    const int cols = m.cols;
    parallel_for_(Range(0, m.rows), [&](const Range& range) {
        for (int row = range.start; row < range.end; ++row)
        {
            float* ptr = m.ptr<float>(row);
            if (clip > 0.f)
            {
                for (int i = 0; i < cols; ++i)
                    ptr[i] = std::min(std::max(ptr[i], -clip), clip);
            }
            if (kind == RNN_RELU)
            {
                for (int i = 0; i < cols; ++i)
                    ptr[i] = std::max(ptr[i], 0.f);
            }
            else if (kind == RNN_SIGMOID)
            {
                for (int i = 0; i < cols; ++i)
                    ptr[i] = 1.f / (1.f + std::exp(-ptr[i]));
            }
            else
            {
                for (int i = 0; i < cols; ++i)
                    ptr[i] = std::tanh(ptr[i]);
            }
        }
    });
}

class RNN2LayerImpl CV_FINAL : public RNN2Layer
{
    enum layout_t : int {
        SEQ_BATCH_HID = 0,
        BATCH_SEQ_HID = 1
    };

    layout_t layout;
    bool bidirectional, reverseOnly;
    bool produceY, produceYh;
    float clip;
    std::vector<RNNActivation> activations;
    int numTimeStamps, numSamples;

public:
    RNN2LayerImpl(const LayerParams& params)
        : bidirectional(false), reverseOnly(false), produceY(true), produceYh(false),
          clip(0.f), numTimeStamps(0), numSamples(0)
    {
        setParamsFrom(params);
        const String direction = params.get<String>("direction", "forward");
        bidirectional = direction == "bidirectional";
        reverseOnly = direction == "reverse";
        layout = (layout_t)params.get<int>("layout", SEQ_BATCH_HID);
        clip = params.get<float>("clip", 0.f);
        produceY = params.get<bool>("produce_y", true);
        produceYh = params.get<bool>("produce_yh", false);

        const int numDirs = 1 + (int)bidirectional;
        activations.assign(numDirs, RNN_TANH);
        DictValue acts = params.get<DictValue>("activations", DictValue(String()));
        if (acts.size() == numDirs && !acts.getStringValue(0).empty())
        {
            for (int i = 0; i < numDirs; i++)
                activations[i] = parseRNNActivation(acts.getStringValue(i));
        }
    }

    bool getMemoryShapes(const std::vector<MatShape>& inputs,
                         const int requiredOutputs,
                         std::vector<MatShape>& outputs,
                         std::vector<MatShape>& internals) const CV_OVERRIDE
    {
        CV_Assert(inputs.size() >= 3);
        const MatShape& X = inputs[0];
        const MatShape& W = inputs[1];
        const MatShape& R = inputs[2];
        CV_CheckEQ(W.dims, 3, "RNN: W must be [num_directions, hidden_size, input_size]");
        CV_CheckEQ(R.dims, 3, "RNN: R must be [num_directions, hidden_size, hidden_size]");

        const int D = W[0];
        const int H = R[2];
        const int T = (layout == BATCH_SEQ_HID) ? X[1] : X[0];
        const int N = (layout == BATCH_SEQ_HID) ? X[0] : X[1];

        MatShape yShape, yhShape;
        if (layout == BATCH_SEQ_HID)
        {
            yShape.push_back(N); yShape.push_back(T); yShape.push_back(D); yShape.push_back(H);
            yhShape.push_back(N); yhShape.push_back(D); yhShape.push_back(H);
        }
        else
        {
            yShape.push_back(T); yShape.push_back(D); yShape.push_back(N); yShape.push_back(H);
            yhShape.push_back(D); yhShape.push_back(N); yhShape.push_back(H);
        }

        const int outCount = std::max(requiredOutputs, 1);
        outputs.assign(outCount, yhShape);
        if (outCount > 1)
            outputs[0] = yShape;
        else if (produceY)
            outputs[0] = yShape;

        internals.clear();
        return false;
    }

    void getTypes(const std::vector<MatType>& inputs,
                  const int requiredOutputs,
                  const int requiredInternals,
                  std::vector<MatType>& outputs,
                  std::vector<MatType>& internals) const CV_OVERRIDE
    {
        CV_Assert(!inputs.empty());
        CV_CheckType(inputs[0], inputs[0] == CV_32F, "RNN supports FP32 only");
        outputs.assign(requiredOutputs, inputs[0]);
        internals.assign(requiredInternals, inputs[0]);
    }

    void forward(InputArrayOfArrays inputs_arr,
                 OutputArrayOfArrays outputs_arr,
                 OutputArrayOfArrays internals_arr) CV_OVERRIDE
    {
        CV_TRACE_FUNCTION();
        CV_UNUSED(internals_arr);

        std::vector<Mat> input, output;
        inputs_arr.getMatVector(input);
        outputs_arr.getMatVector(output);

        CV_Assert(input.size() >= 3);
        Mat X = input[0];
        const Mat& W = input[1];
        const Mat& R = input[2];
        CV_CheckTypeEQ(X.type(), CV_32F, "RNN supports FP32 only");
        CV_Assert(W.isContinuous() && R.isContinuous());

        const int D = W.size[0];
        const int H = R.size[2];
        const int I = W.size[2];

        if (layout == BATCH_SEQ_HID)
            cv::transposeND(X.clone(), {1, 0, 2}, X);
        numTimeStamps = X.size[0];
        numSamples = X.size[1];
        const int T = numTimeStamps, N = numSamples;

        Mat xTs = X.isContinuous() ? X.reshape(1, T * N) : X.clone().reshape(1, T * N);

        Mat B;
        if (input.size() > 3 && !input[3].empty())
            B = input[3].reshape(1, D);
        Mat seqLens;
        if (input.size() > 4 && !input[4].empty())
            input[4].convertTo(seqLens, CV_32S);
        Mat H0;
        if (input.size() > 5 && !input[5].empty())
            H0 = input[5].reshape(1, D * N);

        Mat y, yh;
        resolveOutputs(output, D, N, y, yh);
        Mat y2d  = y.empty()  ? Mat() : y.reshape(1, (int)(y.total() / H));
        Mat yh2d = yh.empty() ? Mat() : yh.reshape(1, (int)(yh.total() / H));

        Mat Wall = W.reshape(1, D * H), Rall = R.reshape(1, D * H);

        for (int dir = 0; dir < D; dir++)
        {
            Mat Wd = Wall.rowRange(dir * H, (dir + 1) * H);
            Mat Rd = Rall.rowRange(dir * H, (dir + 1) * H);
            CV_CheckEQ(Wd.cols, I, "RNN: inconsistent W shape");

            Mat bias = Mat::zeros(1, H, CV_32F);
            if (!B.empty())
            {
                Mat brow = B.row(dir);
                bias = brow.colRange(0, H) + brow.colRange(H, 2 * H);
            }

            Mat h = Mat::zeros(N, H, CV_32F);
            if (!H0.empty())
                H0.rowRange(dir * N, (dir + 1) * N).copyTo(h);

            // xProj[t] = x[t] * Wd^T + (Wb + Rb), computed once for every timestep.
            Mat xProj(T * N, H, CV_32F);
            gemm(xTs, Wd, 1, xProj, 0, xProj, GEMM_2_T);
            gemm(Mat::ones(T * N, 1, CV_32F), bias, 1, xProj, 1, xProj);

            const bool backward = (dir == 1) || (D == 1 && reverseOnly);
            const int tsStart = backward ? T - 1 : 0;
            const int tsEnd   = backward ? -1 : T;
            const int tsInc   = backward ? -1 : 1;

            Mat hPrev(N, H, CV_32F), gate;
            for (int ts = tsStart; ts != tsEnd; ts += tsInc)
            {
                xProj.rowRange(ts * N, (ts + 1) * N).copyTo(gate);
                gemm(h, Rd, 1, gate, 1, gate, GEMM_2_T);
                applyRNNActivation(gate, activations[dir], clip);

                h.copyTo(hPrev);
                gate.copyTo(h);
                // Past its sequence length a row keeps its state and writes zeros to Y.
                if (!seqLens.empty())
                    holdFinishedRows(seqLens, ts, backward, T, hPrev, h);

                writeYStep(y2d, ts, dir, D, T, h, seqLens, backward);
            }
            writeYhDir(yh2d, dir, D, h);
        }
    }

private:
    void resolveOutputs(std::vector<Mat>& output, int D, int N, Mat& y, Mat& yh) const
    {
        if (output.empty())
            return;
        if (output.size() == 1)
        {
            // A lone slot is Y_h when the importer said Y is unused.
            if (produceY) y = output[0];
            else          yh = output[0];
            return;
        }
        y = output[0];
        yh = output[1];
        CV_UNUSED(D); CV_UNUSED(N);
    }

    static void holdFinishedRows(const Mat& seqLens, int ts, bool backward, int T,
                                 const Mat& hPrev, Mat& h)
    {
        const int* lens = seqLens.ptr<int>();
        for (int n = 0; n < h.rows; n++)
        {
            const bool active = backward ? (ts >= T - lens[n]) : (ts < lens[n]);
            if (!active)
                hPrev.row(n).copyTo(h.row(n));
        }
    }

    // Y is [T,D,N,H], or [N,T,D,H] batchwise; both contiguous in H.
    void writeYStep(Mat& y2d, int ts, int dir, int D, int T, const Mat& h,
                    const Mat& seqLens, bool backward) const
    {
        if (y2d.empty())
            return;
        const int N = h.rows;
        for (int n = 0; n < N; n++)
        {
            const int row = (layout == BATCH_SEQ_HID) ? ((n * T + ts) * D + dir)
                                                      : ((ts * D + dir) * N + n);
            bool active = true;
            if (!seqLens.empty())
            {
                const int len = seqLens.ptr<int>()[n];
                active = backward ? (ts >= T - len) : (ts < len);
            }
            if (active) h.row(n).copyTo(y2d.row(row));
            else        y2d.row(row).setTo(0);
        }
    }

    void writeYhDir(Mat& yh2d, int dir, int D, const Mat& h) const
    {
        if (yh2d.empty())
            return;
        for (int n = 0; n < h.rows; n++)
        {
            const int row = (layout == BATCH_SEQ_HID) ? (n * D + dir) : (dir * h.rows + n);
            h.row(n).copyTo(yh2d.row(row));
        }
    }
};

Ptr<RNN2Layer> RNN2Layer::create(const LayerParams& params)
{
    return Ptr<RNN2Layer>(new RNN2LayerImpl(params));
}

}}
