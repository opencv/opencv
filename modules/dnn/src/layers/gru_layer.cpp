#include "../precomp.hpp"
#include <opencv2/dnn/shape_utils.hpp>
#include "layers_common.hpp"

namespace cv {
namespace dnn {

static void tanh(const Mat &src, Mat &dst)
{
    CV_Assert(src.type() == CV_32F);
    dst.create(src.size(), src.type());
    const int nrows = src.rows;
    const int cols = src.cols;
    parallel_for_(Range(0, nrows), [&](const Range& range) {
        for (int row = range.start; row < range.end; ++row)
        {
            const float* srcptr = src.ptr<float>(row);
            float* dstptr = dst.ptr<float>(row);
            int i = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
            const int vlanes = VTraits<v_float32>::vlanes();
            v_float32 one = vx_setall_f32(1.f), two = vx_setall_f32(2.f), minus_two = vx_setall_f32(-2.f);
            for (; i <= cols - vlanes; i += vlanes)
            {
                v_float32 x = vx_load(srcptr + i);
                v_float32 e = v_exp(v_mul(minus_two, x));        // exp(-2x)
                v_float32 t = v_sub(v_div(two, v_add(one, e)), one); // 2/(1+exp(-2x)) - 1
                vx_store(dstptr + i, t);
            }
#endif
            for (; i < cols; ++i)
                dstptr[i] = std::tanh(srcptr[i]);
        }
    });
}

static void sigmoid(const Mat &src, Mat &dst)
{
    CV_Assert(src.type() == CV_32F);
    dst.create(src.size(), src.type());
    const int nrows = src.rows;
    const int cols = src.cols;
    parallel_for_(Range(0, nrows), [&](const Range& range) {
        for (int row = range.start; row < range.end; ++row)
        {
            const float* srcptr = src.ptr<float>(row);
            float* dstptr = dst.ptr<float>(row);
            int i = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
            const int vlanes = VTraits<v_float32>::vlanes();
            v_float32 one = vx_setall_f32(1.f), zero = vx_setzero_f32();
            for (; i <= cols - vlanes; i += vlanes)
            {
                v_float32 x = vx_load(srcptr + i);
                v_float32 t = v_exp(v_sub(zero, x));       // exp(-x)
                t = v_div(one, v_add(one, t));              // 1 / (1 + exp(-x))
                vx_store(dstptr + i, t);
            }
#endif
            for (; i < cols; ++i)
                dstptr[i] = 1.f / (1.f + std::exp(-srcptr[i]));
        }
    });
}

// Fused computation of h_t = z (*) h_(t-1) + (1 - z) (*) n_t
// Single pass over elements instead of 4 separate multiply/subtract/multiply/add calls.
template<typename Dtype>
static void gruComputeH(const Mat &z, const Mat &n, Mat &h)
{
    CV_Assert(z.rows == n.rows && z.rows == h.rows);
    CV_Assert(z.cols == n.cols && z.cols == h.cols);
    for (int row = 0; row < z.rows; ++row)
    {
        const Dtype* zPtr = z.ptr<Dtype>(row);
        const Dtype* nPtr = n.ptr<Dtype>(row);
        Dtype* hPtr = h.ptr<Dtype>(row);
        for (int col = 0; col < z.cols; ++col)
            hPtr[col] = zPtr[col] * hPtr[col] + (Dtype(1) - zPtr[col]) * nPtr[col];
    }
}

class GRULayerImpl CV_FINAL : public GRULayer
{
    enum layout_t : int {
        SEQ_BATCH_HID = 0,
        BATCH_SEQ_HID = 1
    };

    int numTimeStamps, numSamples;
    layout_t layout;
    MatShape outTailShape;
    bool bidirectional;
    bool linearBeforeReset;

public:
    GRULayerImpl(const LayerParams& params) : numTimeStamps(0), numSamples(0)
    {
        setParamsFrom(params);

        bidirectional = params.get<String>("direction", "forward") == "bidirectional";
        linearBeforeReset = params.get<int>("linear_before_reset", 0) != 0;
        layout = (layout_t) params.get<int>("layout", SEQ_BATCH_HID);

        if (!blobs.empty())
        {
            CV_Assert(blobs.size() >= 3);

            blobs[2] = blobs[2].reshape(1, 1);

            const Mat& Wh = blobs[0];
            const Mat& Wx = blobs[1];
            const Mat& bias = blobs[2];
            CV_CheckEQ(Wh.dims, 2, "");
            CV_CheckEQ(Wx.dims, 2, "");
            CV_CheckEQ(Wh.rows, Wx.rows, "");
            CV_CheckEQ(Wh.rows, (1 + static_cast<int>(bidirectional)) * 3 * Wh.cols, "");
            CV_CheckEQ(Wh.rows * 2, (int)bias.total(), "");
            CV_CheckTypeEQ(Wh.type(), Wx.type(), "");
            CV_CheckTypeEQ(Wx.type(), bias.type(), "");

            if (blobs.size() > 3) {
                const Mat& hInternal = blobs[3];
                CV_CheckTypeEQ(Wx.type(), hInternal.type(), "");
            }
        }
        outTailShape.clear();
    }

    bool getMemoryShapes(const std::vector<MatShape> &inputs,
                         const int requiredOutputs,
                         std::vector<MatShape> &outputs,
                         std::vector<MatShape> &internals) const CV_OVERRIDE
    {
        CV_Assert(inputs.size() >= 1);
        const MatShape& inp0 = inputs[0]; // X (sequence input)

        int _numOut, _numInp;
        bool bidir = bidirectional;
        const bool runtimeWeights = hasRuntimeWeights(inputs);
        if (runtimeWeights)
        {
            const MatShape& W_shape = inputs[1];
            const MatShape& R_shape = inputs[2];
            CV_Assert(W_shape.size() == 3 && R_shape.size() == 3);
            _numInp = W_shape[2];
            _numOut = R_shape[2];
            CV_Assert(W_shape[0] == R_shape[0] && W_shape[1] == R_shape[1] && W_shape[1] == 3 * _numOut);
            bidir = (W_shape[0] > 1);
        }
        else
        {
            CV_Assert(blobs.size() >= 2);
            _numOut = blobs[0].size[1];
            _numInp = blobs[1].size[1];
        }

        MatShape outResShape;

        int _numSamples;
        CV_Assert(inp0.size() >= 2 && total(inp0, 2) == _numInp);
        _numSamples = (layout == BATCH_SEQ_HID) ? inp0[0] : inp0[1];

        outResShape.clear();
        if (layout == BATCH_SEQ_HID)
        {
            outResShape.push_back(_numSamples);                      // N
            outResShape.push_back(inp0[1]);                          // T
            outResShape.push_back(1 + static_cast<int>(bidir));      // D
        }
        else
        {
            outResShape.push_back(inp0[0]);                          // T
            outResShape.push_back(1 + static_cast<int>(bidir));      // D
            outResShape.push_back(_numSamples);                      // N
        }
        outResShape.push_back(_numOut);                              // hidden
        MatShape outResShapeLegacy;
        if (layout == BATCH_SEQ_HID)
        {
            outResShapeLegacy.push_back(_numSamples);                // N
            outResShapeLegacy.push_back(inp0[1]);                    // T
        }
        else
        {
            outResShapeLegacy.push_back(inp0[0]);                    // T
            outResShapeLegacy.push_back(_numSamples);                // N
        }
        outResShapeLegacy.push_back(_numOut * (1 + static_cast<int>(bidir))); // hidden * D

        MatShape yhShape;
        if (layout == BATCH_SEQ_HID)
        {
            yhShape.push_back(_numSamples);                          // N
            yhShape.push_back(1 + static_cast<int>(bidir));  // D
        }
        else
        {
            yhShape.push_back(1 + static_cast<int>(bidir));  // D
            yhShape.push_back(_numSamples);                          // N
        }
        yhShape.push_back(_numOut);                                   // hidden

        bool needY, needYh;
        resolveOutputPolicy(requiredOutputs, needY, needYh);
        if (!runtimeWeights)
        {
            needY = true;
            needYh = false;
        }

        const int outCount = std::max(requiredOutputs, 1);
        outputs.assign(outCount, MatShape());
        const bool legacyPackedY = !runtimeWeights;
        if (outCount == 1)
            outputs[0] = needY ? (legacyPackedY ? outResShapeLegacy : outResShape) : yhShape;
        else
        {
            outputs[0] = legacyPackedY ? outResShapeLegacy : outResShape; // slot #0 -> Y
            outputs[1] = yhShape;     // slot #1 -> Y_h
            for (int i = 2; i < outCount; ++i)
                outputs[i] = yhShape;
        }

        internals.assign(1, shape(_numSamples, _numOut));     // hInternal
        internals.push_back(shape(_numSamples, 1));           // dummyOnes
        internals.push_back(shape(_numSamples, 2 * _numOut)); // gates
        internals.push_back(shape(_numSamples, 2 * _numOut)); // gates_b
        internals.push_back(shape(_numSamples, 1 * _numOut)); // h_linear

        return false;
    }

    void getTypes(const std::vector<MatType>& inputs,
                  const int requiredOutputs,
                  const int requiredInternals,
                  std::vector<MatType>& outputs,
                  std::vector<MatType>& internals) const CV_OVERRIDE
    {
        CV_Assert(inputs[0] == CV_32F || inputs[0] == CV_64F);
        outputs.assign(requiredOutputs, inputs[0]);
        internals.assign(requiredInternals, inputs[0]);
    }

    void forward(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr, OutputArrayOfArrays internals_arr) CV_OVERRIDE
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        if (inputs_arr.depth() == CV_16F)
        {
            forward_fallback(inputs_arr, outputs_arr, internals_arr);
            return;
        }

        std::vector<Mat> input, output, internals;
        inputs_arr.getMatVector(input);
        outputs_arr.getMatVector(output);
        internals_arr.getMatVector(internals);

        prepareRuntimeState(input);
        const bool legacyDirectMode = (input.size() == 1);
        const bool useLinearBeforeReset = linearBeforeReset || legacyDirectMode;

        Mat sequence_input = input[0];
        Mat sequence_input_time_major = sequence_input;
        if (layout == BATCH_SEQ_HID)
        {
            cv::transposeND(sequence_input, {1, 0, 2}, sequence_input_time_major);
        }

        const int numDirs = 1 + static_cast<int>(bidirectional);
        const int numOutGlobal = blobs[0].size[1];
        int yIndex = -1, yhIndex = -1;
        for (int oi = 0; oi < (int)output.size(); ++oi)
        {
            if (yIndex < 0 && (isYShape3D(output[oi], numDirs, numOutGlobal) || isYShape4D(output[oi], numDirs, numOutGlobal)))
                yIndex = oi;
            if (yhIndex < 0 && isYhShape(output[oi], numDirs, numOutGlobal))
                yhIndex = oi;
        }
        bool writeY = false;
        bool writeYh = false;
        {
            const size_t nOut = output.size();
            if (nOut == 0)
            {
                writeY = false;
                writeYh = false;
            }
            else if (nOut == 1)
            {
                const bool hasY = (yIndex >= 0);
                const bool hasYh = (yhIndex >= 0);
                writeY = hasY && !hasYh;
                writeYh = !writeY;
            }
            else
            {
                writeY = true;
                writeYh = true;
            }
        }

        if (!writeY) yIndex = -1;
        if (!writeYh) yhIndex = -1;

        Mat y = (yIndex >= 0) ? output[yIndex] : Mat();
        Mat y3d2d;
        if (!y.empty() && y.dims == 3)
            y3d2d = y.reshape(1, numTimeStamps * numSamples);

        Mat yh = (yhIndex >= 0) ? output[yhIndex] : Mat();
        Mat xTs = sequence_input_time_major.reshape(1, numTimeStamps * numSamples);

        for (int i = 0; i < numDirs; ++i)
        {
            const Mat &Wh = blobs[0].rowRange(i * blobs[0].rows / numDirs, (i + 1) * blobs[0].rows / numDirs);
            const Mat &Wx = blobs[1].rowRange(i * blobs[1].rows / numDirs, (i + 1) * blobs[1].rows / numDirs);
            Mat h_0 = blobs[3].rowRange(i * blobs[3].rows / numDirs, (i + 1) * blobs[3].rows / numDirs);

            int numOut = Wh.size[1];
            Mat bias;
            if (blobs.size() > 2 && !blobs[2].empty() && blobs[2].cols >= (i + 1) * 6 * numOut)
                bias = blobs[2].colRange(i * 6 * numOut, (i + 1) * 6 * numOut);
            else
                bias = Mat::zeros(1, 6 * numOut, Wh.type());
            if (h_0.empty() || h_0.rows != numSamples || h_0.cols != numOut)
                h_0 = Mat::zeros(numSamples, numOut, Wh.type());

            const Mat &bx = bias.colRange(0, bias.cols / 2);
            const Mat &bh = bias.colRange(bias.cols / 2, bias.cols);

            Mat hInternal = internals[0], dummyOnes = internals[1], gates = internals[2],
                b_rz = internals[3], n_t = internals[4];
            h_0.copyTo(hInternal);
            dummyOnes.setTo(1.);

            CV_CheckLE(3 * numOut, Wx.rows, "Invalid Wx shape for GRU gates");
            CV_CheckLE(3 * numOut, Wh.rows, "Invalid Wh shape for GRU gates");
            CV_CheckLE(3 * numOut, bx.cols, "Invalid input bias shape for GRU gates");
            CV_CheckLE(3 * numOut, bh.cols, "Invalid recurrent bias shape for GRU gates");
            const Mat& wx_rz = Wx.rowRange(0, 2 * numOut);
            const Mat& wh_rz = Wh.rowRange(0, 2 * numOut);
            b_rz = bx.colRange(0, 2 * numOut) + bh.colRange(0, 2 * numOut);
            const Mat& wx_n = Wx.rowRange(2 * numOut, 3 * numOut);
            const Mat& wh_n = Wh.rowRange(2 * numOut, 3 * numOut);
            const Mat& b_in = bx.colRange(2 * numOut, 3 * numOut);
            const Mat& b_hn = bh.colRange(2 * numOut, 3 * numOut);

            // Precompute input projections for all timesteps at once (batched GEMM).
            const int totalRows = numTimeStamps * numSamples;
            Mat dummyOnesAll = Mat::ones(totalRows, 1, Wh.type());

            // xProj_rz[t] = x[t] * Wx_rz^T + b_rz   for all t
            Mat xProj_rz(totalRows, 2 * numOut, Wh.type());
            gemm(xTs, wx_rz, 1, xProj_rz, 0, xProj_rz, GEMM_2_T);
            gemm(dummyOnesAll, b_rz, 1, xProj_rz, 1, xProj_rz);

            // xProj_n[t] = x[t] * Wx_n^T + b_in   for all t
            Mat xProj_n(totalRows, numOut, Wh.type());
            gemm(xTs, wx_n, 1, xProj_n, 0, xProj_n, GEMM_2_T);
            gemm(dummyOnesAll, b_in, 1, xProj_n, 1, xProj_n);

            int tsStart, tsEnd, tsInc;
            if (i == 1) {
                tsStart = numTimeStamps - 1;
                tsEnd = -1;
                tsInc = -1;
            }
            else {
                tsStart = 0;
                tsEnd = numTimeStamps;
                tsInc = 1;
            }
            for (int ts = tsStart; ts != tsEnd; ts += tsInc)
            {
                Range curRowRange(ts * numSamples, (ts + 1) * numSamples);

                // Use precomputed input projection for this timestep
                Mat xCurrProj_rz = xProj_rz.rowRange(curRowRange);
                Mat xCurrProj_n  = xProj_n.rowRange(curRowRange);

                xCurrProj_rz.copyTo(gates);                                // x * Wx_rz + b_rz (precomputed)
                gemm(hInternal, wh_rz, 1, gates, 1, gates, GEMM_2_T);     // + h_(t-1) * Wh_rz
                sigmoid(gates, gates);                                     // sigmoid()

                Mat z = gates.colRange(0, gates.cols / 2);
                Mat r = gates.colRange(gates.cols / 2, gates.cols);

                if (useLinearBeforeReset)
                {
                    // n_t = tanh(r (*) (h_(t-1) * Wh_n + b_hn) + x * Wx_n + b_in)
                    gemm(hInternal, wh_n, 1, n_t, 0, n_t, GEMM_2_T);      // h_(t-1) * Wh_n
                    gemm(dummyOnes, b_hn, 1, n_t, 1, n_t);                // + b_hn
                    multiply(r, n_t, n_t);                                 // r (*) (...)
                    add(n_t, xCurrProj_n, n_t);                            // + x * Wx_n + b_in (precomputed)
                }
                else
                {
                    // n_t = tanh((r (*) h_(t-1)) * Wh_n + x * Wx_n + b_in + b_hn)
                    multiply(r, hInternal, n_t);                           // r (*) h_(t-1)
                    gemm(n_t, wh_n, 1, n_t, 0, n_t, GEMM_2_T);            // (r*h_(t-1)) * Wh_n
                    add(n_t, xCurrProj_n, n_t);                            // + x * Wx_n + b_in (precomputed)
                    gemm(dummyOnes, b_hn, 1, n_t, 1, n_t);                // + b_hn
                }
                tanh(n_t, n_t);                                            // tanh()

                // h_t = z (*) h_(t-1) + (1 - z) (*) n_t  (fused single-pass)
                if (z.type() == CV_32F)
                    gruComputeH<float>(z, n_t, hInternal);
                else
                    gruComputeH<double>(z, n_t, hInternal);

                writeYStep(y, y3d2d, ts, i, numOut, hInternal);
            }

            writeYhDir(yh, i, numDirs, hInternal);
        }
    }

private:
    template<typename T>
    static bool hasRuntimeWeights(const std::vector<T>& in)
    {
        return in.size() >= 3 && !in[1].empty() && !in[2].empty();
    }

    void resolveOutputPolicy(int requiredOutputs, bool& needY, bool& needYh) const
    {
        if (requiredOutputs == 0)
        {
            needY = true;
            needYh = false;
        }
        else if (requiredOutputs == 1)
        {
            needY = false;
            needYh = true;
        }
        else
        {
            needY = true;
            needYh = true;
        }
    }

    void prepareRuntimeState(const std::vector<Mat>& input)
    {
        CV_Assert(!input.empty());
        const Mat& inp0 = input[0];

        int numOut, numInp;
        if (hasRuntimeWeights(input))
        {
            CV_Assert(input.size() >= 3);
            const Mat& W_orig = input[1];  // [D, 3*H, I]
            const Mat& R_orig = input[2];  // [D, 3*H, H]
            CV_Assert(W_orig.dims == 3 && R_orig.dims == 3);

            const int num_directions = W_orig.size[0];
            numOut = R_orig.size[2];
            numInp = W_orig.size[2];
            bidirectional = (num_directions > 1);

            blobs.resize(4);
            blobs[0] = R_orig.reshape(1, R_orig.size[0] * R_orig.size[1]).clone(); // Wh
            blobs[1] = W_orig.reshape(1, W_orig.size[0] * W_orig.size[1]).clone(); // Wx

            if (input.size() > 3 && !input[3].empty())
                blobs[2] = input[3].reshape(1, 1).clone();
            else if (!blobs[2].empty())
                blobs[2] = blobs[2].reshape(1, 1).clone();
            else
                blobs[2] = Mat::zeros(1, num_directions * 6 * numOut, W_orig.type());

            if (input.size() > 5 && !input[5].empty())
                blobs[3] = input[5].reshape(1, input[5].size[0] * input[5].size[1]).clone();
            else if (!blobs[3].empty())
                blobs[3] = blobs[3].reshape(1, 1).clone();
            else
                blobs[3] = Mat::zeros(1, num_directions * numOut, W_orig.type());
        }
        else
        {
            CV_Assert(blobs.size() >= 2);
            numOut = blobs[0].size[1];
            numInp = blobs[1].size[1];

            if (input.size() > 3 && !input[3].empty())
                blobs[2] = input[3].reshape(1, 1).clone();
            else if (blobs.size() > 2 && !blobs[2].empty())
                blobs[2] = blobs[2].reshape(1, 1).clone();
            else
                blobs[2] = Mat::zeros(1, (1 + static_cast<int>(bidirectional)) * 6 * numOut, blobs[1].type());
        }

        if (outTailShape.empty())
            outTailShape.assign(1, numOut);
        else
            CV_Assert(total(outTailShape) == numOut);

        CV_Assert(inp0.dims >= 2 && (int)inp0.total(2) == numInp);
        if (layout == BATCH_SEQ_HID)
        {
            numSamples = inp0.size[0];
            numTimeStamps = inp0.size[1];
        }
        else
        {
            numTimeStamps = inp0.size[0];
            numSamples = inp0.size[1];
        }

        const int numDirs = 1 + static_cast<int>(bidirectional);
        const int expectedRows = numDirs * numSamples;
        const int expectedCols = numOut;
        if (blobs.size() <= 3 || blobs[3].empty())
            blobs.resize(4), blobs[3] = Mat::zeros(expectedRows, expectedCols, blobs[1].type());

        Mat& h0 = blobs[3];
        if (h0.rows == expectedRows && h0.cols == expectedCols)
            return;
        if (h0.total() == (size_t)(numDirs * expectedCols))
        {
            h0 = Mat::zeros(expectedRows, expectedCols, h0.type());
            return;
        }
        if (h0.rows == numDirs && h0.cols == expectedCols)
        {
            Mat expanded(expectedRows, expectedCols, h0.type());
            for (int dir = 0; dir < numDirs; ++dir)
                for (int sample = 0; sample < numSamples; ++sample)
                    h0.row(dir).copyTo(expanded.row(dir * numSamples + sample));
            h0 = expanded;
            return;
        }
        CV_CheckEQ(h0.rows, expectedRows, "Initial hidden state blob has incorrect dimensions");
        CV_CheckEQ(h0.cols, expectedCols, "Initial hidden state blob has incorrect dimensions");
    }

    bool isYhShape(const Mat& m, int numDirs, int numOutGlobal) const
    {
        if (layout == BATCH_SEQ_HID)
            return m.dims == 3 && m.size[0] == numSamples && m.size[1] == numDirs && m.size[2] == numOutGlobal;
        return m.dims == 3 && m.size[0] == numDirs && m.size[1] == numSamples && m.size[2] == numOutGlobal;
    }

    bool isYShape3D(const Mat& m, int numDirs, int numOutGlobal) const
    {
        if (layout == BATCH_SEQ_HID)
            return m.dims == 3 && m.size[0] == numSamples && m.size[1] == numTimeStamps &&
                   m.size[2] == numOutGlobal * numDirs;
        return m.dims == 3 && m.size[0] == numTimeStamps && m.size[1] == numSamples &&
               m.size[2] == numOutGlobal * numDirs;
    }

    bool isYShape4D(const Mat& m, int numDirs, int numOutGlobal) const
    {
        if (layout == BATCH_SEQ_HID)
            return m.dims == 4 && m.size[0] == numSamples && m.size[1] == numTimeStamps &&
                   m.size[2] == numDirs && m.size[3] == numOutGlobal;
        return m.dims == 4 && m.size[0] == numTimeStamps && m.size[1] == numDirs &&
               m.size[2] == numSamples && m.size[3] == numOutGlobal;
    }

    void writeYStep(Mat& y, Mat& y3d2d, int ts, int dir, int numOut, const Mat& hState) const
    {
        if (y.empty())
            return;

        if (y.dims == 3)
        {
            CV_CheckLE((dir + 1) * numOut, y3d2d.cols, "Invalid Y shape for current direction");
            Mat yRow = y3d2d.rowRange(ts * numSamples, (ts + 1) * numSamples)
                         .colRange(dir * numOut, (dir + 1) * numOut);
            hState.copyTo(yRow);
            return;
        }

        if (layout == BATCH_SEQ_HID)
        {
            Range ranges[4] = { Range::all(), Range(ts, ts + 1), Range(dir, dir + 1), Range::all() };
            Mat ySlice2d = y(ranges).reshape(1, numSamples);
            hState.copyTo(ySlice2d);
        }
        else
        {
            Range ranges[4] = { Range(ts, ts + 1), Range(dir, dir + 1), Range::all(), Range::all() };
            Mat ySlice2d = y(ranges).reshape(1, numSamples);
            hState.copyTo(ySlice2d);
        }
    }

    void writeYhDir(Mat& yh, int dir, int numDirs, const Mat& hState) const
    {
        if (yh.empty())
            return;

        if (layout == BATCH_SEQ_HID)
        {
            Range ranges[3] = { Range::all(), Range(dir, dir + 1), Range::all() };
            Mat yh2d = yh(ranges).reshape(1, numSamples);
            hState.copyTo(yh2d);
        }
        else
        {
            Mat yhDir = yh.reshape(1, numDirs * numSamples).rowRange(dir * numSamples, (dir + 1) * numSamples);
            hState.copyTo(yhDir);
        }
    }
};

Ptr<GRULayer> GRULayer::create(const LayerParams &params) {
    return Ptr<GRULayer>(new GRULayerImpl(params));
}

}
}
