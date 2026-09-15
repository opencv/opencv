// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include <cmath>
#include <opencv2/dnn/shape_utils.hpp>
#include "layers_common.hpp"
#include "cpu_kernels/fast_gemm.hpp"
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
    dst.create(src.size, src.type());
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

// ONNX gate order is I,O,F,C but the recurrence wants I,F,O,C: swap the O/F blocks.
template<typename T>
static void reorderGatesIOFCtoIFOC_(Mat& in)
{
    int first = in.size[0];
    int rest = in.total() / first / 4;
    Mat m = in.reshape(1, {first, 4, rest});
    Mat outputGate = m.col(1), forgetGate = m.col(2);
    std::swap_ranges(outputGate.begin<T>(), outputGate.end<T>(), forgetGate.begin<T>());
}
static void reorderGatesIOFCtoIFOC(Mat& in)
{
    if (in.depth() == CV_64F)
        reorderGatesIOFCtoIFOC_<double>(in);
    else
        reorderGatesIOFCtoIFOC_<float>(in);
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

    bool useAVX, useAVX2, useSVE, useNEON;
    bool constWeights;              // W/R/B are graph constants -> transform them once
    bool weightsPacked;             // whether the cached transform has been built
    std::vector<Mat> weightBlobs;   // cached transformed weights: [Wx, Wh, bias, (pI,pF,pO)]
    FastGemmOpt gemmOpt;            // MLAS GEMM options for the batched input projection
    std::vector<std::vector<float>> packedWx;  // per-direction prepacked Wx (fp32) for fastGemm

    public:
        LSTM2LayerImpl(const LayerParams& params)
        {
            setParamsFrom(params);
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
            } else {
                CV_Assert(activations.size() == 3);
                f_activation = get_activation_function(activations.getStringValue(0));
                g_activation = get_activation_function(activations.getStringValue(1));
                h_activation = get_activation_function(activations.getStringValue(2));
            }

            constWeights = params.get<bool>("const_weights", false);
            weightsPacked = false;
            useAVX  = checkHardwareSupport(CPU_AVX);
            useAVX2 = checkHardwareSupport(CPU_AVX2);
            useSVE  = checkHardwareSupport(CPU_SVE);
            useNEON = checkHardwareSupport(CPU_NEON);
            gemmOpt.init();

            outTailShape.clear();
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
                    outResShape.push_back(_seqLen);
                    outResShape.push_back(1 + static_cast<int>(bidirectional));
                    outResShape.push_back(_batchSize);
                } else {
                    // ONNX layout=1: Y is (batch, seq, dirs, hid) - this must match what forward()
                    // actually writes, the graph engine preallocates the output by this shape
                    _batchSize = inp0[0];
                    _seqLen = inp0[1];
                    outResShape.push_back(_batchSize);
                    outResShape.push_back(_seqLen);
                    outResShape.push_back(1 + static_cast<int>(bidirectional));
                }
            }
            else
            {
                CV_Assert(inp0.size() >= 2 && total(inp0, 1) == _inpSize);
                _batchSize = inp0[0];
                outResShape.push_back(1 + static_cast<int>(bidirectional));
                outResShape.push_back(_batchSize);
            }

            outResShape.push_back(_hidSize);
            outputs.assign(1, outResShape);

            // Yh / Yc: ONNX layout=0 -> (dirs, batch, hid), layout=1 -> (batch, dirs, hid)
            int shp[] = {1 + static_cast<int>(bidirectional), _batchSize, numHidden};
            if (layout == BATCH_SEQ_HID)
                std::swap(shp[0], shp[1]);
            MatShape newShape(shp, shp + sizeof(shp)/sizeof(shp[0]));

            // compute output shape of yc
            if (produceCellOutput)
            {
                outputs.push_back(newShape);
            }
            // compute output shape of yh
            if (produceOutputYh)
            {
                outputs.push_back(newShape);
            }

            // forward() allocates its own per-direction scratch, so no engine internals are needed
            internals.clear();

            return false;
        }

        void getTypes(const std::vector<MatType>& inputs,
                      const int requiredOutputs,
                      const int requiredInternals,
                      std::vector<MatType>& outputs,
                      std::vector<MatType>& internals) const CV_OVERRIDE
        {
            CV_Assert(inputs[0] == CV_32F || inputs[0] == CV_64F);  // Only floating-point types are supported currently
            outputs.assign(requiredOutputs, inputs[0]);
            internals.clear();
        }

        virtual int64 getFLOPS(const std::vector<MatShape> &inputs,
                               const std::vector<MatShape> &outputs) const CV_OVERRIDE
        {
            // LSTM: 4 gates, each gate = input_size*hidden_size + hidden_size*hidden_size MACs
            const MatShape& inp0 = inputs[0];
            const MatShape& Wx = inputs[1];
            int _hidSize = numHidden;
            int _inpSize = Wx[2];
            int _seqLen = inp0[0];
            int _batchSize = inp0[1];
            int numDirs = 1 + static_cast<int>(bidirectional);

            // Per timestep: 4 gates * (2*input_size*hidden_size + 2*hidden_size*hidden_size + hidden_size)
            int64 flopsPerStep = CV_BIG_INT(4) * (CV_BIG_INT(2) * _inpSize * _hidSize +
                                                   CV_BIG_INT(2) * _hidSize * _hidSize +
                                                   _hidSize);
            return (int64)numDirs * _seqLen * _batchSize * flopsPerStep;
        }

        // Run one direction's recurrence into its own scratch so both directions run concurrently.
        // Writes hOutAll columns [i*H, (i+1)*H) and the matching cOut slice.
        void forwardDirection(int i, int numDirs, const Mat& xTs, const Mat& h0All, const Mat& c0All,
                              Mat& hOutAll, Mat& cOut) const
        {
            const int batchSizeTotal = seqLenth * batchSize;
            const int dtype = xTs.type();

            Mat Wx = weightBlobs[0].rowRange(i * weightBlobs[0].rows / numDirs, (i + 1) * weightBlobs[0].rows / numDirs);
            Mat Wh = weightBlobs[1].rowRange(i * weightBlobs[1].rows / numDirs, (i + 1) * weightBlobs[1].rows / numDirs);
            Mat bias = weightBlobs[2].colRange(i * weightBlobs[2].cols / numDirs, (i + 1) * weightBlobs[2].cols / numDirs);
            Mat pI, pF, pO;
            if (usePeephole)
            {
                pI = weightBlobs[3].rowRange(i * weightBlobs[3].rows / numDirs, (i + 1) * weightBlobs[3].rows / numDirs);
                pF = weightBlobs[4].rowRange(i * weightBlobs[4].rows / numDirs, (i + 1) * weightBlobs[4].rows / numDirs);
                pO = weightBlobs[5].rowRange(i * weightBlobs[5].rows / numDirs, (i + 1) * weightBlobs[5].rows / numDirs);
            }

            Mat hInternal(batchSize, numHidden, dtype);
            Mat cInternal(batchSize, numHidden, dtype);
            h0All.rowRange(i * batchSize, (i + 1) * batchSize).copyTo(hInternal);
            c0All.rowRange(i * batchSize, (i + 1) * batchSize).copyTo(cInternal);

            Mat hOutTs(batchSizeTotal, numHidden, dtype);
            Mat cOutTs;
            if (produceCellOutput)
            {
                cOutTs = cOut.reshape(1, batchSizeTotal);
                cOutTs = cOutTs.colRange(i * cOutTs.cols / numDirs, (i + 1) * cOutTs.cols / numDirs);
            }

            // Batched projection: gatesAll = Wx*x + bias for the whole sequence.
            const int gateN = 4 * numHidden, projK = xTs.cols;
            Mat gatesAll;
            if (dtype == CV_32F && (int)packedWx.size() == numDirs && !packedWx[i].empty())
            {
                FastGemmOpt opt = gemmOpt;
                opt.multi_thread = (numDirs == 1);  // directions are parallelized when bidirectional
                repeat(bias, batchSizeTotal, 1, gatesAll);   // each row starts as the bias
                fastGemm(false, batchSizeTotal, gateN, projK, 1.f, xTs.ptr<float>(), projK,
                         packedWx[i].data(), 1.f, gatesAll.ptr<float>(), gateN, opt);
            }
            else  // fp64 / unpacked fallback
            {
                Mat onesCol(batchSizeTotal, 1, dtype, Scalar(1));
                gatesAll.create(batchSizeTotal, gateN, dtype);
                gemm(onesCol, bias, 1.0, noArray(), 0.0, gatesAll);
                gemm(xTs, Wx, 1.0, gatesAll, 1.0, gatesAll, GEMM_2_T);
            }

            // fastGEMM1T (recurrent step) needs contiguous fp32 operands
#if CV_TRY_AVX2 || CV_TRY_AVX
            bool canUseAvxH = hInternal.isContinuous() && gatesAll.isContinuous()
                && Wh.depth() == CV_32F && hInternal.depth() == CV_32F && gatesAll.depth() == CV_32F && Wh.cols >= 8;
#endif
#if CV_TRY_SVE
            bool canUseSveH = hInternal.isContinuous() && gatesAll.isContinuous()
                && Wh.depth() == CV_32F && hInternal.depth() == CV_32F && gatesAll.depth() == CV_32F;
#endif
#if CV_TRY_NEON
            bool canUseNeonH = hInternal.isContinuous() && gatesAll.isContinuous()
                && Wh.depth() == CV_32F && hInternal.depth() == CV_32F && gatesAll.depth() == CV_32F && Wh.cols >= 4;
#endif

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
                Mat gates = gatesAll.rowRange(curRowRange);  // already holds Wx * x_t + b

                // gates += Wh * h_{t-1}
#if CV_TRY_AVX2
                if (useAVX2 && canUseAvxH)
                {
                    for (int n = 0; n < hInternal.rows; n++)
                        opt_AVX2::fastGEMM1T(hInternal.ptr<float>(n), Wh.ptr<float>(), Wh.step1(),
                                             gates.ptr<float>(n), gates.ptr<float>(n), Wh.rows, Wh.cols);
                }
                else
#endif
#if CV_TRY_AVX
                if (useAVX && canUseAvxH)
                {
                    for (int n = 0; n < hInternal.rows; n++)
                        opt_AVX::fastGEMM1T(hInternal.ptr<float>(n), Wh.ptr<float>(), Wh.step1(),
                                            gates.ptr<float>(n), gates.ptr<float>(n), Wh.rows, Wh.cols);
                }
                else
#endif
#if CV_TRY_SVE
                if (useSVE && canUseSveH)
                {
                    for (int n = 0; n < hInternal.rows; n++)
                        opt_SVE::fastGEMM1T(hInternal.ptr<float>(n), Wh.ptr<float>(), Wh.step1(),
                                            gates.ptr<float>(n), gates.ptr<float>(n), Wh.rows, Wh.cols);
                }
                else
#endif
#if CV_TRY_NEON
                if (useNEON && canUseNeonH)
                {
                    for (int n = 0; n < hInternal.rows; n++)
                        opt_NEON::fastGEMM1T(hInternal.ptr<float>(n), Wh.ptr<float>(), Wh.step1(),
                                             gates.ptr<float>(n), gates.ptr<float>(n), Wh.rows, Wh.cols);
                }
                else
#endif
                {
                    gemm(hInternal, Wh, 1, gates, 1, gates, GEMM_2_T);
                }

                Mat gateI = gates.colRange(0*numHidden, 1*numHidden);
                Mat gateF = gates.colRange(1*numHidden, 2*numHidden);
                Mat gateO = gates.colRange(2*numHidden, 3*numHidden);
                Mat gateG = gates.colRange(3*numHidden, 4*numHidden);

                if (forgetBias){
                    add(gateF, forgetBias, gateF);
                }

                if (usePeephole)
                {
                    Mat gatesIF = gates.colRange(0, 2*numHidden);
                    gemm(cInternal, pI, 1, gateI, 1, gateI);
                    gemm(cInternal, pF, 1, gateF, 1, gateF);
                    f_activation(gatesIF, gatesIF);
                }
                else
                {
                    Mat gatesIFO = gates.colRange(0, 3*numHidden);
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

                hInternal.copyTo(hOutTs.rowRange(curRowRange));

                if (produceCellOutput)
                    cInternal.copyTo(cOutTs.rowRange(curRowRange));
            }

            // slice this direction's result into the assembly buffer
            hOutTs.copyTo(hOutAll.colRange(i * numHidden, (i + 1) * numHidden));
        }

        void forward(InputArrayOfArrays inputs_arr,
                     OutputArrayOfArrays outputs_arr,
                     OutputArrayOfArrays internals_arr) CV_OVERRIDE
        {

            std::vector<Mat> input, output;
            inputs_arr.getMatVector(input);
            outputs_arr.getMatVector(output);

            int numInputs = input.size();
            int inpSize = input[0].size[2];

            // determine seqLen and batchSize
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

            // ONNX LSTM inputs 0..7: X,W,R,B,sequence_lens,initial_h,initial_c,P; slots 3..7 optional.
            // Test non-emptiness, not count: some exports keep empty slots for unused inputs.
            auto hasInput = [&](int idx) {
                return idx < numInputs && !input[idx].empty();
            };

            CV_Assert(numInputs >= 3);  // X, W, R are mandatory

            const int numDirs = 1 + static_cast<int>(bidirectional);
            const int batchSizeTotal = seqLenth * batchSize;

            // Weight transform is shape-independent: once for const weights, else every forward.
            if (!constWeights || !weightsPacked)
            {
                packWeights(input);
                weightsPacked = true;
            }

            // Initial states are shape dependent, so (re)build them every forward.
            Mat h0All = hasInput(5) ? input[5].reshape(1, input[5].size[0] * input[5].size[1])
                                    : Mat::zeros(numDirs * batchSize, numHidden, input[0].type());
            Mat c0All = hasInput(6) ? input[6].reshape(1, input[6].size[0] * input[6].size[1])
                                    : Mat::zeros(numDirs * batchSize, numHidden, input[0].type());

            // set outputs to 0
            for (auto& out : output)
                out.setTo(0);

            // seq-major cell-state scratch: (seq, batch, dirs, hid), matching the recurrence.
            int cOutShape[] = {seqLenth, batchSize, numDirs, numHidden};
            Mat cOut = produceCellOutput ? Mat::zeros(4, cOutShape, output[0].type()) : Mat();

            // the recurrence below slices X by timestep, so it needs the seq-major order;
            // under ONNX layout=1 the input arrives as (batch, seq, ...)
            Mat xSeqFirst = input[0];
            if (layout == BATCH_SEQ_HID)
            {
                std::vector<int> perm(input[0].dims);
                std::iota(perm.begin(), perm.end(), 0);
                std::swap(perm[0], perm[1]);
                cv::transposeND(input[0], perm, xSeqFirst);
            }
            Mat xTs = xSeqFirst.reshape(1, batchSizeTotal);

            // seq-major Y assembly buffer; transposed into output[0] below.
            // Never reallocate output[0]'s header or it detaches from the graph.
            Mat hOutAll(batchSizeTotal, numDirs * numHidden, output[0].type());

            // Directions are independent (disjoint output columns): run them in parallel.
            parallel_for_(Range(0, numDirs), [&](const Range& r)
            {
                for (int i = r.start; i < r.end; i++)
                    forwardDirection(i, numDirs, xTs, h0All, c0All, hOutAll, cOut);
            }, numDirs);

            // Reshape to (seq, batch, dirs, hid), then transpose into output[0] per `layout`.
            int shp1[] = {seqLenth, batchSize, numDirs, numHidden};
            Mat y4d = hOutAll.reshape(1, sizeof(shp1)/sizeof(shp1[0]), shp1);
            Mat ySeqFirst;   // (seq, dirs, batch, hid): the layout=0 Y; Yh sliced from it
            if (layout == SEQ_BATCH_HID) {
                cv::transposeND(y4d, {0, 2, 1, 3}, output[0]);
                ySeqFirst = output[0];
            } else {
                cv::transposeND(y4d, {0, 2, 1, 3}, ySeqFirst);
                cv::transposeND(y4d, {1, 0, 2, 3}, output[0]);   // (batch, seq, dirs, hid)
            }

            if (produceOutputYh){
                getCellStateYh(ySeqFirst, output[1], numDirs);
            }

            if (produceCellOutput){
                getCellStateYc(cOut, output[2], numDirs);
            }
        }

        void getCellStateYh(Mat& scr, Mat& dst, int numDirs)
        {
            // TODO: implement
            if (numDirs == 1){
                // take a slice of output[0]
                Mat hOut = scr.rowRange(scr.size[0] - 1, scr.size[0]);

                // reshape 1x1xBxH -> 1xBxH
                int shp[] = {1, batchSize, numHidden};
                hOut = hOut.reshape(1, sizeof(shp)/sizeof(shp[0]), shp);

                if (layout == BATCH_SEQ_HID){
                    cv::transposeND(hOut, {1, 0, 2}, dst);
                }
                else{
                    hOut.copyTo(dst);
                }

            } else {
                // Slice: SxDxBxH -> last sequence, first direction
                Range ranges1[] = {cv::Range(scr.size[0] - 1, scr.size[0]), cv::Range(0, 1), cv::Range::all(), cv::Range::all()};
                Mat part1 = scr(ranges1);

                // Slice: SxDxBxH -> first sequence, last direction
                Range ranges2[] = {cv::Range(0, 1), cv::Range(scr.size[1] - 1, scr.size[1]), cv::Range::all(), cv::Range::all()};
                Mat part2 = scr(ranges2);

                int shp[] = {1, part1.size[2] * part1.size[3]};
                part1 = part1.reshape(1, sizeof(shp)/sizeof(shp[0]), shp);
                part2 = part2.reshape(1, sizeof(shp)/sizeof(shp[0]), shp);

                // build into a temp, then write into the preallocated dst in place (vconcat straight
                // into dst would replace its header and detach it from the graph's output tensor)
                Mat tmp;
                vconcat(part1, part2, tmp);

                int finalShape[] = {2, batchSize, numHidden};
                tmp = tmp.reshape(1, sizeof(finalShape)/sizeof(finalShape[0]), finalShape);

                if (layout == BATCH_SEQ_HID){
                    cv::transposeND(tmp, {1, 0, 2}, dst);
                } else {
                    tmp.copyTo(dst);
                }
            }
        }


        void getCellStateYc(Mat& cOut, Mat& dst, int numDirs)
        {
            // seq, batch, dirs, hidden
            int shp[] = {0, batchSize, numDirs, numHidden};
            cOut = cOut.reshape(1, sizeof(shp)/sizeof(shp[0]), shp);

            // permute to (seq, dirs, batch, hidden); the `layout` only affects the FINAL Yc order
            // below, the last-timestep/last-direction slicing is layout-independent
            cv::Mat newCellState;
            cv::transposeND(cOut, {0, 2, 1, 3}, newCellState);
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

            // (dirs, batch, hid), or (batch, dirs, hid) for the batch-first layout - written into the
            // preallocated dst in place
            if (layout == BATCH_SEQ_HID){
                cv::transposeND(cOut, {1, 0, 2}, dst);
            } else {
                cOut.copyTo(dst);
            }
        }

        // Fill weightBlobs: [0]=Wx, [1]=Wh, [2]=bias, and (peephole) [3]=pI, [4]=pF, [5]=pO.
        // Shape-independent, so it runs once for constant weights.
        void packWeights(const std::vector<Mat>& input)
        {
            int biasShape[] = {1 + static_cast<int>(bidirectional), 8 * numHidden};

            weightBlobs.clear();
            weightBlobs.push_back(input[1].clone());  // W -> Wx
            weightBlobs.push_back(input[2].clone());  // R -> Wh
            weightBlobs.push_back((input.size() > 3 && !input[3].empty())
                                      ? input[3].clone()
                                      : Mat::zeros(2, biasShape, input[0].type()));  // B
            bool hasP = usePeephole && input.size() > 7 && !input[7].empty();
            if (hasP)
                weightBlobs.push_back(input[7].clone());  // P

            Mat& Wx = weightBlobs[0];
            Mat& Wh = weightBlobs[1];
            Mat& b  = weightBlobs[2];

            b = b.reshape(1, b.size[0]);
            Mat bx = b.colRange(0, b.cols / 2);
            Mat bh = b.colRange(b.cols / 2, b.cols);
            b = bx + bh;

            reorderGatesIOFCtoIFOC(Wx);
            reorderGatesIOFCtoIFOC(Wh);
            reorderGatesIOFCtoIFOC(b);

            weightBlobs[0] = Wx.reshape(1, Wx.size[0] * Wx.size[1]);
            weightBlobs[1] = Wh.reshape(1, Wh.size[0] * Wh.size[1]);
            weightBlobs[2] = b.reshape(1, 1);

            // Prepack Wx per direction (fp32) so the projection is one batched GEMM.
            packedWx.clear();
            if (weightBlobs[0].type() == CV_32F)
            {
                const int numDirs = 1 + static_cast<int>(bidirectional);
                const int N = weightBlobs[0].rows / numDirs;   // 4 * numHidden
                packedWx.resize(numDirs);
                for (int d = 0; d < numDirs; d++)
                {
                    Mat WxDir = weightBlobs[0].rowRange(d * N, (d + 1) * N);
                    fastGemmPackB(WxDir, packedWx[d], true, gemmOpt);
                }
            }

            if (!hasP)
                return;

            Mat P = weightBlobs[3];
            weightBlobs[3] = P.colRange(0, numHidden);
            weightBlobs[3] = weightBlobs[3].clone().reshape(1, weightBlobs[3].total());  // Single column.
            weightBlobs[3] = Mat::diag(weightBlobs[3]);

            weightBlobs.push_back(P.colRange(numHidden, 2 * numHidden));
            weightBlobs[4] = weightBlobs[4].clone().reshape(1, weightBlobs[4].total());  // Single column.
            weightBlobs[4] = Mat::diag(weightBlobs[4]);

            weightBlobs.push_back(P.colRange(2 * numHidden, 3 * numHidden));
            weightBlobs[5] = weightBlobs[5].clone().reshape(1, weightBlobs[5].total());  // Single column.
            weightBlobs[5] = Mat::diag(weightBlobs[5]);
        }
};

Ptr<LSTM2Layer> LSTM2Layer::create(const LayerParams& params)
{
    return Ptr<LSTM2Layer>(new LSTM2LayerImpl(params));
};

}}
