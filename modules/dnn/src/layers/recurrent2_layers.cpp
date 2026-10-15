// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include <cmath>
#include <opencv2/dnn/shape_utils.hpp>
#include "layers_common.hpp"
#include "cpu_kernels/fast_gemm.hpp"
#include "cpu_kernels/mlas_gemm.hpp"
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

// One LSTM cell step on the [i | f | o | g] gate row (blocks of H) for hidden units [j0, j0+nj); c and h already point at unit j0.
static void lstmCellFp32(float* g, int H, int j0, int nj, float forgetBias,
                         ActivationFunc sigmoidFn, ActivationFunc tanhFn,
                         float* c, float* h, float* tanhC)
{
    float *gi = g + j0, *gf = g + H + j0, *go = g + 2*H + j0, *gc = g + 3*H + j0;
    if (forgetBias)
        for (int j = 0; j < nj; j++)
            gf[j] += forgetBias;
    sigmoidFn(gi, gi, nj, nullptr);
    sigmoidFn(gf, gf, nj, nullptr);
    sigmoidFn(go, go, nj, nullptr);
    tanhFn(gc, gc, nj, nullptr);
    for (int j = 0; j < nj; j++)
        c[j] = gf[j]*c[j] + gi[j]*gc[j];  // c_t = f_t * c_{t-1} + i_t * g_t
    tanhFn(c, tanhC, nj, nullptr);
    for (int j = 0; j < nj; j++)
        h[j] = go[j]*tanhC[j];            // h_t = o_t * tanh(c_t)
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
    bool defaultActivations;          // sigmoid/tanh/tanh: the gate math can take the fast path
    ActivationFunc sigmoidFn, tanhFn; // dispatched kernels used by the fast gate path

    bool useAVX, useAVX2, useSVE, useNEON;
    bool constWeights;              // W/R/B are graph constants -> transform them once
    bool weightsPacked;             // whether the cached transform has been built
    std::vector<Mat> weightBlobs;   // cached transformed weights: [Wx, Wh, bias, (pI,pF,pO)]
    FastGemmOpt gemmOpt;            // MLAS GEMM options for the batched input projection
    std::vector<std::vector<float>> packedWx;  // per-direction prepacked Wx (fp32) for fastGemm
    std::vector<std::vector<float>> packedWh;  // per-direction prepacked Wh (fp32) for the batched recurrent GEMM
#ifdef HAVE_MLAS
    std::vector<Mat> packedWxMlas;  // per-direction MLAS-packed Wx; preferred over packedWx when present
#endif

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
                defaultActivations = true;
            } else {
                CV_Assert(activations.size() == 3);
                f_activation = get_activation_function(activations.getStringValue(0));
                g_activation = get_activation_function(activations.getStringValue(1));
                h_activation = get_activation_function(activations.getStringValue(2));
                defaultActivations = activations.getStringValue(0) == "Sigmoid" &&
                                     activations.getStringValue(1) == "Tanh" &&
                                     activations.getStringValue(2) == "Tanh";
            }

            constWeights = params.get<bool>("const_weights", false);
            weightsPacked = false;
            useAVX  = checkHardwareSupport(CPU_AVX);
            useAVX2 = checkHardwareSupport(CPU_AVX2);
            useSVE  = checkHardwareSupport(CPU_SVE);
            useNEON = checkHardwareSupport(CPU_NEON);
            gemmOpt.init();
            sigmoidFn = getActivationFunc(ACTIV_SIGMOID);
            tanhFn = getActivationFunc(ACTIV_TANH);

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

        // Batched projection Wx*x + bias for one direction's whole sequence; runs outside the direction loop so the GEMM can use all threads.
        Mat projectInput(int i, int numDirs, const Mat& xTs) const
        {
            const int batchSizeTotal = seqLenth * batchSize;
            const int dtype = xTs.type();

            Mat Wx = weightBlobs[0].rowRange(i * weightBlobs[0].rows / numDirs, (i + 1) * weightBlobs[0].rows / numDirs);
            Mat bias = weightBlobs[2].colRange(i * weightBlobs[2].cols / numDirs, (i + 1) * weightBlobs[2].cols / numDirs);

            const int gateN = 4 * numHidden, projK = xTs.cols;
            Mat gatesAll;
            if (dtype == CV_32F && (int)packedWx.size() == numDirs)
            {
                repeat(bias, batchSizeTotal, 1, gatesAll);   // each row starts as the bias
#ifdef HAVE_MLAS
                if (!packedWxMlas[i].empty() &&
                    mlasSgemmPacked(false, true, batchSizeTotal, gateN, projK, 1.f, xTs.ptr<float>(), projK,
                                    packedWxMlas[i].data, 1.f, gatesAll.ptr<float>(), gateN))
                    return gatesAll;
#endif
                if (!packedWx[i].empty())
                {
                    FastGemmOpt opt = gemmOpt;
                    fastGemm(false, batchSizeTotal, gateN, projK, 1.f, xTs.ptr<float>(), projK,
                             packedWx[i].data(), 1.f, gatesAll.ptr<float>(), gateN, opt);
                    return gatesAll;
                }
            }
            // fp64 / unpacked fallback
            Mat onesCol(batchSizeTotal, 1, dtype, Scalar(1));
            gatesAll.create(batchSizeTotal, gateN, dtype);
            gemm(onesCol, bias, 1.0, noArray(), 0.0, gatesAll);
            gemm(xTs, Wx, 1.0, gatesAll, 1.0, gatesAll, GEMM_2_T);
            return gatesAll;
        }

        // dst[r] += Wh.row(row0 + r) . h for r in [0, nrows): fastGEMM1T for the CPU, cv::gemm otherwise.
        void recurrentGemv(const float* h, const Mat& Wh, int row0, int nrows, float* dst) const
        {
            const float* w = Wh.ptr<float>(row0);
            const size_t wstep = Wh.step1();
            const int K = Wh.cols;
#if CV_TRY_AVX2
            if (useAVX2 && K >= 8) { opt_AVX2::fastGEMM1T(h, w, wstep, dst, dst, nrows, K); return; }
#endif
#if CV_TRY_AVX
            if (useAVX && K >= 8) { opt_AVX::fastGEMM1T(h, w, wstep, dst, dst, nrows, K); return; }
#endif
#if CV_TRY_SVE
            if (useSVE) { opt_SVE::fastGEMM1T(h, w, wstep, dst, dst, nrows, K); return; }
#endif
#if CV_TRY_NEON
            if (useNEON && K >= 4) { opt_NEON::fastGEMM1T(h, w, wstep, dst, dst, nrows, K); return; }
#endif
            Mat hrow(1, K, CV_32F, (void*)h), wrows(nrows, K, CV_32F, (void*)w, wstep * sizeof(float));
            Mat drow(1, nrows, CV_32F, dst);
            gemm(hrow, wrows, 1, drow, 1, drow, GEMM_2_T);
        }

        // Batch-1 fp32 recurrence, one parallel_for_ per timestep; each task owns a hidden-unit chunk of one direction (its four gate rows, c and h).
        void forwardStepParallel(int numDirs, int chunksPerDir, std::vector<Mat>& gatesAll, const Mat& h0All,
                                 const Mat& c0All, Mat& hOutAll, Mat& cOut) const
        {
            const int H = numHidden, T = seqLenth;
            const int chunk = ((H + chunksPerDir - 1) / chunksPerDir + 7) & ~7;  // multiple of fastGEMM1T's 8-row block
            const int ntasks = numDirs * chunksPerDir;
            Mat cState = c0All.clone();                                 // numDirs x H, each task updates its own slice
            Mat cOutRows = produceCellOutput ? cOut.reshape(1, T) : Mat();  // T x (numDirs*H)

            for (int t = 0; t < T; t++)
            {
                parallel_for_(Range(0, ntasks), [&](const Range& r)
                {
                    AutoBuffer<float, 1024> tanhC(chunk);
                    for (int task = r.start; task < r.end; task++)
                    {
                        const int d = task / chunksPerDir, j0 = (task % chunksPerDir) * chunk;
                        const int nj = std::min(chunk, H - j0);
                        if (nj <= 0)
                            continue;
                        const bool rev = reverse || d == 1;
                        const int ts = rev ? T - 1 - t : t;
                        // h_{t-1} is the previous step's row of hOutAll, complete since the last parallel_for_ returned
                        const float* hPrev = t == 0 ? h0All.ptr<float>(d) : hOutAll.ptr<float>(rev ? ts + 1 : ts - 1) + d*H;
                        Mat Wh = weightBlobs[1].rowRange(d * 4*H, (d + 1) * 4*H);
                        float* g = gatesAll[d].ptr<float>(ts);
                        for (int k = 0; k < 4; k++)
                            recurrentGemv(hPrev, Wh, k*H + j0, nj, g + k*H + j0);

                        lstmCellFp32(g, H, j0, nj, forgetBias, sigmoidFn, tanhFn,
                                     cState.ptr<float>(d) + j0, hOutAll.ptr<float>(ts) + d*H + j0, tanhC.data());
                        if (produceCellOutput)
                            memcpy(cOutRows.ptr<float>(ts) + d*H + j0, cState.ptr<float>(d) + j0, nj * sizeof(float));
                    }
                }, ntasks);
            }
        }

        // Run one direction's recurrence into its own scratch so both directions run concurrently.
        // Writes hOutAll columns [i*H, (i+1)*H) and the matching cOut slice.
        void forwardDirection(int i, int numDirs, Mat gatesAll, const Mat& h0All, const Mat& c0All,
                              Mat& hOutAll, Mat& cOut, bool parallelInside) const
        {
            const int batchSizeTotal = seqLenth * batchSize;
            const int dtype = gatesAll.type();

            Mat Wh = weightBlobs[1].rowRange(i * weightBlobs[1].rows / numDirs, (i + 1) * weightBlobs[1].rows / numDirs);
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

            // sigmoid/tanh/tanh without peephole or clip: gate math on raw rows with the dispatched kernels
            const bool fastGates = dtype == CV_32F && defaultActivations && !usePeephole && !useCellClip;
            // batch > 1: one GEMM per step reads Wh once instead of once per batch row
            const bool batchedH = dtype == CV_32F && batchSize > 1 && (int)packedWh.size() == numDirs
                && hInternal.isContinuous() && gatesAll.isContinuous();
            // outside a parallel region the per-step GEMM and the gate rows can use all threads
            const bool parallelRows = parallelInside && batchSize > 1;

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
                if (batchedH)
                {
                    FastGemmOpt opt = gemmOpt;
                    opt.multi_thread = parallelInside;
                    fastGemm(false, batchSize, Wh.rows, Wh.cols, 1.f, hInternal.ptr<float>(), Wh.cols,
                             packedWh[i].data(), 1.f, gates.ptr<float>(), Wh.rows, opt);
                }
                else if (dtype == CV_32F)
                {
                    for (int n = 0; n < hInternal.rows; n++)
                        recurrentGemv(hInternal.ptr<float>(n), Wh, 0, Wh.rows, gates.ptr<float>(n));
                }
                else
                {
                    gemm(hInternal, Wh, 1, gates, 1, gates, GEMM_2_T);
                }

                if (fastGates)
                {
                    const int H = numHidden;
                    auto gateRows = [&](const Range& r)
                    {
                        AutoBuffer<float, 1024> tanhC(H);
                        for (int n = r.start; n < r.end; n++)
                            lstmCellFp32(gates.ptr<float>(n), H, 0, H, forgetBias, sigmoidFn, tanhFn,
                                         cInternal.ptr<float>(n), hInternal.ptr<float>(n), tanhC.data());
                    };
                    if (parallelRows)
                        parallel_for_(Range(0, batchSize), gateRows);
                    else
                        gateRows(Range(0, batchSize));
                    hInternal.copyTo(hOutTs.rowRange(curRowRange));
                    if (produceCellOutput)
                        cInternal.copyTo(cOutTs.rowRange(curRowRange));
                    continue;
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
                cv::transpose(input[0], perm, xSeqFirst);
            }
            Mat xTs = xSeqFirst.reshape(1, batchSizeTotal);

            // seq-major Y assembly buffer; transposed into output[0] below.
            // Never reallocate output[0]'s header or it detaches from the graph.
            Mat hOutAll(batchSizeTotal, numDirs * numHidden, output[0].type());

            // Project every direction first, outside the direction loop, so the GEMM gets all threads.
            std::vector<Mat> gatesAll(numDirs);
            for (int i = 0; i < numDirs; i++)
                gatesAll[i] = projectInput(i, numDirs, xTs);

            const bool fp32 = input[0].type() == CV_32F && h0All.isContinuous() && c0All.isContinuous();
            const bool fastGates = fp32 && defaultActivations && !usePeephole && !useCellClip;

            // Batch 1 with enough hidden units: split each direction across threads at every timestep.
            int chunksPerDir = 1;
            if (fastGates && batchSize == 1 && numHidden >= 128)
                chunksPerDir = std::min(numHidden / 64, std::max(1, std::min(getNumThreads(), 8) / numDirs));
            if (chunksPerDir >= 2)
                forwardStepParallel(numDirs, chunksPerDir, gatesAll, h0All, c0All, hOutAll, cOut);
            else if (batchSize > 1)
            {
                // Pack Wh once for the batched recurrent GEMM; packWeights clears it when the weights change.
                if (fp32 && packedWh.empty())
                {
                    const int N = weightBlobs[1].rows / numDirs;   // 4 * numHidden
                    packedWh.resize(numDirs);
                    for (int d = 0; d < numDirs; d++)
                        fastGemmPackB(weightBlobs[1].rowRange(d * N, (d + 1) * N), packedWh[d], true, gemmOpt);
                }
                // One direction at a time, so each step's GEMM and gate rows can use all threads.
                for (int i = 0; i < numDirs; i++)
                    forwardDirection(i, numDirs, gatesAll[i], h0All, c0All, hOutAll, cOut, true);
            }
            else
            {
                // Directions are independent (disjoint output columns): run them in parallel.
                parallel_for_(Range(0, numDirs), [&](const Range& r)
                {
                    for (int i = r.start; i < r.end; i++)
                        forwardDirection(i, numDirs, gatesAll[i], h0All, c0All, hOutAll, cOut, false);
                }, numDirs);
            }

            // Reshape to (seq, batch, dirs, hid), then transpose into output[0] per `layout`.
            int shp1[] = {seqLenth, batchSize, numDirs, numHidden};
            Mat y4d = hOutAll.reshape(1, sizeof(shp1)/sizeof(shp1[0]), shp1);
            Mat ySeqFirst;   // (seq, dirs, batch, hid): the layout=0 Y; Yh sliced from it
            if (layout == SEQ_BATCH_HID) {
                cv::transpose(y4d, {0, 2, 1, 3}, output[0]);
                ySeqFirst = output[0];
            } else {
                cv::transpose(y4d, {0, 2, 1, 3}, ySeqFirst);
                cv::transpose(y4d, {1, 0, 2, 3}, output[0]);   // (batch, seq, dirs, hid)
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
                // Last step processed: Y[-1] going forward, Y[0] going in reverse.
                const int lastTs = reverse ? 0 : scr.size[0] - 1;
                Mat hOut = scr.rowRange(lastTs, lastTs + 1);

                // reshape 1x1xBxH -> 1xBxH
                int shp[] = {1, batchSize, numHidden};
                hOut = hOut.reshape(1, sizeof(shp)/sizeof(shp[0]), shp);

                if (layout == BATCH_SEQ_HID){
                    cv::transpose(hOut, {1, 0, 2}, dst);
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
                    cv::transpose(tmp, {1, 0, 2}, dst);
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
            cv::transpose(cOut, {0, 2, 1, 3}, newCellState);
            cOut = newCellState;

            if (numDirs == 1)
            {
                // Last step processed: Y[-1] going forward, Y[0] going in reverse.
                const int lastTs = reverse ? 0 : cOut.size[0] - 1;
                Range ranges[] = {cv::Range(lastTs, lastTs + 1), cv::Range::all(), cv::Range::all(), cv::Range::all()};
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
                cv::transpose(cOut, {1, 0, 2}, dst);
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

            // Prepack Wx per direction (fp32) so the projection is one batched GEMM; MLAS when available.
            packedWx.clear();
            packedWh.clear();   // packed on demand in forward() for batches > 1
#ifdef HAVE_MLAS
            packedWxMlas.clear();
#endif
            if (weightBlobs[0].type() == CV_32F)
            {
                const int numDirs = 1 + static_cast<int>(bidirectional);
                const int N = weightBlobs[0].rows / numDirs;   // 4 * numHidden
                packedWx.resize(numDirs);
#ifdef HAVE_MLAS
                packedWxMlas.resize(numDirs);
#endif
                for (int d = 0; d < numDirs; d++)
                {
                    Mat WxDir = weightBlobs[0].rowRange(d * N, (d + 1) * N);
#ifdef HAVE_MLAS
                    const int K = WxDir.cols;   // input size
                    const size_t bytes = gemmOpt.use_rvv ? 0 : mlasSgemmPackBSize(false, true, N, K);
                    if (bytes > 0 && bytes <= static_cast<size_t>(INT_MAX))
                    {
                        packedWxMlas[d].create(1, static_cast<int>(bytes), CV_8U);
                        if (mlasSgemmPackB(false, true, N, K, WxDir.ptr<float>(), K, packedWxMlas[d].data))
                            continue;
                        packedWxMlas[d].release();
                    }
#endif
                    fastGemmPackB(WxDir, packedWx[d], true, gemmOpt);
                }
            }

            if (!hasP)
                return;

            // ONNX stores P as Pi, Po, Pf; the gate loop wants pI, pF, pO.
            Mat P = weightBlobs[3];
            weightBlobs[3] = packPeephole(P, 0, numHidden);                // Pi
            weightBlobs.push_back(packPeephole(P, 2 * numHidden, numHidden));  // Pf
            weightBlobs.push_back(packPeephole(P, numHidden, numHidden));      // Po
        }

        // One H x H diagonal per direction, so the per-direction rowRange stays square.
        static Mat packPeephole(const Mat& P, int col0, int numHidden)
        {
            const int numDirs = P.rows;
            Mat packed = Mat::zeros(numDirs * numHidden, numHidden, P.type());
            for (int d = 0; d < numDirs; d++)
            {
                Mat vals = P.row(d).colRange(col0, col0 + numHidden).clone().reshape(1, numHidden);
                Mat::diag(vals).copyTo(packed.rowRange(d * numHidden, (d + 1) * numHidden));
            }
            return packed;
        }
};

Ptr<LSTM2Layer> LSTM2Layer::create(const LayerParams& params)
{
    return Ptr<LSTM2Layer>(new LSTM2LayerImpl(params));
};

}}
