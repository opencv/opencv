// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "layers_common.hpp"
#include <opencv2/dnn/shape_utils.hpp>

namespace cv
{
namespace dnn
{

class EltwiseLayerInt8Impl CV_FINAL : public EltwiseLayerInt8
{
public:
    enum EltwiseOp
    {
        PROD = 0,
        SUM = 1,
        MAX = 2
    } op;
    std::vector<float> coeffs;
    std::vector<int> zeropoints;

    enum OutputChannelsMode
    {
        ELTWISE_CHANNNELS_SAME = 0,              //!< number of channels from inputs must be the same and equal to output's number of channels
        ELTWISE_CHANNNELS_INPUT_0,               //!< number of channels from inputs may be different,
                                                 //!< output's number of channels is equal to number of channels of first input
                                                 //!< number of channels of other inputs should not be greater than number of channels of first input
        ELTWISE_CHANNNELS_INPUT_0_TRUNCATE,      //!< number of channels from inputs may be different,
                                                 //!< output's number of channels is equal to number of channels of first input
                                                 //!< there is restriction on number of channels of other inputs
                                                 //!< extra channels of other inputs is ignored
        ELTWISE_CHANNNELS_USE_MAX,               //!< number of channels from inputs may be different,
                                                 //!< output's number of channels is equal to maximal number of input channels
                                                 //!< @note supported operation: `SUM`
    } channelsModeInput;


    mutable OutputChannelsMode channelsMode;     //!< "optimized" channels mode (switch to ELTWISE_CHANNNELS_SAME if number of input channels are equal)
    mutable /*size_t*/int outputChannels;

    EltwiseLayerInt8Impl(const LayerParams& params)
        : outputChannels(0)
    {
        setParamsFrom(params);
        offset = params.get<float>("offset", 0.f);
        hasVecInput = false;
        op = SUM;
        if (params.has("operation"))
        {
            String operation = toLowerCase(params.get<String>("operation"));
            if (operation == "prod")
                op = PROD;
            else if (operation == "sum")
                op = SUM;
            else if (operation == "max")
                op = MAX;
            else
                CV_Error(cv::Error::StsBadArg, "Unknown operation type \"" + operation + "\"");
        }

        if (params.has("coeff"))
        {
            DictValue paramCoeff = params.get("coeff");
            int i, n = paramCoeff.size();
            coeffs.resize(n);
            for (i = 0; i < n; i++)
            {
                coeffs[i] = paramCoeff.get<float>(i);
            }
        }

        if (params.has("input_zeropoints"))
        {
            DictValue zp = params.get("input_zeropoints");
            int i, n = zp.size();
            zeropoints.resize(n);
            for (i = 0; i < n; i++)
            {
                zeropoints[i] = zp.get<int>(i);
            }
        }

        channelsModeInput = ELTWISE_CHANNNELS_SAME;
        if (params.has("output_channels_mode"))
        {
            String v = toLowerCase(params.get<String>("output_channels_mode"));
            if (v == "same")
            {
                channelsModeInput = ELTWISE_CHANNNELS_SAME;
            }
            else if (v == "input_0")
            {
                channelsModeInput = ELTWISE_CHANNNELS_INPUT_0;
            }
            else if (v == "input_0_truncate")
            {
                channelsModeInput = ELTWISE_CHANNNELS_INPUT_0_TRUNCATE;
            }
            else if (v == "max_input_channels")
            {
                channelsModeInput = ELTWISE_CHANNNELS_USE_MAX;
                if (op != SUM)
                    CV_Error(cv::Error::StsBadArg, "[" + type + "]:(" + name + ") 'max' channels mode is limited to SUM operation only");
            }
            else
                CV_Error(cv::Error::StsBadArg, "[" + type + "]:(" + name + ") unknown channels mode: \"" + v + "\"");
        }
        channelsMode = channelsModeInput;

        // TODO Must have checks for other unknown options
    }

    virtual bool supportBackend(int backendId) CV_OVERRIDE
    {
        return backendId == DNN_BACKEND_OPENCV;
    }

    bool getMemoryShapes(const std::vector<MatShape> &inputs,
                         const int requiredOutputs,
                         std::vector<MatShape> &outputs,
                         std::vector<MatShape> &internals) const CV_OVERRIDE
    {
        CV_Assert(inputs.size() >= 2);
        CV_Assert(inputs[0].size() >= 2);
        CV_Assert(coeffs.size() == 0 || coeffs.size() == inputs.size());
        CV_Assert(op == SUM || op == PROD || coeffs.size() == 0);

        int dims = inputs[0].size();
        // Number of channels in output shape is determined by the first input tensor.
        bool variableChannels = false;
        int numChannels = inputs[0][1];
        for (size_t i = 1; i < inputs.size(); i++)
        {
            CV_Assert(inputs[0][0] == inputs[i][0]);  // batch sizes are equal

            int input_channels = inputs[i][1];
            if (numChannels != input_channels)
                variableChannels = true;

            if (channelsModeInput == ELTWISE_CHANNNELS_SAME)
            {
                CV_Assert(numChannels == input_channels);
            }
            else if (channelsModeInput == ELTWISE_CHANNNELS_INPUT_0)
            {
                CV_Assert(numChannels >= input_channels);
            }
            else if (channelsModeInput == ELTWISE_CHANNNELS_INPUT_0_TRUNCATE)
            {
                // nothing to check
            }
            else if (channelsModeInput == ELTWISE_CHANNNELS_USE_MAX)
            {
                numChannels = std::max(numChannels, input_channels);
            }
            else
            {
                CV_Assert(0 && "Internal error");
            }
        }

        channelsMode = variableChannels ? channelsModeInput : ELTWISE_CHANNNELS_SAME;
        outputChannels = numChannels;

        outputs.assign(1, inputs[0]);
        outputs[0][1] = numChannels;

        if (dims > 2)
        {
            size_t vecIdx = 0;
            bool isVecFound = false;
            for (size_t i = 0; i < inputs.size(); i++)
            {
                bool allOnes = isAllOnes(inputs[i], 2, dims);
                if (!allOnes && !isVecFound)
                {
                    vecIdx = i;
                    isVecFound = true;
                }

                if (!allOnes && i != vecIdx)
                {
                    for (size_t j = 2; j < dims; j++)
                    {
                         CV_Assert(inputs[vecIdx][j] == inputs[i][j]);
                    }
                }
            }

            if (channelsModeInput == ELTWISE_CHANNNELS_SAME && isVecFound)
            {
                for (size_t j = 2; j < dims; j++)
                {
                    outputs[0][j] = inputs[vecIdx][j];
                }
            }
        }

        return false;
    }

    void finalize(InputArrayOfArrays inputs_arr, OutputArrayOfArrays) CV_OVERRIDE
    {
        std::vector<Mat> inputs;
        inputs_arr.getMatVector(inputs);

        for (size_t i = 0; i < inputs.size(); i++)
        {
            MatShape inpShape = shape(inputs[i].size);
            if (isAllOnes(inpShape, 2, inputs[i].dims))
            {
                hasVecInput = true;
                return;
            }
        }
    }

    class EltwiseInvoker : public ParallelLoopBody
    {
        EltwiseLayerInt8Impl& self;
        std::vector<const Mat*> srcs;
        std::vector<int> srcNumChannels;
        int nsrcs;
        Mat* dst;
        Mat* buf;
        std::vector<float> coeffs;
        std::vector<int> zeropoints;
        int nstripes;
        const Mat* activLUT;
        const ActivationLayerInt8* activ;
        int channels;
        size_t planeSize;
        float offset;

        EltwiseInvoker(EltwiseLayerInt8Impl& self_)
            : self(self_)
            , nsrcs(0), dst(0), buf(0), nstripes(0), activLUT(0), activ(0), channels(0)
            , planeSize(0), offset(0)
        {}

    public:
        static void run(EltwiseLayerInt8Impl& self,
                        const Mat* srcs, int nsrcs, Mat& buf, Mat& dst,
                        int nstripes, float offset)
        {
            const EltwiseOp op = self.op;
            CV_Check(dst.dims, 1 < dst.dims && dst.dims <= 5, ""); CV_CheckTypeEQ(dst.type(), CV_8SC1, ""); CV_Assert(dst.isContinuous());
            CV_Assert(self.coeffs.empty() || self.coeffs.size() == (size_t)nsrcs);
            CV_CheckGE(nsrcs, 2, "");

            CV_Assert(self.outputChannels == dst.size[1]);

            EltwiseInvoker p(self);
            p.srcs.resize(nsrcs);
            p.srcNumChannels.resize(nsrcs);
            p.coeffs = self.coeffs;  // can be sorted
            p.zeropoints = self.zeropoints;

            bool sortInputs = false;
            for( int i = 0; i < nsrcs; i++ )
            {
                p.srcs[i] = &srcs[i];
                CV_CheckEQ(srcs[i].dims, dst.dims, "");
                CV_Assert(srcs[i].isContinuous());
                CV_Assert(srcs[i].type() == dst.type());
                p.srcNumChannels[i] = (srcs[i].dims >= 4) ? srcs[i].size[1] : 1;

                if (self.channelsMode == ELTWISE_CHANNNELS_SAME)
                {
                    CV_Assert(srcs[i].size == dst.size);
                }
                else if (self.channelsMode == ELTWISE_CHANNNELS_INPUT_0)
                {
                    if (i == 0)
                        CV_Assert(srcs[0].size == dst.size);
                    CV_Assert(self.outputChannels >= p.srcNumChannels[i]);
                    sortInputs = true;
                }
                else if (self.channelsMode == ELTWISE_CHANNNELS_INPUT_0_TRUNCATE)
                {
                    if (i == 0)
                        CV_Assert(srcs[0].size == dst.size);
                    sortInputs = true;
                }
                else if (self.channelsMode == ELTWISE_CHANNNELS_USE_MAX)
                {
                    CV_Assert(op == SUM);
                    CV_Assert(self.outputChannels >= p.srcNumChannels[i]);
                    sortInputs = true;
                }
                else
                {
                    CV_Assert(0 && "Internal error");
                }

                if (sortInputs)
                {
                    // Sort srcs and coefficients in the desc order by number of channels
                    for (int j = i; j >= 1; j--)
                    {
                        if (std::min(self.outputChannels, p.srcs[j - 1]->size[1]) < std::min(self.outputChannels, p.srcs[j]->size[1]))
                        {
                            std::swap(p.srcs[j - 1], p.srcs[j]);
                            std::swap(p.srcNumChannels[j - 1], p.srcNumChannels[j]);
                            if (!p.coeffs.empty())
                                std::swap(p.coeffs[j - 1], p.coeffs[j]);
                            if (!p.zeropoints.empty())
                                std::swap(p.zeropoints[j - 1], p.zeropoints[j]);
                        }
                        else
                            break;
                    }
                }
            }

            p.nsrcs = nsrcs;
            p.dst = &dst;
            p.buf = &buf;
            p.nstripes = nstripes;
            p.offset = offset;
            p.channels = (dst.dims >= 4 ? dst.size[1] : 1);

            p.planeSize = dst.total(dst.dims >= 4 ? 2 : 1);
            CV_CheckEQ(dst.total(), dst.size[0] * p.channels * p.planeSize, "");
            p.activLUT = &self.activationLUT;
            p.activ = !self.activationLUT.empty() ? self.activ.get() : 0;

            parallel_for_(Range(0, nstripes), p, nstripes);
        }

        void operator()(const Range& r) const CV_OVERRIDE
        {
            const EltwiseOp op = self.op;
            size_t total = dst->size[0]*planeSize;
            size_t stripeSize = (total + nstripes - 1)/nstripes;
            size_t stripeStart = r.start*stripeSize;
            size_t stripeEnd = std::min(r.end*stripeSize, total);
            const float* coeffsptr = !coeffs.empty() ? &coeffs[0] : 0;
            const int* zeropointsptr = !zeropoints.empty() ? &zeropoints[0] : 0;
            const int8_t* lutptr = !activLUT->empty() ? activLUT->ptr<int8_t>() : 0;
            int8_t* dstptr0 = dst->ptr<int8_t>();
            float* bufptr0 = buf->ptr<float>();
            int blockSize0 = 1 << 12;
            CV_Assert(op != PROD || zeropointsptr);
            CV_Assert((op != PROD && op != SUM) || coeffsptr);
            for (size_t ofs = stripeStart; ofs < stripeEnd; )
            {
                int sampleIdx = (int)(ofs / planeSize);
                int delta = (int)ofs - sampleIdx * planeSize;
                int blockSize = std::min(blockSize0, std::min((int)(stripeEnd - ofs), (int)planeSize - delta));
                if( blockSize <= 0 )
                    break;
                ofs += blockSize;

                for (int c = 0; c < channels; c++)
                {
                    size_t dstIdx = delta + (sampleIdx*channels + c)*planeSize;
                    int8_t* dstptr = dstptr0 + dstIdx;
                    float* bufptr = bufptr0 + dstIdx;

                    // process first two inputs
                    {
                        const int8_t* srcptr0 = srcs[0]->ptr<int8_t>() + dstIdx;

                        const int inputIdx = 1;
                        int src1_channels = srcNumChannels[inputIdx];
                        if (c >= src1_channels)
                        {
                            // no data from second input
                            if (!coeffsptr)
                            {
                                for (int j = 0; j < blockSize; j++)
                                {
                                    dstptr[j] = srcptr0[j];
                                }
                            }
                            else
                            {
                                float c0 = coeffsptr[0];
                                int z0 = op == PROD ? zeropointsptr[0] : 0;
                                for (int j = 0; j < blockSize; j++)
                                {
                                    bufptr[j] = c0 * (srcptr0[j] - z0);
                                }
                            }
                        }
                        else
                        {
                            size_t srcIdx = delta + (sampleIdx * src1_channels + c) * planeSize;
                            const int8_t* srcptrI = srcs[inputIdx]->ptr<int8_t>() + srcIdx;

                            if (op == PROD)
                            {
                                float c0 = coeffsptr[0];
                                float c1 = coeffsptr[1];
                                int z0 = zeropointsptr[0];
                                int z1 = zeropointsptr[1];
                                for (int j = 0; j < blockSize; j++)
                                {
                                    bufptr[j] = (c0*(srcptr0[j] - z0)) * (c1*(srcptrI[j] - z1));
                                }
                            }
                            else if (op == MAX)
                            {
                                for (int j = 0; j < blockSize; j++)
                                {
                                    dstptr[j] = std::max(srcptr0[j], srcptrI[j]);
                                }
                            }
                            else if (op == SUM)
                            {
                                float c0 = coeffsptr[0];
                                float c1 = coeffsptr[1];
                                for (int j = 0; j < blockSize; j++)
                                {
                                    bufptr[j] = c0*srcptr0[j] + c1*srcptrI[j];
                                }
                            }
                            else
                                CV_Error(Error::StsInternal, "");
                        }
                    }

                    // aggregate other inputs (3+)
                    for (size_t inputIdx = 2; inputIdx < nsrcs; inputIdx++)
                    {
                        int srcI_channels = srcNumChannels[inputIdx];
                        if (c >= srcI_channels)
                            continue;  // no data from second input
                        size_t srcIdx = delta + (sampleIdx * srcI_channels + c) * planeSize;
                        const int8_t* srcptrI = srcs[inputIdx]->ptr<int8_t>() + srcIdx;

                        if (op == PROD)
                        {
                            float cI = coeffsptr[inputIdx];
                            int zI = zeropointsptr[inputIdx];
                            for (int j = 0; j < blockSize; j++)
                            {
                                bufptr[j] *= cI*(srcptrI[j] - zI);
                            }
                        }
                        else if (op == MAX)
                        {
                            for (int j = 0; j < blockSize; j++)
                            {
                                dstptr[j] = std::max(dstptr[j], srcptrI[j]);
                            }
                        }
                        else if (op == SUM)
                        {
                            float cI = coeffsptr[inputIdx];
                            for (int j = 0; j < blockSize; j++)
                            {
                                bufptr[j] += cI * srcptrI[j];
                            }
                        }
                        else
                            CV_Error(Error::StsInternal, "");
                    }

                    // add offset and saturate cast to int8
                    if (op == SUM || op == PROD)
                    {
                        for (int j = 0; j < blockSize; j++)
                        {
                            dstptr[j] = saturate_cast<int8_t>(std::round(bufptr[j] + offset));
                        }
                    }
                }
                if( activ )
                {
                    int8_t* ptr = dstptr0 + delta + sampleIdx*channels*planeSize;
                    activ->forwardSlice(ptr, lutptr, ptr, blockSize, planeSize, 0, channels);
                }
            }
        }
    };

    void forward(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr, OutputArrayOfArrays internals_arr) CV_OVERRIDE
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        std::vector<Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);

        CV_Assert(outputs.size() == 1);
        const int nstripes = getNumThreads();

        if (channelsModeInput == ELTWISE_CHANNNELS_SAME && inputs[0].dims > 2)
        {
            for (size_t i = 0; i < inputs.size(); i++)
            {
                MatShape inpShape = shape(inputs[i].size);
                bool allOnes = isAllOnes(inpShape, 2, inputs[i].dims);

                if (allOnes)
                {
                    Mat tmpInput = inputs[i];
                    MatShape outShape = shape(outputs[0].size);
                    size_t xSize = outShape[2];
                    for (size_t j = 3; j < outShape.size(); j++)
                        xSize *= outShape[j];

                    int dimVec[3] = {outShape[0], outShape[1], (int) xSize};
                    std::vector<int> matSizesVec(&dimVec[0], &dimVec[0] + 3);
                    inputs[i] = Mat(matSizesVec, tmpInput.type());

                    std::vector<int> idx(outShape.size(), 0);
                    std::vector<int> outIdx(inpShape.size(), 0);

                    for (size_t j = 0; j < outShape[0]; j++)
                    {
                        outIdx[0] = idx[0] = j;
                        for(size_t k = 0; k < outShape[1]; k++)
                        {
                            outIdx[1] = idx[1] = k;
                            for (size_t x = 0; x < xSize; x++)
                            {
                                outIdx[2] = x;
                                inputs[i].at<int8_t>(outIdx.data()) = tmpInput.at<int8_t>(idx.data());
                            }
                        }
                    }
                    inputs[i] = inputs[i].reshape(0, outShape);
                }
            }
        }

        Mat buf = Mat(shape(outputs[0]), CV_32F); // to store intermediate results
        EltwiseInvoker::run(*this, &inputs[0], (int)inputs.size(), buf, outputs[0], nstripes, offset);
    }

    virtual int64 getFLOPS(const std::vector<MatShape> &inputs,
                           const std::vector<MatShape> &outputs) const CV_OVERRIDE
    {
        CV_UNUSED(outputs); // suppress unused variable warning
        CV_Assert(inputs.size());

        // FIXIT: handle inputs with different number of channels
        long flops = inputs.size() * total(inputs[0]);

        return flops;
    }

    bool setActivation(const Ptr<ActivationLayer>& layer) CV_OVERRIDE
    {
        Ptr<ActivationLayerInt8> activ_int8 = layer.dynamicCast<ActivationLayerInt8>();
        if (!activ_int8.empty())
        {
            activ = activ_int8;
            if (!activ_int8->blobs.empty())
                activationLUT = activ_int8->blobs[0];
            return true;
        }
        return false;
    }

    Mat activationLUT;
    Ptr<ActivationLayerInt8> activ;

private:
    bool hasVecInput;
    float offset;
};

Ptr<EltwiseLayerInt8> EltwiseLayerInt8::create(const LayerParams& params)
{
    return Ptr<EltwiseLayerInt8>(new EltwiseLayerInt8Impl(params));
}

}
}
