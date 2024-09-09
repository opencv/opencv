// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "layers_common.hpp"
#include "../net_impl.hpp"
//#include "../op_cuda.hpp"
//#include "../op_inf_engine.hpp"
//#include "../ie_ngraph.hpp"
//#include "../op_webnn.hpp"
//#include "../op_timvx.hpp"
//#include "../op_cann.hpp"

//#include <opencv2/dnn/shape_utils.hpp>

#ifdef HAVE_CUDA
#include "../cuda4dnn/primitives/reshape.hpp"
using namespace cv::dnn::cuda4dnn;
#endif

namespace cv
{
namespace dnn
{

class Reshape2LayerImpl CV_FINAL : public Reshape2Layer
{
public:
    bool dynamicShapeSpec;

    Reshape2LayerImpl(const LayerParams& params)
    {
        dynamicShapeSpec = true;
        setParamsFrom(params);
        if (params.has("shape"))
        {
            dynamicShapeSpec = false;

            const DictValue &paramShape = params.get("shape");
            int i, ndims = paramShape.size();
            newShapeDesc.resize(ndims);
            for (i = 0; i < ndims; i++) {
                int sz = paramShape.get<int>(i);
                if (sz <= 0)
                    dynamicShapeSpec = true;
                newShapeDesc[i] = sz;
            }
        }
    }

    virtual bool dynamicOutputShapes() const CV_OVERRIDE
    {
        return dynamicShapeSpec;
    }

    virtual bool supportBackend(int backendId) CV_OVERRIDE
    {
        return backendId == DNN_BACKEND_OPENCV;
    }

    bool haveShapeSpec() const
    {
        return newShapeDesc.dims >= 0;
    }

    MatShape getOutShape(const MatShape& inpShape, const MatShape& shapeSpec) const
    {
        MatShape outShape = shapeSpec;
        int m1idx = -1, haveZeros = 0;
        int i, ndims = outShape.dims;
        int64_t outTotal = 1;
        for (i = 0; i < ndims; i++) {
            if (outShape[i] < 0) {
                CV_Assert(outShape[i] == -1);
                if (m1idx >= 0) {
                    CV_Error(Error::StsBadArg, "invalid shape spec, there must be at most one '-1'");
                }
                m1idx = i;
            }
            else {
                if (outShape[i] == 0) {
                    haveZeros++;
                    if (i >= inpShape.dims) {
                        CV_Error(Error::StsBadArg, "cannot copy dimension from the input tensor");
                    }
                    outShape[i] = inpShape[i];
                }
                outTotal *= outShape[i];
            }
        }

        int64_t inpTotal = (int64_t)inpShape.total();
        if (m1idx >= 0) {
            int64_t autoSize = inpTotal/outTotal;
            CV_Assert(autoSize <= INT_MAX && autoSize*outTotal == inpTotal);
            outShape[m1idx] = (int)autoSize;
        } else {
            CV_Assert(outTotal == inpTotal);
        }

        return outShape;
    }

    MatShape shapeFromMat(const Mat& shapeTensor) const
    {
        int shapeType = shapeTensor.type();
        CV_Assert(shapeType == CV_32S || shapeType == CV_64S);
        CV_Assert(shapeTensor.dims <= 1);
        size_t ndims = shapeTensor.total();
        CV_Assert(ndims <= (size_t)MatShape::MAX_DIMS);
        MatShape shape((int)ndims);
        for (int i = 0; i < (int)ndims; i++) {
            shape[i] = shapeType == CV_32S ? shapeTensor.at<int>(i) :
                (int)shapeTensor.at<int64_t>(i);
        }
        return shape;
    }

    bool getMemoryShapes(const std::vector<MatShape> &inputs,
                         const int requiredOutputs,
                         std::vector<MatShape> &outputs,
                         std::vector<MatShape> &internals) const CV_OVERRIDE
    {
        bool haveShapeSpec_ = haveShapeSpec();
        CV_Assert((inputs.size() == 1 && haveShapeSpec_) ||
                  (inputs.size() == 2 && !haveShapeSpec_));
        MatShape shapeSpec = newShapeDesc, outShape;

        if (inputs.size() == 2)
        {
            CV_Assert(this->inputs.size() == 2);
            Net::Impl* netimpl_ = reinterpret_cast<Net::Impl*>(netimpl);
            Mat shapeTensor = netimpl_->argTensor(this->inputs[1]);
            shapeSpec = shapeFromMat(shapeTensor);
        } else {
            CV_Assert(shapeSpec.dims >= 0);
        }
        outputs.assign(1, getOutShape(inputs[0], shapeSpec));
        internals.clear();
        return true;
    }

    void getTypes(const std::vector<MatType>& inputs,
        const int requiredOutputs,
        const int requiredInternals,
        std::vector<MatType>& outputs,
        std::vector<MatType>& internals) const CV_OVERRIDE
    {
        size_t ninputs = inputs.size();
        CV_Assert(ninputs == 1 || ninputs == 2);
        outputs.assign(requiredOutputs, inputs[0]);
        CV_Assert(requiredInternals == 0);
        internals.clear();
    }

    void finalize(InputArrayOfArrays, OutputArrayOfArrays outputs_arr) CV_OVERRIDE
    {
    }

    void forward(InputArrayOfArrays inputs_arr,
                 OutputArrayOfArrays outputs_arr,
                 OutputArrayOfArrays) CV_OVERRIDE
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        Size size = inputs_arr.size();
        int ninputs = size.area();
        bool haveShapeSpec_ = haveShapeSpec();
        CV_Assert((ninputs == 1 && haveShapeSpec_) ||
                  (ninputs == 2 && !haveShapeSpec_));

        MatShape inpShape = inputs_arr.shape(0);
        MatShape shapeSpec = newShapeDesc;
        if (!haveShapeSpec_) {
            Mat shapeTensor = inputs_arr.getMat(1);
            shapeSpec = shapeFromMat(shapeTensor);
        }
        MatShape outShape = getOutShape(inpShape, shapeSpec);
        reshapeAndCopyFirst(inputs_arr, outputs_arr, outShape);
    }
};

Ptr<Reshape2Layer> Reshape2Layer::create(const LayerParams& params)
{
    return Ptr<Reshape2Layer>(new Reshape2LayerImpl(params));
}

}
}
