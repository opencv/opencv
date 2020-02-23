#include "../precomp.hpp"
#include "../op_inf_engine.hpp"
#include "../ie_ngraph.hpp"

namespace cv
{
    namespace dnn
    {

        class CopyLayerImpl CV_FINAL : public CopyLayer
        {
            public:

                CopyLayerImpl(const LayerParams& params)
                {
                    setParamsFrom(params);
                    intype = params.get("intype", params.get<String>("_type"));
                    outtype = params.get("outtype", params.get<String>("_type"));
                    forceCopy = params.get<bool>("forceCopy", false);
                    dontCast = params.get<bool>("dontCast", false);
                }

                bool getMemoryShapes(const std::vector<MatShape> &inputs,
                         const int requiredOutputs,
                         std::vector<MatShape> &outputs,
                         std::vector<MatShape> &internals) const CV_OVERRIDE
                {
                    // For only one tensor as input
                    CV_Assert(inputs.size() == 1);
                    outputs.resize(1, inputs[0]);
                    return false;
                }
        };

        Ptr<Layer> CopyLayer::create(const LayerParams& params)
        {
            return Ptr<CopyLayer>(new CopyLayerImpl(params));
        }

    }
}
