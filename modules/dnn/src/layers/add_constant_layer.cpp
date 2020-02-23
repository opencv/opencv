#include "../precomp.hpp"
#include "layers_common.hpp"
#include "../op_inf_engine.hpp"
#include "../ie_ngraph.hpp"

namespace cv
{
    namespace dnn
    {

        class AddConstantLayerImpl CV_FINAL : public AddConstantLayer
        {
            public:

                AddConstantLayerImpl(const LayerParams& params)
                {
                    setParamsFrom(params);
                    constant_scalar = params.get<float>("constant_scalar");
                    inplace = params.get<bool>("inplace", false);
                }
        };

        Ptr<Layer> AddConstantLayer::create(const LayerParams& params)
        {
            return Ptr<AddConstantLayer>(new AddConstantLayerImpl(params));
        }

    }
}
