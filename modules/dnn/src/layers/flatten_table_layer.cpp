#include "../precomp.hpp"
#include "../op_inf_engine.hpp"
#include "../ie_ngraph.hpp"

namespace cv
{
    namespace dnn
    {

        class FlattenTableLayerImpl CV_FINAL : public FlattenTableLayer
        {
            public:

                FlattenTableLayerImpl(const LayerParams& params)
                {
                    setParamsFrom(params);
                }
        };

        Ptr<Layer> FlattenTableLayer::create(const LayerParams& params)
        {
            return Ptr<FlattenTableLayer>(new FlattenTableLayerImpl(params));
        }

    }
}
