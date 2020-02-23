#include "../precomp.hpp"
#include "../op_inf_engine.hpp"
#include "../ie_ngraph.hpp"

namespace cv
{
    namespace dnn
    {

        class NarrowTableLayerImpl CV_FINAL : public NarrowTableLayer
        {
            public:

                NarrowTableLayerImpl(const LayerParams& params)
                {
                    setParamsFrom(params);
                }
        };

        Ptr<Layer> NarrowTableLayer::create(const LayerParams& params)
        {
            return Ptr<NarrowTableLayer>(new NarrowTableLayerImpl(params));
        }

    }
}
