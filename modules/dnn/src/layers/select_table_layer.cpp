#include "../precomp.hpp"
#include "../op_inf_engine.hpp"
#include "../ie_ngraph.hpp"

namespace cv
{
    namespace dnn
    {

        class SelectTableLayerImpl CV_FINAL : public SelectTableLayer
        {
            public:

                SelectTableLayerImpl(const LayerParams& params)
                {
                    setParamsFrom(params);
                }
        };

        Ptr<Layer> SelectTableLayer::create(const LayerParams& params)
        {
            return Ptr<SelectTableLayer>(new SelectTableLayerImpl(params));
        }

    }
}
