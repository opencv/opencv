#include "../precomp.hpp"
#include "../op_inf_engine.hpp"
#include "../ie_ngraph.hpp"

namespace cv
{
    namespace dnn
    {

        class ParallelTableLayerImpl CV_FINAL : public ParallelTableLayer
        {
            public:

                ParallelTableLayerImpl(const LayerParams& params)
                {
                    setParamsFrom(params);
                }
        };

        Ptr<Layer> ParallelTableLayer::create(const LayerParams& params)
        {
            return Ptr<ParallelTableLayer>(new ParallelTableLayerImpl(params));
        }

    }
}
