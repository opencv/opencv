#include "../precomp.hpp"
#include "../op_inf_engine.hpp"
#include "../ie_ngraph.hpp"
#include<iostream>

namespace cv
{
    namespace dnn
    {

        class PixelShuffleLayerImpl CV_FINAL : public PixelShuffleLayer
        {
            public:

                PixelShuffleLayerImpl(const LayerParams& params)
                {
                    setParamsFrom(params);
                    upscaleFactor = params.get<float>("upscaleFactor");
                }
        };

        Ptr<Layer> PixelShuffleLayer::create(const LayerParams& params)
        {
            return Ptr<PixelShuffleLayer>(new PixelShuffleLayerImpl(params));
        }

    }
}
