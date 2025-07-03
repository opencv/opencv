#include "../precomp.hpp"
#include "../net_impl.hpp"
#include "layers_common.hpp"
#include <opencv2/dnn.hpp>
#include<iostream>

namespace cv { namespace dnn {

class IfLayerImpl CV_FINAL : public IfLayer
{
public:
    IfLayerImpl(const LayerParams& params)
    {
        // pull cond
        condIndex = params.get<int>("cond");

        // unpack then/else from blobs[0] & blobs[1]
        constexpr int PTR_BYTES = int(sizeof(Ptr<Graph>));
        thenGraph = *reinterpret_cast<const Ptr<Graph>*>(params.blobs[0].ptr());
        elseGraph = *reinterpret_cast<const Ptr<Graph>*>(params.blobs[1].ptr());

        // now that all the pieces are set, you can do the usual
        setParamsFrom(params);
    }


    Ptr<Graph> then_else(bool flag) const CV_OVERRIDE
    {
        std::cout<<"got the call in then_else function, flag: "<<flag<<std::endl;
        return flag ? thenGraph : elseGraph;
    }

private:
    Ptr<Graph> thenGraph;
    Ptr<Graph> elseGraph;
    int condIndex;
};

Ptr<IfLayer> IfLayer::create(const LayerParams& params)
{
    std::cout<<"In the if create function"<<std::endl;
    return makePtr<IfLayerImpl>(params);
}

}}  // namespace cv::dnn
