// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "../precomp.hpp"
#include "../net_impl.hpp"
#include "layers_common.hpp"
#include <opencv2/dnn.hpp>

namespace cv { namespace dnn {

class IfLayerImpl CV_FINAL : public IfLayer
{
public:
    explicit IfLayerImpl(const LayerParams& params)
    {
        setParamsFrom(params);
    }
    virtual ~IfLayerImpl() = default;

    std::vector<Ptr<Graph>>* subgraphs() const CV_OVERRIDE { return &thenelse; }

    bool getMemoryShapes(const std::vector<MatShape>& /*inputs*/,
                         const int requiredOutputs,
                         std::vector<MatShape>& outputs,
                         std::vector<MatShape>& internals) const CV_OVERRIDE
    {
        outputs.assign(std::max(1, requiredOutputs), MatShape());
        internals.clear();
        return false;
    }

    bool dynamicOutputShapes() const CV_OVERRIDE { return true; }

    int branch(InputArray arr) const CV_OVERRIDE
    {
        Mat buf, *inp;
        if (arr.kind() == _InputArray::MAT) {
            inp = (Mat*)arr.getObj();
        } else {
            buf = arr.getMat();
            inp = &buf;
        }
        CV_Assert(inp->total() == 1u);
        bool flag;
        switch (inp->depth())
        {
        case CV_8U: case CV_8S: case CV_Bool:
            flag = *inp->ptr<char>() != 0; break;
        case CV_16U: case CV_16S:
            flag = *inp->ptr<short>() != 0; break;
        case CV_16F:
            flag = *inp->ptr<hfloat>() != 0; break;
        case CV_16BF:
            flag = *inp->ptr<hfloat>() != 0; break;
        case CV_32U: case CV_32S:
            flag = *inp->ptr<int>() != 0; break;
        case CV_32F:
            flag = *inp->ptr<float>() != 0; break;
        case CV_64U: case CV_64S:
            flag = *inp->ptr<long long>() != 0; break;
        case CV_64F:
            flag = *inp->ptr<double>() != 0; break;
        default:
            CV_Error_(Error::StsBadArg,
                    ("If-layer condition: unsupported tensor type %s",
                    typeToString(inp->type()).c_str()));
        }
        return (int)!flag;
    }

private:
    mutable std::vector<Ptr<Graph>> thenelse;
};

Ptr<IfLayer> IfLayer::create(const LayerParams& params)
{
    return makePtr<IfLayerImpl>(params);
}

}}  // namespace cv::dnn
