// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "../net_impl.hpp"
#include "layers_common.hpp"
#include <opencv2/dnn.hpp>

namespace cv { namespace dnn {
CV__DNN_INLINE_NS_BEGIN

class LoopLayerImpl CV_FINAL : public LoopLayer
{
public:
    explicit LoopLayerImpl(const LayerParams& params) { setParamsFrom(params); }
    virtual ~LoopLayerImpl() = default;

    // Single subgraph: the loop body
    std::vector<Ptr<Graph> >* subgraphs() const CV_OVERRIDE { return &body_; }

    bool getMemoryShapes(const std::vector<MatShape>&,
                         const int requiredOutputs,
                         std::vector<MatShape>& outputs,
                         std::vector<MatShape>& internals) const CV_OVERRIDE
    {
        outputs.assign(std::max(1, requiredOutputs), MatShape());
        internals.clear();
        return false;
    }

    bool dynamicOutputShapes() const CV_OVERRIDE { return true; }

    // OPTIMIZATION: Reduced to a simple numeric check via double
    bool cond(InputArray arr) const CV_OVERRIDE
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
                      ("Loop condition: unsupported tensor type %s",
                       typeToString(inp->type()).c_str()));
        }
        return flag;
    }

private:
    mutable std::vector<Ptr<Graph> > body_;
};

Ptr<LoopLayer> LoopLayer::create(const LayerParams& params)
{
    return makePtr<LoopLayerImpl>(params);
}
CV__DNN_INLINE_NS_END
}} // namespace cv::dnn
