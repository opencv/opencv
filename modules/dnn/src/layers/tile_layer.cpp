// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "layers_common.hpp"
#include "../op_inf_engine.hpp"
#include "../ie_ngraph.hpp"


#include <opencv2/dnn/shape_utils.hpp>

namespace cv { namespace dnn {

class TileLayerImpl CV_FINAL : public TileLayer
{
public:
    TileLayerImpl(const LayerParams& params)
    {
        setParamsFrom(params);
        if (params.has("repeats"))
        {
            DictValue param_repeats = params.get("repeats");
            int n_repeats = param_repeats.size();

            CV_Assert(n_repeats > 0);
            repeats.resize(n_repeats);
            for (int i = 0; i < n_repeats; i++)
                repeats[i] = param_repeats.get<int>(i);
        }
        else
            CV_Error(Error::StsNotImplemented, "Tile: repeats needs to be treated as parameter but it is missing.");
    }

    virtual bool supportBackend(int backendId) CV_OVERRIDE
    {
        return backendId == DNN_BACKEND_OPENCV ||
               backendId == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH;
    }

    virtual bool getMemoryShapes(const std::vector<MatShape> &inputs,
                                 const int requiredOutputs,
                                 std::vector<MatShape> &outputs,
                                 std::vector<MatShape> &internals) const CV_OVERRIDE
    {
        CV_CheckEQ(inputs.size(), 1ull, "Tile: one input is expected");
        int nrepeats = (int)repeats.size();

        // repeats must have the same length as input's dimension number
        if (inputs[0].size() > 1) {
            CV_CheckEQ(inputs[0].size(), repeats.size(), "Tile: repeats must be a 1D tensor of the same length as input's dimension number");
            outputs.assign(1, inputs[0]);
            for (int i = 0; i < nrepeats; i++)
            {
                outputs[0][i] *= repeats[i];
            }
        } else {
            CV_CheckGE(nrepeats, 1, "Tile: Provide at least one repeat along any dimension");
            outputs.assign(1, MatShape(repeats));
            if (inputs[0].size() == 1)
                outputs[0][nrepeats - 1] *= inputs[0][0];
        }

        return false;
    }

    void getTypes(const std::vector<MatType>& inputs,
        const int requiredOutputs,
        const int requiredInternals,
        std::vector<MatType>& outputs,
        std::vector<MatType>& internals) const CV_OVERRIDE
    {
        outputs.assign(requiredOutputs, inputs[0]);
    }

    void forward(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr, OutputArrayOfArrays internals_arr) CV_OVERRIDE
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        std::vector<Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);

        const Mat& data = inputs[0];
        Mat& out = outputs[0];

        Mat tmp = data.clone();
        MatShape tmp_shape = shape(tmp);
        MatShape out_shape = shape(out);
        int rep_i, ndims = data.dims;
        int dims = 1;
        if (ndims > 1){
            for (int i = 0; i < ndims; i++)
            {
                rep_i = repeats[i];
                if (rep_i != 1)
                {
                    tmp = tmp.reshape(0, dims);
                    tmp = cv::repeat(tmp, 1, rep_i);
                }
                dims *= out_shape[i];
            }
            tmp = tmp.reshape(0, out_shape);
        } else {
            for (int i = 0; i < repeats.size(); i++){
                tmp = tmp.reshape(0, dims);
                tmp = cv::repeat(tmp, repeats[i], 1);
                dims *= out_shape[i];
            }
            tmp = tmp.reshape(0, out_shape);
        }
        tmp.copyTo(out);
    }

#ifdef HAVE_DNN_NGRAPH
    virtual Ptr<BackendNode> initNgraph(const std::vector<Ptr<BackendWrapper> >& inputs,
                                        const std::vector<Ptr<BackendNode> >& nodes) CV_OVERRIDE
    {
        auto repeats_node = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{repeats.size()}, repeats.data());
        auto tile = std::make_shared<ov::op::v0::Tile>(nodes[0].dynamicCast<InfEngineNgraphNode>()->node, repeats_node);
        return Ptr<BackendNode>(new InfEngineNgraphNode(tile));
    }
#endif  // HAVE_DNN_NGRAPH


private:
    std::vector<int> repeats;
};

Ptr<TileLayer> TileLayer::create(const LayerParams& params)
{
    return makePtr<TileLayerImpl>(params);
}

}} // namespace cv::dnn
