#include "../precomp.hpp"
#include "layers_common.hpp"

#include <limits.h> // for INT_MAX
#include <string>
#include "../nms.inl.hpp"

namespace cv
{
namespace dnn
{

class NonMaxSuppressionLayerImpl CV_FINAL : public NonMaxSuppressionLayer
{
public:
    NonMaxSuppressionLayerImpl(const LayerParams& params)
    {
        setParamsFrom(params);

        // 0: [y1, x1, y2, x2] for TF models; 1: [cx, cy, w, h] for PyTorch models
        center_point_box = params.get<int>("center_point_box", 0);
        max_output_boxes_per_class = params.get<int>("max_output_boxes_per_class", INT_MAX);
        iou_threshold = params.get<float>("iou_threshold", 0); // keep if iou <= iou_threshold
        score_threshold = params.get<float>("score_threshold", 0); // keep if score >= score_threshold

        // WARNINGS: magic number that works for most of the cases
        top_k = 5000;
        keep_top_k = 650;
    }

    virtual bool supportBackend(int backendId) CV_OVERRIDE
    {
        return backendId == DNN_BACKEND_OPENCV;
    }

    bool getMemoryShapes(const std::vector<MatShape> &inputs,
                         const int requiredOutputs,
                         std::vector<MatShape> &outputs,
                         std::vector<MatShape> &internals) const CV_OVERRIDE
    {
        // inputs[0]: boxes, [num_batches, num_boxes, 4]
        // inputs[1]: scores, [num_batches, num_classes, num_boxes]
        CV_Assert(inputs.size() == 2); // support with boxes & scores as inputs only
        CV_Assert(inputs[0][0] == inputs[1][0]); // same batch size
        CV_Assert(inputs[0][1] == inputs[1][2]); // same spatial dimension

        int _num_batches = inputs[0][0];
        int _num_classes = inputs[1][1];
        // outputs[0]: selected_indices, num_selected_indices * [batch_index, class_index, box_index]
        // consider the case whose _num_batches == 1 & _num_classes == 1
        outputs.resize(1, shape(keep_top_k, 3));

        return false;
    }

    static inline float rect2dOverlap(const Rect2d& a, const Rect2d& b)
    {
        return 1.f - static_cast<float>(jaccardDistance(a, b));
    }

    void forward(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr, OutputArrayOfArrays internals_arr) CV_OVERRIDE
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        std::vector<Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);

        int num_batches = inputs[0].size[0];
        int num_boxes = inputs[0].size[1];

        std::vector<Rect2d> boxes;
        std::vector<float> scores;
        // Retrieve bboxes
        boxes.resize(num_boxes);
        const float* ptr_boxes = (float*)inputs[0].data;
        if (center_point_box == 1) // num_boxes * [cx, cy, w, h]
        {
            float cx, cy, w, h;
            for (size_t i = 0; i < boxes.size(); i++)
            {
                Rect2d& box = boxes[i];
                cx = ptr_boxes[i * 4];
                cy = ptr_boxes[i * 4 + 1];
                w  = ptr_boxes[i * 4 + 2];
                h  = ptr_boxes[i * 4 + 3];

                box.x = cx - 0.5 * w;
                box.y = cy - 0.5 * h;
                box.width = w;
                box.height = h;
            }
        }
        else // num_boxes * [y1, x1, y2, x2]
        {
            float x1, y1, x2, y2;
            for (size_t i = 0; i < boxes.size(); i++)
            {
                Rect2d& box = boxes[i];
                y1 = ptr_boxes[i * 4];
                x1 = ptr_boxes[i * 4 + 1];
                y2 = ptr_boxes[i * 4 + 2];
                x2 = ptr_boxes[i * 4 + 3];

                box.x = x1;
                box.y = y1;
                box.width = x2 - x1;
                box.height = y2 - y1;
            }
        }
        // Retrieve scores
        const float* ptr_scores = (float*)inputs[1].data;
        if (inputs[1].isContinuous())
        {
            std::cout << "It is continuous!!!" << std::endl;
            scores.assign(ptr_scores, ptr_scores + inputs[1].total());
        }
        else
        {
            scores.resize(num_boxes);
            for (size_t i = 0; i < scores.size(); i++)
            {
                scores[i] = ptr_scores[i];
            }
        }

        // NMS
        std::vector<int> keep_indices;
        NMSFast_(boxes, scores, score_threshold, iou_threshold, 1.0, top_k, keep_indices, rect2dOverlap, keep_top_k);

        // Store to output
        outputs[0].setTo(-1);
        if (keep_indices.size() == 0)
            return;

        float* outputsData = outputs[0].ptr<float>();
        for (int i = 0; i < keep_indices.size(); i++)
        {
            outputsData[i * 3] = 0;
            outputsData[i * 3 + 1] = 0;
            outputsData[i * 3 + 2] = keep_indices[i];
        }
        outputs_arr.assign(outputs);
    }
};

Ptr<NonMaxSuppressionLayer> NonMaxSuppressionLayer::create(const LayerParams& params)
{
    return Ptr<NonMaxSuppressionLayer>(new NonMaxSuppressionLayerImpl(params));
}

}
}
