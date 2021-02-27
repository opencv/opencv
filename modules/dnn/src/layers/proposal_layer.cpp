// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2017, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.
#include "../precomp.hpp"
#include "layers_common.hpp"
#include "../op_inf_engine.hpp"

#ifdef HAVE_DNN_NGRAPH
#include "../ie_ngraph.hpp"
#if INF_ENGINE_VER_MAJOR_GT(INF_ENGINE_RELEASE_2020_4)
#include <ngraph/op/proposal.hpp>
#else
#include <ngraph/op/experimental/layers/proposal.hpp>
#endif
#endif

namespace cv { namespace dnn {

class ProposalLayerImpl CV_FINAL : public ProposalLayer
{
public:
    ProposalLayerImpl(const LayerParams& params)
    {
        setParamsFrom(params);

        featStride = params.get<uint32_t>("feat_stride", 16);
        baseSize = params.get<uint32_t>("base_size", 16);
        // uint32_t minSize = params.get<uint32_t>("min_size", 16);
        keepTopBeforeNMS = params.get<uint32_t>("pre_nms_topn", 6000);
        keepTopAfterNMS = params.get<uint32_t>("post_nms_topn", 300);
        nmsThreshold = params.get<float>("nms_thresh", 0.7);
        ratios = params.get("ratio");
        scales = params.get("scale");

        {
            LayerParams lp;
            lp.set("step", featStride);
            lp.set("flip", false);
            lp.set("clip", false);
            lp.set("normalized_bbox", false);
            lp.set("offset", 0.5 * baseSize / featStride);

            // Unused values.
            float variance[] = {0.1f, 0.1f, 0.2f, 0.2f};
            lp.set("variance", DictValue::arrayReal<float*>(&variance[0], 4));

            // Compute widths and heights explicitly.
            std::vector<float> widths, heights;
            widths.reserve(ratios.size() * scales.size());
            heights.reserve(ratios.size() * scales.size());
            for (int i = 0; i < ratios.size(); ++i)
            {
                float ratio = ratios.get<float>(i);
                float width = std::floor(baseSize / sqrt(ratio) + 0.5f);
                float height = std::floor(width * ratio + 0.5f);
                for (int j = 0; j < scales.size(); ++j)
                {
                    float scale = scales.get<float>(j);
                    widths.push_back(scale * width);
                    heights.push_back(scale * height);
                }
            }
            lp.set("width", DictValue::arrayReal<float*>(&widths[0], widths.size()));
            lp.set("height", DictValue::arrayReal<float*>(&heights[0], heights.size()));

            priorBoxLayer = PriorBoxLayer::create(lp);
        }
        {
            int order[] = {0, 2, 3, 1};
            LayerParams lp;
            lp.set("order", DictValue::arrayInt<int*>(&order[0], 4));

            deltasPermute = PermuteLayer::create(lp);
            scoresPermute = PermuteLayer::create(lp);
        }
        {
            LayerParams lp;
            lp.set("code_type", "CENTER_SIZE");
            lp.set("num_classes", 1);
            lp.set("share_location", true);
            lp.set("background_label_id", 1);  // We won't pass background scores so set it out of range [0, num_classes)
            lp.set("variance_encoded_in_target", true);
            lp.set("keep_top_k", keepTopAfterNMS);
            lp.set("top_k", keepTopBeforeNMS);
            lp.set("nms_threshold", nmsThreshold);
            lp.set("normalized_bbox", false);
            lp.set("clip", true);

            detectionOutputLayer = DetectionOutputLayer::create(lp);
        }
    }

    virtual bool supportBackend(int backendId) CV_OVERRIDE
    {
#ifdef HAVE_INF_ENGINE
        if (backendId == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019 || backendId == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
        {
            bool isMyriad = preferableTarget == DNN_TARGET_MYRIAD || preferableTarget == DNN_TARGET_HDDL;
            return !isMyriad;
        }
#endif
        return backendId == DNN_BACKEND_OPENCV;
    }

    bool getMemoryShapes(const std::vector<MatShape> &inputs,
                         const int requiredOutputs,
                         std::vector<MatShape> &outputs,
                         std::vector<MatShape> &internals) const CV_OVERRIDE
    {
        // We need to allocate the following blobs:
        // - output priors from PriorBoxLayer
        // - permuted priors
        // - permuted scores
        CV_Assert(inputs.size() == 3);

        const MatShape& scores = inputs[0];
        const MatShape& bboxDeltas = inputs[1];

        std::vector<MatShape> layerInputs, layerOutputs, layerInternals;

        // Prior boxes layer.
        layerInputs.assign(1, scores);
        priorBoxLayer->getMemoryShapes(layerInputs, 1, layerOutputs, layerInternals);
        CV_Assert(layerOutputs.size() == 1);
        CV_Assert(layerInternals.empty());
        internals.push_back(layerOutputs[0]);

        // Scores permute layer.
        CV_Assert(scores.size() == 4);
        MatShape objectScores = scores;
        CV_Assert((scores[1] & 1) == 0);  // Number of channels is even.
        objectScores[1] /= 2;
        layerInputs.assign(1, objectScores);
        scoresPermute->getMemoryShapes(layerInputs, 1, layerOutputs, layerInternals);
        CV_Assert(layerOutputs.size() == 1);
        CV_Assert(layerInternals.empty());
        internals.push_back(layerOutputs[0]);

        // BBox predictions permute layer.
        layerInputs.assign(1, bboxDeltas);
        deltasPermute->getMemoryShapes(layerInputs, 1, layerOutputs, layerInternals);
        CV_Assert(layerOutputs.size() == 1);
        CV_Assert(layerInternals.empty());
        internals.push_back(layerOutputs[0]);

        // Detections layer.
        internals.push_back(shape(1, 1, keepTopAfterNMS, 7));

        outputs.resize(2);
        outputs[0] = shape(keepTopAfterNMS, 5);
        outputs[1] = shape(keepTopAfterNMS, 1);
        return false;
    }

    void finalize(InputArrayOfArrays inputs_arr, OutputArrayOfArrays) CV_OVERRIDE
    {
        std::vector<Mat> inputs;
        inputs_arr.getMatVector(inputs);

        std::vector<Mat> layerInputs;
        std::vector<Mat> layerOutputs;

        // Scores permute layer.
        Mat scores = getObjectScores(inputs[0]);
        layerInputs.assign(1, scores);
        layerOutputs.assign(1, Mat(shape(scores.size[0], scores.size[2],
                                         scores.size[3], scores.size[1]), CV_32FC1));
        scoresPermute->finalize(layerInputs, layerOutputs);

        // BBox predictions permute layer.
        const Mat& bboxDeltas = inputs[1];
        CV_Assert(bboxDeltas.dims == 4);
        layerInputs.assign(1, bboxDeltas);
        layerOutputs.assign(1, Mat(shape(bboxDeltas.size[0], bboxDeltas.size[2],
                                         bboxDeltas.size[3], bboxDeltas.size[1]), CV_32FC1));
        deltasPermute->finalize(layerInputs, layerOutputs);
    }

#ifdef HAVE_OPENCL
    bool forward_ocl(InputArrayOfArrays inputs_, OutputArrayOfArrays outputs_, OutputArrayOfArrays internals_)
    {
        std::vector<UMat> inputs;
        std::vector<UMat> outputs;
        std::vector<UMat> internals;

        if (inputs_.depth() == CV_16S)
            return false;

        inputs_.getUMatVector(inputs);
        outputs_.getUMatVector(outputs);
        internals_.getUMatVector(internals);

        CV_Assert(inputs.size() == 3);
        CV_Assert(internals.size() == 4);
        const UMat& scores = inputs[0];
        const UMat& bboxDeltas = inputs[1];
        const UMat& imInfo = inputs[2];
        UMat& priorBoxes = internals[0];
        UMat& permuttedScores = internals[1];
        UMat& permuttedDeltas = internals[2];
        UMat& detections = internals[3];

        CV_Assert(imInfo.total() >= 2);
        // We've chosen the smallest data type because we need just a shape from it.
        Mat szMat;
        imInfo.copyTo(szMat);
        int rows = (int)szMat.at<float>(0);
        int cols = (int)szMat.at<float>(1);
        umat_fakeImageBlob.create(shape(1, 1, rows, cols), CV_8UC1);
        umat_fakeImageBlob.setTo(0);

        // Generate prior boxes.
        std::vector<UMat> layerInputs(2), layerOutputs(1, priorBoxes);
        layerInputs[0] = scores;
        layerInputs[1] = umat_fakeImageBlob;
        priorBoxLayer->forward(layerInputs, layerOutputs, internals);

        // Permute scores.
        layerInputs.assign(1, getObjectScores(scores));
        layerOutputs.assign(1, permuttedScores);
        scoresPermute->forward(layerInputs, layerOutputs, internals);

        // Permute deltas.
        layerInputs.assign(1, bboxDeltas);
        layerOutputs.assign(1, permuttedDeltas);
        deltasPermute->forward(layerInputs, layerOutputs, internals);

        // Sort predictions by scores and apply NMS. DetectionOutputLayer allocates
        // output internally because of different number of objects after NMS.
        layerInputs.resize(4);
        layerInputs[0] = permuttedDeltas;
        layerInputs[1] = permuttedScores;
        layerInputs[2] = priorBoxes;
        layerInputs[3] = umat_fakeImageBlob;

        layerOutputs[0] = detections;
        detectionOutputLayer->forward(layerInputs, layerOutputs, internals);

        // DetectionOutputLayer produces 1x1xNx7 output where N might be less or
        // equal to keepTopAfterNMS. We fill the rest by zeros.
        const int numDets = layerOutputs[0].total() / 7;
        CV_Assert(numDets <= keepTopAfterNMS);

        MatShape s = shape(numDets, 7);
        layerOutputs[0] = layerOutputs[0].reshape(1, s.size(), &s[0]);

        // The boxes.
        UMat dst = outputs[0].rowRange(0, numDets);
        layerOutputs[0].colRange(3, 7).copyTo(dst.colRange(1, 5));
        dst.col(0).setTo(0);  // First column are batch ids. Keep it zeros too.

        // The scores.
        dst = outputs[1].rowRange(0, numDets);
        layerOutputs[0].col(2).copyTo(dst);

        return true;
    }
#endif

    void forward(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr, OutputArrayOfArrays internals_arr) CV_OVERRIDE
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        CV_OCL_RUN(IS_DNN_OPENCL_TARGET(preferableTarget) &&
                   OCL_PERFORMANCE_CHECK(ocl::Device::getDefault().isIntel()),
                   forward_ocl(inputs_arr, outputs_arr, internals_arr))

        if (inputs_arr.depth() == CV_16S)
        {
            forward_fallback(inputs_arr, outputs_arr, internals_arr);
            return;
        }

        std::vector<Mat> inputs, outputs, internals;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);
        internals_arr.getMatVector(internals);

        CV_Assert(inputs.size() == 3);
        CV_Assert(internals.size() == 4);
        const Mat& scores = inputs[0];
        const Mat& bboxDeltas = inputs[1];
        const Mat& imInfo = inputs[2];
        Mat& priorBoxes = internals[0];
        Mat& permuttedScores = internals[1];
        Mat& permuttedDeltas = internals[2];
        Mat& detections = internals[3];

        CV_Assert(imInfo.total() >= 2);
        // We've chosen the smallest data type because we need just a shape from it.
        // We don't allocate memory but just need the shape is correct.
        Mat fakeImageBlob(shape(1, 1, imInfo.at<float>(0), imInfo.at<float>(1)), CV_8UC1, NULL);

        // Generate prior boxes.
        std::vector<Mat> layerInputs(2), layerOutputs(1, priorBoxes);
        layerInputs[0] = scores;
        layerInputs[1] = fakeImageBlob;
        priorBoxLayer->forward(layerInputs, layerOutputs, internals);

        // Permute scores.
        layerInputs.assign(1, getObjectScores(scores));
        layerOutputs.assign(1, permuttedScores);
        scoresPermute->forward(layerInputs, layerOutputs, internals);

        // Permute deltas.
        layerInputs.assign(1, bboxDeltas);
        layerOutputs.assign(1, permuttedDeltas);
        deltasPermute->forward(layerInputs, layerOutputs, internals);

        // Sort predictions by scores and apply NMS. DetectionOutputLayer allocates
        // output internally because of different number of objects after NMS.
        layerInputs.resize(4);
        layerInputs[0] = permuttedDeltas;
        layerInputs[1] = permuttedScores;
        layerInputs[2] = priorBoxes;
        layerInputs[3] = fakeImageBlob;

        layerOutputs[0] = detections;
        detectionOutputLayer->forward(layerInputs, layerOutputs, internals);

        // DetectionOutputLayer produces 1x1xNx7 output where N might be less or
        // equal to keepTopAfterNMS. We fill the rest by zeros.
        const int numDets = layerOutputs[0].total() / 7;
        CV_Assert(numDets <= keepTopAfterNMS);

        // The boxes.
        layerOutputs[0] = layerOutputs[0].reshape(1, numDets);
        Mat dst = outputs[0].rowRange(0, numDets);
        layerOutputs[0].colRange(3, 7).copyTo(dst.colRange(1, 5));
        dst.col(0).setTo(0);  // First column are batch ids. Keep it zeros too.

        // The scores.
        dst = outputs[1].rowRange(0, numDets);
        layerOutputs[0].col(2).copyTo(dst);
    }

#ifdef HAVE_DNN_IE_NN_BUILDER_2019
    virtual Ptr<BackendNode> initInfEngine(const std::vector<Ptr<BackendWrapper> >&) CV_OVERRIDE
    {
        InferenceEngine::Builder::ProposalLayer ieLayer(name);

        ieLayer.setBaseSize(baseSize);
        ieLayer.setFeatStride(featStride);
        ieLayer.setMinSize(16);
        ieLayer.setNMSThresh(nmsThreshold);
        ieLayer.setPostNMSTopN(keepTopAfterNMS);
        ieLayer.setPreNMSTopN(keepTopBeforeNMS);

        std::vector<float> scalesVec(scales.size());
        for (int i = 0; i < scales.size(); ++i)
            scalesVec[i] = scales.get<float>(i);
        ieLayer.setScale(scalesVec);

        std::vector<float> ratiosVec(ratios.size());
        for (int i = 0; i < ratios.size(); ++i)
            ratiosVec[i] = ratios.get<float>(i);
        ieLayer.setRatio(ratiosVec);

        return Ptr<BackendNode>(new InfEngineBackendNode(ieLayer));
    }
#endif  // HAVE_DNN_IE_NN_BUILDER_2019


#ifdef HAVE_DNN_NGRAPH
    virtual Ptr<BackendNode> initNgraph(const std::vector<Ptr<BackendWrapper> >& inputs,
                                        const std::vector<Ptr<BackendNode> >& nodes) CV_OVERRIDE
    {
        CV_Assert(nodes.size() == 3);
        ngraph::op::ProposalAttrs attr;
        attr.base_size     = baseSize;
        attr.nms_thresh    = nmsThreshold;
        attr.feat_stride   = featStride;
        attr.min_size      = 16;
        attr.pre_nms_topn  = keepTopBeforeNMS;
        attr.post_nms_topn = keepTopAfterNMS;

        std::vector<float> ratiosVec(ratios.size());
        for (int i = 0; i < ratios.size(); ++i)
            ratiosVec[i] = ratios.get<float>(i);
        attr.ratio = ratiosVec;

        std::vector<float> scalesVec(scales.size());
        for (int i = 0; i < scales.size(); ++i)
            scalesVec[i] = scales.get<float>(i);
        attr.scale = scalesVec;

        auto& class_probs  = nodes[0].dynamicCast<InfEngineNgraphNode>()->node;
        auto& class_logits = nodes[1].dynamicCast<InfEngineNgraphNode>()->node;
        auto& image_shape  = nodes[2].dynamicCast<InfEngineNgraphNode>()->node;

        CV_Assert_N(image_shape->get_shape().size() == 2, image_shape->get_shape().front() == 1);
        auto shape   = std::make_shared<ngraph::op::Constant>(ngraph::element::i64,
                       ngraph::Shape{1},
                       std::vector<int64_t>{(int64_t)image_shape->get_shape().back()});
        auto reshape = std::make_shared<ngraph::op::v1::Reshape>(image_shape, shape, true);

        auto proposal = std::make_shared<ngraph::op::Proposal>(class_probs, class_logits, reshape, attr);
        return Ptr<BackendNode>(new InfEngineNgraphNode(proposal));
    }
#endif  // HAVE_DNN_NGRAPH

private:
    // A first half of channels are background scores. We need only a second one.
    static Mat getObjectScores(const Mat& m)
    {
        CV_Assert(m.dims == 4);
        CV_Assert(m.size[0] == 1);
        int channels = m.size[1];
        CV_Assert((channels & 1) == 0);
        return slice(m, Range::all(), Range(channels / 2, channels));
    }

#ifdef HAVE_OPENCL
    static UMat getObjectScores(const UMat& m)
    {
        CV_Assert(m.dims == 4);
        CV_Assert(m.size[0] == 1);
        int channels = m.size[1];
        CV_Assert((channels & 1) == 0);

        Range r = Range(channels / 2, channels);
        Range ranges[4] = { Range::all(), r, Range::all(), Range::all() };
        return m(&ranges[0]);
    }
#endif

    Ptr<PriorBoxLayer> priorBoxLayer;
    Ptr<DetectionOutputLayer> detectionOutputLayer;

    Ptr<PermuteLayer> deltasPermute;
    Ptr<PermuteLayer> scoresPermute;
    uint32_t keepTopBeforeNMS, keepTopAfterNMS, featStride, baseSize;
    float nmsThreshold;
    DictValue ratios, scales;
#ifdef HAVE_OPENCL
    UMat umat_fakeImageBlob;
#endif
};


Ptr<ProposalLayer> ProposalLayer::create(const LayerParams& params)
{
    return Ptr<ProposalLayer>(new ProposalLayerImpl(params));
}

}  // namespace dnn
}  // namespace cv
