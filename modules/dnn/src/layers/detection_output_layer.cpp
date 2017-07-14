/*M ///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
// Copyright (C) 2017, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "../precomp.hpp"
#include "layers_common.hpp"
#include <float.h>
#include <string>
#include <caffe.pb.h>

namespace cv
{
namespace dnn
{

namespace util
{

template <typename T>
static inline bool SortScorePairDescend(const std::pair<float, T>& pair1,
                          const std::pair<float, T>& pair2)
{
    return pair1.first > pair2.first;
}

} // namespace

class DetectionOutputLayerImpl : public DetectionOutputLayer
{
public:
    unsigned _numClasses;
    bool _shareLocation;
    int _numLocClasses;

    int _backgroundLabelId;

    typedef caffe::PriorBoxParameter_CodeType CodeType;
    CodeType _codeType;

    bool _varianceEncodedInTarget;
    int _keepTopK;
    float _confidenceThreshold;

    float _nmsThreshold;
    int _topK;

    enum { _numAxes = 4 };
    static const std::string _layerName;

    typedef std::map<int, std::vector<caffe::NormalizedBBox> > LabelBBox;

    bool getParameterDict(const LayerParams &params,
                          const std::string &parameterName,
                          DictValue& result)
    {
        if (!params.has(parameterName))
        {
            return false;
        }

        result = params.get(parameterName);
        return true;
    }

    template<typename T>
    T getParameter(const LayerParams &params,
                   const std::string &parameterName,
                   const size_t &idx=0,
                   const bool required=true,
                   const T& defaultValue=T())
    {
        DictValue dictValue;
        bool success = getParameterDict(params, parameterName, dictValue);
        if(!success)
        {
            if(required)
            {
                std::string message = _layerName;
                message += " layer parameter does not contain ";
                message += parameterName;
                message += " parameter.";
                CV_ErrorNoReturn(Error::StsBadArg, message);
            }
            else
            {
                return defaultValue;
            }
        }
        return dictValue.get<T>(idx);
    }

    void getCodeType(const LayerParams &params)
    {
        String codeTypeString = params.get<String>("code_type").toLowerCase();
        if (codeTypeString == "corner")
            _codeType = caffe::PriorBoxParameter_CodeType_CORNER;
        else if (codeTypeString == "center_size")
            _codeType = caffe::PriorBoxParameter_CodeType_CENTER_SIZE;
        else
            _codeType = caffe::PriorBoxParameter_CodeType_CORNER;
    }

    DetectionOutputLayerImpl(const LayerParams &params)
    {
        _numClasses = getParameter<unsigned>(params, "num_classes");
        _shareLocation = getParameter<bool>(params, "share_location");
        _numLocClasses = _shareLocation ? 1 : _numClasses;
        _backgroundLabelId = getParameter<int>(params, "background_label_id");
        _varianceEncodedInTarget = getParameter<bool>(params, "variance_encoded_in_target", 0, false, false);
        _keepTopK = getParameter<int>(params, "keep_top_k");
        _confidenceThreshold = getParameter<float>(params, "confidence_threshold", 0, false, -FLT_MAX);
        _topK = getParameter<int>(params, "top_k", 0, false, -1);

        getCodeType(params);

        // Parameters used in nms.
        _nmsThreshold = getParameter<float>(params, "nms_threshold");
        CV_Assert(_nmsThreshold > 0.);

        setParamsFrom(params);
    }

    void checkInputs(const std::vector<Mat*> &inputs)
    {
        for (size_t i = 1; i < inputs.size(); i++)
        {
            CV_Assert(inputs[i]->size == inputs[0]->size);
        }
    }

    bool getMemoryShapes(const std::vector<MatShape> &inputs,
                         const int requiredOutputs,
                         std::vector<MatShape> &outputs,
                         std::vector<MatShape> &internals) const
    {
        CV_Assert(inputs.size() > 0);
        CV_Assert(inputs[0][0] == inputs[1][0]);

        int numPriors = inputs[2][2] / 4;
        CV_Assert((numPriors * _numLocClasses * 4) == inputs[0][1]);
        CV_Assert(int(numPriors * _numClasses) == inputs[1][1]);

        // num() and channels() are 1.
        // Since the number of bboxes to be kept is unknown before nms, we manually
        // set it to (fake) 1.
        // Each row is a 7 dimension std::vector, which stores
        // [image_id, label, confidence, xmin, ymin, xmax, ymax]
        outputs.resize(1, shape(1, 1, 1, 7));

        return false;
    }

    void forward(std::vector<Mat*> &inputs, std::vector<Mat> &outputs, std::vector<Mat> &internals)
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        std::vector<LabelBBox> allDecodedBBoxes;
        std::vector<std::vector<std::vector<float> > > allConfidenceScores;

        int num = inputs[0]->size[0];

        // extract predictions from input layers
        {
            int numPriors = inputs[2]->size[2] / 4;

            const float* locationData = inputs[0]->ptr<float>();
            const float* confidenceData = inputs[1]->ptr<float>();
            const float* priorData = inputs[2]->ptr<float>();

            // Retrieve all location predictions
            std::vector<LabelBBox> allLocationPredictions;
            GetLocPredictions(locationData, num, numPriors, _numLocClasses,
                              _shareLocation, allLocationPredictions);

            // Retrieve all confidences
            GetConfidenceScores(confidenceData, num, numPriors, _numClasses, allConfidenceScores);

            // Retrieve all prior bboxes
            std::vector<caffe::NormalizedBBox> priorBBoxes;
            std::vector<std::vector<float> > priorVariances;
            GetPriorBBoxes(priorData, numPriors, priorBBoxes, priorVariances);

            // Decode all loc predictions to bboxes
            DecodeBBoxesAll(allLocationPredictions, priorBBoxes, priorVariances, num,
                            _shareLocation, _numLocClasses, _backgroundLabelId,
                            _codeType, _varianceEncodedInTarget, false, allDecodedBBoxes);
        }

        size_t numKept = 0;
        std::vector<std::map<int, std::vector<int> > > allIndices;
        for (int i = 0; i < num; ++i)
        {
            numKept += processDetections_(allDecodedBBoxes[i], allConfidenceScores[i], allIndices);
        }

        if (numKept == 0)
        {
            return;
        }
        int outputShape[] = {1, 1, (int)numKept, 7};
        outputs[0].create(4, outputShape, CV_32F);
        float* outputsData = outputs[0].ptr<float>();

        size_t count = 0;
        for (int i = 0; i < num; ++i)
        {
            count += outputDetections_(i, &outputsData[count * 7],
                                       allDecodedBBoxes[i], allConfidenceScores[i],
                                       allIndices[i]);
        }
        CV_Assert(count == numKept);
    }

    size_t outputDetections_(
            const int i, float* outputsData,
            const LabelBBox& decodeBBoxes, const std::vector<std::vector<float> >& confidenceScores,
            const std::map<int, std::vector<int> >& indicesMap
    )
    {
        size_t count = 0;
        for (std::map<int, std::vector<int> >::const_iterator it = indicesMap.begin(); it != indicesMap.end(); ++it)
        {
            int label = it->first;
            if (confidenceScores.size() <= label)
                CV_ErrorNoReturn_(cv::Error::StsError, ("Could not find confidence predictions for label %d", label));
            const std::vector<float>& scores = confidenceScores[label];
            int locLabel = _shareLocation ? -1 : label;
            LabelBBox::const_iterator label_bboxes = decodeBBoxes.find(locLabel);
            if (label_bboxes == decodeBBoxes.end())
                CV_ErrorNoReturn_(cv::Error::StsError, ("Could not find location predictions for label %d", locLabel));
            const std::vector<int>& indices = it->second;

            for (size_t j = 0; j < indices.size(); ++j, ++count)
            {
                int idx = indices[j];
                const caffe::NormalizedBBox& decode_bbox = label_bboxes->second[idx];
                outputsData[count * 7] = i;
                outputsData[count * 7 + 1] = label;
                outputsData[count * 7 + 2] = scores[idx];
                outputsData[count * 7 + 3] = decode_bbox.xmin();
                outputsData[count * 7 + 4] = decode_bbox.ymin();
                outputsData[count * 7 + 5] = decode_bbox.xmax();
                outputsData[count * 7 + 6] = decode_bbox.ymax();
            }
        }
        return count;
    }

    size_t processDetections_(
            const LabelBBox& decodeBBoxes, const std::vector<std::vector<float> >& confidenceScores,
            std::vector<std::map<int, std::vector<int> > >& allIndices
    )
    {
        std::map<int, std::vector<int> > indices;
        size_t numDetections = 0;
        for (int c = 0; c < (int)_numClasses; ++c)
        {
            if (c == _backgroundLabelId)
                continue; // Ignore background class.
            if (c >= confidenceScores.size())
                CV_ErrorNoReturn_(cv::Error::StsError, ("Could not find confidence predictions for label %d", c));

            const std::vector<float>& scores = confidenceScores[c];
            int label = _shareLocation ? -1 : c;

            LabelBBox::const_iterator label_bboxes = decodeBBoxes.find(label);
            if (label_bboxes == decodeBBoxes.end())
                CV_ErrorNoReturn_(cv::Error::StsError, ("Could not find location predictions for label %d", label));
            ApplyNMSFast(label_bboxes->second, scores, _confidenceThreshold, _nmsThreshold, 1.0, _topK, indices[c]);
            numDetections += indices[c].size();
        }
        if (_keepTopK > -1 && numDetections > (size_t)_keepTopK)
        {
            std::vector<std::pair<float, std::pair<int, int> > > scoreIndexPairs;
            for (std::map<int, std::vector<int> >::iterator it = indices.begin();
                 it != indices.end(); ++it)
            {
                int label = it->first;
                const std::vector<int>& labelIndices = it->second;
                if (label >= confidenceScores.size())
                    CV_ErrorNoReturn_(cv::Error::StsError, ("Could not find location predictions for label %d", label));
                const std::vector<float>& scores = confidenceScores[label];
                for (size_t j = 0; j < labelIndices.size(); ++j)
                {
                    size_t idx = labelIndices[j];
                    CV_Assert(idx < scores.size());
                    scoreIndexPairs.push_back(std::make_pair(scores[idx], std::make_pair(label, idx)));
                }
            }
            // Keep outputs k results per image.
            std::sort(scoreIndexPairs.begin(), scoreIndexPairs.end(),
                      util::SortScorePairDescend<std::pair<int, int> >);
            scoreIndexPairs.resize(_keepTopK);

            std::map<int, std::vector<int> > newIndices;
            for (size_t j = 0; j < scoreIndexPairs.size(); ++j)
            {
                int label = scoreIndexPairs[j].second.first;
                int idx = scoreIndexPairs[j].second.second;
                newIndices[label].push_back(idx);
            }
            allIndices.push_back(newIndices);
            return (size_t)_keepTopK;
        }
        else
        {
            allIndices.push_back(indices);
            return numDetections;
        }
    }


    // **************************************************************
    // Utility functions
    // **************************************************************

    // Compute bbox size
    template<bool normalized>
    static float BBoxSize(const caffe::NormalizedBBox& bbox)
    {
        if (bbox.xmax() < bbox.xmin() || bbox.ymax() < bbox.ymin())
        {
            return 0; // If bbox is invalid (e.g. xmax < xmin or ymax < ymin), return 0.
        }
        else
        {
            if (bbox.has_size())
            {
                return bbox.size();
            }
            else
            {
                float width = bbox.xmax() - bbox.xmin();
                float height = bbox.ymax() - bbox.ymin();
                if (normalized)
                {
                    return width * height;
                }
                else
                {
                    // If bbox is not within range [0, 1].
                    return (width + 1) * (height + 1);
                }
            }
        }
    }


    // Decode a bbox according to a prior bbox
    template<bool variance_encoded_in_target>
    static void DecodeBBox(
        const caffe::NormalizedBBox& prior_bbox, const std::vector<float>& prior_variance,
        const CodeType code_type,
        const bool clip_bbox, const caffe::NormalizedBBox& bbox,
        caffe::NormalizedBBox& decode_bbox)
    {
        float bbox_xmin = variance_encoded_in_target ? bbox.xmin() : prior_variance[0] * bbox.xmin();
        float bbox_ymin = variance_encoded_in_target ? bbox.ymin() : prior_variance[1] * bbox.ymin();
        float bbox_xmax = variance_encoded_in_target ? bbox.xmax() : prior_variance[2] * bbox.xmax();
        float bbox_ymax = variance_encoded_in_target ? bbox.ymax() : prior_variance[3] * bbox.ymax();
        switch(code_type)
        {
            case caffe::PriorBoxParameter_CodeType_CORNER:
                decode_bbox.set_xmin(prior_bbox.xmin() + bbox_xmin);
                decode_bbox.set_ymin(prior_bbox.ymin() + bbox_ymin);
                decode_bbox.set_xmax(prior_bbox.xmax() + bbox_xmax);
                decode_bbox.set_ymax(prior_bbox.ymax() + bbox_ymax);
                break;
            case caffe::PriorBoxParameter_CodeType_CENTER_SIZE:
            {
                float prior_width = prior_bbox.xmax() - prior_bbox.xmin();
                CV_Assert(prior_width > 0);
                float prior_height = prior_bbox.ymax() - prior_bbox.ymin();
                CV_Assert(prior_height > 0);
                float prior_center_x = (prior_bbox.xmin() + prior_bbox.xmax()) * .5;
                float prior_center_y = (prior_bbox.ymin() + prior_bbox.ymax()) * .5;

                float decode_bbox_center_x, decode_bbox_center_y;
                float decode_bbox_width, decode_bbox_height;
                decode_bbox_center_x = bbox_xmin * prior_width + prior_center_x;
                decode_bbox_center_y = bbox_ymin * prior_height + prior_center_y;
                decode_bbox_width = exp(bbox_xmax) * prior_width;
                decode_bbox_height = exp(bbox_ymax) * prior_height;
                decode_bbox.set_xmin(decode_bbox_center_x - decode_bbox_width * .5);
                decode_bbox.set_ymin(decode_bbox_center_y - decode_bbox_height * .5);
                decode_bbox.set_xmax(decode_bbox_center_x + decode_bbox_width * .5);
                decode_bbox.set_ymax(decode_bbox_center_y + decode_bbox_height * .5);
                break;
            }
            default:
                CV_ErrorNoReturn(Error::StsBadArg, "Unknown type.");
        };
        if (clip_bbox)
        {
            // Clip the caffe::NormalizedBBox such that the range for each corner is [0, 1]
            decode_bbox.set_xmin(std::max(std::min(decode_bbox.xmin(), 1.f), 0.f));
            decode_bbox.set_ymin(std::max(std::min(decode_bbox.ymin(), 1.f), 0.f));
            decode_bbox.set_xmax(std::max(std::min(decode_bbox.xmax(), 1.f), 0.f));
            decode_bbox.set_ymax(std::max(std::min(decode_bbox.ymax(), 1.f), 0.f));
        }
        decode_bbox.clear_size();
        decode_bbox.set_size(BBoxSize<true>(decode_bbox));
    }

    // Decode a set of bboxes according to a set of prior bboxes
    static void DecodeBBoxes(
        const std::vector<caffe::NormalizedBBox>& prior_bboxes,
        const std::vector<std::vector<float> >& prior_variances,
        const CodeType code_type, const bool variance_encoded_in_target,
        const bool clip_bbox, const std::vector<caffe::NormalizedBBox>& bboxes,
        std::vector<caffe::NormalizedBBox>& decode_bboxes)
    {
        CV_Assert(prior_bboxes.size() == prior_variances.size());
        CV_Assert(prior_bboxes.size() == bboxes.size());
        size_t num_bboxes = prior_bboxes.size();
        CV_Assert(num_bboxes == 0 || prior_variances[0].size() == 4);
        decode_bboxes.clear(); decode_bboxes.resize(num_bboxes);
        if(variance_encoded_in_target)
        {
            for (int i = 0; i < num_bboxes; ++i)
                DecodeBBox<true>(prior_bboxes[i], prior_variances[i], code_type,
                                 clip_bbox, bboxes[i], decode_bboxes[i]);
        }
        else
        {
            for (int i = 0; i < num_bboxes; ++i)
                DecodeBBox<false>(prior_bboxes[i], prior_variances[i], code_type,
                                  clip_bbox, bboxes[i], decode_bboxes[i]);
        }
    }

    // Decode all bboxes in a batch
    static void DecodeBBoxesAll(const std::vector<LabelBBox>& all_loc_preds,
        const std::vector<caffe::NormalizedBBox>& prior_bboxes,
        const std::vector<std::vector<float> >& prior_variances,
        const int num, const bool share_location,
        const int num_loc_classes, const int background_label_id,
        const CodeType code_type, const bool variance_encoded_in_target,
        const bool clip, std::vector<LabelBBox>& all_decode_bboxes)
    {
        CV_Assert(all_loc_preds.size() == num);
        all_decode_bboxes.clear();
        all_decode_bboxes.resize(num);
        for (int i = 0; i < num; ++i)
        {
            // Decode predictions into bboxes.
            const LabelBBox& loc_preds = all_loc_preds[i];
            LabelBBox& decode_bboxes = all_decode_bboxes[i];
            for (int c = 0; c < num_loc_classes; ++c)
            {
                int label = share_location ? -1 : c;
                if (label == background_label_id)
                    continue; // Ignore background class.
                LabelBBox::const_iterator label_loc_preds = loc_preds.find(label);
                if (label_loc_preds == loc_preds.end())
                    CV_ErrorNoReturn_(cv::Error::StsError, ("Could not find location predictions for label %d", label));
                DecodeBBoxes(prior_bboxes, prior_variances,
                             code_type, variance_encoded_in_target, clip,
                             label_loc_preds->second, decode_bboxes[label]);
            }
        }
    }

    // Get prior bounding boxes from prior_data
    //    prior_data: 1 x 2 x num_priors * 4 x 1 blob.
    //    num_priors: number of priors.
    //    prior_bboxes: stores all the prior bboxes in the format of caffe::NormalizedBBox.
    //    prior_variances: stores all the variances needed by prior bboxes.
    static void GetPriorBBoxes(const float* priorData, const int& numPriors,
                        std::vector<caffe::NormalizedBBox>& priorBBoxes,
                        std::vector<std::vector<float> >& priorVariances)
    {
        priorBBoxes.clear(); priorBBoxes.resize(numPriors);
        priorVariances.clear(); priorVariances.resize(numPriors);
        for (int i = 0; i < numPriors; ++i)
        {
            int startIdx = i * 4;
            caffe::NormalizedBBox& bbox = priorBBoxes[i];
            bbox.set_xmin(priorData[startIdx]);
            bbox.set_ymin(priorData[startIdx + 1]);
            bbox.set_xmax(priorData[startIdx + 2]);
            bbox.set_ymax(priorData[startIdx + 3]);
            bbox.set_size(BBoxSize<true>(bbox));
        }

        for (int i = 0; i < numPriors; ++i)
        {
            int startIdx = (numPriors + i) * 4;
            // not needed here: priorVariances[i].clear();
            for (int j = 0; j < 4; ++j)
            {
                priorVariances[i].push_back(priorData[startIdx + j]);
            }
        }
    }

    // Get location predictions from loc_data.
    //    loc_data: num x num_preds_per_class * num_loc_classes * 4 blob.
    //    num: the number of images.
    //    num_preds_per_class: number of predictions per class.
    //    num_loc_classes: number of location classes. It is 1 if share_location is
    //      true; and is equal to number of classes needed to predict otherwise.
    //    share_location: if true, all classes share the same location prediction.
    //    loc_preds: stores the location prediction, where each item contains
    //      location prediction for an image.
    static void GetLocPredictions(const float* locData, const int num,
                           const int numPredsPerClass, const int numLocClasses,
                           const bool shareLocation, std::vector<LabelBBox>& locPreds)
    {
        locPreds.clear();
        if (shareLocation)
        {
            CV_Assert(numLocClasses == 1);
        }
        locPreds.resize(num);
        for (int i = 0; i < num; ++i, locData += numPredsPerClass * numLocClasses * 4)
        {
            LabelBBox& labelBBox = locPreds[i];
            for (int p = 0; p < numPredsPerClass; ++p)
            {
                int startIdx = p * numLocClasses * 4;
                for (int c = 0; c < numLocClasses; ++c)
                {
                    int label = shareLocation ? -1 : c;
                    if (labelBBox.find(label) == labelBBox.end())
                    {
                        labelBBox[label].resize(numPredsPerClass);
                    }
                    caffe::NormalizedBBox& bbox = labelBBox[label][p];
                    bbox.set_xmin(locData[startIdx + c * 4]);
                    bbox.set_ymin(locData[startIdx + c * 4 + 1]);
                    bbox.set_xmax(locData[startIdx + c * 4 + 2]);
                    bbox.set_ymax(locData[startIdx + c * 4 + 3]);
                }
            }
        }
    }

    // Get confidence predictions from conf_data.
    //    conf_data: num x num_preds_per_class * num_classes blob.
    //    num: the number of images.
    //    num_preds_per_class: number of predictions per class.
    //    num_classes: number of classes.
    //    conf_preds: stores the confidence prediction, where each item contains
    //      confidence prediction for an image.
    static void GetConfidenceScores(const float* confData, const int num,
                             const int numPredsPerClass, const int numClasses,
                             std::vector<std::vector<std::vector<float> > >& confPreds)
    {
        confPreds.clear(); confPreds.resize(num);
        for (int i = 0; i < num; ++i, confData += numPredsPerClass * numClasses)
        {
            std::vector<std::vector<float> >& labelScores = confPreds[i];
            labelScores.resize(numClasses);
            for (int c = 0; c < numClasses; ++c)
            {
                std::vector<float>& classLabelScores = labelScores[c];
                classLabelScores.resize(numPredsPerClass);
                for (int p = 0; p < numPredsPerClass; ++p)
                {
                    classLabelScores[p] = confData[p * numClasses + c];
                }
            }
        }
    }

    // Do non maximum suppression given bboxes and scores.
    // Inspired by Piotr Dollar's NMS implementation in EdgeBox.
    // https://goo.gl/jV3JYS
    //    bboxes: a set of bounding boxes.
    //    scores: a set of corresponding confidences.
    //    score_threshold: a threshold used to filter detection results.
    //    nms_threshold: a threshold used in non maximum suppression.
    //    top_k: if not -1, keep at most top_k picked indices.
    //    indices: the kept indices of bboxes after nms.
    static void ApplyNMSFast(const std::vector<caffe::NormalizedBBox>& bboxes,
          const std::vector<float>& scores, const float score_threshold,
          const float nms_threshold, const float eta, const int top_k,
          std::vector<int>& indices)
    {
        CV_Assert(bboxes.size() == scores.size());

        // Get top_k scores (with corresponding indices).
        std::vector<std::pair<float, int> > score_index_vec;
        GetMaxScoreIndex(scores, score_threshold, top_k, score_index_vec);

        // Do nms.
        float adaptive_threshold = nms_threshold;
        indices.clear();
        while (score_index_vec.size() != 0) {
            const int idx = score_index_vec.front().second;
            bool keep = true;
            for (int k = 0; k < (int)indices.size() && keep; ++k) {
                const int kept_idx = indices[k];
                float overlap = JaccardOverlap<true>(bboxes[idx], bboxes[kept_idx]);
                keep = overlap <= adaptive_threshold;
            }
            if (keep)
                indices.push_back(idx);
            score_index_vec.erase(score_index_vec.begin());
            if (keep && eta < 1 && adaptive_threshold > 0.5) {
              adaptive_threshold *= eta;
            }
        }
    }

    // Get max scores with corresponding indices.
    //    scores: a set of scores.
    //    threshold: only consider scores higher than the threshold.
    //    top_k: if -1, keep all; otherwise, keep at most top_k.
    //    score_index_vec: store the sorted (score, index) pair.
    static void GetMaxScoreIndex(const std::vector<float>& scores, const float threshold, const int top_k,
                          std::vector<std::pair<float, int> >& score_index_vec)
    {
        CV_DbgAssert(score_index_vec.empty());
        // Generate index score pairs.
        for (size_t i = 0; i < scores.size(); ++i)
        {
            if (scores[i] > threshold)
            {
                score_index_vec.push_back(std::make_pair(scores[i], i));
            }
        }

        // Sort the score pair according to the scores in descending order
        std::stable_sort(score_index_vec.begin(), score_index_vec.end(),
                         util::SortScorePairDescend<int>);

        // Keep top_k scores if needed.
        if (top_k > -1 && top_k < (int)score_index_vec.size())
        {
            score_index_vec.resize(top_k);
        }
    }

    // Compute the jaccard (intersection over union IoU) overlap between two bboxes.
    template<bool normalized>
    static float JaccardOverlap(const caffe::NormalizedBBox& bbox1,
                         const caffe::NormalizedBBox& bbox2)
    {
        caffe::NormalizedBBox intersect_bbox;
        if (bbox2.xmin() > bbox1.xmax() || bbox2.xmax() < bbox1.xmin() ||
            bbox2.ymin() > bbox1.ymax() || bbox2.ymax() < bbox1.ymin())
        {
            // Return [0, 0, 0, 0] if there is no intersection.
            intersect_bbox.set_xmin(0);
            intersect_bbox.set_ymin(0);
            intersect_bbox.set_xmax(0);
            intersect_bbox.set_ymax(0);
        }
        else
        {
            intersect_bbox.set_xmin(std::max(bbox1.xmin(), bbox2.xmin()));
            intersect_bbox.set_ymin(std::max(bbox1.ymin(), bbox2.ymin()));
            intersect_bbox.set_xmax(std::min(bbox1.xmax(), bbox2.xmax()));
            intersect_bbox.set_ymax(std::min(bbox1.ymax(), bbox2.ymax()));
        }

        float intersect_width, intersect_height;
        intersect_width = intersect_bbox.xmax() - intersect_bbox.xmin();
        intersect_height = intersect_bbox.ymax() - intersect_bbox.ymin();
        if (intersect_width > 0 && intersect_height > 0)
        {
            if (!normalized)
            {
                intersect_width++;
                intersect_height++;
            }
            float intersect_size = intersect_width * intersect_height;
            float bbox1_size = BBoxSize<true>(bbox1);
            float bbox2_size = BBoxSize<true>(bbox2);
            return intersect_size / (bbox1_size + bbox2_size - intersect_size);
        }
        else
        {
            return 0.;
        }
    }
};

const std::string DetectionOutputLayerImpl::_layerName = std::string("DetectionOutput");

Ptr<DetectionOutputLayer> DetectionOutputLayer::create(const LayerParams &params)
{
    return Ptr<DetectionOutputLayer>(new DetectionOutputLayerImpl(params));
}

}
}
