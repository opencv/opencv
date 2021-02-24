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
#include "../op_inf_engine.hpp"

#include <float.h>
#include <string>
#include "../nms.inl.hpp"

#ifdef HAVE_OPENCL
#include "opencl_kernels_dnn.hpp"
#endif

#ifdef HAVE_DNN_NGRAPH
#include "../ie_ngraph.hpp"
#if INF_ENGINE_VER_MAJOR_GT(INF_ENGINE_RELEASE_2020_4)
#include <ngraph/op/detection_output.hpp>
#else
#include <ngraph/op/experimental/layers/detection_output.hpp>
#endif

#endif

namespace cv
{
namespace dnn
{

namespace util
{

class NormalizedBBox
{
public:
    float xmin, ymin, xmax, ymax;

    NormalizedBBox()
        : xmin(0), ymin(0), xmax(0), ymax(0), has_size_(false), size_(0) {}

    float size() const { return size_; }

    bool has_size() const { return has_size_; }

    void set_size(float value) { size_ = value; has_size_ = true; }

    void clear_size() { size_ = 0; has_size_ = false; }

private:
    bool has_size_;
    float size_;
};

template <typename T>
static inline bool SortScorePairDescend(const std::pair<float, T>& pair1,
                          const std::pair<float, T>& pair2)
{
    return pair1.first > pair2.first;
}

static inline float caffe_box_overlap(const util::NormalizedBBox& a, const util::NormalizedBBox& b);

static inline float caffe_norm_box_overlap(const util::NormalizedBBox& a, const util::NormalizedBBox& b);

} // namespace

class DetectionOutputLayerImpl CV_FINAL : public DetectionOutputLayer
{
public:
    unsigned _numClasses;
    bool _shareLocation;
    int _numLocClasses;

    int _backgroundLabelId;

    cv::String _codeType;

    bool _varianceEncodedInTarget;
    int _keepTopK;
    float _confidenceThreshold;

    float _nmsThreshold;
    int _topK;
    // Whenever predicted bounding boxes are represented in YXHW instead of XYWH layout.
    bool _locPredTransposed;
    // It's true whenever predicted bounding boxes and proposals are normalized to [0, 1].
    bool _bboxesNormalized;
    bool _clip;
    bool _groupByClasses;

    enum { _numAxes = 4 };
    static const std::string _layerName;

    typedef std::map<int, std::vector<util::NormalizedBBox> > LabelBBox;

    inline int getNumOfTargetClasses() {
        unsigned numBackground =
            (_backgroundLabelId >= 0 && _backgroundLabelId < _numClasses) ? 1 : 0;
        return (_numClasses - numBackground);
    }

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
                CV_Error(Error::StsBadArg, message);
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
        if (codeTypeString == "center_size")
            _codeType = "CENTER_SIZE";
        else
            _codeType = "CORNER";
    }

    DetectionOutputLayerImpl(const LayerParams &params)
    {
        _numClasses = getParameter<unsigned>(params, "num_classes");
        _shareLocation = getParameter<bool>(params, "share_location");
        _numLocClasses = _shareLocation ? 1 : _numClasses;
        _backgroundLabelId = getParameter<int>(params, "background_label_id");
        _varianceEncodedInTarget = getParameter<bool>(params, "variance_encoded_in_target", 0, false, false);
        _keepTopK = getParameter<int>(params, "keep_top_k");
        _confidenceThreshold = getParameter<float>(params, "confidence_threshold", 0, false, 0);
        _topK = getParameter<int>(params, "top_k", 0, false, -1);
        _locPredTransposed = getParameter<bool>(params, "loc_pred_transposed", 0, false, false);
        _bboxesNormalized = getParameter<bool>(params, "normalized_bbox", 0, false, true);
        _clip = getParameter<bool>(params, "clip", 0, false, false);
        _groupByClasses = getParameter<bool>(params, "group_by_classes", 0, false, true);

        getCodeType(params);

        // Parameters used in nms.
        _nmsThreshold = getParameter<float>(params, "nms_threshold");
        CV_Assert(_nmsThreshold > 0.);

        setParamsFrom(params);
    }

    virtual bool supportBackend(int backendId) CV_OVERRIDE
    {
        return backendId == DNN_BACKEND_OPENCV ||
               ((backendId == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019 || backendId == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH) && !_locPredTransposed && _bboxesNormalized);
    }

    bool getMemoryShapes(const std::vector<MatShape> &inputs,
                         const int requiredOutputs,
                         std::vector<MatShape> &outputs,
                         std::vector<MatShape> &internals) const CV_OVERRIDE
    {
        const int num = inputs[0][0];
        CV_Assert(inputs.size() >= 3);
        CV_Assert(num == inputs[1][0]);

        int numPriors = inputs[2][2] / 4;
        CV_Assert((numPriors * _numLocClasses * 4) == total(inputs[0], 1));
        CV_Assert(int(numPriors * _numClasses) == total(inputs[1], 1));
        CV_Assert(inputs[2][1] == 1 + (int)(!_varianceEncodedInTarget));

        // num() and channels() are 1.
        // Since the number of bboxes to be kept is unknown before nms, we manually
        // set it to maximal number of detections, [keep_top_k] parameter multiplied by batch size.
        // Each row is a 7 dimension std::vector, which stores
        // [image_id, label, confidence, xmin, ymin, xmax, ymax]
        outputs.resize(1, shape(1, 1, _keepTopK * num, 7));

        return false;
    }

#ifdef HAVE_OPENCL
    // Decode all bboxes in a batch
    bool ocl_DecodeBBoxesAll(UMat& loc_mat, UMat& prior_mat,
                             const int num, const int numPriors, const bool share_location,
                             const int num_loc_classes, const int background_label_id,
                             const cv::String& code_type, const bool variance_encoded_in_target,
                             const bool clip, std::vector<LabelBBox>& all_decode_bboxes)
    {
        UMat outmat = UMat(loc_mat.dims, loc_mat.size, CV_32F);
        size_t nthreads = loc_mat.total();
        String kernel_name;

        if (code_type == "CORNER")
            kernel_name = "DecodeBBoxesCORNER";
        else if (code_type == "CENTER_SIZE")
            kernel_name = "DecodeBBoxesCENTER_SIZE";
        else
            return false;

        for (int i = 0; i < num; ++i)
        {
            ocl::Kernel kernel(kernel_name.c_str(), ocl::dnn::detection_output_oclsrc);
            kernel.set(0, (int)nthreads);
            kernel.set(1, ocl::KernelArg::PtrReadOnly(loc_mat));
            kernel.set(2, ocl::KernelArg::PtrReadOnly(prior_mat));
            kernel.set(3, (int)variance_encoded_in_target);
            kernel.set(4, (int)numPriors);
            kernel.set(5, (int)share_location);
            kernel.set(6, (int)num_loc_classes);
            kernel.set(7, (int)background_label_id);
            kernel.set(8, (int)clip);
            kernel.set(9, (int)_locPredTransposed);
            kernel.set(10, ocl::KernelArg::PtrWriteOnly(outmat));

            if (!kernel.run(1, &nthreads, NULL, false))
                return false;
        }

        all_decode_bboxes.clear();
        all_decode_bboxes.resize(num);
        {
            Mat mat = outmat.getMat(ACCESS_READ);
            const float* decode_data = mat.ptr<float>();
            for (int i = 0; i < num; ++i)
            {
                LabelBBox& decode_bboxes = all_decode_bboxes[i];
                for (int c = 0; c < num_loc_classes; ++c)
                {
                    int label = share_location ? -1 : c;
                    decode_bboxes[label].resize(numPriors);
                    for (int p = 0; p < numPriors; ++p)
                    {
                        int startIdx = p * num_loc_classes * 4;
                        util::NormalizedBBox& bbox = decode_bboxes[label][p];
                        bbox.xmin = decode_data[startIdx + c * 4];
                        bbox.ymin = decode_data[startIdx + c * 4 + 1];
                        bbox.xmax = decode_data[startIdx + c * 4 + 2];
                        bbox.ymax = decode_data[startIdx + c * 4 + 3];
                    }
                }
            }
        }
        return true;
    }

    void ocl_GetConfidenceScores(const UMat& inp1, const int num,
                                 const int numPredsPerClass, const int numClasses,
                                 std::vector<Mat>& confPreds)
    {
        int shape[] = { numClasses, numPredsPerClass };
        for (int i = 0; i < num; i++)
            confPreds.push_back(Mat(2, shape, CV_32F));

        shape[0] = num * numPredsPerClass;
        shape[1] = inp1.total() / shape[0];
        UMat umat = inp1.reshape(1, 2, &shape[0]);
        for (int i = 0; i < num; ++i)
        {
            Range ranges[] = { Range(i * numPredsPerClass, (i + 1) * numPredsPerClass), Range::all() };
            transpose(umat(ranges), confPreds[i]);
        }
    }

    bool forward_ocl(InputArrayOfArrays inps, OutputArrayOfArrays outs, OutputArrayOfArrays internals)
    {
        std::vector<UMat> inputs;
        std::vector<UMat> outputs;
        outs.getUMatVector(outputs);

        bool use_half = (inps.depth() == CV_16S);
        if (use_half)
        {
            std::vector<UMat> orig_inputs;
            inps.getUMatVector(orig_inputs);

            inputs.resize(orig_inputs.size());
            for (size_t i = 0; i < orig_inputs.size(); i++)
                convertFp16(orig_inputs[i], inputs[i]);
        }
        else
        {
            inps.getUMatVector(inputs);
        }

        std::vector<LabelBBox> allDecodedBBoxes;
        std::vector<Mat> allConfidenceScores;

        int num = inputs[0].size[0];

        // extract predictions from input layers
        {
            int numPriors = inputs[2].size[2] / 4;

            // Retrieve all confidences
            ocl_GetConfidenceScores(inputs[1], num, numPriors, _numClasses, allConfidenceScores);

            // Decode all loc predictions to bboxes
            bool ret = ocl_DecodeBBoxesAll(inputs[0], inputs[2], num, numPriors,
                                           _shareLocation, _numLocClasses, _backgroundLabelId,
                                           _codeType, _varianceEncodedInTarget, _clip,
                                           allDecodedBBoxes);
            if (!ret)
                return false;
        }

        size_t numKept = 0;
        std::vector<std::map<int, std::vector<int> > > allIndices;
        for (int i = 0; i < num; ++i)
        {
            numKept += processDetections_(allDecodedBBoxes[i], allConfidenceScores[i], allIndices);
        }

        if (numKept == 0)
        {
            outputs[0].setTo(0);
            return true;
        }

        UMat umat = use_half ? UMat::zeros(4, outputs[0].size, CV_32F) : outputs[0];

        if (!use_half)
            umat.setTo(0);

        // If there are valid detections
        if (numKept > 0)
        {
            Mat mat = umat.getMat(ACCESS_WRITE);
            float* outputsData = mat.ptr<float>();

            size_t count = 0;
            for (int i = 0; i < num; ++i)
            {
                count += outputDetections_(i, &outputsData[count * 7],
                                           allDecodedBBoxes[i], allConfidenceScores[i],
                                           allIndices[i], _groupByClasses);
            }
            CV_Assert(count == numKept);
        }

        if (use_half)
        {
            UMat half_umat;
            convertFp16(umat, half_umat);
            outs.assign(std::vector<UMat>(1, half_umat));
        }

        return true;
    }
#endif

    void forward(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr, OutputArrayOfArrays internals_arr) CV_OVERRIDE
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        if (_bboxesNormalized)
        {
            CV_OCL_RUN(IS_DNN_OPENCL_TARGET(preferableTarget),
                       forward_ocl(inputs_arr, outputs_arr, internals_arr))
        }
        if (inputs_arr.depth() == CV_16S)
        {
            forward_fallback(inputs_arr, outputs_arr, internals_arr);
            return;
        }

        std::vector<Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);

        std::vector<LabelBBox> allDecodedBBoxes;
        std::vector<Mat> allConfidenceScores;

        int num = inputs[0].size[0];

        // extract predictions from input layers
        {
            int numPriors = inputs[2].size[2] / 4;

            const float* locationData = inputs[0].ptr<float>();
            const float* confidenceData = inputs[1].ptr<float>();
            const float* priorData = inputs[2].ptr<float>();

            // Retrieve all location predictions
            std::vector<LabelBBox> allLocationPredictions;
            GetLocPredictions(locationData, num, numPriors, _numLocClasses,
                              _shareLocation, _locPredTransposed, allLocationPredictions);

            // Retrieve all confidences
            GetConfidenceScores(confidenceData, num, numPriors, _numClasses, allConfidenceScores);

            // Retrieve all prior bboxes
            std::vector<util::NormalizedBBox> priorBBoxes;
            std::vector<std::vector<float> > priorVariances;
            GetPriorBBoxes(priorData, numPriors, _bboxesNormalized, priorBBoxes, priorVariances);

            // Decode all loc predictions to bboxes
            util::NormalizedBBox clipBounds;
            if (_clip)
            {
                CV_Assert(_bboxesNormalized || inputs.size() >= 4);
                clipBounds.xmin = clipBounds.ymin = 0.0f;
                if (_bboxesNormalized)
                    clipBounds.xmax = clipBounds.ymax = 1.0f;
                else
                {
                    // Input image sizes;
                    CV_Assert(inputs[3].dims == 4);
                    clipBounds.xmax = inputs[3].size[3] - 1;
                    clipBounds.ymax = inputs[3].size[2] - 1;
                }
            }
            DecodeBBoxesAll(allLocationPredictions, priorBBoxes, priorVariances, num,
                            _shareLocation, _numLocClasses, _backgroundLabelId,
                            _codeType, _varianceEncodedInTarget, _clip, clipBounds,
                            _bboxesNormalized, allDecodedBBoxes);
        }

        size_t numKept = 0;
        std::vector<std::map<int, std::vector<int> > > allIndices;
        for (int i = 0; i < num; ++i)
        {
            numKept += processDetections_(allDecodedBBoxes[i], allConfidenceScores[i], allIndices);
        }

        outputs[0].setTo(0);

        // If there is no detections
        if (numKept == 0)
            return;

        float* outputsData = outputs[0].ptr<float>();

        size_t count = 0;
        for (int i = 0; i < num; ++i)
        {
            count += outputDetections_(i, &outputsData[count * 7],
                                       allDecodedBBoxes[i], allConfidenceScores[i],
                                       allIndices[i], _groupByClasses);
        }
        CV_Assert(count == numKept);
        // Sync results back due changed output shape.
        outputs_arr.assign(outputs);
    }

    size_t outputDetections_(
            const int i, float* outputsData,
            const LabelBBox& decodeBBoxes, Mat& confidenceScores,
            const std::map<int, std::vector<int> >& indicesMap,
            bool groupByClasses
    )
    {
        std::vector<int> dstIndices;
        std::vector<std::pair<float, int> > allScores;
        for (std::map<int, std::vector<int> >::const_iterator it = indicesMap.begin(); it != indicesMap.end(); ++it)
        {
            int label = it->first;
            if (confidenceScores.rows <= label)
                CV_Error_(cv::Error::StsError, ("Could not find confidence predictions for label %d", label));
            const std::vector<float>& scores = confidenceScores.row(label);
            const std::vector<int>& indices = it->second;

            const int numAllScores = allScores.size();
            allScores.reserve(numAllScores + indices.size());
            for (size_t j = 0; j < indices.size(); ++j)
            {
                allScores.push_back(std::make_pair(scores[indices[j]], numAllScores + j));
            }
        }
        if (!groupByClasses)
            std::sort(allScores.begin(), allScores.end(), util::SortScorePairDescend<int>);

        dstIndices.resize(allScores.size());
        for (size_t j = 0; j < dstIndices.size(); ++j)
        {
            dstIndices[allScores[j].second] = j;
        }

        size_t count = 0;
        for (std::map<int, std::vector<int> >::const_iterator it = indicesMap.begin(); it != indicesMap.end(); ++it)
        {
            int label = it->first;
            if (confidenceScores.rows <= label)
                CV_Error_(cv::Error::StsError, ("Could not find confidence predictions for label %d", label));
            const std::vector<float>& scores = confidenceScores.row(label);
            int locLabel = _shareLocation ? -1 : label;
            LabelBBox::const_iterator label_bboxes = decodeBBoxes.find(locLabel);
            if (label_bboxes == decodeBBoxes.end())
                CV_Error_(cv::Error::StsError, ("Could not find location predictions for label %d", locLabel));
            const std::vector<int>& indices = it->second;

            for (size_t j = 0; j < indices.size(); ++j, ++count)
            {
                int idx = indices[j];
                int dstIdx = dstIndices[count];
                const util::NormalizedBBox& decode_bbox = label_bboxes->second[idx];
                outputsData[dstIdx * 7] = i;
                outputsData[dstIdx * 7 + 1] = label;
                outputsData[dstIdx * 7 + 2] = scores[idx];
                outputsData[dstIdx * 7 + 3] = decode_bbox.xmin;
                outputsData[dstIdx * 7 + 4] = decode_bbox.ymin;
                outputsData[dstIdx * 7 + 5] = decode_bbox.xmax;
                outputsData[dstIdx * 7 + 6] = decode_bbox.ymax;
            }
        }
        return count;
    }

    size_t processDetections_(
            const LabelBBox& decodeBBoxes, Mat& confidenceScores,
            std::vector<std::map<int, std::vector<int> > >& allIndices
    )
    {
        std::map<int, std::vector<int> > indices;
        size_t numDetections = 0;
        for (int c = 0; c < (int)_numClasses; ++c)
        {
            if (c == _backgroundLabelId)
                continue; // Ignore background class.
            if (c >= confidenceScores.rows)
                CV_Error_(cv::Error::StsError, ("Could not find confidence predictions for label %d", c));

            const std::vector<float> scores = confidenceScores.row(c);
            int label = _shareLocation ? -1 : c;

            LabelBBox::const_iterator label_bboxes = decodeBBoxes.find(label);
            if (label_bboxes == decodeBBoxes.end())
                CV_Error_(cv::Error::StsError, ("Could not find location predictions for label %d", label));
            int limit = (getNumOfTargetClasses() == 1) ? _keepTopK : std::numeric_limits<int>::max();
            if (_bboxesNormalized)
                NMSFast_(label_bboxes->second, scores, _confidenceThreshold, _nmsThreshold, 1.0, _topK,
                         indices[c], util::caffe_norm_box_overlap, limit);
            else
                NMSFast_(label_bboxes->second, scores, _confidenceThreshold, _nmsThreshold, 1.0, _topK,
                         indices[c], util::caffe_box_overlap, limit);
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
                if (label >= confidenceScores.rows)
                    CV_Error_(cv::Error::StsError, ("Could not find location predictions for label %d", label));
                const std::vector<float>& scores = confidenceScores.row(label);
                for (size_t j = 0; j < labelIndices.size(); ++j)
                {
                    size_t idx = labelIndices[j];
                    CV_Assert(idx < scores.size());
                    scoreIndexPairs.push_back(std::make_pair(scores[idx], std::make_pair(label, idx)));
                }
            }
            // Keep outputs k results per image.
            if ((_keepTopK * 8) > scoreIndexPairs.size()) {
                std::sort(scoreIndexPairs.begin(), scoreIndexPairs.end(),
                          util::SortScorePairDescend<std::pair<int, int> >);
            } else {
                std::partial_sort(scoreIndexPairs.begin(), scoreIndexPairs.begin() + _keepTopK, scoreIndexPairs.end(),
                          util::SortScorePairDescend<std::pair<int, int> >);
            }
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
    static float BBoxSize(const util::NormalizedBBox& bbox, bool normalized)
    {
        if (bbox.xmax < bbox.xmin || bbox.ymax < bbox.ymin)
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
                float width = bbox.xmax - bbox.xmin;
                float height = bbox.ymax - bbox.ymin;
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
        const util::NormalizedBBox& prior_bbox, const std::vector<float>& prior_variance,
        const cv::String& code_type,
        const bool clip_bbox, const util::NormalizedBBox& clip_bounds,
        const bool normalized_bbox, const util::NormalizedBBox& bbox,
        util::NormalizedBBox& decode_bbox)
    {
        float bbox_xmin = variance_encoded_in_target ? bbox.xmin : prior_variance[0] * bbox.xmin;
        float bbox_ymin = variance_encoded_in_target ? bbox.ymin : prior_variance[1] * bbox.ymin;
        float bbox_xmax = variance_encoded_in_target ? bbox.xmax : prior_variance[2] * bbox.xmax;
        float bbox_ymax = variance_encoded_in_target ? bbox.ymax : prior_variance[3] * bbox.ymax;
        if (code_type == "CORNER")
        {
            decode_bbox.xmin = prior_bbox.xmin + bbox_xmin;
            decode_bbox.ymin = prior_bbox.ymin + bbox_ymin;
            decode_bbox.xmax = prior_bbox.xmax + bbox_xmax;
            decode_bbox.ymax = prior_bbox.ymax + bbox_ymax;
        }
        else if (code_type == "CENTER_SIZE")
        {
            float prior_width = prior_bbox.xmax - prior_bbox.xmin;
            float prior_height = prior_bbox.ymax - prior_bbox.ymin;
            if (!normalized_bbox)
            {
                prior_width += 1.0f;
                prior_height += 1.0f;
            }
            float prior_center_x = prior_bbox.xmin + prior_width * .5;
            float prior_center_y = prior_bbox.ymin + prior_height * .5;

            float decode_bbox_center_x, decode_bbox_center_y;
            float decode_bbox_width, decode_bbox_height;
            decode_bbox_center_x = bbox_xmin * prior_width + prior_center_x;
            decode_bbox_center_y = bbox_ymin * prior_height + prior_center_y;
            decode_bbox_width = exp(bbox_xmax) * prior_width;
            decode_bbox_height = exp(bbox_ymax) * prior_height;
            decode_bbox.xmin = decode_bbox_center_x - decode_bbox_width * .5;
            decode_bbox.ymin = decode_bbox_center_y - decode_bbox_height * .5;
            decode_bbox.xmax = decode_bbox_center_x + decode_bbox_width * .5;
            decode_bbox.ymax = decode_bbox_center_y + decode_bbox_height * .5;
        }
        else
            CV_Error(Error::StsBadArg, "Unknown type.");

        if (clip_bbox)
        {
            // Clip the util::NormalizedBBox.
            decode_bbox.xmin = std::max(std::min(decode_bbox.xmin, clip_bounds.xmax), clip_bounds.xmin);
            decode_bbox.ymin = std::max(std::min(decode_bbox.ymin, clip_bounds.ymax), clip_bounds.ymin);
            decode_bbox.xmax = std::max(std::min(decode_bbox.xmax, clip_bounds.xmax), clip_bounds.xmin);
            decode_bbox.ymax = std::max(std::min(decode_bbox.ymax, clip_bounds.ymax), clip_bounds.ymin);
        }
        decode_bbox.clear_size();
        decode_bbox.set_size(BBoxSize(decode_bbox, normalized_bbox));
    }

    // Decode a set of bboxes according to a set of prior bboxes
    static void DecodeBBoxes(
        const std::vector<util::NormalizedBBox>& prior_bboxes,
        const std::vector<std::vector<float> >& prior_variances,
        const cv::String& code_type, const bool variance_encoded_in_target,
        const bool clip_bbox, const util::NormalizedBBox& clip_bounds,
        const bool normalized_bbox, const std::vector<util::NormalizedBBox>& bboxes,
        std::vector<util::NormalizedBBox>& decode_bboxes)
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
                                 clip_bbox, clip_bounds, normalized_bbox,
                                 bboxes[i], decode_bboxes[i]);
        }
        else
        {
            for (int i = 0; i < num_bboxes; ++i)
                DecodeBBox<false>(prior_bboxes[i], prior_variances[i], code_type,
                                  clip_bbox, clip_bounds, normalized_bbox,
                                  bboxes[i], decode_bboxes[i]);
        }
    }

    // Decode all bboxes in a batch
    static void DecodeBBoxesAll(const std::vector<LabelBBox>& all_loc_preds,
        const std::vector<util::NormalizedBBox>& prior_bboxes,
        const std::vector<std::vector<float> >& prior_variances,
        const int num, const bool share_location,
        const int num_loc_classes, const int background_label_id,
        const cv::String& code_type, const bool variance_encoded_in_target,
        const bool clip, const util::NormalizedBBox& clip_bounds,
        const bool normalized_bbox, std::vector<LabelBBox>& all_decode_bboxes)
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
                    CV_Error_(cv::Error::StsError, ("Could not find location predictions for label %d", label));
                DecodeBBoxes(prior_bboxes, prior_variances,
                             code_type, variance_encoded_in_target, clip, clip_bounds,
                             normalized_bbox, label_loc_preds->second, decode_bboxes[label]);
            }
        }
    }

    // Get prior bounding boxes from prior_data
    //    prior_data: 1 x 2 x num_priors * 4 x 1 blob.
    //    num_priors: number of priors.
    //    prior_bboxes: stores all the prior bboxes in the format of util::NormalizedBBox.
    //    prior_variances: stores all the variances needed by prior bboxes.
    static void GetPriorBBoxes(const float* priorData, const int& numPriors,
                        bool normalized_bbox, std::vector<util::NormalizedBBox>& priorBBoxes,
                        std::vector<std::vector<float> >& priorVariances)
    {
        priorBBoxes.clear(); priorBBoxes.resize(numPriors);
        priorVariances.clear(); priorVariances.resize(numPriors);
        for (int i = 0; i < numPriors; ++i)
        {
            int startIdx = i * 4;
            util::NormalizedBBox& bbox = priorBBoxes[i];
            bbox.xmin = priorData[startIdx];
            bbox.ymin = priorData[startIdx + 1];
            bbox.xmax = priorData[startIdx + 2];
            bbox.ymax = priorData[startIdx + 3];
            bbox.set_size(BBoxSize(bbox, normalized_bbox));
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
    //    loc_pred_transposed: if true, represent four bounding box values as
    //                         [y,x,height,width] or [x,y,width,height] otherwise.
    //    loc_preds: stores the location prediction, where each item contains
    //      location prediction for an image.
    static void GetLocPredictions(const float* locData, const int num,
                           const int numPredsPerClass, const int numLocClasses,
                           const bool shareLocation, const bool locPredTransposed,
                           std::vector<LabelBBox>& locPreds)
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
            int start = shareLocation ? -1 : 0;
            for (int c = 0; c < numLocClasses; ++c) {
                labelBBox[start++].resize(numPredsPerClass);
            }
            for (int p = 0; p < numPredsPerClass; ++p)
            {
                int startIdx = p * numLocClasses * 4;
                for (int c = 0; c < numLocClasses; ++c)
                {
                    int label = shareLocation ? -1 : c;
                    util::NormalizedBBox& bbox = labelBBox[label][p];
                    if (locPredTransposed)
                    {
                        bbox.ymin = locData[startIdx + c * 4];
                        bbox.xmin = locData[startIdx + c * 4 + 1];
                        bbox.ymax = locData[startIdx + c * 4 + 2];
                        bbox.xmax = locData[startIdx + c * 4 + 3];
                    }
                    else
                    {
                        bbox.xmin = locData[startIdx + c * 4];
                        bbox.ymin = locData[startIdx + c * 4 + 1];
                        bbox.xmax = locData[startIdx + c * 4 + 2];
                        bbox.ymax = locData[startIdx + c * 4 + 3];
                    }
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
                             std::vector<Mat>& confPreds)
    {
        int shape[] = { numClasses, numPredsPerClass };
        for (int i = 0; i < num; i++)
            confPreds.push_back(Mat(2, shape, CV_32F));

        for (int i = 0; i < num; ++i, confData += numPredsPerClass * numClasses)
        {
            Mat labelScores = confPreds[i];
            for (int c = 0; c < numClasses; ++c)
            {
                for (int p = 0; p < numPredsPerClass; ++p)
                {
                    labelScores.at<float>(c, p) = confData[p * numClasses + c];
                }
            }
        }
    }

    // Compute the jaccard (intersection over union IoU) overlap between two bboxes.
    template<bool normalized>
    static float JaccardOverlap(const util::NormalizedBBox& bbox1,
                         const util::NormalizedBBox& bbox2)
    {
        util::NormalizedBBox intersect_bbox;
        intersect_bbox.xmin = std::max(bbox1.xmin, bbox2.xmin);
        intersect_bbox.ymin = std::max(bbox1.ymin, bbox2.ymin);
        intersect_bbox.xmax = std::min(bbox1.xmax, bbox2.xmax);
        intersect_bbox.ymax = std::min(bbox1.ymax, bbox2.ymax);

        float intersect_size = BBoxSize(intersect_bbox, normalized);
        if (intersect_size > 0)
        {
            float bbox1_size = BBoxSize(bbox1, normalized);
            float bbox2_size = BBoxSize(bbox2, normalized);
            return intersect_size / (bbox1_size + bbox2_size - intersect_size);
        }
        else
        {
            return 0.;
        }
    }

#ifdef HAVE_DNN_IE_NN_BUILDER_2019
    virtual Ptr<BackendNode> initInfEngine(const std::vector<Ptr<BackendWrapper> >&) CV_OVERRIDE
    {
        InferenceEngine::Builder::DetectionOutputLayer ieLayer(name);

        ieLayer.setNumClasses(_numClasses);
        ieLayer.setShareLocation(_shareLocation);
        ieLayer.setBackgroudLabelId(_backgroundLabelId);
        ieLayer.setNMSThreshold(_nmsThreshold);
        ieLayer.setTopK(_topK > 0 ? _topK : _keepTopK);
        ieLayer.setKeepTopK(_keepTopK);
        ieLayer.setConfidenceThreshold(_confidenceThreshold);
        ieLayer.setVariantEncodedInTarget(_varianceEncodedInTarget);
        ieLayer.setCodeType("caffe.PriorBoxParameter." + _codeType);
        ieLayer.setInputPorts(std::vector<InferenceEngine::Port>(3));

        InferenceEngine::Builder::Layer l = ieLayer;
        l.getParameters()["eta"] = std::string("1.0");
        l.getParameters()["clip"] = _clip;

        return Ptr<BackendNode>(new InfEngineBackendNode(l));
    }
#endif  // HAVE_DNN_IE_NN_BUILDER_2019


#ifdef HAVE_DNN_NGRAPH
    virtual Ptr<BackendNode> initNgraph(const std::vector<Ptr<BackendWrapper> >& inputs, const std::vector<Ptr<BackendNode> >& nodes) CV_OVERRIDE
    {
        CV_Assert(nodes.size() == 3);
        auto& box_logits  = nodes[0].dynamicCast<InfEngineNgraphNode>()->node;
        auto& class_preds = nodes[1].dynamicCast<InfEngineNgraphNode>()->node;
        auto& proposals   = nodes[2].dynamicCast<InfEngineNgraphNode>()->node;

        ngraph::op::DetectionOutputAttrs attrs;
        attrs.num_classes                = _numClasses;
        attrs.background_label_id        = _backgroundLabelId;
        attrs.top_k                      = _topK > 0 ? _topK : _keepTopK;
        attrs.variance_encoded_in_target = _varianceEncodedInTarget;
        attrs.keep_top_k                 = {_keepTopK};
        attrs.nms_threshold              = _nmsThreshold;
        attrs.confidence_threshold       = _confidenceThreshold;
        attrs.share_location             = _shareLocation;
        attrs.clip_before_nms            = _clip;
        attrs.code_type                  = std::string{"caffe.PriorBoxParameter." + _codeType};
        attrs.normalized                 = true;

        auto det_out = std::make_shared<ngraph::op::DetectionOutput>(box_logits, class_preds,
                       proposals, attrs);
        return Ptr<BackendNode>(new InfEngineNgraphNode(det_out));
    }
#endif  // HAVE_DNN_NGRAPH
};

float util::caffe_box_overlap(const util::NormalizedBBox& a, const util::NormalizedBBox& b)
{
    return DetectionOutputLayerImpl::JaccardOverlap<false>(a, b);
}

float util::caffe_norm_box_overlap(const util::NormalizedBBox& a, const util::NormalizedBBox& b)
{
    return DetectionOutputLayerImpl::JaccardOverlap<true>(a, b);
}

const std::string DetectionOutputLayerImpl::_layerName = std::string("DetectionOutput");

Ptr<DetectionOutputLayer> DetectionOutputLayer::create(const LayerParams &params)
{
    return Ptr<DetectionOutputLayer>(new DetectionOutputLayerImpl(params));
}

}
}
