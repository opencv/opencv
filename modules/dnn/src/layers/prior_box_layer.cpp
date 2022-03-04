/*M///////////////////////////////////////////////////////////////////////////////////////
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
#include "../op_cuda.hpp"
#include "../op_inf_engine.hpp"

#ifdef HAVE_DNN_NGRAPH
#include "../ie_ngraph.hpp"
#if INF_ENGINE_VER_MAJOR_GT(INF_ENGINE_RELEASE_2020_4)
#include <ngraph/op/prior_box.hpp>
#include <ngraph/op/prior_box_clustered.hpp>
#else
#include <ngraph/op/experimental/layers/prior_box.hpp>
#include <ngraph/op/experimental/layers/prior_box_clustered.hpp>
#endif
#endif

#include "../op_vkcom.hpp"

#include <float.h>
#include <algorithm>
#include <cmath>

#ifdef HAVE_OPENCL
#include "opencl_kernels_dnn.hpp"
#endif

#ifdef HAVE_CUDA
#include "../cuda4dnn/primitives/prior_box.hpp"
using namespace cv::dnn::cuda4dnn;
#endif

namespace cv
{
namespace dnn
{

class PriorBoxLayerImpl CV_FINAL : public PriorBoxLayer
{
public:
    static bool getParameterDict(const LayerParams &params,
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

    void getAspectRatios(const LayerParams &params)
    {
        DictValue aspectRatioParameter;
        bool aspectRatioRetieved = getParameterDict(params, "aspect_ratio", aspectRatioParameter);
        if (!aspectRatioRetieved)
            return;

        for (int i = 0; i < aspectRatioParameter.size(); ++i)
        {
            float aspectRatio = aspectRatioParameter.get<float>(i);
            bool alreadyExists = fabs(aspectRatio - 1.f) < 1e-6f;

            for (size_t j = 0; j < _aspectRatios.size() && !alreadyExists; ++j)
            {
                alreadyExists = fabs(aspectRatio - _aspectRatios[j]) < 1e-6;
            }
            if (!alreadyExists)
            {
                _aspectRatios.push_back(aspectRatio);
                if (_flip)
                {
                    _aspectRatios.push_back(1./aspectRatio);
                }
            }
        }
    }

    static void getParams(const std::string& name, const LayerParams &params,
                          std::vector<float>* values)
    {
        DictValue dict;
        if (getParameterDict(params, name, dict))
        {
            values->resize(dict.size());
            for (int i = 0; i < dict.size(); ++i)
            {
                (*values)[i] = dict.get<float>(i);
            }
        }
        else
            values->clear();
    }

    void getVariance(const LayerParams &params)
    {
        DictValue varianceParameter;
        bool varianceParameterRetrieved = getParameterDict(params, "variance", varianceParameter);
        CV_Assert(varianceParameterRetrieved);

        int varianceSize = varianceParameter.size();
        if (varianceSize > 1)
        {
            // Must and only provide 4 variance.
            CV_Assert(varianceSize == 4);

            for (int i = 0; i < varianceSize; ++i)
            {
                float variance = varianceParameter.get<float>(i);
                CV_Assert(variance > 0);
                _variance.push_back(variance);
            }
        }
        else
        {
            if (varianceSize == 1)
            {
                float variance = varianceParameter.get<float>(0);
                CV_Assert(variance > 0);
                _variance.push_back(variance);
            }
            else
            {
                // Set default to 0.1.
                _variance.push_back(0.1f);
            }
        }
    }

    PriorBoxLayerImpl(const LayerParams &params)
    {
        setParamsFrom(params);
        _flip = getParameter<bool>(params, "flip", 0, false, true);
        _clip = getParameter<bool>(params, "clip", 0, false, true);
        _bboxesNormalized = getParameter<bool>(params, "normalized_bbox", 0, false, true);

        getParams("min_size", params, &_minSize);
        getAspectRatios(params);
        getVariance(params);

        if (params.has("max_size"))
        {
            getParams("max_size", params, &_maxSize);
            CV_Assert(_minSize.size() == _maxSize.size());
            for (int i = 0; i < _maxSize.size(); i++)
                CV_Assert(_minSize[i] < _maxSize[i]);
        }

        std::vector<float> widths, heights;
        getParams("width", params, &widths);
        getParams("height", params, &heights);
        _explicitSizes = !widths.empty();
        CV_Assert(widths.size() == heights.size());

        if (_explicitSizes)
        {
            CV_Assert(_aspectRatios.empty());
            CV_Assert(!params.has("min_size"));
            CV_Assert(!params.has("max_size"));
            _boxWidths = widths;
            _boxHeights = heights;
        }
        else
        {
            CV_Assert(!_minSize.empty());
            for (int i = 0; i < _minSize.size(); ++i)
            {
                float minSize = _minSize[i];
                CV_Assert(minSize > 0);
                _boxWidths.push_back(minSize);
                _boxHeights.push_back(minSize);

                if (_maxSize.size() > 0)
                {
                    float size = sqrt(minSize * _maxSize[i]);
                    _boxWidths.push_back(size);
                    _boxHeights.push_back(size);
                }

                // rest of priors
                for (size_t r = 0; r < _aspectRatios.size(); ++r)
                {
                    float arSqrt = sqrt(_aspectRatios[r]);
                    _boxWidths.push_back(minSize * arSqrt);
                    _boxHeights.push_back(minSize / arSqrt);
                }
            }
        }
        CV_Assert(_boxWidths.size() == _boxHeights.size());
        _numPriors = _boxWidths.size();

        if (params.has("step_h") || params.has("step_w")) {
          CV_Assert(!params.has("step"));
          _stepY = getParameter<float>(params, "step_h");
          CV_Assert(_stepY > 0.);
          _stepX = getParameter<float>(params, "step_w");
          CV_Assert(_stepX > 0.);
        } else if (params.has("step")) {
          const float step = getParameter<float>(params, "step");
          CV_Assert(step > 0);
          _stepY = step;
          _stepX = step;
        } else {
          _stepY = 0;
          _stepX = 0;
        }
        if (params.has("offset_h") || params.has("offset_w"))
        {
            CV_Assert_N(!params.has("offset"), params.has("offset_h"), params.has("offset_w"));
            getParams("offset_h", params, &_offsetsY);
            getParams("offset_w", params, &_offsetsX);
            CV_Assert(_offsetsX.size() == _offsetsY.size());
            _numPriors *= std::max((size_t)1, 2 * (_offsetsX.size() - 1));
        }
        else
        {
            float offset = getParameter<float>(params, "offset", 0, false, 0.5);
            _offsetsX.assign(1, offset);
            _offsetsY.assign(1, offset);
        }
    }

    virtual bool supportBackend(int backendId) CV_OVERRIDE
    {
#ifdef HAVE_DNN_NGRAPH
        if (backendId == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
            return _explicitSizes || _stepX == _stepY;
#endif
        return backendId == DNN_BACKEND_OPENCV ||
               backendId == DNN_BACKEND_CUDA ||
               (backendId == DNN_BACKEND_VKCOM && haveVulkan());
    }

    bool getMemoryShapes(const std::vector<MatShape> &inputs,
                         const int requiredOutputs,
                         std::vector<MatShape> &outputs,
                         std::vector<MatShape> &internals) const CV_OVERRIDE
    {
        CV_Assert(!inputs.empty());

        int layerHeight = inputs[0][2];
        int layerWidth = inputs[0][3];

        // Since all images in a batch has same height and width, we only need to
        // generate one set of priors which can be shared across all images.
        size_t outNum = 1;
        // 2 channels. First channel stores the mean of each prior coordinate.
        // Second channel stores the variance of each prior coordinate.
        size_t outChannels = 2;

        outputs.resize(1, shape(outNum, outChannels,
                                layerHeight * layerWidth * _numPriors * 4));

        return false;
    }

    void finalize(InputArrayOfArrays inputs_arr, OutputArrayOfArrays) CV_OVERRIDE
    {
        std::vector<Mat> inputs;
        inputs_arr.getMatVector(inputs);

        CV_CheckGT(inputs.size(), (size_t)1, "");
        CV_CheckEQ(inputs[0].dims, 4, ""); CV_CheckEQ(inputs[1].dims, 4, "");
        int layerWidth = inputs[0].size[3];
        int layerHeight = inputs[0].size[2];

        int imageWidth = inputs[1].size[3];
        int imageHeight = inputs[1].size[2];

        _stepY = _stepY == 0 ? (static_cast<float>(imageHeight) / layerHeight) : _stepY;
        _stepX = _stepX == 0 ? (static_cast<float>(imageWidth) / layerWidth) : _stepX;
    }

#ifdef HAVE_OPENCL
    bool forward_ocl(InputArrayOfArrays inps, OutputArrayOfArrays outs, OutputArrayOfArrays internals)
    {
        std::vector<UMat> inputs;
        std::vector<UMat> outputs;

        bool use_half = (inps.depth() == CV_16S);
        inps.getUMatVector(inputs);
        outs.getUMatVector(outputs);

        int _layerWidth = inputs[0].size[3];
        int _layerHeight = inputs[0].size[2];

        int _imageWidth = inputs[1].size[3];
        int _imageHeight = inputs[1].size[2];

        if (umat_offsetsX.empty())
        {
            Mat offsetsX(1, _offsetsX.size(), CV_32FC1, &_offsetsX[0]);
            Mat offsetsY(1, _offsetsY.size(), CV_32FC1, &_offsetsY[0]);
            Mat variance(1, _variance.size(), CV_32FC1, &_variance[0]);
            Mat widths(1, _boxWidths.size(), CV_32FC1, &_boxWidths[0]);
            Mat heights(1, _boxHeights.size(), CV_32FC1, &_boxHeights[0]);

            offsetsX.copyTo(umat_offsetsX);
            offsetsY.copyTo(umat_offsetsY);
            variance.copyTo(umat_variance);
            widths.copyTo(umat_widths);
            heights.copyTo(umat_heights);
        }

        String opts;
        if (use_half)
            opts = "-DDtype=half -DDtype4=half4 -Dconvert_T=convert_half4";
        else
            opts = "-DDtype=float -DDtype4=float4 -Dconvert_T=convert_float4";

        size_t nthreads = _layerHeight * _layerWidth;
        ocl::Kernel kernel("prior_box", ocl::dnn::prior_box_oclsrc, opts);

        kernel.set(0, (int)nthreads);
        kernel.set(1, (float)_stepX);
        kernel.set(2, (float)_stepY);
        kernel.set(3, ocl::KernelArg::PtrReadOnly(umat_offsetsX));
        kernel.set(4, ocl::KernelArg::PtrReadOnly(umat_offsetsY));
        kernel.set(5, (int)_offsetsX.size());
        kernel.set(6, ocl::KernelArg::PtrReadOnly(umat_widths));
        kernel.set(7, ocl::KernelArg::PtrReadOnly(umat_heights));
        kernel.set(8, (int)_boxWidths.size());
        kernel.set(9, ocl::KernelArg::PtrWriteOnly(outputs[0]));
        kernel.set(10, (int)_layerHeight);
        kernel.set(11, (int)_layerWidth);
        kernel.set(12, (int)_imageHeight);
        kernel.set(13, (int)_imageWidth);
        kernel.run(1, &nthreads, NULL, false);

        // clip the prior's coordinate such that it is within [0, 1]
        if (_clip)
        {
            ocl::Kernel kernel("clip", ocl::dnn::prior_box_oclsrc, opts);
            size_t nthreads = _layerHeight * _layerWidth * _numPriors * 4;
            if (!kernel.args((int)nthreads, ocl::KernelArg::PtrReadWrite(outputs[0]))
                       .run(1, &nthreads, NULL, false))
                return false;
        }

        // set the variance.
        {
            ocl::Kernel kernel("set_variance", ocl::dnn::prior_box_oclsrc, opts);
            int offset = total(shape(outputs[0]), 2);
            size_t nthreads = _layerHeight * _layerWidth * _numPriors;
            kernel.set(0, (int)nthreads);
            kernel.set(1, (int)offset);
            kernel.set(2, (int)_variance.size());
            kernel.set(3, ocl::KernelArg::PtrReadOnly(umat_variance));
            kernel.set(4, ocl::KernelArg::PtrWriteOnly(outputs[0]));
            if (!kernel.run(1, &nthreads, NULL, false))
                return false;
        }
        return true;
    }
#endif

    void forward(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr, OutputArrayOfArrays internals_arr) CV_OVERRIDE
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        CV_OCL_RUN(IS_DNN_OPENCL_TARGET(preferableTarget),
                   forward_ocl(inputs_arr, outputs_arr, internals_arr))

        if (inputs_arr.depth() == CV_16S)
        {
            forward_fallback(inputs_arr, outputs_arr, internals_arr);
            return;
        }

        std::vector<Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);

        CV_Assert(inputs.size() == 2);

        int _layerWidth = inputs[0].size[3];
        int _layerHeight = inputs[0].size[2];

        int _imageWidth = inputs[1].size[3];
        int _imageHeight = inputs[1].size[2];

        float* outputPtr = outputs[0].ptr<float>();
        float _boxWidth, _boxHeight;
        for (size_t h = 0; h < _layerHeight; ++h)
        {
            for (size_t w = 0; w < _layerWidth; ++w)
            {
                for (size_t i = 0; i < _boxWidths.size(); ++i)
                {
                    _boxWidth = _boxWidths[i];
                    _boxHeight = _boxHeights[i];
                    for (int j = 0; j < _offsetsX.size(); ++j)
                    {
                        float center_x = (w + _offsetsX[j]) * _stepX;
                        float center_y = (h + _offsetsY[j]) * _stepY;
                        outputPtr = addPrior(center_x, center_y, _boxWidth, _boxHeight, _imageWidth,
                                             _imageHeight, _bboxesNormalized, outputPtr);
                    }
                }
            }
        }
        // clip the prior's coordinate such that it is within [0, 1]
        if (_clip)
        {
            int _outChannelSize = _layerHeight * _layerWidth * _numPriors * 4;
            outputPtr = outputs[0].ptr<float>();
            for (size_t d = 0; d < _outChannelSize; ++d)
            {
                outputPtr[d] = std::min<float>(std::max<float>(outputPtr[d], 0.), 1.);
            }
        }
        // set the variance.
        outputPtr = outputs[0].ptr<float>(0, 1);
        if(_variance.size() == 1)
        {
            Mat secondChannel(1, outputs[0].size[2], CV_32F, outputPtr);
            secondChannel.setTo(Scalar::all(_variance[0]));
        }
        else
        {
            int count = 0;
            for (size_t h = 0; h < _layerHeight; ++h)
            {
                for (size_t w = 0; w < _layerWidth; ++w)
                {
                    for (size_t i = 0; i < _numPriors; ++i)
                    {
                        for (int j = 0; j < 4; ++j)
                        {
                            outputPtr[count] = _variance[j];
                            ++count;
                        }
                    }
                }
            }
        }
    }


#ifdef HAVE_DNN_NGRAPH
    virtual Ptr<BackendNode> initNgraph(const std::vector<Ptr<BackendWrapper> >& inputs, const std::vector<Ptr<BackendNode> >& nodes) CV_OVERRIDE
    {
        CV_Assert(nodes.size() == 2);
        auto layer = nodes[0].dynamicCast<InfEngineNgraphNode>()->node;
        auto image = nodes[1].dynamicCast<InfEngineNgraphNode>()->node;
        auto layer_shape = std::make_shared<ngraph::op::ShapeOf>(layer);
        auto image_shape = std::make_shared<ngraph::op::ShapeOf>(image);

        auto lower_bounds = std::make_shared<ngraph::op::Constant>(ngraph::element::i64, ngraph::Shape{1}, std::vector<int64_t>{2});
        auto upper_bounds = std::make_shared<ngraph::op::Constant>(ngraph::element::i64, ngraph::Shape{1}, std::vector<int64_t>{4});
        auto strides      = std::make_shared<ngraph::op::Constant>(ngraph::element::i64, ngraph::Shape{1}, std::vector<int64_t>{1});

        auto slice_layer = std::make_shared<ngraph::op::v1::StridedSlice>(layer_shape,
                                            lower_bounds, upper_bounds, strides, std::vector<int64_t>{}, std::vector<int64_t>{});
        auto slice_image = std::make_shared<ngraph::op::v1::StridedSlice>(image_shape,
                                            lower_bounds, upper_bounds, strides, std::vector<int64_t>{}, std::vector<int64_t>{});

        if (_explicitSizes)
        {
            CV_Assert_N(!_boxWidths.empty(), !_boxHeights.empty(), !_variance.empty());
            CV_Assert(_boxWidths.size() == _boxHeights.size());
            ngraph::op::PriorBoxClusteredAttrs attrs;
            attrs.widths = _boxWidths;
            attrs.heights = _boxHeights;
            attrs.clip = _clip;
            CV_CheckEQ(_offsetsX.size(), (size_t)1, ""); CV_CheckEQ(_offsetsY.size(), (size_t)1, ""); CV_CheckEQ(_offsetsX[0], _offsetsY[0], "");
            attrs.offset = _offsetsX[0];
            attrs.step_heights = _stepY;
            attrs.step_widths = _stepX;
            attrs.variances = _variance;

            auto priorBox = std::make_shared<ngraph::op::PriorBoxClustered>(slice_layer, slice_image, attrs);
            auto axis = std::make_shared<ngraph::op::Constant>(ngraph::element::i64, ngraph::Shape{1}, std::vector<int64_t>{0});
            auto unsqueeze = std::make_shared<ngraph::op::v0::Unsqueeze>(priorBox, axis);
            return Ptr<BackendNode>(new InfEngineNgraphNode(unsqueeze));
        }
        else
        {
            ngraph::op::PriorBoxAttrs attrs;
            attrs.min_size = _minSize;
            attrs.max_size = _maxSize;
            // doesn't work with empty aspectRatio
            attrs.aspect_ratio = !_aspectRatios.empty()? _aspectRatios : std::vector<float>{1.0f};
            attrs.clip = _clip;
            attrs.flip = false;
            attrs.variance = _variance;
            CV_CheckEQ(_offsetsX.size(), (size_t)1, ""); CV_CheckEQ(_offsetsY.size(), (size_t)1, ""); CV_CheckEQ(_offsetsX[0], _offsetsY[0], "");
            attrs.offset = _offsetsX[0];

            attrs.step = _stepX;
            attrs.scale_all_sizes = !_aspectRatios.empty();

            auto priorBox = std::make_shared<ngraph::op::PriorBox>(slice_layer, slice_image, attrs);
            auto axis = std::make_shared<ngraph::op::Constant>(ngraph::element::i64, ngraph::Shape{1}, std::vector<int64_t>{0});
            auto unsqueeze = std::make_shared<ngraph::op::v0::Unsqueeze>(priorBox, axis);
            return Ptr<BackendNode>(new InfEngineNgraphNode(unsqueeze));
        }
    }
#endif  // HAVE_DNN_NGRAPH


#ifdef HAVE_CUDA
    Ptr<BackendNode> initCUDA(
        void *context_,
        const std::vector<Ptr<BackendWrapper>>& inputs,
        const std::vector<Ptr<BackendWrapper>>& outputs
    ) override
    {
        auto context = reinterpret_cast<csl::CSLContext*>(context_);

        auto feature_map_wrapper = inputs[0].dynamicCast<CUDABackendWrapper>();
        auto feature_map_shape = feature_map_wrapper->getShape();

        auto image_wrapper = inputs[1].dynamicCast<CUDABackendWrapper>();
        auto image_shape = image_wrapper->getShape();

        PriorBoxConfiguration config;
        config.feature_map_width = feature_map_shape.rbegin()[0];
        config.feature_map_height = feature_map_shape.rbegin()[1];
        config.image_width = image_shape.rbegin()[0];
        config.image_height = image_shape.rbegin()[1];

        config.num_priors = _numPriors;
        config.box_widths = _boxWidths;
        config.box_heights = _boxHeights;
        config.offsets_x = _offsetsX;
        config.offsets_y = _offsetsY;
        config.stepX = _stepX;
        config.stepY = _stepY;

        config.variance = _variance;

        config.clip = _clip;
        config.normalize = _bboxesNormalized;

        return make_cuda_node<cuda4dnn::PriorBoxOp>(preferableTarget, std::move(context->stream), config);
    }
#endif


#ifdef HAVE_VULKAN
    virtual Ptr<BackendNode> initVkCom(const std::vector<Ptr<BackendWrapper> > &input) CV_OVERRIDE
    {
        std::shared_ptr<vkcom::OpBase> op(new vkcom::OpPriorBox(_stepX, _stepY,
                                                                _clip, _numPriors,
                                                                _variance, _offsetsX,
                                                                _offsetsY, _boxWidths,
                                                                _boxHeights));
        return Ptr<BackendNode>(new VkComBackendNode(input, op));
    }
#endif // HAVE_VULKAN


    virtual int64 getFLOPS(const std::vector<MatShape> &inputs,
                           const std::vector<MatShape> &outputs) const CV_OVERRIDE
    {
        CV_UNUSED(outputs); // suppress unused variable warning
        long flops = 0;

        for (int i = 0; i < inputs.size(); i++)
        {
            flops += total(inputs[i], 2) * _numPriors * 4;
        }

        return flops;
    }

private:
    std::vector<float> _minSize;
    std::vector<float> _maxSize;

    float _stepX, _stepY;

    std::vector<float> _aspectRatios;
    std::vector<float> _variance;
    std::vector<float> _offsetsX;
    std::vector<float> _offsetsY;
    // Precomputed final widths and heights based on aspect ratios or explicit sizes.
    std::vector<float> _boxWidths;
    std::vector<float> _boxHeights;

#ifdef HAVE_OPENCL
    UMat umat_offsetsX;
    UMat umat_offsetsY;
    UMat umat_widths;
    UMat umat_heights;
    UMat umat_variance;
#endif

    bool _flip;
    bool _clip;
    bool _explicitSizes;
    bool _bboxesNormalized;

    size_t _numPriors;

    static const size_t _numAxes = 4;
    static const std::string _layerName;

    static float* addPrior(float center_x, float center_y, float width, float height,
                           float imgWidth, float imgHeight, bool normalized, float* dst)
    {
        if (normalized)
        {
            dst[0] = (center_x - width * 0.5f) / imgWidth;    // xmin
            dst[1] = (center_y - height * 0.5f) / imgHeight;  // ymin
            dst[2] = (center_x + width * 0.5f) / imgWidth;    // xmax
            dst[3] = (center_y + height * 0.5f) / imgHeight;  // ymax
        }
        else
        {
            dst[0] = center_x - width * 0.5f;          // xmin
            dst[1] = center_y - height * 0.5f;         // ymin
            dst[2] = center_x + width * 0.5f - 1.0f;   // xmax
            dst[3] = center_y + height * 0.5f - 1.0f;  // ymax
        }
        return dst + 4;
    }
};

const std::string PriorBoxLayerImpl::_layerName = std::string("PriorBox");

Ptr<PriorBoxLayer> PriorBoxLayer::create(const LayerParams &params)
{
    return Ptr<PriorBoxLayer>(new PriorBoxLayerImpl(params));
}

}
}
