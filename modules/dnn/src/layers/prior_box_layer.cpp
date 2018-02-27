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
#include "op_inf_engine.hpp"
#include <float.h>
#include <algorithm>
#include <cmath>
#include "opencl_kernels_dnn.hpp"

namespace cv
{
namespace dnn
{

class PriorBoxLayerImpl : public PriorBoxLayer
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
            bool alreadyExists = false;

            for (size_t j = 0; j < _aspectRatios.size(); ++j)
            {
                if (fabs(aspectRatio - _aspectRatios[j]) < 1e-6)
                {
                    alreadyExists = true;
                    break;
                }
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
        _minSize = getParameter<float>(params, "min_size", 0, false, 0);
        _flip = getParameter<bool>(params, "flip", 0, false, true);
        _clip = getParameter<bool>(params, "clip", 0, false, true);
        _bboxesNormalized = getParameter<bool>(params, "normalized_bbox", 0, false, true);

        _aspectRatios.clear();

        getAspectRatios(params);
        getVariance(params);

        _maxSize = -1;
        if (params.has("max_size"))
        {
            _maxSize = params.get("max_size").get<float>(0);
            CV_Assert(_maxSize > _minSize);
        }

        std::vector<float> widths, heights;
        getParams("width", params, &widths);
        getParams("height", params, &heights);
        _explicitSizes = !widths.empty();
        CV_Assert(widths.size() == heights.size());

        if (_explicitSizes)
        {
            CV_Assert(_aspectRatios.empty(), !params.has("min_size"), !params.has("max_size"));
            _boxWidths = widths;
            _boxHeights = heights;
        }
        else
        {
            CV_Assert(!_aspectRatios.empty(), _minSize > 0);
            _boxWidths.resize(1 + (_maxSize > 0 ? 1 : 0) + _aspectRatios.size());
            _boxHeights.resize(_boxWidths.size());
            _boxWidths[0] = _boxHeights[0] = _minSize;

            int i = 1;
            if (_maxSize > 0)
            {
                // second prior: aspect_ratio = 1, size = sqrt(min_size * max_size)
                _boxWidths[i] = _boxHeights[i] = sqrt(_minSize * _maxSize);
                i += 1;
            }

            // rest of priors
            for (size_t r = 0; r < _aspectRatios.size(); ++r)
            {
                float arSqrt = sqrt(_aspectRatios[r]);
                _boxWidths[i + r] = _minSize * arSqrt;
                _boxHeights[i + r] = _minSize / arSqrt;
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
            CV_Assert(!params.has("offset"), params.has("offset_h"), params.has("offset_w"));
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

    virtual bool supportBackend(int backendId)
    {
        return backendId == DNN_BACKEND_DEFAULT ||
               backendId == DNN_BACKEND_INFERENCE_ENGINE && haveInfEngine() && !_explicitSizes;
    }

    bool getMemoryShapes(const std::vector<MatShape> &inputs,
                         const int requiredOutputs,
                         std::vector<MatShape> &outputs,
                         std::vector<MatShape> &internals) const
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

#ifdef HAVE_OPENCL
    bool forward_ocl(InputArrayOfArrays inps, OutputArrayOfArrays outs, OutputArrayOfArrays internals)
    {
        std::vector<UMat> inputs;
        std::vector<UMat> outputs;

        inps.getUMatVector(inputs);
        outs.getUMatVector(outputs);

        int _layerWidth = inputs[0].size[3];
        int _layerHeight = inputs[0].size[2];

        int _imageWidth = inputs[1].size[3];
        int _imageHeight = inputs[1].size[2];

        float stepX, stepY;
        if (_stepX == 0 || _stepY == 0)
        {
            stepX = static_cast<float>(_imageWidth) / _layerWidth;
            stepY = static_cast<float>(_imageHeight) / _layerHeight;
        } else {
            stepX = _stepX;
            stepY = _stepY;
        }

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

        size_t nthreads = _layerHeight * _layerWidth;

        ocl::Kernel kernel("prior_box", ocl::dnn::prior_box_oclsrc);
        kernel.set(0, (int)nthreads);
        kernel.set(1, (float)stepX);
        kernel.set(2, (float)stepY);
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

        // clip the prior's coordidate such that it is within [0, 1]
        if (_clip)
        {
            Mat mat = outputs[0].getMat(ACCESS_READ);
            int aspect_count = (_maxSize > 0) ? 1 : 0;
            int offset = nthreads * 4 * _offsetsX.size() * (1 + aspect_count + _aspectRatios.size());
            float* outputPtr = mat.ptr<float>() + offset;
            int _outChannelSize = _layerHeight * _layerWidth * _numPriors * 4;
            for (size_t d = 0; d < _outChannelSize; ++d)
            {
                outputPtr[d] = std::min<float>(std::max<float>(outputPtr[d], 0.), 1.);
            }
        }

        // set the variance.
        {
            ocl::Kernel kernel("set_variance", ocl::dnn::prior_box_oclsrc);
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

    void forward(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr, OutputArrayOfArrays internals_arr)
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        CV_OCL_RUN((preferableTarget == DNN_TARGET_OPENCL) &&
                   OCL_PERFORMANCE_CHECK(ocl::Device::getDefault().isIntel()),
                   forward_ocl(inputs_arr, outputs_arr, internals_arr))

        Layer::forward_fallback(inputs_arr, outputs_arr, internals_arr);
    }

    void forward(std::vector<Mat*> &inputs, std::vector<Mat> &outputs, std::vector<Mat> &internals)
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        CV_Assert(inputs.size() == 2);

        int _layerWidth = inputs[0]->size[3];
        int _layerHeight = inputs[0]->size[2];

        int _imageWidth = inputs[1]->size[3];
        int _imageHeight = inputs[1]->size[2];

        float stepX, stepY;
        if (_stepX == 0 || _stepY == 0) {
          stepX = static_cast<float>(_imageWidth) / _layerWidth;
          stepY = static_cast<float>(_imageHeight) / _layerHeight;
        } else {
          stepX = _stepX;
          stepY = _stepY;
        }

        int _outChannelSize = _layerHeight * _layerWidth * _numPriors * 4;

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
                        float center_x = (w + _offsetsX[j]) * stepX;
                        float center_y = (h + _offsetsY[j]) * stepY;
                        outputPtr = addPrior(center_x, center_y, _boxWidth, _boxHeight, _imageWidth,
                                             _imageHeight, _bboxesNormalized, outputPtr);
                    }
                }
            }
        }
        // clip the prior's coordidate such that it is within [0, 1]
        if (_clip)
        {
            for (size_t d = 0; d < _outChannelSize; ++d)
            {
                outputPtr[d] = std::min<float>(std::max<float>(outputPtr[d], 0.), 1.);
            }
        }
        // set the variance.
        outputPtr = outputs[0].ptr<float>(0, 1);
        if(_variance.size() == 1)
        {
            Mat secondChannel(outputs[0].size[2], outputs[0].size[3], CV_32F, outputPtr);
            secondChannel.setTo(Scalar(_variance[0]));
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

    virtual Ptr<BackendNode> initInfEngine(const std::vector<Ptr<BackendWrapper> >&)
    {
#ifdef HAVE_INF_ENGINE
        InferenceEngine::LayerParams lp;
        lp.name = name;
        lp.type = "PriorBox";
        lp.precision = InferenceEngine::Precision::FP32;
        std::shared_ptr<InferenceEngine::CNNLayer> ieLayer(new InferenceEngine::CNNLayer(lp));

        ieLayer->params["min_size"] = format("%f", _minSize);
        ieLayer->params["max_size"] = _maxSize > 0 ? format("%f", _maxSize) : "";

        CV_Assert(!_aspectRatios.empty());
        ieLayer->params["aspect_ratio"] = format("%f", _aspectRatios[0]);
        for (int i = 1; i < _aspectRatios.size(); ++i)
            ieLayer->params["aspect_ratio"] += format(",%f", _aspectRatios[i]);

        ieLayer->params["flip"] = _flip ? "1" : "0";
        ieLayer->params["clip"] = _clip ? "1" : "0";

        CV_Assert(!_variance.empty());
        ieLayer->params["variance"] = format("%f", _variance[0]);
        for (int i = 1; i < _variance.size(); ++i)
            ieLayer->params["variance"] += format(",%f", _variance[i]);

        ieLayer->params["step"] = _stepX == _stepY ? format("%f", _stepX) : "0";
        ieLayer->params["step_h"] = _stepY;
        ieLayer->params["step_w"] = _stepX;

        CV_Assert(_offsetsX.size() == 1, _offsetsY.size() == 1, _offsetsX[0] == _offsetsY[0]);
        ieLayer->params["offset"] = format("%f", _offsetsX[0]);;

        return Ptr<BackendNode>(new InfEngineBackendNode(ieLayer));
#endif  // HAVE_INF_ENGINE
        return Ptr<BackendNode>();
    }

    virtual int64 getFLOPS(const std::vector<MatShape> &inputs,
                           const std::vector<MatShape> &outputs) const
    {
        (void)outputs; // suppress unused variable warning
        long flops = 0;

        for (int i = 0; i < inputs.size(); i++)
        {
            flops += total(inputs[i], 2) * _numPriors * 4;
        }

        return flops;
    }

private:
    float _minSize;
    float _maxSize;

    float _stepX, _stepY;

    std::vector<float> _aspectRatios;
    std::vector<float> _variance;
    std::vector<float> _offsetsX;
    std::vector<float> _offsetsY;
    // Precomputed final widhts and heights based on aspect ratios or explicit sizes.
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
