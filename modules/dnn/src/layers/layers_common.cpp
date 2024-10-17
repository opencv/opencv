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

namespace cv
{
namespace dnn
{

namespace util
{

std::string makeName(const std::string& str1, const std::string& str2)
{
    return str1 + str2;
}

bool getParameter(const LayerParams &params, const std::string& nameBase, const std::string& nameAll,
                  std::vector<size_t>& parameter, bool hasDefault = false, const std::vector<size_t>& defaultValue = std::vector<size_t>(2, 0))
{
    std::string nameH = makeName(nameBase, std::string("_h"));
    std::string nameW = makeName(nameBase, std::string("_w"));
    std::string nameAll_ = nameAll;
    if (nameAll_ == "")
        nameAll_ = nameBase;

    if (params.has(nameH) && params.has(nameW))
    {
        CV_Assert(params.get<int>(nameH) >= 0 && params.get<int>(nameW) >= 0);
        parameter.push_back(params.get<int>(nameH));
        parameter.push_back(params.get<int>(nameW));
        return true;
    }
    else
    {
        if (params.has(nameAll_))
        {
            DictValue param = params.get(nameAll_);
            for (int i = 0; i < param.size(); i++) {
                CV_Assert(param.get<int>(i) >= 0);
                parameter.push_back(param.get<int>(i));
            }
            if (parameter.size() == 1)
                parameter.resize(2, parameter[0]);
            return true;
        }
        else
        {
            if (hasDefault)
            {
                parameter = defaultValue;
                return true;
            }
            else
            {
                return false;
            }
        }
    }
}

void getKernelSize(const LayerParams &params, std::vector<size_t>& kernel)
{
    if (!util::getParameter(params, "kernel", "kernel_size", kernel))
        CV_Error(cv::Error::StsBadArg, "kernel_size (or kernel_h and kernel_w) not specified");

    for (int i = 0; i < kernel.size(); i++)
        CV_Assert(kernel[i] > 0);
}

void getStrideAndPadding(const LayerParams &params, std::vector<size_t>& pads_begin, std::vector<size_t>& pads_end,
                         std::vector<size_t>& strides, cv::String& padMode, size_t kernel_size = 2)
{
    if (params.has("pad_l") && params.has("pad_t") && params.has("pad_r") && params.has("pad_b")) {
        CV_Assert(params.get<int>("pad_t") >= 0 && params.get<int>("pad_l") >= 0 &&
                  params.get<int>("pad_b") >= 0 && params.get<int>("pad_r") >= 0);
        pads_begin.push_back(params.get<int>("pad_t"));
        pads_begin.push_back(params.get<int>("pad_l"));
        pads_end.push_back(params.get<int>("pad_b"));
        pads_end.push_back(params.get<int>("pad_r"));
    }
    else {
        util::getParameter(params, "pad", "pad", pads_begin, true, std::vector<size_t>(kernel_size, 0));
        if (pads_begin.size() < 4)
            pads_end = pads_begin;
        else
        {
            pads_end = std::vector<size_t>(pads_begin.begin() + pads_begin.size() / 2, pads_begin.end());
            pads_begin.resize(pads_begin.size() / 2);
        }
        CV_Assert(pads_begin.size() == pads_end.size());
    }
    util::getParameter(params, "stride", "stride", strides, true, std::vector<size_t>(kernel_size, 1));

    padMode = "";
    if (params.has("pad_mode"))
    {
        padMode = params.get<String>("pad_mode");
    }

    for (int i = 0; i < strides.size(); i++)
        CV_Assert(strides[i] > 0);
}
}

void getPoolingKernelParams(const LayerParams &params, std::vector<size_t>& kernel, std::vector<bool>& globalPooling,
                            std::vector<size_t>& pads_begin, std::vector<size_t>& pads_end,
                            std::vector<size_t>& strides, cv::String &padMode)
{
    bool is_global = params.get<bool>("global_pooling", false);
    globalPooling.assign({
        params.get<bool>("global_pooling_d", is_global),
        params.get<bool>("global_pooling_h", is_global),
        params.get<bool>("global_pooling_w", is_global)
    });

    if (globalPooling[0] || globalPooling[1] || globalPooling[2])
    {
        util::getStrideAndPadding(params, pads_begin, pads_end, strides, padMode);
        if ((globalPooling[0] && params.has("kernel_d")) ||
            (globalPooling[1] && params.has("kernel_h")) ||
            (globalPooling[2] && params.has("kernel_w")) ||
            params.has("kernel_size")) {
            CV_Error(cv::Error::StsBadArg, "In global_pooling mode, kernel_size (or kernel_h and kernel_w) cannot be specified");
        }

        kernel.resize(3);
        kernel[0] = params.get<int>("kernel_d", 1);
        kernel[1] = params.get<int>("kernel_h", 1);
        kernel[2] = params.get<int>("kernel_w", 1);

        for (int i = 0, j = globalPooling.size() - pads_begin.size(); i < pads_begin.size(); i++, j++) {
            if ((pads_begin[i] != 0 || pads_end[i] != 0) && globalPooling[j])
                CV_Error(cv::Error::StsBadArg, "In global_pooling mode, pads must be = 0");
        }
        for (int i = 0, j = globalPooling.size() - strides.size(); i < strides.size(); i++, j++) {
            if (strides[i] != 1 && globalPooling[j])
                CV_Error(cv::Error::StsBadArg, "In global_pooling mode, strides must be = 1");
        }
    }
    else
    {
        util::getKernelSize(params, kernel);
        util::getStrideAndPadding(params, pads_begin, pads_end, strides, padMode, kernel.size());
    }
}

void getConvolutionKernelParams(const LayerParams &params, std::vector<size_t>& kernel, std::vector<size_t>& pads_begin,
                                std::vector<size_t>& pads_end, std::vector<size_t>& strides,
                                std::vector<size_t>& dilations, cv::String &padMode, std::vector<size_t>& adjust_pads,
                                bool& useWinograd)
{
    util::getKernelSize(params, kernel);
    util::getStrideAndPadding(params, pads_begin, pads_end, strides, padMode, kernel.size());
    util::getParameter(params, "dilation", "dilation", dilations, true, std::vector<size_t>(kernel.size(), 1));
    util::getParameter(params, "adj", "adj", adjust_pads, true, std::vector<size_t>(kernel.size(), 0));
    useWinograd = params.get<bool>("use_winograd", useWinograd);

    for (int i = 0; i < dilations.size(); i++)
        CV_Assert(dilations[i] > 0);
}

// From TensorFlow code:
// Total padding on rows and cols is
// Pr = (R' - 1) * S + Kr - R
// Pc = (C' - 1) * S + Kc - C
// where (R', C') are output dimensions, (R, C) are input dimensions, S
// is stride, (Kr, Kc) are filter dimensions.
// We pad Pr/2 on the left and Pr - Pr/2 on the right, Pc/2 on the top
// and Pc - Pc/2 on the bottom.  When Pr or Pc is odd, this means
// we pad more on the right and bottom than on the top and left.
void getConvPoolOutParams(const std::vector<int>& inp, const std::vector<size_t>& kernel,
                          const std::vector<size_t>& stride, const String &padMode,
                          const std::vector<size_t>& dilation, std::vector<int>& out)
{
    if (padMode == "VALID")
    {
        for (int i = 0; i < inp.size(); i++)
            out.push_back((inp[i] - dilation[i] * (kernel[i] - 1) - 1 + stride[i]) / stride[i]);
    }
    else if (padMode == "SAME")
    {
        for (int i = 0; i < inp.size(); i++)
            out.push_back((inp[i] - 1 + stride[i]) / stride[i]);
    }
    else
    {
        CV_Error(Error::StsError, "Unsupported padding mode");
    }
}

void getConvPoolPaddings(const std::vector<int>& inp, const std::vector<size_t>& kernel,
                         const std::vector<size_t>& strides, const String &padMode,
                         std::vector<size_t>& pads_begin, std::vector<size_t>& pads_end)
{
    if (padMode == "SAME" || padMode == "VALID")
    {
        pads_begin.assign(kernel.size(), 0);
        pads_end.assign(kernel.size(), 0);
    }
    if (padMode == "SAME")
    {
        CV_Assert_N(kernel.size() == strides.size(), kernel.size() == inp.size());
        for (int i = 0; i < pads_begin.size(); i++) {
            // There are test cases with stride > kernel.
            if (strides[i] <= kernel[i])
            {
                int pad = (kernel[i] - 1 - (inp[i] - 1 + strides[i]) % strides[i]) / 2;
                pads_begin[i] = pads_end[i] = pad;
            }
        }
    }
}

double getWeightScale(const Mat& weightsMat)
{
    double realMin, realMax;

    cv::minMaxIdx(weightsMat, &realMin, &realMax);
    realMin = std::min(realMin, 0.0);
    realMax = std::max(realMax, 0.0);

    return (realMax == realMin) ? 1.0 : std::max(-realMin, realMax)/127;
}

void tensorToIntVec(const Mat& tensor, std::vector<int>& vec)
{
    if (tensor.empty()) {
        vec.clear();
    } else {
        int type = tensor.type();
        CV_Assert(type == CV_32S || type == CV_64S);
        CV_Assert(tensor.dims <= 1);
        int size = (int)tensor.total();
        vec.resize(size);
        for (int i = 0; i < size; i++) {
            vec[i] = type == CV_32S ? tensor.at<int>(i) :
                saturate_cast<int>(tensor.at<int64_t>(i));
        }
    }
}

void tensorToFloatVec(const Mat& tensor, std::vector<float>& vec)
{
    if (tensor.empty()) {
        vec.clear();
    } else {
        int type = tensor.type();
        MatShape shape = tensor.shape();
        CV_Assert(type == CV_32F || type == CV_16F);
        CV_Assert(shape.dims <= 1);
        int size = (int)shape.total();
        vec.resize(size);
        for (int i = 0; i < size; i++) {
            vec[i] = type == CV_32F ? tensor.at<float>(i) :
                (float)tensor.at<hfloat>(i);
        }
    }
}

void reshapeAndCopyFirst(InputArrayOfArrays inputs,
                         OutputArrayOfArrays outputs,
                         const MatShape& shape)
{
    int inpKind = inputs.kind(), outKind = outputs.kind();
    CV_Assert(inpKind == outKind);
    CV_Assert(inpKind == _InputArray::STD_VECTOR_MAT ||
              inpKind == _InputArray::STD_VECTOR_UMAT);
    CV_Assert(inputs.isContinuous(0));
    int inpType = inputs.type(0);
    if (inpKind == _InputArray::STD_VECTOR_MAT) {
        Mat inp = inputs.getMat(0);
        std::vector<Mat>& outref = outputs.getMatVecRef();
        outref.resize(1);
        outref[0].fit(shape, inpType);
        CV_Assert(outref[0].isContinuous());
        Mat inp_ = inp.reshape(0, shape);
        if (inp_.data != outref[0].data)
            inp_.copyTo(outref[0]);
    }
    else {
        UMat inp = inputs.getUMat(0);
        std::vector<UMat>& outref = outputs.getUMatVecRef();
        outref.resize(1);
        outref[0].fit(shape, inpType);
        CV_Assert(outref[0].isContinuous());
        UMat inp_ = inp.reshape(0, shape);
        inp_.copyTo(outref[0]);
    }
}

MatShape tensorToShape(const Mat& shapeTensor)
{
    std::vector<int> shapeSpecVec;
    tensorToIntVec(shapeTensor, shapeSpecVec);
    return MatShape(shapeSpecVec);
}

void tensorToScalar(const Mat& tensor, int type, void* value)
{
    CV_Assert(tensor.total() == 1);
    int type0 = tensor.type();
    int depth = CV_MAT_DEPTH(type), cn = CV_MAT_CN(type);
    CV_Assert(cn == 1);
    double v = 0;
    int64_t iv = 0;
    bool isflt = type0 == CV_32F || type0 == CV_64F || type0 == CV_16F || type0 == CV_16BF;

    if (type0 == CV_8U)
        iv = *tensor.ptr<uint8_t>();
    else if (type0 == CV_8S)
        iv = *tensor.ptr<uint8_t>();
    else if (type0 == CV_16U)
        iv = *tensor.ptr<uint8_t>();
    else if (type0 == CV_16S)
        iv = *tensor.ptr<int16_t>();
    else if (type0 == CV_32U)
        iv = *tensor.ptr<uint32_t>();
    else if (type0 == CV_32S)
        iv = *tensor.ptr<int32_t>();
    else if (type0 == CV_64S)
        iv = *tensor.ptr<int64_t>();
    else if (type0 == CV_32F)
        v = *tensor.ptr<float>();
    else if (type0 == CV_64F)
        v = *tensor.ptr<double>();
    else if (type0 == CV_16F)
        v = (float)*tensor.ptr<hfloat>();
    else if (type0 == CV_16BF)
        v = (float)*tensor.ptr<bfloat>();
    else if (type0 == CV_Bool)
        iv = *tensor.ptr<uint8_t>() != 0;
    else {
        CV_Error_(Error::StsNotImplemented, ("type %s is not supported", typeToString(type0).c_str()));
    }

    if (depth == CV_8U)
        *reinterpret_cast<uint8_t*>(value) = isflt ? saturate_cast<uint8_t>(v) : saturate_cast<uint8_t>(iv);
    else if (depth == CV_8S)
        *reinterpret_cast<int8_t*>(value) = isflt ? saturate_cast<int8_t>(v) : saturate_cast<int8_t>(iv);
    else if (depth == CV_16U)
        *reinterpret_cast<uint16_t*>(value) = isflt ? saturate_cast<uint16_t>(v) : saturate_cast<uint16_t>(iv);
    else if (depth == CV_16S)
        *reinterpret_cast<int16_t*>(value) = isflt ? saturate_cast<int16_t>(v) : saturate_cast<int16_t>(iv);
    else if (depth == CV_32U)
        *reinterpret_cast<uint32_t*>(value) = isflt ? saturate_cast<uint32_t>(v) : saturate_cast<uint32_t>(iv);
    else if (depth == CV_32S)
        *reinterpret_cast<int32_t*>(value) = isflt ? saturate_cast<int32_t>(v) : saturate_cast<int32_t>(iv);
    else if (depth == CV_64U)
        *reinterpret_cast<uint64_t*>(value) = isflt ? saturate_cast<uint64_t>(v) : saturate_cast<uint64_t>(iv);
    else if (depth == CV_64S)
        *reinterpret_cast<int64_t*>(value) = isflt ? saturate_cast<int64_t>(v) : iv;
    else if (depth == CV_32F)
        *reinterpret_cast<float*>(value) = isflt ? (float)v : saturate_cast<float>(iv);
    else if (depth == CV_64F)
        *reinterpret_cast<double*>(value) = isflt ? v : saturate_cast<double>(iv);
    else if (depth == CV_16F)
        *reinterpret_cast<hfloat*>(value) = isflt ? saturate_cast<hfloat>(v) : saturate_cast<hfloat>(iv);
    else if (depth == CV_16BF)
        *reinterpret_cast<bfloat*>(value) = isflt ? saturate_cast<bfloat>(v) : saturate_cast<bfloat>(iv);
    else if (depth == CV_Bool)
        *reinterpret_cast<uint8_t*>(value) = isflt ? (uint8_t)(v != 0) : (uint8_t)(iv != 0);
    else {
        CV_Error_(Error::StsNotImplemented, ("type %s is not supported", typeToString(depth).c_str()));
    }
}

}
}
