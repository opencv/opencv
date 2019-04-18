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
                  std::vector<int>& parameter, bool hasDefault = false, const int& defaultValue = 0)
{
    std::string nameH = makeName(nameBase, std::string("_h"));
    std::string nameW = makeName(nameBase, std::string("_w"));
    std::string nameAll_ = nameAll;
    if (nameAll_ == "")
        nameAll_ = nameBase;

    if (params.has(nameH) && params.has(nameW))
    {
        parameter.push_back(params.get<int>(nameH));
        parameter.push_back(params.get<int>(nameW));
        return true;
    }
    else
    {
        if (params.has(nameAll_))
        {
            DictValue param = params.get(nameAll_);
            for (int i = 0; i < param.size(); i++)
                parameter.push_back(param.get<int>(i));
            if (parameter.size() == 1) {
                parameter.push_back(parameter.back());
            }
            return true;
        }
        else
        {
            if (hasDefault)
            {
                parameter.push_back(defaultValue);
                parameter.push_back(defaultValue);
                return true;
            }
                return false;
        }
    }
}

void getKernelSize(const LayerParams &params, std::vector<int>& kernel)
{
      if (!util::getParameter(params, "kernel", "kernel_size", kernel))
      {
         CV_Error(cv::Error::StsBadArg, "kernel_size (or kernel_h and kernel_w) not specified");
      }
      for (int i = 0; i < kernel.size(); i++) {
          CV_Assert(kernel[i] > 0);
      }
}

void getStrideAndPadding(const LayerParams &params, std::vector<int>& pads, std::vector<int>& strides, cv::String& padMode)
{
    if (params.has("pad_l") && params.has("pad_t") && params.has("pad_r") && params.has("pad_b")) {
        pads.push_back(params.get<int>("pad_t"));
        pads.push_back(params.get<int>("pad_l"));
        pads.push_back(params.get<int>("pad_b"));
        pads.push_back(params.get<int>("pad_r"));
    }
    else {
        util::getParameter(params, "pad", "pad", pads, true, 0);
        if (pads.size() < 4) {
            pads.insert(pads.end(), pads.begin(), pads.end());
        }
    }
    util::getParameter(params, "stride", "stride", strides, true, 1);

    padMode = "";
    if (params.has("pad_mode"))
        padMode = params.get<String>("pad_mode");


    for (int i = 0; i < pads.size(); i++)
        CV_Assert(pads[i] >= 0);
    for (int i = 0; i < strides.size(); i++)
        CV_Assert(strides[i] > 0);
}
}

void getPoolingKernelParams(const LayerParams &params, std::vector<int>& kernel, bool &globalPooling,
                            std::vector<int>& pads, std::vector<int>& strides, cv::String &padMode)
{
    util::getStrideAndPadding(params, pads, strides, padMode);
    globalPooling = params.has("global_pooling") &&
                    params.get<bool>("global_pooling");
    if (globalPooling)
    {
        if(params.has("kernel_h") || params.has("kernel_w") || params.has("kernel_size"))
        {
            CV_Error(cv::Error::StsBadArg, "In global_pooling mode, kernel_size (or kernel_h and kernel_w) cannot be specified");
        }
        for (int i = 0; i < pads.size(); i++) {
            if (pads[i] != 0)
                CV_Error(cv::Error::StsBadArg, "In global_pooling mode, pads must be = 0");
        }
        for (int i = 0; i < strides.size(); i++) {
            if (strides[i] != 1)
                CV_Error(cv::Error::StsBadArg, "In global_pooling mode, strides must be = 1");
        }
    }
    else
    {
        util::getKernelSize(params, kernel);
        if (strides.size() == kernel.size() - 1) {
            strides.push_back(strides.back());
        }
        if (pads.size() == 2 * kernel.size() - 2) {
            pads.push_back(pads.back());
            pads.push_back(pads.back());
        }
        CV_Assert(kernel.size() == strides.size() && kernel.size() == pads.size() / 2);
    }
}

void getConvolutionKernelParams(const LayerParams &params, std::vector<int>& kernel, std::vector<int>& pads,
                                std::vector<int>& strides, std::vector<int>& dilations, cv::String &padMode)
{
    util::getKernelSize(params, kernel);
    util::getStrideAndPadding(params, pads, strides, padMode);
    util::getParameter(params, "dilation", "dilation", dilations, true, 1);

    if (dilations.size() == kernel.size() - 1) {
        dilations.push_back(dilations.back());
    }
    if (strides.size() == kernel.size() - 1) {
        strides.push_back(strides.back());
    }
    if (pads.size() == 2 * kernel.size() - 2) {
        pads.push_back(pads.back());
        pads.push_back(pads.back());
    }
    CV_Assert(kernel.size() == strides.size() && kernel.size() == pads.size() / 2 && kernel.size() == dilations.size());
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
void getConvPoolOutParams(const std::vector<int>& inp, const std::vector<int>& kernel,
                          const std::vector<int>& stride, const String &padMode,
                          const std::vector<int>& dilation, std::vector<int>& out)
{
    if (padMode == "VALID")
    {
        for (int i = 0; i < inp.size(); i++)
            out.push_back((inp[i] - dilation[i] * (kernel[i] - 1) - 1 + stride[i]) / stride[i] );
    }
    else if (padMode == "SAME")
    {
        for (int i = 0; i < inp.size(); i++)
            out.push_back((inp[i] - 1 + stride[i]) / stride[i]);
    }
    else
        CV_Error(Error::StsError, "Unsupported padding mode");
}

void getConvPoolPaddings(const Size& inp, const Size& out,
                         const Size &kernel, const Size &stride,
                         const String &padMode, const Size &dilation, int &padT, int &padL, int &padB, int &padR)
{
    if (padMode == "VALID")
    {
        padT = padL = padB = padR = 0;
    }
    else if (padMode == "SAME")
    {
        int Ph = std::max(0, (out.height - 1) * stride.height + (dilation.height * (kernel.height - 1) + 1) - inp.height);
        int Pw = std::max(0, (out.width - 1) * stride.width + (dilation.width * (kernel.width - 1) + 1) - inp.width);
        // For odd values of total padding, add more padding at the 'right'
        // side of the given dimension.
        padT = padB = Ph / 2;
        padL = padR = Pw / 2;
    }
}

void getConvPoolPaddings(const std::vector<int>& inp, const std::vector<int>& out,
                         const std::vector<int>& kernel, const std::vector<int>& stride,
                         const String &padMode, const std::vector<int>& dilation, std::vector<int>& pads)
{
    CV_Assert(pads.size() / 2 == kernel.size());
    if (padMode == "VALID")
        std::fill(pads.begin(), pads.end(), 0);
    else if (padMode == "SAME")
    {
        for (int i = 0; i < kernel.size(); i++)
            pads[i] = pads[i + pads.size() / 2] = std::max(0, (out[i] - 1) * stride[i] + (dilation[i] * (kernel[i] - 1) + 1) - inp[i]) / 2;
    }
}

}
}
