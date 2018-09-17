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
                  int &parameterH, int &parameterW, bool hasDefault = false, const int& defaultValue = 0)
{
    std::string nameH = makeName(nameBase, std::string("_h"));
    std::string nameW = makeName(nameBase, std::string("_w"));
    std::string nameAll_ = nameAll;
    if(nameAll_ == "")
    {
        nameAll_ = nameBase;
    }

    if (params.has(nameH) && params.has(nameW))
    {
        parameterH = params.get<int>(nameH);
        parameterW = params.get<int>(nameW);
        return true;
    }
    else
    {
        if (params.has(nameAll_))
        {
            DictValue param = params.get(nameAll_);
            parameterH = param.get<int>(0);
            if (param.size() == 1)
            {
                parameterW = parameterH;
            }
            else if (param.size() == 2)
            {
                parameterW = param.get<int>(1);
            }
            else
            {
                return false;
            }
            return true;
        }
        else
        {
            if(hasDefault)
            {
                parameterH = parameterW = defaultValue;
                return true;
            }
            else
            {
                return false;
            }
        }
    }
}

void getKernelSize(const LayerParams &params, int &kernelH, int &kernelW)
{
    if(!util::getParameter(params, "kernel", "kernel_size", kernelH, kernelW))
    {
        CV_Error(cv::Error::StsBadArg, "kernel_size (or kernel_h and kernel_w) not specified");
    }

    CV_Assert(kernelH > 0 && kernelW > 0);
}

void getStrideAndPadding(const LayerParams &params, int &padT, int &padL, int &padB, int &padR, int &strideH, int &strideW, cv::String& padMode)
{
    if (params.has("pad_l") && params.has("pad_t") && params.has("pad_r") && params.has("pad_b")) {
        padT = params.get<int>("pad_t");
        padL = params.get<int>("pad_l");
        padB = params.get<int>("pad_b");
        padR = params.get<int>("pad_r");
    }
    else {
        util::getParameter(params, "pad", "pad", padT, padL, true, 0);
        padB = padT;
        padR = padL;
    }
    util::getParameter(params, "stride", "stride", strideH, strideW, true, 1);

    padMode = "";
    if (params.has("pad_mode"))
    {
        padMode = params.get<String>("pad_mode");
    }

    CV_Assert(padT >= 0 && padL >= 0 && padB >= 0 && padR >= 0 && strideH > 0 && strideW > 0);
}
}


void getPoolingKernelParams(const LayerParams &params, int &kernelH, int &kernelW, bool &globalPooling,
                            int &padT, int &padL, int &padB, int &padR, int &strideH, int &strideW, cv::String &padMode)
{
    util::getStrideAndPadding(params, padT, padL, padB, padR, strideH, strideW, padMode);

    globalPooling = params.has("global_pooling") &&
                    params.get<bool>("global_pooling");

    if (globalPooling)
    {
        if(params.has("kernel_h") || params.has("kernel_w") || params.has("kernel_size"))
        {
            CV_Error(cv::Error::StsBadArg, "In global_pooling mode, kernel_size (or kernel_h and kernel_w) cannot be specified");
        }
        if(padT != 0 || padL != 0 || padB != 0 || padR != 0 || strideH != 1 || strideW != 1)
        {
            CV_Error(cv::Error::StsBadArg, "In global_pooling mode, pads must be = 0, and stride_h and stride_w must be = 1");
        }
    }
    else
    {
        util::getKernelSize(params, kernelH, kernelW);
    }
}

void getConvolutionKernelParams(const LayerParams &params, int &kernelH, int &kernelW, int &padT, int &padL, int &padB, int &padR,
                                int &strideH, int &strideW, int &dilationH, int &dilationW, cv::String &padMode)
{
    util::getKernelSize(params, kernelH, kernelW);
    util::getStrideAndPadding(params, padT, padL, padB, padR, strideH, strideW, padMode);
    util::getParameter(params, "dilation", "dilation", dilationH, dilationW, true, 1);

    CV_Assert(dilationH > 0 && dilationW > 0);
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
void getConvPoolOutParams(const Size& inp, const Size &kernel,
                          const Size &stride, const String &padMode,
                          const Size &dilation, Size& out)
{
    if (padMode == "VALID")
    {
        out.height = (inp.height - (dilation.height * (kernel.height - 1) + 1) + stride.height) / stride.height;
        out.width = (inp.width - (dilation.width * (kernel.width - 1) + 1) + stride.width) / stride.width;
    }
    else if (padMode == "SAME")
    {
        out.height = (inp.height - 1 + stride.height) / stride.height;
        out.width = (inp.width - 1 + stride.width) / stride.width;
    }
    else
    {
        CV_Error(Error::StsError, "Unsupported padding mode");
    }
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
        padT= padB = Ph / 2;
        padL = padR = Pw / 2;
    }
}

}
}
