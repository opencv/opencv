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
#include <opencv2/dnn/shape_utils.hpp>

namespace cv
{
namespace dnn
{

class MVNLayerImpl : public MVNLayer
{
public:
    MVNLayerImpl(const LayerParams& params)
    {
        setParamsFrom(params);
        normVariance = params.get<bool>("normalize_variance", true);
        acrossChannels = params.get<bool>("across_channels", false);
        eps = params.get<double>("eps", 1e-9);
    }

    void forward(std::vector<Mat *> &inputs, std::vector<Mat> &outputs, std::vector<Mat> &internals)
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        for (size_t inpIdx = 0; inpIdx < inputs.size(); inpIdx++)
        {
            Mat &inpBlob = *inputs[inpIdx];
            Mat &outBlob = outputs[inpIdx];

            int splitDim = (acrossChannels) ? 1 : 2;
            int i, newRows = 1;
            for( i = 0; i < splitDim; i++ )
                newRows *= inpBlob.size[i];
            Mat inpMat = inpBlob.reshape(1, newRows);
            Mat outMat = outBlob.reshape(1, newRows);

            Scalar mean, dev;
            for ( i = 0; i < newRows; i++)
            {
                Mat inpRow = inpMat.row(i);
                Mat outRow = outMat.row(i);

                cv::meanStdDev(inpRow, mean, (normVariance) ? dev : noArray());
                double alpha = (normVariance) ? 1/(eps + dev[0]) : 1;
                inpRow.convertTo(outRow, outRow.type(), alpha, -mean[0] * alpha);
            }
        }
    }

    virtual int64 getFLOPS(const std::vector<MatShape> &inputs,
                           const std::vector<MatShape> &outputs) const
    {
        (void)outputs; // suppress unused variable warning
        long flops = 0;
        for(int i = 0; i < inputs.size(); i++)
        {
            flops += 6*total(inputs[i]) + 3*total(inputs[i], 0, normVariance ? 2 : 1);
        }
        return flops;
    }
};

Ptr<MVNLayer> MVNLayer::create(const LayerParams& params)
{
    return Ptr<MVNLayer>(new MVNLayerImpl(params));
}

}
}
