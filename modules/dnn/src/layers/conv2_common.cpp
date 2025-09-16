// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "../net_impl.hpp"
#include "layers_common.hpp"
#include "conv2_common.hpp"
#include <math.h>

namespace cv { namespace dnn {

AutoPadding getAutoPadding(const LayerParams& params)
{
    std::string auto_pad = params.get<std::string>("auto_pad", "NOTSET");
    if (auto_pad == "NOTSET")
        return AUTO_PAD_NONE;
    if (auto_pad == "SAME_UPPER")
        return AUTO_PAD_SAME_UPPER;
    if (auto_pad == "SAME_LOWER")
        return AUTO_PAD_SAME_LOWER;
    if (auto_pad != "VALID") {
        CV_Error_(Error::StsBadArg, ("invalid auto_pad value '%s'", auto_pad.c_str()));
    }
    return AUTO_PAD_VALID;
}

// computes shape of the output tensor of convolution
// (including depth-wise convolution), max pooling or average pooling operations
MatShape convInferShape(const MatShape& inpshape, const MatShape& wshape,
                        const std::vector<int>& kernel_shape, int ngroups,
                        const std::vector<int>& strides,
                        const std::vector<int>& dilations,
                        const std::vector<int>& pads,
                        AutoPadding auto_pad, bool ceil_mode)
{
    int blockLayout = inpshape.layout == DATA_LAYOUT_BLOCK;
    int ndims = inpshape.dims;
    size_t nspatialdims = (size_t)(ndims - 2 - blockLayout);
    MatShape outshape = inpshape;
    int kshape[MatShape::MAX_DIMS];

    if (!kernel_shape.empty()) {
        size_t kshape_size = kernel_shape.size();
        CV_Assert(kshape_size == nspatialdims || kshape_size == nspatialdims+2);
        for (size_t i = 0; i < nspatialdims; i++)
            kshape[i] = kernel_shape[kshape_size - nspatialdims + i];
    } else {
        CV_Assert(!wshape.empty() && wshape.dims == nspatialdims + 2);
        for (size_t i = 0; i < nspatialdims; i++)
            kshape[i] = wshape[wshape.dims - nspatialdims + i];
    }

    if (ngroups == 0 || wshape.empty()) {
        outshape[1] = inpshape[1];
    } else if (blockLayout) {
        int C0 = inpshape[ndims-1];
        outshape[1] = (wshape[0] + C0 - 1)/C0;
    } else {
        outshape[1] = wshape[0];
    }

    CV_Assert(strides.empty() || strides.size() == nspatialdims);
    CV_Assert(dilations.empty() || dilations.size() == nspatialdims);
    CV_Assert(auto_pad == AUTO_PAD_NONE || pads.empty());
    CV_Assert(pads.empty() || pads.size() == nspatialdims*2);

    for (size_t i = 0; i < nspatialdims; i++) {
        int inp_i = inpshape[i+2], k_i = kshape[i];
        int stride = strides.empty() ? 1 : strides[i];
        int dilation = dilations.empty() ? 1 : dilations[i];
        int out_i;
        if (auto_pad == AUTO_PAD_NONE || auto_pad == AUTO_PAD_VALID) {
            int pad = 0;
            if (!pads.empty()) {
                pad = pads[i] + pads[i + nspatialdims];
            }
            out_i = (inp_i + pad - 1 - dilation * (k_i - 1) + (ceil_mode ? stride - 1 : 0)) / stride + 1;
        } else {
            if (ceil_mode)
                out_i = (inp_i + stride - 1)/stride;
            else
                out_i = (inp_i - 1)/stride + 1;
        }
        outshape[i + 2] = out_i;
    }

    if (blockLayout) {
        outshape.C = ngroups == 0 || wshape.empty() ? inpshape.C : wshape[0];
    } else {
        outshape.C = 0;
    }

    return outshape;
}


void initPoolingState(const MatShape& inpshape,
                      const MatShape& outshape,
                      const std::vector<int>& kernel_shape,
                      const std::vector<int>& strides,
                      const std::vector<int>& dilations,
                      const std::vector<int>& pads,
                      AutoPadding auto_pad, bool ceil_mode,
                      int mindims, ConvState& cs)
{
    //size_t kshape_size =
    //CV_Assert(kernel_shape.size() <= (size_t)ConvState::MAX_CONV_DIMS);
    CV_Error(Error::StsNotImplemented, "");
}

}
}
