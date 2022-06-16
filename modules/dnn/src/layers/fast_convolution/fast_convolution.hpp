// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_FAST_CONVOLUTION_HPP
#define OPENCV_FAST_CONVOLUTION_HPP

namespace cv { namespace dnn {

typedef struct FastConv2d
{
    int ngroups;
    int K, C, Hk, Wk;
    int stride_y, stride_x;
    int dilation_y, dilation_x;
    int pad_top, pad_bottom, pad_left, pad_right;

    AutoBuffer<float> weightsBuf, weightsWino63Buf;
    float* weightsPtr; // For generic Conv 2d
    float* weightsWino63Ptr; // For Winograd F(6x6, 3x3).

    AutoBuffer<float> biasBuf;
    float* biasPtr;
    bool ifWinograd63 = false;
    bool useAVX = checkHardwareSupport(CPU_AVX);
    bool useAVX2 = checkHardwareSupport(CPU_AVX2);
    bool useNEON = checkHardwareSupport(CPU_NEON);
} FastConv2d;

// return a FastConv2d instance.
Ptr<FastConv2d> initFastConv2d(
        int ngroups,
        int K, int C, int Hk, int Wk,
        int stride_x, int stride_y,
        int dilation_x, int dilation_y,
        const std::vector<size_t>& pads_begin,
        const std::vector<size_t>& pads_end,
        float* weightsPtr,
        float* biasPtr);

// It contains different computing branches, like winograd, 1x1 conv.
void runFastConv2d(InputArray _input, OutputArray _output,
                 const Ptr<FastConv2d>& conv, int ntasks, const Ptr<ActivationLayer>& actLayer);

}} // namespace cv::dnn

#endif //OPENCV_FAST_CONVOLUTION_HPP
