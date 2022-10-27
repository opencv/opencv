// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_FAST_CONVOLUTION_HPP
#define OPENCV_FAST_CONVOLUTION_HPP

#include "opencv2/core/hal/intrin.hpp"

#ifndef CONV_PRAM
#define CONV_PRAM
#if CV_NEON && CV_NEON_AARCH64  // 32 registers.
#define CONV_MR 4
#define CONV_NR 28
#elif CV_NEON              // 16 registers.
#define CONV_MR 4
#define CONV_NR 12
#else // SIMD 128, AVX or AVX2
#define CONV_MR 4
#define CONV_NR 24
#endif

// Winograd Params
enum {
    _FX_WINO_STEP=6,
    _FX_WINO_KSIZE=3,
    _FX_WINO_SIZE=_FX_WINO_STEP+_FX_WINO_KSIZE-1,
    _FX_WINO_AREA=_FX_WINO_SIZE*_FX_WINO_SIZE,

    _FX_WINO_KBLOCK = 4,
#if (CV_NEON && CV_NEON_AARCH64) || CV_TRY_AVX2
    _FX_WINO_IBLOCK = 6,
#else
    _FX_WINO_IBLOCK = 3,
#endif

#if CV_TRY_AVX2
    _FX_WINO_ATOM_F32 = 8,
#else
    _FX_WINO_ATOM_F32 = 4,
#endif

    _FX_WINO_NATOMS_F32 = _FX_WINO_AREA / _FX_WINO_ATOM_F32, // for AVX2, it is 8, otherwise, it's 16.
};
enum { _FX_CONV_TYPE_GENERIC=0, _FX_CONV_TYPE_DEPTHWISE=1, _FX_CONV_TYPE_WINOGRAD3X3=2 };
#endif

namespace cv {
namespace dnn {

struct FastConv2d
{
    int ngroups;
    int K, C, Hk, Wk;
    int stride_y, stride_x;
    int dilation_y, dilation_x;
    int pad_top, pad_bottom, pad_left, pad_right;

    std::vector<float> weightsBuf;     // For generic Conv 2D
    float* weightsBufPtr;
    std::vector<float> weightsWinoBuf; // For Winograd F(6x6, 3x3).
    float* weightsWinoBufPtr;
    std::vector<float> biasBuf;
    int conv_type;
#if CV_SIMD128
    bool useSIMD128 = true;
#else
    bool useSIMD128 = false;
#endif

#if CV_TRY_AVX2
    bool useAVX2 = checkHardwareSupport(CPU_AVX2);
#else
    bool useAVX2 = false;
#endif

#if CV_NEON
    bool useNEON = checkHardwareSupport(CPU_NEON);
#else
    bool useNEON = false;
#endif
};

// return a FastConv2d instance.
Ptr<FastConv2d> initFastConv2d(
        int ngroups,
        int K, int C, int Hk, int Wk,
        int stride_x, int stride_y,
        int dilation_x, int dilation_y,
        const std::vector<size_t>& pads_begin,
        const std::vector<size_t>& pads_end,
        InputArray weightsMat,
        float* srcBias, bool useWinograd);

// It contains different computing branches, like winograd, 1x1 conv.
void runFastConv2d(InputArray _input, OutputArray _output, const Ptr<FastConv2d>& conv, int ntasks,
                   const Ptr<ActivationLayer>& actLayer, bool fusedAdd);

void runDepthwise(InputArray _input, OutputArray _output, const Ptr<FastConv2d>& conv, float minval, float maxval,
        ActivationLayer* activ, bool ifMinMaxAct);

int runWinograd63(InputArray _input, InputArray _fusedAddMat, OutputArray _output, const Ptr<FastConv2d>& conv, int ntasks,
                  float minval, float maxval, ActivationLayer* activ, bool ifMinMaxAct);

} // namespace dnn

namespace opt_AVX2
{
#if CV_TRY_AVX2
void convBlock_AVX2(int np, const float* a, const float* b, float* c, int ldc, bool init_c);

void depthWiseBlock_AVX2(const float *inptr, float *outptr, const float *weights, float biasval, int *ofstab, int *yxtab,
                float minval, float maxval, int Hi, int Wi, int H0, int W0, int ksize, int pad_top, int pad_left,
                int dilation_y, int stride_x, int stride_y, int inner_xleft, int inner_xright, int inner_ytop,
                int inner_ybottom, bool ifMinMaxAct, bool useSIMD, bool is3x3);

void _fx_winograd_accum_f32(const float* inwptr, const float* wptr, float* outbuf, int Cg, int iblock);
void _fx_winograd_BtXB_8x8_f32(const float* inptr, int inpstep, float* outptr, int Cg);
void _fx_winograd_AtXA_8x8_f32(const float* inptr, int inpstep, float* bpptr, int bpstep, float* outptr, int outstep,
                               float bias, float minval, float maxval, bool ifMinMaxAct);

#endif
} // namespace opt_AVX2

} // namespace cv

#endif //OPENCV_FAST_CONVOLUTION_HPP
