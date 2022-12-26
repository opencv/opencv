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
enum { _FX_CONV_TYPE_GENERIC=0, _FX_CONV_TYPE_DEPTHWISE=1, _FX_CONV_TYPE_WINOGRAD3X3=2, _FX_CONV_TYPE_DEPTHWISE_REMAIN=3 };
enum { CONV_1D = 0, CONV_2D = 1, CONV_3D = 2 };
#endif

namespace cv {
namespace dnn {

struct FastConv
{
    int ngroups;
    int K, C, Hk, Wk, Dk;
    int stride_h, stride_w, stride_d;
    int dilation_h, dilation_w, dilation_d;
    int pad_top, pad_bottom, pad_left, pad_right, pad_front, pad_behind;

    std::vector<float> weightsBuf;     // For generic Conv 2D
    float* weightsBufPtr;
    std::vector<float> weightsWinoBuf; // For Winograd F(6x6, 3x3).
    float* weightsWinoBufPtr;
    std::vector<float> biasBuf;
    int conv_type;
    int conv_dim;  // Flag for conv1d, conv2d, or conv3d.
#if CV_SIMD128
    bool useSIMD128 = true;
#else
    bool useSIMD128 = false;
#endif

#if CV_NEON
    bool useNEON = checkHardwareSupport(CPU_NEON);
#else
    bool useNEON = false;
#endif

    bool useAVX   = checkHardwareSupport(CPU_AVX);
    bool useAVX2  = checkHardwareSupport(CPU_AVX2);
    bool useRVV   = checkHardwareSupport(CPU_RVV);
};

// return a FastConv instance.
Ptr<FastConv> initFastConv(
        InputArray weightsMat,
        float* srcBias,
        int ngroups,
        int K, int C,
        const std::vector<size_t>& kernel_size,
        const std::vector<size_t>& strides,
        const std::vector<size_t>& dilations,
        const std::vector<size_t>& pads_begin,
        const std::vector<size_t>& pads_end,
        int conv_dim,
        bool useWinograd);

// It contains different computing branches, like winograd, 1x1 conv.
void runFastConv(InputArray _input, OutputArray _output, const Ptr<FastConv>& conv, int ntasks,
                   const Ptr<ActivationLayer>& actLayer, const std::vector<float>& reluslope, bool fusedAdd);

void runDepthwise(InputArray _input, OutputArray _output, const Ptr<FastConv>& conv, ActivationLayer* activ,
                  const std::vector<float>& reluslope);

int runWinograd63(InputArray _input, InputArray _fusedAddMat, OutputArray _output, const Ptr<FastConv>& conv, int ntasks,
                  float minval, float maxval, ActivationLayer* activ, bool ifMinMaxAct);

namespace opt_AVX2
{
#if CV_TRY_AVX2
void convBlock_AVX2(int np, const float* a, const float* b, float* c, int ldc, bool init_c);

void convBlockMR1(int np, const float* a, const float* b, float *c, const float bias, bool init_c, const float minval,
                  const float maxval, bool ifMinMaxAct);

void _fx_winograd_accum_f32(const float* inwptr, const float* wptr, float* outbuf, int Cg, int iblock);
void _fx_winograd_BtXB_8x8_f32(const float* inptr, int inpstep, float* outptr, int Cg);
void _fx_winograd_AtXA_8x8_f32(const float* inptr, int inpstep, float* bpptr, int bpstep, float* outptr, int outstep,
                               float bias, float minval, float maxval, bool ifMinMaxAct);

#endif
} // namespace opt_AVX2

} // namespace dnn
} // namespace cv

#endif //OPENCV_FAST_CONVOLUTION_HPP
