// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#if defined(__riscv) && defined(__riscv_dsp) && defined(__ANDES)
# include <nds_intrinsic.h>
# define CV_RVP052 1

namespace cv {
namespace dnn {
namespace opt_RVP052 {

void fastConv( const int8_t* weights, size_t wstep, const int* bias,
               const int8_t* rowbuf, int* output, const int* outShape,
               int blockSize, int vecsize, int vecsize_aligned, int outZp,
               const float* multiplier, bool initOutput, bool finalOutput );
void fastDepthwiseConv( const int8_t* wptr,
                        int kernel_h, int kernel_w,
                        int stride_h, int stride_w,
                        int dilation_h, int dilation_w,
                        int pad_t, int pad_l,
                        const int* biasptr, const float* multptr,
                        const int8_t* inptr_,
                        int height, int width,
                        int* outptr_,
                        int out_d, int outH, int outW,
                        int inpZp, int outZp );
void fastGEMM1T( const int8_t* vec, const int8_t* weights,
                 size_t wstep, const int* bias, const float* multiplier,
                 int* dst, int nvecs, int vecsize, int outZp );

}}}

#else
# define CV_RVP052 0
#endif
