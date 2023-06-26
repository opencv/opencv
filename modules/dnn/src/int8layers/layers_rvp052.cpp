// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "./layers_rvp052.hpp"

#if CV_RVP052

namespace cv {
namespace dnn {
namespace opt_RVP052 {

void fastConv(const int8_t *weights, size_t wstep, const int *bias,
              const int8_t *rowbuf, int *output, const int *outShape,
              int blockSize, int vecsize, int vecsize_aligned, int outZp,
              const float *multiplier, bool initOutput, bool finalOutput)
{
    int outCn = outShape[1];
    size_t outPlaneSize = outShape[2] * outShape[3];
    for (int i = 0; i < outCn; i += 2)
    {
        const int8_t *wptr0 = weights + i * wstep;
        const int8_t *wptr1 = wptr0 + wstep;
        int *outptr0 = output + i * outPlaneSize;
        int *outptr1 = outptr0 + outPlaneSize;
        int bias0 = bias[i], bias1 = bias[i + 1];
        float mult0 = multiplier[i], mult1 = multiplier[i + 1];

        if (i + 1 >= outCn)
        {
            wptr1 = wptr0;
            outptr1 = outptr0;
            bias1 = bias0;
            mult1 = mult0;
        }
        int j = 0;
        for (; j < blockSize; j++)
        {
            const int8_t *rptr = rowbuf + j * vecsize_aligned;
            int s00 = initOutput ? bias0 : outptr0[j];
            int s10 = initOutput ? bias1 : outptr1[j];

            int32x2_t vsx0 = {s00, s10};

            for (int k = 0; k < vecsize; k += 4)
            {
                int8x4_t vrptr[2] = {*(int8x4_t*)(rptr + k), *(int8x4_t*)(rptr + k)};
                int8x4_t vwptr[2] = {*(int8x4_t*)(wptr0 + k), *(int8x4_t*)(wptr1 + k)};
                vsx0 = __nds__v_smaqa(vsx0, *(int8x8_t*)vwptr, *(int8x8_t*)vrptr);
            }

            if (finalOutput)
            {
                vsx0[0] = outZp + (int)std::round(vsx0[0] * mult0);
                vsx0[1] = outZp + (int)std::round(vsx0[1] * mult1);
                vsx0 = __nds__v_sclip32(vsx0, 7);
            }

            outptr0[j] = vsx0[0];
            outptr1[j] = vsx0[1];
        }
    }
}

void fastDepthwiseConv(const int8_t *wptr,
                       int kernel_h, int kernel_w,
                       int stride_h, int stride_w,
                       int dilation_h, int dilation_w,
                       int pad_t, int pad_l,
                       const int *biasptr, const float *multptr,
                       const int8_t *inptr_,
                       int height, int width,
                       int *outptr_,
                       int out_d, int outH, int outW,
                       int inpZp, int outZp)
{
    const int8_t w00_ = wptr[0], w01_ = wptr[1], w02_ = wptr[2],
                 w10 = wptr[3], w11 = wptr[4], w12 = wptr[5],
                 w20_ = wptr[6], w21_ = wptr[7], w22_ = wptr[8];
    int outW1 = min(outW, (width - dilation_w * (kernel_w - 1) + pad_l) / stride_w);
    int bias = biasptr[out_d], biasCopy;
    float mult = multptr[out_d];

    for (int out_i = 0; out_i < outH; out_i++)
    {
        int in_i = out_i * stride_h - pad_t, out_j = 0;
        const int8_t *imgptr0 = inptr_ + in_i * width;
        const int8_t *imgptr1 = imgptr0 + dilation_h * width;
        const int8_t *imgptr2 = imgptr0 + (dilation_h * 2) * width;
        int8_t w00 = w00_, w01 = w01_, w02 = w02_;
        int8_t w20 = w20_, w21 = w21_, w22 = w22_;
        int out;
        biasCopy = bias;

        if (in_i < 0)
        {
            biasCopy += inpZp * (w00 + w01 + w02);
            w00 = w01 = w02 = 0;
            imgptr0 = imgptr1;
        }
        else if (in_i + dilation_h * (kernel_h - 1) >= height)
        {
            biasCopy += inpZp * (w20 + w21 + w22);
            w20 = w21 = w22 = 0;
            imgptr2 = imgptr1;
        }
        int *outptr = outptr_ + out_i * outW;
        if (pad_l > 0)
        {
            out = (int)imgptr0[0] * w01 + (int)imgptr0[dilation_w] * w02 +
                  (int)imgptr1[0] * w11 + (int)imgptr1[dilation_w] * w12 +
                  (int)imgptr2[0] * w21 + (int)imgptr2[dilation_w] * w22 +
                  biasCopy + inpZp * (w00 + w10 + w20);
            outptr[0] = __nds__sclip32(outZp + (int)std::round(out * mult), 7);
            out_j = 1;
        }

        int8x8_t vwx0 = (int8x8_t){w00, w10, w20, 0, w00, w10, w20, 0};
        int8x8_t vwx1 = (int8x8_t){w01, w11, w21, 0, w01, w11, w21, 0};
        int8x8_t vwx2 = (int8x8_t){w02, w12, w22, 0, w02, w12, w22, 0};
        int8x8_t vimgx0, vimgx1, vimgx2;
        int32x2_t vout = {0, 0};
        for (; out_j < outW1; out_j+=2)
        {
            int in_j = out_j * stride_w - pad_l;
            vimgx0 = (int8x8_t){imgptr0[in_j], imgptr1[in_j], imgptr2[in_j], 0,
                                imgptr0[in_j + stride_w], imgptr1[in_j + stride_w], imgptr2[in_j + stride_w], 0};
            vimgx1 = (int8x8_t){imgptr0[in_j + dilation_w], imgptr1[in_j + dilation_w], imgptr2[in_j + dilation_w], 0,
                                imgptr0[in_j + dilation_w + stride_w], imgptr1[in_j + dilation_w + stride_w], imgptr2[in_j + dilation_w + stride_w], 0};
            vimgx2 = (int8x8_t){imgptr0[in_j + dilation_w * 2], imgptr1[in_j + dilation_w * 2], imgptr2[in_j + dilation_w * 2], 0,
                                imgptr0[in_j + dilation_w * 2 + stride_w], imgptr1[in_j + dilation_w * 2 + stride_w], imgptr2[in_j + dilation_w * 2 + stride_w], 0};

            vout = (int32x2_t){biasCopy, biasCopy};
            vout = __nds__v_smaqa(vout, vwx0, vimgx0);
            vout = __nds__v_smaqa(vout, vwx1, vimgx1);
            vout = __nds__v_smaqa(vout, vwx2, vimgx2);

            outptr[out_j] = __nds__sclip32(outZp + (int)std::round(vout[0] * mult), 7);
            outptr[out_j + 1] = __nds__sclip32(outZp + (int)std::round(vout[1] * mult), 7);
        }

        while (out_j > outW1) out_j--;

        for (; out_j < outW; out_j++)
        {
            int in_j0 = out_j * stride_w - pad_l, in_j1 = in_j0 + dilation_w, in_j2 = in_j0 + dilation_w * 2;
            int s0 = 1, s1 = 1, s2 = 1;
            if (in_j0 >= width)
            {
                in_j0 = 0;
                s0 = 0;
                biasCopy += inpZp * (w00 + w10 + w20);
            }
            if (in_j1 >= width)
            {
                in_j1 = 0;
                s1 = 0;
                biasCopy += inpZp * (w01 + w11 + w21);
            }
            if (in_j2 >= width)
            {
                in_j2 = 0;
                s2 = 0;
                biasCopy += inpZp * (w02 + w12 + w22);
            }
            out = (int)imgptr0[in_j0] * w00 * s0 + (int)imgptr0[in_j1] * w01 * s1 + (int)imgptr0[in_j2] * w02 * s2 +
                  (int)imgptr1[in_j0] * w10 * s0 + (int)imgptr1[in_j1] * w11 * s1 + (int)imgptr1[in_j2] * w12 * s2 +
                  (int)imgptr2[in_j0] * w20 * s0 + (int)imgptr2[in_j1] * w21 * s1 + (int)imgptr2[in_j2] * w22 * s2 + biasCopy;
            outptr[out_j] = __nds__sclip32(outZp + (int)std::round(out * mult), 7);
        }
    }
}

// dst = vec * weights^t + bias
void fastGEMM1T( const int8_t* vec, const int8_t* weights,
                 size_t wstep, const int* bias, const float* multiplier,
                 int* dst, int nvecs, int vecsize, int outZp )
{
    int i = 0;

    for( ; i <= nvecs - 2; i += 2 )
    {
        const int8_t* wptr0 = weights + i * wstep;
        const int8_t* wptr1 = weights + (i + 1) * wstep;

        int32x2_t vs0 = *(int32x2_t*)(bias + i);

        for( int k = 0; k < vecsize; k += 4 )
        {
            int8x4_t vvec[2] = {*(int8x4_t*)(vec + k), *(int8x4_t*)(vec + k)};
            int8x4_t vwptr[2] = {*(int8x4_t*)(wptr0 + k), *(int8x4_t*)(wptr1 + k)};
            vs0 = __nds__v_smaqa(vs0, *(int8x8_t*)vwptr, *(int8x8_t*)vvec);
        }

        int32x2_t vdst = {(int)std::round(vs0[0] * multiplier[i]), (int)std::round(vs0[1] * multiplier[i + 1])};

        vdst = __nds__v_sclip32(vdst + outZp, 7);

        *(int32x2_t*)(dst + i) = vdst;
    }

    for( ; i < nvecs; i++ )
    {
        const int8_t* wptr = weights + i * wstep;
        int s0 = bias[i];

        for( int k = 0; k < vecsize; k += 4 )
        {
            int8x4_t vvec[2] = {*(int8x4_t*)(vec + k), 0};
            int8x4_t vwptr[2] = {*(int8x4_t*)(wptr + k), 0};
            s0 = __nds__smaqa(s0, *(unsigned long*)vwptr, *(unsigned long*)vvec);
        }

        dst[i] = __nds__sclip32(outZp + (int)std::round(s0 * multiplier[i]), 7);
    }
}

}}} // namespace

#endif
