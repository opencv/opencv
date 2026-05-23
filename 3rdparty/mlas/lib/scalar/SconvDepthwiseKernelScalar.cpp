/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    SconvDepthwiseKernelScalar.cpp

Abstract:

    This module implements the kernels for the single precision direct
    convolution kernels.

--*/

#include "mlasi.h"

static
void
MlasConv2dSingleChannel_CHW_Kernel3x3_Pad01_Dilation1(
    const MLAS_CONV_PARAMETERS* Parameters,
    const float* Input,
    const float* Filter,
    float* Output,
    const float* Zeros
    )
/*++

Routine Description:

    This routine is an inner kernel to compute convolution on one channel input with one filter channel.

Arguments:

    Parameters - conv parameters calculated based on conv parameters like padding, strides, dilations, etc.

    Input - input channel data start. Input is NCHW, so this pointer point to single H x W image data.

    Filter - Whole filters are of F x CpG x FH x FW, this filter point to single FH x FW filter data.

    Output - whole output are of N x F x OH x OW. This pointer point to single OH x OW output image data.

    Zeroes - Point to working buffer where all 0.0f are filled.

--*/
{
    const size_t W = Parameters->InputShape[1];
    const float beta = Parameters->Beta;

    if (W > 1) {

        const float w00 = Filter[0];
        const float w01 = Filter[1];
        const float w02 = Filter[2];
        const float w10 = Filter[3];
        const float w11 = Filter[4];
        const float w12 = Filter[5];
        const float w20 = Filter[6];
        const float w21 = Filter[7];
        const float w22 = Filter[8];

        const size_t H = Parameters->InputShape[0];
        const size_t pad_top = Parameters->Padding[0];
        const size_t pad_left = Parameters->Padding[1];
        const size_t stride_h = Parameters->StrideShape[0];
        const size_t stride_w = Parameters->StrideShape[1];

        // We treat pad_left, pad_top are hard require.
        // While pad_right and pad_bottom could be adjusted if they do not 100% match other parameters.
        const size_t pad_right = (((Parameters->OutputShape[1] - 1) * stride_w + 3) > (pad_left + W)) ? 1 : 0;

        const float* row0 = (pad_top > 0) ? Zeros : (Input - pad_left);
        // Need to handle effective pad_bottom is 2 when H == 1
        const float* row1 = (H + pad_top <= 1) ? Zeros : (Input + (1 - pad_top) * W) - pad_left;
        const float* row2 = (H + pad_top <= 2) ? Zeros : (row1 + W);

        for (size_t h = 0, out_row = Parameters->OutputShape[0]; out_row > 0; --out_row) {
            auto out_col = Parameters->OutputShape[1];

            if (pad_left == 1) {
                float dotsum = w01 * row0[1] + w02 * row0[2] + w11 * row1[1] + w12 * row1[2] +
                               w21 * row2[1] + w22 * row2[2] + (beta == 0.f ? 0.f : *Output * beta);
                *Output++ = dotsum;
                out_col--;
                row0 += stride_w;
                row1 += stride_w;
                row2 += stride_w;
            }

            for (; out_col > pad_right; out_col--) {
                float dotsum = w00 * row0[0] + w01 * row0[1] + w02 * row0[2] + w10 * row1[0] +
                               w11 * row1[1] + w12 * row1[2] + w20 * row2[0] + w21 * row2[1] +
                               w22 * row2[2] + (beta == 0.f ? 0.f : *Output * beta);
                *Output++ = dotsum;
                row0 += stride_w;
                row1 += stride_w;
                row2 += stride_w;
            }

            if (out_col == 1) { // pad_right == 1
                float dotsum = w00 * row0[0] + w01 * row0[1] + w10 * row1[0] + w11 * row1[1] +
                               w20 * row2[0] + w21 * row2[1] + (beta == 0.f ? 0.f : *Output * beta);
                *Output++ = dotsum;
            }

            h += stride_h;
            row0 = (Input + (h - pad_top) * W) - pad_left;
            row1 = row0 + W;
            row2 = (h + 2 >= H + pad_top) ? Zeros : (row1 + W);
        }

    } else { // W == 1

        const size_t H = Parameters->InputShape[0];
        const size_t pad_left = Parameters->Padding[1];
        const size_t pad_top = Parameters->Padding[0];
        const size_t stride_h = Parameters->StrideShape[0];
        size_t out_row = Parameters->OutputShape[0];

        // Make sure pad_bottom is consistent with other parameters.
        size_t pad_bottom = ((out_row - 1) * stride_h + 3) > (pad_top + H) ?
                                ((out_row - 1) * stride_h + 3) - (pad_top + H) : 0;

        const float w0 = Filter[pad_left ? 1 : 0];
        const float w1 = Filter[pad_left ? 4 : 3];
        const float w2 = Filter[pad_left ? 7 : 6];
        auto init_v = (beta == 0.f ? 0.f : *Output * beta);

        if (pad_top == 1) {
            *Output++ = w1 * Input[0] + w2 * ((H + pad_top <= 2) ? 0.0f : Input[1]) + init_v;
            out_row--;
        }

        for (const float* row = Input + pad_top * stride_h - pad_top; out_row > pad_bottom; --out_row) {
            // All pixels are in the input col
            auto init = (beta == 0.f ? 0.f : *Output * beta);
            *Output++ = w0 * row[0] + w1 * row[1] + w2 * row[2] + init;
            row += stride_h;
        }

        if (out_row > 0) {
            // last 1 or 2 rows are from the padding zero row.
            // out_row == 1 when arrive here
            if (pad_bottom == 1) {
                const float* row = Input + H - 2;
                *Output++ = w0 * row[0] + w1 * row[1] + init_v;
            } else { // pad_bottom == 2 and H == 1 and padding_top == 0
                *Output++ = w0 * Input[0] + init_v;
            }
        }
    }

}


void
MlasConvDepthwiseFloat_CHW(
    const MLAS_CONV_PARAMETERS* Parameters,
    const float* Input,
    const float* Filter,
    float* Output,
    const float* Zeros
    )
/*++

Routine Description:

    This routine is an inner kernel to compute depthwise convolution for one filter channel on one input channel.

Arguments:

    Parameters - conv parameters calculated based on conv parameters like padding, strides, dilations, etc.

    Input - input channel data start. Input is NCHW, so this pointer point to single H x W image data.

    Filter - Whole filters are of F x CpG x FH x FW, this filter point to single FH x FW filter data.

    Output - whole output are of N x F x OH x OW. This pointer point to single OH x OW output image data.

    Zeroes - Point to working buffer where all 0.0f are filled.

Note:
    No checking here as it is inner loop. Logic in generating Parameters controls the check.

    Currently only support 2d kernel 3x3.
    Will add general case and more special case if needed later.

--*/
{
    MlasConv2dSingleChannel_CHW_Kernel3x3_Pad01_Dilation1(Parameters, Input, Filter, Output, Zeros);
}
