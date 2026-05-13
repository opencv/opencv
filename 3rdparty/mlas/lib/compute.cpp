/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    compute.cpp

Abstract:

    This module implements miscellaneous computation routines.

    Our usage requires building platform specific versions of the algorithm to
    target different instruction sets. The implementation below targets the
    base instruction set (typically SSE2) while assembly implementations target
    newer instruction sets (such as FMA3).

--*/

#include "mlasi.h"
#include "softmax.h"

//
// Bundles the constants for use by kernels written in assembly.
//

MLAS_INTERNAL_DATA const struct {
    float LowerRange;
    float UpperRange;
    float LowerRangeSumExp;
    float UpperRangeSumExp;
    float RoundingBias;
    float Log2Reciprocal;
    float Log2High;
    float Log2Low;
    float poly_0;
    float poly_1;
    float poly_2;
    float poly_3;
    float poly_4;
    float poly_56;
    int32_t MinimumExponent;
    int32_t MaximumExponent;
} MlasExpConstants = {
    -103.9720840454f,
    88.7762626647950f,
    -88.3762626647949f,
    88.3762626647949f,
    MLAS_ROUNDING_BIAS_MAGIC,
    1.44269504088896341f,
    -6.93145752e-1f,
    -1.42860677e-6f,
    0x1.694000p-10,
    0x1.125edcp-7,
    0x1.555b5ap-5,
    0x1.555450p-3,
    0x1.fffff6p-2,
    0x1.000000p+0,
    int32_t(0xC1000000),
    int32_t(0x3F800000),
};

MLAS_INTERNAL_DATA const float MlasMinimumF32Value = std::numeric_limits<float>::lowest();

//
// Define the parameters to execute segments of a softmax operation on worker
// threads.
//

template <typename T>
struct MLAS_SOFTMAX_WORK_BLOCK {
    ptrdiff_t ThreadCountN;
    bool LogSoftmax;
    bool SmoothSoftmax;
    float Sink;
    const T* Input;
    T* Output;
    size_t N;
    size_t D;
};

MLAS_FORCEINLINE
MLAS_FLOAT32X4
MlasComputeExpVector(
    MLAS_FLOAT32X4 Vector
)
/*++

Routine Description:

    This routine computes the exponential function for the supplied vector.

    This merges ideas from multiple vectorized expf() implementations:

        1.  The original polynomials of expf() are extracted from MlasComputeErf, which
            was based on an answer to the following Stack Overflow post:

            https://stackoverflow.com/questions/35148198/efficient-faithfully-rounded-implementation-of-error-function-erff

        2.  The author of the answer further refined the polynomials at:

            https://forums.developer.nvidia.com/t/a-more-accurate-performance-competitive-implementation-of-expf/47528/5

            Using these polynomials yields even closer results to the Microsoft
            UCRT version of std::expf() than the values from the above post.

        3.  XNNPACK has a further useful refinement to extend the effective
            range of results from [-88.376, 88.376] to [-103.972, 88.776] by
            splitting the step of exponent reconstruction into two pieces. This
            yields results similar to an AVX512 implementation using VSCALEFPS.

Arguments:

    Vector - Supplies the values to operate on.

Return Value:

    Returns the exponential function of the input.

--*/
{
    Vector = MlasClampFloat32x4(Vector, MlasExpConstants.LowerRange, MlasExpConstants.UpperRange);

    //
    // Range reduction of the input by computing "(2 ^ m) * exp(reduced)".
    //

    const auto RoundingBias = MlasBroadcastFloat32x4(MlasExpConstants.RoundingBias);

    auto biased = MlasMultiplyAddFloat32x4(Vector, MlasExpConstants.Log2Reciprocal, RoundingBias);
    auto m = MlasSubtractFloat32x4(biased, RoundingBias);

    Vector = MlasMultiplyAddFloat32x4(m, MlasExpConstants.Log2High, Vector);
    Vector = MlasMultiplyAddFloat32x4(m, MlasExpConstants.Log2Low, Vector);

    //
    // Compute the scaling factors used to reconstruct the "(2 ^ m)" value
    // from above. To cover the entire single precision floating point range,
    // two scaling factors are needed to handle exponents [-150, 128].
    //

    const auto MinimumExponent = MlasBroadcastInt32x4(MlasExpConstants.MinimumExponent);
    const auto MaximumExponent = MlasBroadcastInt32x4(MlasExpConstants.MaximumExponent);

    auto overflow = MlasShiftLeftInt32x4<23>(MlasReinterpretAsInt32x4(biased));
    auto normal = overflow;
#if defined(MLAS_SSE2_INTRINSICS)
    // N.B. PMINSD/PMAXSD were not added until SSE 4.1, but the lower 16 bits
    // are zero, so they can be ignored for this computation, so use PMINSW/PMAXSW
    // instead.
    normal = _mm_min_epi16(normal, MaximumExponent);
    normal = _mm_max_epi16(normal, MinimumExponent);
#elif defined(MLAS_LSX_INTRINSICS)
    normal = __lsx_vmin_h(normal, MaximumExponent);
    normal = __lsx_vmax_h(normal, MinimumExponent);
#else
    normal = MlasMinimumInt32x4(normal, MaximumExponent);
    normal = MlasMaximumInt32x4(normal, MinimumExponent);
#endif
    overflow = MlasSubtractInt32x4(overflow, normal);
    overflow = MlasAddInt32x4(overflow, MaximumExponent);
    normal = MlasAddInt32x4(normal, MaximumExponent);

    //
    // Compute the polynomial approximation of exp(reduced) and reconstruct
    // the final result using the above scaling factors. The final term of
    // the polynomial (poly_6=1.0f) is merged as the multiply/add of the
    // overflow exponent (reference XNNPACK).
    //

    auto p = MlasBroadcastFloat32x4(MlasExpConstants.poly_0);
    p = MlasMultiplyAddFloat32x4(p, Vector, MlasExpConstants.poly_1);
    p = MlasMultiplyAddFloat32x4(p, Vector, MlasExpConstants.poly_2);
    p = MlasMultiplyAddFloat32x4(p, Vector, MlasExpConstants.poly_3);
    p = MlasMultiplyAddFloat32x4(p, Vector, MlasExpConstants.poly_4);
    p = MlasMultiplyAddFloat32x4(p, Vector, MlasExpConstants.poly_56);

    Vector = MlasMultiplyFloat32x4(Vector, MlasReinterpretAsFloat32x4(overflow));
    p = MlasMultiplyAddFloat32x4(p, Vector, MlasReinterpretAsFloat32x4(overflow));
    p = MlasMultiplyFloat32x4(p, MlasReinterpretAsFloat32x4(normal));

    return p;
}

void
MLASCALL
MlasComputeExpF32Kernel(
    const float* Input,
    float* Output,
    size_t N
)
/*++

Routine Description:

    This routine implements the generic kernel for the exponential function.

Arguments:

    Input - Supplies the input buffer.

    Output - Supplies the output buffer.

    N - Supplies the number of elements to process.

Return Value:

    None.

--*/
{
    while (N > 0) {
        MLAS_FLOAT32X4 Vector;

        if (N >= 4) {
            Vector = MlasLoadFloat32x4(Input);
        } else {
#if defined(MLAS_SSE2_INTRINSICS)
            // N.B. SSE2 lacks a broadcast load instruction, so avoid a shuffle
            // and use zeroes for the upper elements.
            Vector = _mm_load_ss(Input);
#elif defined(MLAS_LSX_INTRINSICS)
            Vector = (MLAS_FLOAT32X4)__lsx_vldrepl_w(Input, 0);
#else
            Vector = MlasBroadcastFloat32x4(Input);
#endif
        }

        Vector = MlasComputeExpVector(Vector);

        if (N >= 4) {
            MlasStoreFloat32x4(Output, Vector);

            Input += 4;
            Output += 4;
            N -= 4;

        } else {
            MlasStoreLaneFloat32x4<0>(Output, Vector);

            Input += 1;
            Output += 1;
            N -= 1;
        }
    }
}

template <>
void
MLASCALL
MlasComputeExp<float>(
    const float* Input,
    float* Output,
    size_t N
)
/*++

Routine Description:

    This routine computes the exponential function.

    N.B. This implementation supports in place updates of the output buffer.

Arguments:

    Input - Supplies the input buffer.

    Output - Supplies the output buffer.

    N - Supplies the number of elements to process.

Return Value:

    None.

--*/
{
#if defined(MLAS_TARGET_AMD64)
    GetMlasPlatform().ComputeExpF32Kernel(Input, Output, N);
#else
    MlasComputeExpF32Kernel(Input, Output, N);
#endif
}

template <>
void MLASCALL
MlasComputeExp<MLAS_FP16>(
    const MLAS_FP16* Input,
    MLAS_FP16* Output,
    size_t N
) {
    const auto* dispatch = GetMlasPlatform().SoftmaxDispatch;
    if (dispatch == nullptr || dispatch->Exp_Fp16 == nullptr) {
        MLAS_THROW_EX(std::runtime_error, "Exp_Fp16 is not supported.");
    }
    dispatch->Exp_Fp16(Input, Output, N);
}

MLAS_FORCEINLINE
MLAS_FLOAT32X4
MlasComputeSumExpVector(
    MLAS_FLOAT32X4 Vector,
    MLAS_FLOAT32X4 NegativeMaximumVector
)
/*++

Routine Description:

    This routine computes the exponential function for the supplied vector.

    This function handles a narrower range of inputs compared to
    MlasComputeExpVector in order to improve efficiency.

Arguments:

    Vector - Supplies the values to operate on.

    NegativeMaximumVector - Supplies the broadcasted negative maximum
        value that is added to each element before computing the exponential
        function.

Return Value:

    Returns the exponential function of the input.

--*/
{
    //
    // Subtract the maximum value from every element.
    //
    // N.B. For each of use by the assembly kernels, this value has been negated
    // so add the value instead.
    //

    Vector = MlasAddFloat32x4(Vector, NegativeMaximumVector);

    //
    // Clamp to the lower range of this function.
    //
    // The value should already be negative or equal to zero as every value has
    // been reduced by the maximum value.
    //

#if defined(MLAS_SSE2_INTRINSICS)
    // N.B. MINPS and MAXPS propagates the value from the second vector if the
    // value is a NaN.
#endif
    Vector = MlasMaximumFloat32x4(MlasBroadcastFloat32x4(MlasExpConstants.LowerRangeSumExp), Vector);

    //
    // Range reduction of the input by computing "(2 ^ m) * exp(reduced)".
    //

    const auto RoundingBias = MlasBroadcastFloat32x4(MlasExpConstants.RoundingBias);

    auto biased = MlasMultiplyAddFloat32x4(Vector, MlasExpConstants.Log2Reciprocal, RoundingBias);
    auto m = MlasSubtractFloat32x4(biased, RoundingBias);

    Vector = MlasMultiplyAddFloat32x4(m, MlasExpConstants.Log2High, Vector);
    Vector = MlasMultiplyAddFloat32x4(m, MlasExpConstants.Log2Low, Vector);

    //
    // Compute the scaling factor used to reconstruct the "(2 ^ m)" value
    // from above. The effective range of this function is smaller than
    // MlasComputeExp to reduce the number of operations.
    //

    auto normal = MlasShiftLeftInt32x4<23>(MlasReinterpretAsInt32x4(biased));
    normal = MlasAddInt32x4(normal, MlasBroadcastInt32x4(MlasExpConstants.MaximumExponent));

    //
    // Compute the polynomial approximation of exp(reduced) and reconstruct
    // the final result using the above scale factor.
    //

    auto p = MlasBroadcastFloat32x4(MlasExpConstants.poly_0);
    p = MlasMultiplyAddFloat32x4(p, Vector, MlasExpConstants.poly_1);
    p = MlasMultiplyAddFloat32x4(p, Vector, MlasExpConstants.poly_2);
    p = MlasMultiplyAddFloat32x4(p, Vector, MlasExpConstants.poly_3);
    p = MlasMultiplyAddFloat32x4(p, Vector, MlasExpConstants.poly_4);
    p = MlasMultiplyAddFloat32x4(p, Vector, MlasExpConstants.poly_56);
    p = MlasMultiplyAddFloat32x4(p, Vector, MlasExpConstants.poly_56);

    p = MlasMultiplyFloat32x4(p, MlasReinterpretAsFloat32x4(normal));

    return p;
}

float
MLASCALL
MlasComputeSumExpF32Kernel(
    const float* Input,
    float* Output,
    size_t N,
    const float* NegativeMaximum
)
/*++

Routine Description:

    This routine implements the generic kernel for the sum of exponential
    functions.

Arguments:

    Input - Supplies the input buffer.

    Output - Optionally supplies the output buffer. When used for Softmax,
        the output buffer is used to store the intermediate exp() results. When
        used for LogSoftmax, the intermediate exp() results are not required.

    N - Supplies the number of elements to process.

    NegativeMaximum - Supplies the address of the negative maximum
        value that is added to each element before computing the exponential
        function.

Return Value:

    Returns the sum of the exponential functions.

--*/
{
    MLAS_FLOAT32X4 NegativeMaximumVector = MlasBroadcastFloat32x4(*NegativeMaximum);
    float Accumulator = 0.0f;

    if (N >= 4) {
        MLAS_FLOAT32X4 AccumulatorVector = MlasZeroFloat32x4();

#if !defined(MLAS_SSE2_INTRINSICS)

        //
        // Unroll the loop for architectures that can benefit from improved
        // instruction level parallelism.
        //
        // N.B. The extra code size is not worth the benefit for SSE2 as the
        // MLAS_TARGET_AMD64 build already has specialized AVX2/AVX512F kernels
        // that do this.
        //

        while (N >= 8) {
            MLAS_FLOAT32X4 Vector0 = MlasLoadFloat32x4(Input);
            MLAS_FLOAT32X4 Vector1 = MlasLoadFloat32x4(Input + 4);

            Vector0 = MlasComputeSumExpVector(Vector0, NegativeMaximumVector);
            Vector1 = MlasComputeSumExpVector(Vector1, NegativeMaximumVector);
            AccumulatorVector = MlasAddFloat32x4(AccumulatorVector, Vector0);
            AccumulatorVector = MlasAddFloat32x4(AccumulatorVector, Vector1);

            if (Output != nullptr) {
                MlasStoreFloat32x4(Output, Vector0);
                MlasStoreFloat32x4(Output + 4, Vector1);
                Output += 8;
            }

            Input += 8;
            N -= 8;
        }

#endif

        while (N >= 4) {
            MLAS_FLOAT32X4 Vector = MlasLoadFloat32x4(Input);

            Vector = MlasComputeSumExpVector(Vector, NegativeMaximumVector);
            AccumulatorVector = MlasAddFloat32x4(AccumulatorVector, Vector);

            if (Output != nullptr) {
                MlasStoreFloat32x4(Output, Vector);
                Output += 4;
            }

            Input += 4;
            N -= 4;
        }

        Accumulator = MlasReduceAddFloat32x4(AccumulatorVector);
    }

    while (N > 0) {
#if defined(MLAS_SSE2_INTRINSICS)
        // N.B. SSE2 lacks a broadcast load instruction, so avoid a shuffle and
        // use zeroes for the upper elements.
        MLAS_FLOAT32X4 Vector = _mm_load_ss(Input);
#elif defined(MLAS_LSX_INTRINSICS)
        MLAS_FLOAT32X4 Vector = (MLAS_FLOAT32X4)__lsx_vldrepl_w(Input, 0);
#else
        MLAS_FLOAT32X4 Vector = MlasBroadcastFloat32x4(Input);
#endif

        Vector = MlasComputeSumExpVector(Vector, NegativeMaximumVector);
        Accumulator += MlasExtractLaneFloat32x4<0>(Vector);

        if (Output != nullptr) {
            MlasStoreLaneFloat32x4<0>(Output, Vector);
            Output += 1;
        }

        Input += 1;
        N -= 1;
    }
    return Accumulator;
}

float
MLASCALL
MlasReduceMaximumF32Kernel(
    const float* Input,
    size_t N
)
/*++

Routine Description:

    This routine implements the generic kernel to find the maximum value of
    the supplied buffer.

Arguments:

    Input - Supplies the input buffer.

    N - Supplies the number of elements to process.

Return Value:

    Returns the maximum value of the supplied buffer.

--*/
{
    float Maximum = MlasMinimumF32Value;

    if (N >= 4) {
        MLAS_FLOAT32X4 MaximumVector0 = MlasBroadcastFloat32x4(Maximum);

        if (N >= 16) {
            MLAS_FLOAT32X4 MaximumVector1 = MaximumVector0;
            MLAS_FLOAT32X4 MaximumVector2 = MaximumVector0;
            MLAS_FLOAT32X4 MaximumVector3 = MaximumVector0;

            while (N >= 16) {
                MaximumVector0 = MlasMaximumFloat32x4(MaximumVector0, MlasLoadFloat32x4(Input));
                MaximumVector1 = MlasMaximumFloat32x4(MaximumVector1, MlasLoadFloat32x4(Input + 4));
                MaximumVector2 = MlasMaximumFloat32x4(MaximumVector2, MlasLoadFloat32x4(Input + 8));
                MaximumVector3 = MlasMaximumFloat32x4(MaximumVector3, MlasLoadFloat32x4(Input + 12));

                Input += 16;
                N -= 16;
            }

            MaximumVector0 = MlasMaximumFloat32x4(MaximumVector0, MaximumVector1);
            MaximumVector2 = MlasMaximumFloat32x4(MaximumVector2, MaximumVector3);
            MaximumVector0 = MlasMaximumFloat32x4(MaximumVector0, MaximumVector2);
        }

        while (N >= 4) {
            MaximumVector0 = MlasMaximumFloat32x4(MaximumVector0, MlasLoadFloat32x4(Input));

            Input += 4;
            N -= 4;
        }

        Maximum = MlasReduceMaximumFloat32x4(MaximumVector0);
    }

    while (N > 0) {
        Maximum = std::max(Maximum, *Input);

        Input += 1;
        N -= 1;
    }
    return Maximum;
}

void
MLASCALL
MlasReduceMinimumMaximumF32Kernel(
    const float* Input,
    float* Min,
    float* Max,
    size_t N
)
{
    float tmp_min = std::numeric_limits<float>::max();
    float tmp_max = std::numeric_limits<float>::lowest();

    if (N >= 4) {
        MLAS_FLOAT32X4 MaximumVector0 = MlasBroadcastFloat32x4(tmp_max);
        MLAS_FLOAT32X4 MinimumVector0 = MlasBroadcastFloat32x4(tmp_min);

        if (N >= 16) {
            MLAS_FLOAT32X4 MaximumVector1 = MaximumVector0;
            MLAS_FLOAT32X4 MaximumVector2 = MaximumVector0;
            MLAS_FLOAT32X4 MaximumVector3 = MaximumVector0;

            MLAS_FLOAT32X4 MinimumVector1 = MinimumVector0;
            MLAS_FLOAT32X4 MinimumVector2 = MinimumVector0;
            MLAS_FLOAT32X4 MinimumVector3 = MinimumVector0;

            while (N >= 16) {
                MLAS_FLOAT32X4 InputVector0 = MlasLoadFloat32x4(Input);
                MLAS_FLOAT32X4 InputVector1 = MlasLoadFloat32x4(Input + 4);
                MLAS_FLOAT32X4 InputVector2 = MlasLoadFloat32x4(Input + 8);
                MLAS_FLOAT32X4 InputVector3 = MlasLoadFloat32x4(Input + 12);

                MaximumVector0 = MlasMaximumFloat32x4(MaximumVector0, InputVector0);
                MaximumVector1 = MlasMaximumFloat32x4(MaximumVector1, InputVector1);
                MaximumVector2 = MlasMaximumFloat32x4(MaximumVector2, InputVector2);
                MaximumVector3 = MlasMaximumFloat32x4(MaximumVector3, InputVector3);

                MinimumVector0 = MlasMinimumFloat32x4(MinimumVector0, InputVector0);
                MinimumVector1 = MlasMinimumFloat32x4(MinimumVector1, InputVector1);
                MinimumVector2 = MlasMinimumFloat32x4(MinimumVector2, InputVector2);
                MinimumVector3 = MlasMinimumFloat32x4(MinimumVector3, InputVector3);

                Input += 16;
                N -= 16;
            }

            MaximumVector0 = MlasMaximumFloat32x4(MaximumVector0, MaximumVector1);
            MaximumVector2 = MlasMaximumFloat32x4(MaximumVector2, MaximumVector3);
            MaximumVector0 = MlasMaximumFloat32x4(MaximumVector0, MaximumVector2);

            MinimumVector0 = MlasMinimumFloat32x4(MinimumVector0, MinimumVector1);
            MinimumVector2 = MlasMinimumFloat32x4(MinimumVector2, MinimumVector3);
            MinimumVector0 = MlasMinimumFloat32x4(MinimumVector0, MinimumVector2);
        }

        while (N >= 4) {
            MLAS_FLOAT32X4 InputVector0 = MlasLoadFloat32x4(Input);
            MaximumVector0 = MlasMaximumFloat32x4(MaximumVector0, InputVector0);

            MinimumVector0 = MlasMinimumFloat32x4(MinimumVector0, InputVector0);

            Input += 4;
            N -= 4;
        }

        tmp_min = MlasReduceMinimumFloat32x4(MinimumVector0);
        tmp_max = MlasReduceMaximumFloat32x4(MaximumVector0);
    }

    while (N > 0) {
        tmp_max = std::max(tmp_max, *Input);
        tmp_min = std::min(tmp_min, *Input);

        Input += 1;
        N -= 1;
    }

    *Min = tmp_min;
    *Max = tmp_max;
}

void
MLASCALL
MlasComputeSoftmaxOutputF32Kernel(
    float* Output,
    size_t N,
    const float* Parameters
)
/*++

Routine Description:

    This routine implements the generic kernel to produce the final output for
    the softmax operation.

Arguments:

    Output - Supplies the output buffer.

    N - Supplies the number of elements to process.

    Parameters - Supplies an array containing the scale value.

Return Value:

    None.

--*/
{
    const float Scale = Parameters[0];

    const MLAS_FLOAT32X4 ScaleVector = MlasBroadcastFloat32x4(Scale);

    while (N >= 16) {
        MLAS_FLOAT32X4 Vector0 = MlasMultiplyFloat32x4(ScaleVector, MlasLoadFloat32x4(Output));
        MLAS_FLOAT32X4 Vector1 = MlasMultiplyFloat32x4(ScaleVector, MlasLoadFloat32x4(Output + 4));
        MLAS_FLOAT32X4 Vector2 = MlasMultiplyFloat32x4(ScaleVector, MlasLoadFloat32x4(Output + 8));
        MLAS_FLOAT32X4 Vector3 = MlasMultiplyFloat32x4(ScaleVector, MlasLoadFloat32x4(Output + 12));

        MlasStoreFloat32x4(Output, Vector0);
        MlasStoreFloat32x4(Output + 4, Vector1);
        MlasStoreFloat32x4(Output + 8, Vector2);
        MlasStoreFloat32x4(Output + 12, Vector3);

        Output += 16;
        N -= 16;
    }

    while (N >= 4) {
        MlasStoreFloat32x4(Output, MlasMultiplyFloat32x4(ScaleVector, MlasLoadFloat32x4(Output)));

        Output += 4;
        N -= 4;
    }

    while (N > 0) {
        *Output *= Scale;

        Output += 1;
        N -= 1;
    }
}

void
MLASCALL
MlasComputeLogSoftmaxOutputF32Kernel(
    const float* Input,
    float* Output,
    size_t N,
    const float* Parameters
)
/*++

Routine Description:

    This routine implements the generic kernel to produce the final output for
    the log softmax operation.

Arguments:

    Input - Supplies the input buffer.

    Output - Supplies the output buffer.

    N - Supplies the number of elements to process.

    Parameters - Supplies an array containing the negative maximum and
        logarithm values.

Return Value:

    None.

--*/
{
    const float NegativeMaximum = Parameters[0];
    const float Logarithm = Parameters[1];

    const MLAS_FLOAT32X4 NegativeMaximumVector = MlasBroadcastFloat32x4(NegativeMaximum);
    const MLAS_FLOAT32X4 LogarithmVector = MlasBroadcastFloat32x4(Logarithm);

    while (N >= 16) {
        MLAS_FLOAT32X4 Vector0 = MlasLoadFloat32x4(Input);
        MLAS_FLOAT32X4 Vector1 = MlasLoadFloat32x4(Input + 4);
        MLAS_FLOAT32X4 Vector2 = MlasLoadFloat32x4(Input + 8);
        MLAS_FLOAT32X4 Vector3 = MlasLoadFloat32x4(Input + 12);

        Vector0 = MlasAddFloat32x4(Vector0, NegativeMaximumVector);
        Vector1 = MlasAddFloat32x4(Vector1, NegativeMaximumVector);
        Vector2 = MlasAddFloat32x4(Vector2, NegativeMaximumVector);
        Vector3 = MlasAddFloat32x4(Vector3, NegativeMaximumVector);

        Vector0 = MlasSubtractFloat32x4(Vector0, LogarithmVector);
        Vector1 = MlasSubtractFloat32x4(Vector1, LogarithmVector);
        Vector2 = MlasSubtractFloat32x4(Vector2, LogarithmVector);
        Vector3 = MlasSubtractFloat32x4(Vector3, LogarithmVector);

        MlasStoreFloat32x4(Output, Vector0);
        MlasStoreFloat32x4(Output + 4, Vector1);
        MlasStoreFloat32x4(Output + 8, Vector2);
        MlasStoreFloat32x4(Output + 12, Vector3);

        Input += 16;
        Output += 16;
        N -= 16;
    }

    while (N >= 4) {
        MLAS_FLOAT32X4 Vector = MlasLoadFloat32x4(Input);
        Vector = MlasAddFloat32x4(Vector, NegativeMaximumVector);
        Vector = MlasSubtractFloat32x4(Vector, LogarithmVector);
        MlasStoreFloat32x4(Output, Vector);

        Input += 4;
        Output += 4;
        N -= 4;
    }

    while (N > 0) {
        *Output = *Input + NegativeMaximum - Logarithm;

        Input += 1;
        Output += 1;
        N -= 1;
    }
}

template <typename T>
void
MlasComputeSoftmaxThreaded(
    void* Context,
    ptrdiff_t Index
);

template <>
void
MlasComputeSoftmaxThreaded<float>(
    void* Context,
    ptrdiff_t Index
)
/*++

Routine Description:

    This routine is invoked from a worker thread to execute a segment of a
    softmax or log softmax operation.

Arguments:

    Context - Supplies the pointer to the context for the threaded operation.

    ThreadId - Supplies the current index of the threaded operation.

Return Value:

    None.

--*/
{
    const auto* WorkBlock = (MLAS_SOFTMAX_WORK_BLOCK<float>*)Context;
    
    //
    // Partition the operation along the N dimension.
    //

    size_t n;
    size_t CountN;

    MlasPartitionWork(Index, WorkBlock->ThreadCountN, WorkBlock->N, &n, &CountN);

    //
    // Compute the softmax or log softmax function.
    //

    const size_t D = WorkBlock->D;
    const bool LogSoftmax = WorkBlock->LogSoftmax;
    const bool SmoothSoftmax = WorkBlock->SmoothSoftmax;
    const float Sink = WorkBlock->Sink;

    const float* Input = WorkBlock->Input + n * D;
    float* Output = WorkBlock->Output + n * D;

#if defined(MLAS_SSE2_INTRINSICS)
    // TODO: Use std::hardware_constructive_interference_size
    constexpr size_t CacheLineSize = 64;
    constexpr size_t ElementsPerCacheLine = CacheLineSize / sizeof(float);
#endif

    while (CountN > 0) {
#if defined(MLAS_SSE2_INTRINSICS)
        //
        // Prefetch the next row of the input buffer.
        //

        for (size_t i = 0; i * ElementsPerCacheLine < D; i++) {
            _mm_prefetch((char*)(Input + D) + i * CacheLineSize, _MM_HINT_T0);
        }
#endif

        //
        // Find the maximum value for the row.
        //
        float Maximum;

#if defined(MLAS_TARGET_AMD64) || defined(MLAS_TARGET_LARCH64) || defined(MLAS_USE_SVE) || defined(MLAS_TARGET_RISCV64)
        Maximum = GetMlasPlatform().ReduceMaximumF32Kernel(Input, D);
#else 
        Maximum = MlasReduceMaximumF32Kernel(Input, D);
#endif
        if (SmoothSoftmax && Sink > Maximum) {
            Maximum = Sink;
        }

        float NegativeMaximum = -Maximum;

        //
        // Compute the exponential function for each element of the row (save to Temp if provided) and
        // compute the sum of these exponential functions.
        //
        float* Temp = LogSoftmax ? nullptr : Output;
        float Accumulation;

#if defined(MLAS_TARGET_AMD64) || defined(MLAS_USE_SVE) || defined(MLAS_TARGET_RISCV64)
        Accumulation = GetMlasPlatform().ComputeSumExpF32Kernel(Input, Temp, D, &NegativeMaximum);
#else
        Accumulation = MlasComputeSumExpF32Kernel(Input, Temp, D, &NegativeMaximum);
#endif

        if (SmoothSoftmax) {
            Accumulation += expf(Sink + NegativeMaximum);
        }

        if (LogSoftmax) {
            //
            // Compute the log softmax output.
            //
            float Parameters[] = {NegativeMaximum, std::log(Accumulation)};

#if defined(MLAS_TARGET_AMD64) || defined(MLAS_TARGET_LARCH64) || defined(MLAS_USE_SVE) || defined(MLAS_TARGET_RISCV64)
            GetMlasPlatform().ComputeLogSoftmaxOutputF32Kernel(Input, Output, D, Parameters);
#else 

            MlasComputeLogSoftmaxOutputF32Kernel(Input, Output, D, Parameters);
#endif
        } else {
            //
            // Normalize the softmax output.
            //
            float Parameters[] = {1.0f / Accumulation};

#if defined(MLAS_TARGET_AMD64) || defined(MLAS_TARGET_LARCH64) || defined(MLAS_USE_SVE) || defined(MLAS_TARGET_RISCV64)
            GetMlasPlatform().ComputeSoftmaxOutputF32Kernel(Output, D, Parameters);
#else
            MlasComputeSoftmaxOutputF32Kernel(Output, D, Parameters);
#endif
        }

        Input += D;
        Output += D;
        CountN--;
    }
}

template <>
void
MlasComputeSoftmaxThreaded<MLAS_FP16>(
    void* Context,
    ptrdiff_t Index
)
/*++

Routine Description:

    This routine is invoked from a worker thread to execute a segment of a
    softmax or log softmax operation.

Arguments:

    Context - Supplies the pointer to the context for the threaded operation.

    ThreadId - Supplies the current index of the threaded operation.

Return Value:

    None.

--*/
{
    const auto* WorkBlock = (MLAS_SOFTMAX_WORK_BLOCK<MLAS_FP16>*)Context;
    size_t n;
    size_t CountN;
    MlasPartitionWork(Index, WorkBlock->ThreadCountN, WorkBlock->N, &n, &CountN);

    const size_t D = WorkBlock->D;
    const bool LogSoftmax = WorkBlock->LogSoftmax;
    const bool SmoothSoftmax = WorkBlock->SmoothSoftmax;

    const MLAS_FP16* Input = WorkBlock->Input + n * D;
    MLAS_FP16* Output = WorkBlock->Output + n * D;

    const auto* dispatch = GetMlasPlatform().SoftmaxDispatch;
    if (dispatch == nullptr ||
        dispatch->ReduceMax_Fp16 == nullptr ||
        dispatch->SumExp_Fp16 == nullptr ||
        (LogSoftmax && dispatch->LogSoftmax_Fp16 == nullptr) ||
        (!LogSoftmax && dispatch->Softmax_Fp16 == nullptr)) {
        MLAS_THROW_EX(std::runtime_error, "Lacks kernels for fp16 softmax.");
    }

    while (CountN > 0) {
        MLAS_FP16 Maximum = dispatch->ReduceMax_Fp16(Input, D);
        MLAS_FP16 NegativeMaximum = Maximum.Negate();
        if (SmoothSoftmax && !NegativeMaximum.IsNegative()) {
            NegativeMaximum = MLAS_FP16::FromBits(0);
        }

        MLAS_FP16* Temp = LogSoftmax ? nullptr : Output;
        MLAS_FP16 Accumulation = dispatch->SumExp_Fp16(Input, Temp, D, NegativeMaximum);
        float accumulation_fp32 = Accumulation.ToFloat();

        if (SmoothSoftmax) {
            accumulation_fp32 += expf(NegativeMaximum.ToFloat());
        }

        if (LogSoftmax) {
            dispatch->LogSoftmax_Fp16(Input, Output, D, NegativeMaximum, MLAS_FP16(std::log(accumulation_fp32)));
        } else {
            dispatch->Softmax_Fp16(Output, Output, D, MLAS_FP16(accumulation_fp32));
        }

        Input += D;
        Output += D;
        CountN--;
    }
}

template <typename T>
void
MLASCALL
MlasComputeSoftmax(
    const T* Input,
    T* Output,
    size_t N,
    size_t D,
    bool LogSoftmax,
    bool SmoothSoftmax,
    float Sink,
    MLAS_THREADPOOL* ThreadPool
)
/*++

Routine Description:

    This routine computes the softmax or log softmax function.

    N.B. This implementation supports in place updates of the output buffer.

Arguments:

    Input - Supplies the input buffer.

    Output - Supplies the output buffer.

    N - Supplies the number of rows to process.

    D - Supplies the number of columns per row to process.

    LogSoftmax - Supplies true if this is a log softmax operation, else false
        if this is a softmax operation.

    SmoothSoftmax - Supplies true if a smooth factor is used in softmax operation.

    Sink - Supplies the smooth factor to use in the softmax operation.

    ThreadPool - Supplies the thread pool object to use, else nullptr if the
        base library threading support should be used.

Return Value:

    None.

--*/
{
    MLAS_SOFTMAX_WORK_BLOCK<T> WorkBlock;

    //
    // Capture the softmax parameters to the work block.
    //

    WorkBlock.LogSoftmax = LogSoftmax;
    WorkBlock.SmoothSoftmax = SmoothSoftmax;
    WorkBlock.Input = Input;
    WorkBlock.Output = Output;
    WorkBlock.N = N;
    WorkBlock.D = D;
    WorkBlock.Sink = Sink;

    //
    // Compute the number of target threads given the complexity of the softmax
    // operation. Limit the number of threads to the number of rows and try to
    // keep each thread processing a minimum number of elements before using
    // another thread.
    //

    ptrdiff_t ThreadCountN = MlasGetMaximumThreadCount(ThreadPool);

    if (size_t(ThreadCountN) > N) {
        ThreadCountN = ptrdiff_t(N);
    }

    constexpr size_t MinimumElementsPerThread = 16384;

    size_t BlockCount = ((N * D) / MinimumElementsPerThread) + 1;

    if (size_t(ThreadCountN) > BlockCount) {
        ThreadCountN = ptrdiff_t(BlockCount);
    }

    WorkBlock.ThreadCountN = ThreadCountN;

    MlasExecuteThreaded(MlasComputeSoftmaxThreaded<T>, &WorkBlock, ThreadCountN, ThreadPool);
}

template
void
MLASCALL
MlasComputeSoftmax<float>(
    const float* Input,
    float* Output,
    size_t N,
    size_t D,
    bool LogSoftmax,
    bool SmoothSoftmax,
    float Sink,
    MLAS_THREADPOOL* ThreadPool
);

template
void
MLASCALL
MlasComputeSoftmax<MLAS_FP16>(
    const MLAS_FP16* Input,
    MLAS_FP16* Output,
    size_t N,
    size_t D,
    bool LogSoftmax,
    bool SmoothSoftmax,
    float Sink,
    MLAS_THREADPOOL* ThreadPool
);

template <>
bool
MLASCALL
MlasGQASupported<MLAS_FP16>(
    CBLAS_TRANSPOSE TransA,
    CBLAS_TRANSPOSE TransB
) {
    if (!MlasHGemmSupported(TransA, TransB)) {
        return false;
    }

    const auto* softmax_dispatch = GetMlasPlatform().SoftmaxDispatch;
    if (softmax_dispatch == nullptr ||
        softmax_dispatch->Tanh_Fp16 == nullptr ||
        softmax_dispatch->Softcap_Fp16 == nullptr ||
        softmax_dispatch->SumExp_Fp16 == nullptr ||
        softmax_dispatch->Softmax_Fp16 == nullptr ||
        softmax_dispatch->ReduceMax_Fp16 == nullptr) {
        return false;
    }

    return true;
}

template <>
bool
MLASCALL
MlasGQASupported<float>(
    CBLAS_TRANSPOSE TransA,
    CBLAS_TRANSPOSE TransB
) {
    MLAS_UNREFERENCED_PARAMETER(TransA);
    MLAS_UNREFERENCED_PARAMETER(TransB);
    return true;
}
