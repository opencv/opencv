/*++

Copyright 2025 FUJITSU LIMITED

Module Name:

    mlasi_sve.h

Abstract:

    This module contains the procedure prototypes for the SVE intrinsics.

--*/

#pragma once

#include "../mlasi.h"
#include <arm_sve.h>  // SVE intrinsic header

#ifndef __clang__
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+sve")

// Use Clang-specific per-function attribute
#ifdef __clang__
#define MLAS_SVE_TARGET __attribute__((target("arch=armv8.2-a+sve")))
#else
#define MLAS_SVE_TARGET
#endif

typedef svfloat32_t MLAS_SVFLOAT32;
typedef svint32_t MLAS_SVINT32;
typedef svuint32_t MLAS_SVUINT32;
typedef svbool_t MLAS_SVBOOL;
typedef svfloat16_t MLAS_SVFLOAT16;
typedef svuint16_t MLAS_SVUINT16;

void
MLASCALL
MlasSveErfFP16Kernel(
    const MLAS_FP16* Input,
    MLAS_FP16* Output,
    size_t N
);

void 
MLASCALL 
MlasSveTanhFP16Kernel(
    const MLAS_FP16* Input,
    MLAS_FP16* Output,
    size_t N
);

void 
MLASCALL 
MlasSveGeluFP16Kernel(
    const MLAS_FP16* Input,
    MLAS_FP16* Output,
    MLAS_FP16* Temp,
    size_t N,
    MLAS_GELU_ALGORITHM Algo
);
// function declarations
MLAS_FORCEINLINE
MLAS_SVFLOAT32
MlasSveComputeExpVector(
    MLAS_SVBOOL Pred,
    MLAS_SVFLOAT32 Vector
);

void
MLASCALL
MlasSveComputeExpF32Kernel(
    const float* Input,
    float* Output,
    size_t N
);

MLAS_FORCEINLINE
MLAS_SVFLOAT32
MlasSveComputeSumExpVector(
    MLAS_SVBOOL Pred,
    MLAS_SVFLOAT32 Vector,
    MLAS_SVFLOAT32 NegativeMaximumVector
);

float
MLASCALL
MlasSveComputeSumExpF32Kernel(
    const float* Input,
    float* Output,
    size_t N,
    const float* NegativeMaximum
);

float MLASCALL
MlasSveReduceMaximumF32Kernel(
    const float* Input,
    size_t N
);

void
MLASCALL
MlasSveReduceMinimumMaximumF32Kernel(
    const float* Input,
    float* Min,
    float* Max,
    size_t N
);

void
MLASCALL
MlasSveComputeSoftmaxOutputF32Kernel(
    float* Output,
    size_t N,
    const float* Parameters
);

void
MLASCALL
MlasSveComputeLogSoftmaxOutputF32Kernel(
    const float* Input,
    float* Output,
    size_t N,
    const float* Parameters
);

void
MLASCALL
MlasSveErfKernel(
    const float* Input,
    float* Output,
    size_t N
);

void 
MLASCALL
MlasSveLogisticKernel(
    const float* Input,
    float* Output,
    size_t N
);

//MLAS API for SVE intrinsics

MLAS_SVE_TARGET
MLAS_FORCEINLINE
MLAS_SVINT32
MlasSveReinterpretAsInt32(MLAS_SVFLOAT32 Vector)
{
    return svreinterpret_s32_f32(Vector);
}

MLAS_SVE_TARGET
MLAS_FORCEINLINE
MLAS_SVUINT32
MlasSveReinterpretAsUInt32(MLAS_SVFLOAT32 Vector)
{
    return svreinterpret_u32_f32(Vector);
}

// Reinterprets an unsigned 32-bit vector as a 32-bit floating-point vector.
MLAS_SVE_TARGET
MLAS_FORCEINLINE
MLAS_SVFLOAT32
MlasSveReinterpretAsFLOAT32(MLAS_SVUINT32 Vector)
{
    return svreinterpret_f32_u32(Vector);
}

MLAS_SVE_TARGET
MLAS_FORCEINLINE
MLAS_SVINT32
MlasSveCastToInt32(MLAS_SVBOOL Pred, MLAS_SVFLOAT32 Vector)
{
    return svcvt_s32_f32_z(Pred, Vector);
}

MLAS_SVE_TARGET
MLAS_FORCEINLINE
MLAS_SVFLOAT32
MlasSveCastToFloat32(MLAS_SVBOOL Pred, MLAS_SVINT32 Vector)
{
    return svcvt_f32_s32_z(Pred, Vector);
}

MLAS_SVE_TARGET
MLAS_FORCEINLINE
MLAS_SVINT32
MlasSveBroadcastInt32(int32_t Value)
{
    return svdup_n_s32(Value);
}

MLAS_SVE_TARGET
MLAS_FORCEINLINE
MLAS_SVINT32
MlasSveLoadInt32(MLAS_SVBOOL Pred, const int32_t* Buffer)
{
    return svld1_s32(Pred, Buffer);
}

MLAS_SVE_TARGET
MLAS_FORCEINLINE
void
MlasSveStoreInt32(MLAS_SVBOOL Pred, int32_t* Buffer, MLAS_SVINT32 Vector)
{
    svst1_s32(Pred, Buffer, Vector);
}

MLAS_SVE_TARGET
MLAS_FORCEINLINE
MLAS_SVINT32
MlasSveAddInt32(MLAS_SVBOOL Pred, MLAS_SVINT32 Vector1, MLAS_SVINT32 Vector2)
{   
    return svadd_s32_m(Pred, Vector1, Vector2);  
}

MLAS_SVE_TARGET
MLAS_FORCEINLINE
MLAS_SVINT32
MlasSveSubtractInt32(MLAS_SVBOOL Pred, MLAS_SVINT32 Vector1, MLAS_SVINT32 Vector2)
{
    return svsub_s32_m(Pred, Vector1, Vector2);
}

MLAS_SVE_TARGET
MLAS_FORCEINLINE
MLAS_SVINT32
MlasSveAndInt32(MLAS_SVBOOL Pred, MLAS_SVINT32 Vector1, MLAS_SVINT32 Vector2)
{
    return svand_s32_m(Pred, Vector1, Vector2);
}

MLAS_SVE_TARGET
MLAS_FORCEINLINE
MLAS_SVUINT32
MlasSveAndUInt32(MLAS_SVBOOL Pred, MLAS_SVUINT32 Vector1, MLAS_SVUINT32 Vector2)
{
    return svand_u32_m(Pred, Vector1, Vector2);
}

MLAS_SVE_TARGET
MLAS_FORCEINLINE
MLAS_SVINT32
MlasSveOrInt32(MLAS_SVBOOL Pred, MLAS_SVINT32 Vector1, MLAS_SVINT32 Vector2)
{
    return svorr_s32_m(Pred, Vector1, Vector2);
}

MLAS_SVE_TARGET
MLAS_FORCEINLINE
MLAS_SVINT32
MlasSveAndNotInt32(MLAS_SVBOOL Pred, MLAS_SVINT32 VectorNot, MLAS_SVINT32 Vector)
{
    return svand_s32_m(Pred, svnot_s32_z(Pred, VectorNot), Vector);
}

MLAS_SVE_TARGET
MLAS_FORCEINLINE
MLAS_SVINT32
MlasSveXorInt32(MLAS_SVBOOL Pred, MLAS_SVINT32 Vector1, MLAS_SVINT32 Vector2)
{
    return sveor_s32_m(Pred, Vector1, Vector2);
}

MLAS_SVE_TARGET
MLAS_FORCEINLINE
MLAS_SVINT32
MlasSveBlendInt32(MLAS_SVBOOL Pred, MLAS_SVINT32 Vector1, MLAS_SVINT32 Vector2, MLAS_SVINT32 Selection)
{
    return MlasSveOrInt32(
        Pred, 
        MlasSveAndInt32(Pred, Vector2, Selection), 
        MlasSveAndNotInt32(Pred, Selection, Vector1)
    );
}

template<unsigned ShiftCount>
MLAS_SVE_TARGET
MLAS_FORCEINLINE
MLAS_SVUINT32
MlasSveShiftLeftUInt32(MLAS_SVBOOL Pred, MLAS_SVUINT32 Vector)
{
    return svlsl_n_u32_z(Pred, Vector, ShiftCount);
}

template<unsigned ShiftCount>
MLAS_SVE_TARGET
MLAS_FORCEINLINE
MLAS_SVINT32
MlasSveShiftLeftInt32(MLAS_SVBOOL Pred, MLAS_SVINT32 Vector)
{
    return svlsl_n_s32_z(Pred, Vector, ShiftCount);
}

MLAS_SVE_TARGET
MLAS_FORCEINLINE
MLAS_SVUINT32
MlasSveShiftRightInt32(MLAS_SVBOOL Pred, MLAS_SVUINT32 Vector, uint ShiftCount)
{
    return svlsr_n_u32_m(Pred, Vector, ShiftCount);
}

MLAS_SVE_TARGET
MLAS_FORCEINLINE
MLAS_SVINT32
MlasSveMaximumInt32(MLAS_SVBOOL Pred, MLAS_SVINT32 Vector1, MLAS_SVINT32 Vector2)
{
    return svmax_s32_m(Pred, Vector1, Vector2);
}

MLAS_SVE_TARGET
MLAS_FORCEINLINE
MLAS_SVINT32
MlasSveMinimumInt32(MLAS_SVBOOL Pred, MLAS_SVINT32 Vector1, MLAS_SVINT32 Vector2)
{
    return svmin_s32_m(Pred, Vector1, Vector2);
}

MLAS_SVE_TARGET
MLAS_FORCEINLINE
MLAS_SVFLOAT32
MlasSveReinterpretAsFloat32(MLAS_SVINT32 Vector)
{
    return svreinterpret_f32_s32(Vector);
}

MLAS_SVE_TARGET
MLAS_FORCEINLINE
MLAS_SVFLOAT32
MlasSveBroadcastFloat32(float Value)
{
    return svdup_n_f32(Value);
}

MLAS_SVE_TARGET
MLAS_FORCEINLINE
MLAS_SVUINT32
MlasSveBroadcastUINT32(uint Value)
{
    return svdup_n_u32(Value);
}

MLAS_SVE_TARGET
MLAS_FORCEINLINE
MLAS_SVFLOAT32
MlasSveBroadcastFloat32(const float* Value)
{
    return svld1_f32(svptrue_b32(), Value);
}

MLAS_SVE_TARGET
MLAS_FORCEINLINE
MLAS_SVFLOAT32
MlasSveZeroFloat32(void)
{
    return svdup_n_f32(0.0f);
}

MLAS_SVE_TARGET
MLAS_FORCEINLINE
MLAS_SVFLOAT32
MlasSveLoadFloat32(MLAS_SVBOOL Pred, const float* Buffer)
{
    return svld1_f32(Pred, Buffer);
}

MLAS_SVE_TARGET
MLAS_FORCEINLINE
void
MlasSveStoreFloat32(MLAS_SVBOOL Pred, float* Buffer, MLAS_SVFLOAT32 Vector)
{
    svst1_f32(Pred, Buffer, Vector);
}

template<unsigned Lane>
MLAS_SVE_TARGET
MLAS_FORCEINLINE
void
MlasSveStoreLaneFloat32(float* Buffer, MLAS_SVFLOAT32 Vector)
{
    svbool_t Pred = svwhilelt_b32(Lane, Lane + 1);
    svst1_f32(Pred, Buffer, Vector);
}

MLAS_SVE_TARGET
MLAS_FORCEINLINE
void
MlasSveStoreLowHalfFloat32(float* Buffer, MLAS_SVFLOAT32 Vector)
{
    svbool_t Pred = svwhilelt_b32(0, (int32_t)svcntw() / 2);
    svst1_f32(Pred, Buffer, Vector);
}

template<unsigned Lane>
MLAS_SVE_TARGET
MLAS_FORCEINLINE
float
MlasSveExtractLaneFloat32(MLAS_SVFLOAT32 Vector)
{
    float TmpBuffer[1];
    svbool_t Pred = svwhilelt_b32(Lane, Lane + 1);
    svst1_f32(Pred, TmpBuffer, Vector);
    return TmpBuffer[0];
}

MLAS_SVE_TARGET
MLAS_FORCEINLINE
MLAS_SVFLOAT32
MlasSveInterleaveLowFloat32(MLAS_SVFLOAT32 Vector1, MLAS_SVFLOAT32 Vector2)
{
    return svzip1_f32(Vector1, Vector2);
}

MLAS_SVE_TARGET
MLAS_FORCEINLINE
MLAS_SVFLOAT32
MlasSveInterleaveHighFloat32(MLAS_SVFLOAT32 Vector1, MLAS_SVFLOAT32 Vector2)
{
    return svzip2_f32(Vector1, Vector2);
}

MLAS_SVE_TARGET
MLAS_FORCEINLINE
MLAS_SVFLOAT32
MlasSveAddFloat32(MLAS_SVBOOL Pred, MLAS_SVFLOAT32 Vector1, MLAS_SVFLOAT32 Vector2)
{
    return svadd_f32_m(Pred, Vector1, Vector2);
}

MLAS_SVE_TARGET
MLAS_FORCEINLINE
MLAS_SVFLOAT32
MlasSveSubtractFloat32(MLAS_SVBOOL Pred, MLAS_SVFLOAT32 Vector1, MLAS_SVFLOAT32 Vector2)
{
    return svsub_f32_m(Pred, Vector1, Vector2);
}

MLAS_SVE_TARGET
MLAS_FORCEINLINE
MLAS_SVFLOAT32
MlasSveMultiplyFloat32(MLAS_SVBOOL Pred, MLAS_SVFLOAT32 Vector1, MLAS_SVFLOAT32 Vector2)
{
    return svmul_f32_m(Pred, Vector1, Vector2);  
}

MLAS_SVE_TARGET
MLAS_FORCEINLINE
MLAS_SVFLOAT32
MlasSveExpFloat32(MLAS_SVUINT32 Vector)
{
    return svexpa_f32(Vector);
}

MLAS_SVE_TARGET
MLAS_FORCEINLINE
MLAS_SVFLOAT32
MlasSveScaleFloat32(MLAS_SVBOOL Pred,  MLAS_SVFLOAT32 Vector1, MLAS_SVINT32 Vector2)
{
    return svscale_f32_m(Pred, Vector1, Vector2);
}

MLAS_SVE_TARGET
MLAS_FORCEINLINE
MLAS_SVFLOAT32
MlasSveRoundINTFloat32(MLAS_SVBOOL Pred, MLAS_SVFLOAT32 Vector)
{
   return svrintm_f32_z(Pred, Vector);
}

MLAS_SVE_TARGET
MLAS_FORCEINLINE
MLAS_SVFLOAT32
MlasSveMultiplyAddFloat32(MLAS_SVBOOL Pred, MLAS_SVFLOAT32 Vector1, MLAS_SVFLOAT32 Vector2, MLAS_SVFLOAT32 Vector3)
{
    return svmla_f32_m(Pred, Vector3, Vector1, Vector2);
}

MLAS_SVE_TARGET
MLAS_FORCEINLINE
MLAS_SVFLOAT32
MlasSveMultiplyAddFloat32(MLAS_SVBOOL Pred, MLAS_SVFLOAT32 Vector1, float Scalar2, MLAS_SVFLOAT32 Vector3)
{
    return MlasSveMultiplyAddFloat32(Pred, Vector1, MlasSveBroadcastFloat32(Scalar2), Vector3);
}

MLAS_SVE_TARGET
MLAS_FORCEINLINE
MLAS_SVFLOAT32
MlasSveMultiplyAddFloat32(MLAS_SVBOOL Pred, MLAS_SVFLOAT32 Vector1, MLAS_SVFLOAT32 Vector2, float Scalar3)
{
    return MlasSveMultiplyAddFloat32(Pred, Vector1, Vector2, MlasSveBroadcastFloat32(Scalar3));
}

MLAS_SVE_TARGET
MLAS_FORCEINLINE
MLAS_SVFLOAT32
MlasSveDivideFloat32(MLAS_SVBOOL Pred, MLAS_SVFLOAT32 Vector1, MLAS_SVFLOAT32 Vector2)
{
    return svdiv_f32_m(Pred, Vector1, Vector2);
}

MLAS_SVE_TARGET
MLAS_FORCEINLINE
MLAS_SVFLOAT32
MlasSveGreaterThanFloat32(MLAS_SVBOOL Pred, MLAS_SVFLOAT32 Vector1, MLAS_SVFLOAT32 Vector2)
{
    // Compare Vector1 and Vector2, return a predicate vector
    svbool_t cmp_mask = svcmpgt_f32(Pred, Vector1, Vector2);

    //Convert predicate to uint32_t mask
    svuint32_t mask_bits = svdup_u32_z(cmp_mask, 0xFFFFFFFF);

    //Reinterpret to float32
    return svreinterpret_f32_u32(mask_bits);
}

MLAS_SVE_TARGET
MLAS_FORCEINLINE
MLAS_SVFLOAT32
MlasSveAndFloat32(MLAS_SVBOOL Pred, MLAS_SVFLOAT32 Vector1, MLAS_SVFLOAT32 Vector2)
{
    return MlasSveReinterpretAsFloat32(
        MlasSveAndInt32(
            Pred, 
            MlasSveReinterpretAsInt32(Vector1),
            MlasSveReinterpretAsInt32(Vector2)
        )
    );
}

MLAS_SVE_TARGET
MLAS_FORCEINLINE
MLAS_SVFLOAT32
MlasSveOrFloat32(MLAS_SVBOOL Pred, MLAS_SVFLOAT32 Vector1, MLAS_SVFLOAT32 Vector2)
{
    return MlasSveReinterpretAsFloat32(
        MlasSveOrInt32(
            Pred,
            MlasSveReinterpretAsInt32(Vector1),
            MlasSveReinterpretAsInt32(Vector2)
        )
    );
}

MLAS_SVE_TARGET
MLAS_FORCEINLINE
MLAS_SVFLOAT32
MlasSveAndNotFloat32(MLAS_SVBOOL Pred, MLAS_SVFLOAT32 Vector1, MLAS_SVFLOAT32 Vector2)
{
    return MlasSveReinterpretAsFloat32(
        MlasSveAndNotInt32(
            Pred,
            MlasSveReinterpretAsInt32(Vector1),
            MlasSveReinterpretAsInt32(Vector2)
        )
    );
}

MLAS_SVE_TARGET
MLAS_FORCEINLINE
MLAS_SVFLOAT32
MlasSveXorFloat32(MLAS_SVBOOL Pred, MLAS_SVFLOAT32 Vector1, MLAS_SVFLOAT32 Vector2)
{
    return MlasSveReinterpretAsFloat32(
        MlasSveXorInt32(
            Pred,
            MlasSveReinterpretAsInt32(Vector1),
            MlasSveReinterpretAsInt32(Vector2)
        )
    );
}

MLAS_SVE_TARGET
MLAS_FORCEINLINE
MLAS_SVFLOAT32
MlasSveBlendFloat32(MLAS_SVBOOL Pred, MLAS_SVFLOAT32 Vector1, MLAS_SVFLOAT32 Vector2, MLAS_SVFLOAT32 Selection)
{
    return MlasSveOrFloat32(
        Pred, 
        MlasSveAndFloat32(Pred, Vector2, Selection),
        MlasSveAndFloat32(Pred, Vector1, Selection)
    );
}

MLAS_SVE_TARGET
MLAS_FORCEINLINE
MLAS_SVFLOAT32
MlasSveMaximumFloat32(MLAS_SVBOOL Pred, MLAS_SVFLOAT32 Vector1, MLAS_SVFLOAT32 Vector2)
{
    return svmax_f32_m(Pred, Vector1, Vector2);
}

MLAS_SVE_TARGET
MLAS_FORCEINLINE
MLAS_SVFLOAT32
MlasSveMinimumFloat32(MLAS_SVBOOL Pred, MLAS_SVFLOAT32 Vector1, MLAS_SVFLOAT32 Vector2)
{
    return svmin_f32_m(Pred, Vector1, Vector2);
}

MLAS_SVE_TARGET
MLAS_FORCEINLINE
MLAS_SVFLOAT32
MlasSveClampFloat32(MLAS_SVBOOL Pred, MLAS_SVFLOAT32 Value, float LowerRange, float UpperRange)
{
    Value = MlasSveMaximumFloat32(Pred, MlasSveBroadcastFloat32(LowerRange), Value);
    Value = MlasSveMinimumFloat32(Pred, MlasSveBroadcastFloat32(UpperRange), Value);
    return Value;
}

MLAS_SVE_TARGET
MLAS_FORCEINLINE
float
MlasSveReduceAddFloat32(MLAS_SVBOOL Pred, MLAS_SVFLOAT32 Vector)
{
    return svaddv_f32(Pred, Vector);
}

MLAS_SVE_TARGET
MLAS_FORCEINLINE
float
MlasSveReduceMaximumFloat32(MLAS_SVBOOL Pred, MLAS_SVFLOAT32 Vector)
{
    return svmaxv_f32(Pred, Vector);
}

MLAS_SVE_TARGET
MLAS_FORCEINLINE
float
MlasSveReduceMinimumFloat32(MLAS_SVBOOL Pred, MLAS_SVFLOAT32 Vector)
{
    return svminv_f32(Pred, Vector);
}

MLAS_SVE_TARGET
MLAS_FORCEINLINE
MLAS_SVFLOAT32
MlasSvePowerOf2Float32(MLAS_SVBOOL Pred, MLAS_SVFLOAT32 Vector)
{
    MLAS_SVINT32 emm0 = MlasSveAddInt32(
        Pred, 
        MlasSveCastToInt32(Pred, Vector), 
        MlasSveBroadcastInt32(127)
    );
    return MlasSveReinterpretAsFloat32(MlasSveShiftLeftInt32<23>(Pred, emm0));
}

MLAS_SVE_TARGET
MLAS_FORCEINLINE
MLAS_SVFLOAT32
MlasSveSelect(svbool_t Pred, MLAS_SVFLOAT32 TrueValue, MLAS_SVFLOAT32 FalseValue)
{
    return svsel_f32(Pred, TrueValue, FalseValue);
}

MLAS_SVE_TARGET
MLAS_FORCEINLINE
MLAS_SVBOOL
MlasSveCompareLessThan(svbool_t Pred, MLAS_SVFLOAT32 A, MLAS_SVFLOAT32 B)
{
    return svcmplt_f32(Pred, A, B);
}

MLAS_SVE_TARGET
MLAS_FORCEINLINE
MLAS_SVBOOL
MlasSveCompareGreaterThan(svbool_t Pred, MLAS_SVFLOAT32 A, MLAS_SVFLOAT32 B)
{
    return svcmpgt_f32(Pred, A, B);
}

// GCC: Pop options after SVE-specific functions
#ifndef __clang__
#pragma GCC pop_options
#endif

#endif

