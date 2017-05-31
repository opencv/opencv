/* 
// Copyright 2014 2015 Intel Corporation All Rights Reserved.
// 
// The source code, information and material ("Material") contained herein is
// owned by Intel Corporation or its suppliers or licensors, and title
// to such Material remains with Intel Corporation or its suppliers or
// licensors. The Material contains proprietary information of Intel
// or its suppliers and licensors. The Material is protected by worldwide
// copyright laws and treaty provisions. No part of the Material may be used,
// copied, reproduced, modified, published, uploaded, posted, transmitted,
// distributed or disclosed in any way without Intel's prior express written
// permission. No license under any patent, copyright or other intellectual
// property rights in the Material is granted to or conferred upon you,
// either expressly, by implication, inducement, estoppel or otherwise.
// Any license under such intellectual property rights must be express and
// approved by Intel in writing.
// 
// Unless otherwise agreed by Intel in writing,
// you may not remove or alter this notice or any other notice embedded in
// Materials by Intel or Intel's suppliers or licensors in any way.
// 
*/

#if !defined( __IPPICV_TYPES_H__ )
#define __IPPICV_TYPES_H__

#ifdef __cplusplus
extern "C" {
#endif

#if defined( __IPPTYPES_H__ )
#error "You try to include ippicv_types.h and ipptypes.h together. You can't use IPP and ICV libraries together, please use only one and please use header files only for one library"
#endif

/*****************************************************************************/
/*                   Below are ippCore domain specific definitions           */
/*****************************************************************************/
#define   ippCPUID_MMX        0x00000001   /* Intel Architecture MMX technology supported  */
#define   ippCPUID_SSE        0x00000002   /* Streaming SIMD Extensions                    */
#define   ippCPUID_SSE2       0x00000004   /* Streaming SIMD Extensions 2                  */
#define   ippCPUID_SSE3       0x00000008   /* Streaming SIMD Extensions 3                  */
#define   ippCPUID_SSSE3      0x00000010   /* Supplemental Streaming SIMD Extensions 3     */
#define   ippCPUID_MOVBE      0x00000020   /* The processor supports MOVBE instruction     */
#define   ippCPUID_SSE41      0x00000040   /* Streaming SIMD Extensions 4.1                */
#define   ippCPUID_SSE42      0x00000080   /* Streaming SIMD Extensions 4.2                */
#define   ippCPUID_AVX        0x00000100   /* Advanced Vector Extensions instruction set   */
#define   ippAVX_ENABLEDBYOS  0x00000200   /* The operating system supports AVX            */
#define   ippCPUID_AES        0x00000400   /* AES instruction                              */
#define   ippCPUID_CLMUL      0x00000800   /* PCLMULQDQ instruction                        */
#define   ippCPUID_ABR        0x00001000   /* Reserved                                     */
#define   ippCPUID_RDRAND     0x00002000   /* Read Random Number instructions              */
#define   ippCPUID_F16C       0x00004000   /* Float16 instructions                         */
#define   ippCPUID_AVX2       0x00008000   /* Advanced Vector Extensions 2 instruction set */
#define   ippCPUID_ADCOX      0x00010000   /* ADCX and ADOX instructions                   */
#define   ippCPUID_RDSEED     0x00020000   /* The RDSEED instruction                       */
#define   ippCPUID_PREFETCHW  0x00040000   /* The PREFETCHW instruction                    */
#define   ippCPUID_SHA        0x00080000   /* Intel (R) SHA Extensions                     */
#define   ippCPUID_AVX512F    0x00100000   /* AVX-512 Foundation instructions              */
#define   ippCPUID_AVX512CD   0x00200000   /* AVX-512 Conflict Detection instructions      */
#define   ippCPUID_AVX512ER   0x00400000   /* AVX-512 Exponential & Reciprocal instructions*/
#define   ippCPUID_AVX512PF   0x00800000   /* AVX-512 Prefetch instructions                */
#define   ippCPUID_AVX512BW   0x01000000   /* AVX-512 Byte & Word instructions             */
#define   ippCPUID_AVX512DQ   0x02000000   /* AVX-512 DWord & QWord instructions           */
#define   ippCPUID_AVX512VL   0x04000000   /* AVX-512 Vector Length extensions             */
#define   ippCPUID_KNC        0x80000000   /* Intel(R) Xeon Phi(TM) Coprocessor            */
#if defined( _WIN32 ) || defined ( _WIN64 )
  #define   ippCPUID_NOCHECK    0x8000000000000000L   /* Force ippSetCpuFeatures to set CPU features without check */
#else
  #define   ippCPUID_NOCHECK    0x8000000000000000LL
#endif

#if defined( _WIN32 ) || defined ( _WIN64 )
  #define   ippCPUID_GETINFO_A  0x616f666e69746567L   /* Force ipp_GetCpuFeatures to work as cpuid instruction */
#else
  #define   ippCPUID_GETINFO_A  0x616f666e69746567LL
#endif

#define IPP_COUNT_OF( obj )  (sizeof(obj)/sizeof(obj[0]))

/*****************************************************************************/
/*                   Below are ippSP domain specific definitions             */
/*****************************************************************************/

typedef enum {
    ippRndZero,
    ippRndNear,
    ippRndFinancial,
    ippRndHintAccurate=0x10
} IppRoundMode;


typedef enum {
    ippAlgHintNone,
    ippAlgHintFast,
    ippAlgHintAccurate
} IppHintAlgorithm;

typedef enum {
    ippCmpLess,
    ippCmpLessEq,
    ippCmpEq,
    ippCmpGreaterEq,
    ippCmpGreater
} IppCmpOp;

typedef enum {
    ippAlgAuto    = 0x00000000,
    ippAlgDirect  = 0x00000001,
    ippAlgFFT     = 0x00000002,
    ippAlgMask    = 0x000000FF
} IppAlgType;

typedef enum {
    ippsNormNone  = 0x00000000, /* default */
    ippsNormA     = 0x00000100, /* biased normalization */
    ippsNormB     = 0x00000200, /* unbiased normalization */
    ippsNormMask  = 0x0000FF00
} IppsNormOp;

typedef enum {
    ippNormInf  =   0x00000001,
    ippNormL1   =   0x00000002,
    ippNormL2   =   0x00000004
} IppNormType;

enum {
    IPP_FFT_DIV_FWD_BY_N = 1,
    IPP_FFT_DIV_INV_BY_N = 2,
    IPP_FFT_DIV_BY_SQRTN = 4,
    IPP_FFT_NODIV_BY_ANY = 8
};

enum {
    IPP_DIV_FWD_BY_N = 1,
    IPP_DIV_INV_BY_N = 2,
    IPP_DIV_BY_SQRTN = 4,
    IPP_NODIV_BY_ANY = 8
};

typedef struct {
    Ipp32f rho;
    Ipp32f theta;
} IppPointPolar;

typedef enum {ippWinBartlett,ippWinBlackman,ippWinHamming,ippWinHann,ippWinRect} IppWinType;

typedef enum { ippButterworth, ippChebyshev1 } IppsIIRFilterType;

typedef enum  { ippZCR=0,   ippZCXor,   ippZCC } IppsZCType;

#if !defined( _OWN_BLDPCS )

typedef struct {
    int left;
    int right;
} IppsROI;

typedef struct RandUniState_8u IppsRandUniState_8u;
typedef struct RandUniState_16s IppsRandUniState_16s;
typedef struct RandUniState_32f IppsRandUniState_32f;
typedef struct RandUniState_64f IppsRandUniState_64f;

typedef struct RandGaussState_8u IppsRandGaussState_8u;
typedef struct RandGaussState_16s IppsRandGaussState_16s;
typedef struct RandGaussState_32f IppsRandGaussState_32f;
typedef struct RandGaussState_64f IppsRandGaussState_64f;

typedef struct FFTSpec_C_32fc   IppsFFTSpec_C_32fc;
typedef struct FFTSpec_C_32f    IppsFFTSpec_C_32f;
typedef struct FFTSpec_R_32f    IppsFFTSpec_R_32f;

typedef struct FFTSpec_C_64fc   IppsFFTSpec_C_64fc;
typedef struct FFTSpec_C_64f    IppsFFTSpec_C_64f;
typedef struct FFTSpec_R_64f    IppsFFTSpec_R_64f;

typedef struct DFTSpec_C_32fc       IppsDFTSpec_C_32fc;
typedef struct DFTSpec_C_32f        IppsDFTSpec_C_32f;
typedef struct DFTSpec_R_32f        IppsDFTSpec_R_32f;

typedef struct DFTSpec_C_64fc       IppsDFTSpec_C_64fc;
typedef struct DFTSpec_C_64f        IppsDFTSpec_C_64f;
typedef struct DFTSpec_R_64f        IppsDFTSpec_R_64f;

typedef struct DCTFwdSpec_32f IppsDCTFwdSpec_32f;
typedef struct DCTInvSpec_32f IppsDCTInvSpec_32f;

typedef struct DCTFwdSpec_64f IppsDCTFwdSpec_64f;
typedef struct DCTInvSpec_64f IppsDCTInvSpec_64f;

typedef struct sWTFwdState_32f    IppsWTFwdState_32f;
typedef struct sWTFwdState_8u32f  IppsWTFwdState_8u32f;
typedef struct sWTFwdState_16s32f IppsWTFwdState_16s32f;
typedef struct sWTFwdState_16u32f IppsWTFwdState_16u32f;
typedef struct sWTInvState_32f    IppsWTInvState_32f;
typedef struct sWTInvState_32f8u  IppsWTInvState_32f8u;
typedef struct sWTInvState_32f16s IppsWTInvState_32f16s;
typedef struct sWTInvState_32f16u IppsWTInvState_32f16u;

typedef struct IIRState_32f      IppsIIRState_32f;
typedef struct IIRState_32fc     IppsIIRState_32fc;
typedef struct IIRState32f_16s   IppsIIRState32f_16s;
typedef struct IIRState32fc_16sc IppsIIRState32fc_16sc;
typedef struct IIRState_64f      IppsIIRState_64f;
typedef struct IIRState_64fc     IppsIIRState_64fc;
typedef struct IIRState64f_32f   IppsIIRState64f_32f;
typedef struct IIRState64fc_32fc IppsIIRState64fc_32fc;
typedef struct IIRState64f_32s   IppsIIRState64f_32s;
typedef struct IIRState64fc_32sc IppsIIRState64fc_32sc;
typedef struct IIRState64f_16s   IppsIIRState64f_16s;
typedef struct IIRState64fc_16sc IppsIIRState64fc_16sc;

typedef struct FIRSpec_32f        IppsFIRSpec_32f;
typedef struct FIRSpec_64f        IppsFIRSpec_64f;
typedef struct FIRSpec_32fc       IppsFIRSpec_32fc;
typedef struct FIRSpec_64fc       IppsFIRSpec_64fc;

typedef struct FIRLMSState_32f    IppsFIRLMSState_32f;
typedef struct FIRLMSState32f_16s IppsFIRLMSState32f_16s;

typedef struct HilbertSpec IppsHilbertSpec;

typedef struct FIRSparseState_32f IppsFIRSparseState_32f;
typedef struct IIRSparseState_32f IppsIIRSparseState_32f;

typedef struct ResamplingPolyphase_16s      IppsResamplingPolyphase_16s;
typedef struct ResamplingPolyphaseFixed_16s IppsResamplingPolyphaseFixed_16s;
typedef struct ResamplingPolyphase_32f      IppsResamplingPolyphase_32f;
typedef struct ResamplingPolyphaseFixed_32f IppsResamplingPolyphaseFixed_32f;

#endif /* _OWN_BLDPCS */

/*****************************************************************************/
/*                   Below are ippIP domain specific definitions             */
/*****************************************************************************/
#define IPP_TEMPORAL_COPY      0x0
#define IPP_NONTEMPORAL_STORE  0x01
#define IPP_NONTEMPORAL_LOAD   0x02

typedef int IppEnum;

#define IPP_DEG_TO_RAD( deg ) ( (deg)/180.0 * IPP_PI )

typedef enum {
    ippiNormNone        = 0x00000000, /* default */
    ippiNorm            = 0x00000100, /* normalized form */
    ippiNormCoefficient = 0x00000200, /* correlation coefficient in the range [-1.0 ... 1.0] */
    ippiNormMask        = 0x0000FF00
} IppiNormOp;

typedef enum {
   ippiROIFull   = 0x00000000,
   ippiROIValid  = 0x00010000,
   ippiROISame   = 0x00020000,
   ippiROIMask   = 0x00FF0000
} IppiROIShape;

typedef enum {
   ippC0    =  0,
   ippC1    =  1,
   ippC2    =  2,
   ippC3    =  3,
   ippC4    =  4,
   ippP2    =  5,
   ippP3    =  6,
   ippP4    =  7,
   ippAC1   =  8,
   ippAC4   =  9,
   ippA0C4  = 10,
   ippAP4   = 11
} IppChannels;

typedef enum _IppiBorderType {
    ippBorderConst     =  0,
    ippBorderRepl      =  1,
    ippBorderWrap      =  2,
    ippBorderMirror    =  3, /* left border: 012... -> 21012... */
    ippBorderMirrorR   =  4, /* left border: 012... -> 210012... */
    ippBorderInMem     =  6,
    ippBorderTransp    =  7,
    ippBorderInMemTop     =  0x0010,
    ippBorderInMemBottom  =  0x0020,
    ippBorderInMemLeft    =  0x0040,
    ippBorderInMemRight   =  0x0080
} IppiBorderType;

typedef enum {
    ippAxsHorizontal,
    ippAxsVertical,
    ippAxsBoth,
    ippAxs45,
    ippAxs135
} IppiAxis;

typedef struct {
    int x;
    int y;
    int width;
    int height;
} IppiRect;

typedef struct {
    int x;
    int y;
} IppiPoint;

typedef struct {
    int width;
    int height;
} IppiSize;

typedef struct {
    Ipp32f x;
    Ipp32f y;
} IppiPoint_32f;

typedef enum  _IppiMaskSize {
    ippMskSize1x3 = 13,
    ippMskSize1x5 = 15,
    ippMskSize3x1 = 31,
    ippMskSize3x3 = 33,
    ippMskSize5x1 = 51,
    ippMskSize5x5 = 55
} IppiMaskSize;

enum {
    IPPI_INTER_NN     = 1,
    IPPI_INTER_LINEAR = 2,
    IPPI_INTER_CUBIC  = 4,
    IPPI_INTER_CUBIC2P_BSPLINE,     /* two-parameter cubic filter (B=1, C=0) */
    IPPI_INTER_CUBIC2P_CATMULLROM,  /* two-parameter cubic filter (B=0, C=1/2) */
    IPPI_INTER_CUBIC2P_B05C03,      /* two-parameter cubic filter (B=1/2, C=3/10) */
    IPPI_INTER_SUPER  = 8,
    IPPI_INTER_LANCZOS = 16,
    IPPI_ANTIALIASING  = (1 << 29),
    IPPI_SUBPIXEL_EDGE = (1 << 30),
    IPPI_SMOOTH_EDGE   = (1 << 31)
};

typedef enum {
    ippNearest = IPPI_INTER_NN,
    ippLinear = IPPI_INTER_LINEAR,
    ippCubic = IPPI_INTER_CUBIC2P_CATMULLROM,
    ippLanczos = IPPI_INTER_LANCZOS,
    ippHahn = 0,
    ippSuper = IPPI_INTER_SUPER
} IppiInterpolationType;

typedef enum {
    ippPolyphase_1_2,
    ippPolyphase_3_5,
    ippPolyphase_2_3,
    ippPolyphase_7_10,
    ippPolyphase_3_4
} IppiFraction;

enum {
    IPP_FASTN_ORIENTATION = 0x0001,
    IPP_FASTN_NMS         = 0x0002,
    IPP_FASTN_CIRCLE      = 0X0004,
    IPP_FASTN_SCORE_MODE0 = 0X0020
};

#if !defined( _OWN_BLDPCS )

typedef enum {
    ippAlphaOver,
    ippAlphaIn,
    ippAlphaOut,
    ippAlphaATop,
    ippAlphaXor,
    ippAlphaPlus,
    ippAlphaOverPremul,
    ippAlphaInPremul,
    ippAlphaOutPremul,
    ippAlphaATopPremul,
    ippAlphaXorPremul,
    ippAlphaPlusPremul
} IppiAlphaType;

typedef struct DeconvFFTState_32f_C1R IppiDeconvFFTState_32f_C1R;
typedef struct DeconvFFTState_32f_C3R IppiDeconvFFTState_32f_C3R;
typedef struct DeconvLR_32f_C1R IppiDeconvLR_32f_C1R;
typedef struct DeconvLR_32f_C3R IppiDeconvLR_32f_C3R;

typedef enum {
    ippiFilterBilateralGauss = 100,
    ippiFilterBilateralGaussFast = 101
} IppiFilterBilateralType;

typedef struct FilterBilateralSpec IppiFilterBilateralSpec;

typedef enum {
    ippDistNormL1   =   0x00000002
} IppiDistanceMethodType;

typedef enum {
    ippResizeFilterHann,
    ippResizeFilterLanczos
} IppiResizeFilterType;

typedef struct ResizeFilterState IppiResizeFilterState;

typedef struct {
    Ipp32u borderLeft;
    Ipp32u borderTop;
    Ipp32u borderRight;
    Ipp32u borderBottom;
} IppiBorderSize;

typedef enum {
    ippWarpForward,
    ippWarpBackward,
} IppiWarpDirection;

typedef enum {
    ippWarpAffine,
    ippWarpPerspective,
    ippWarpBilinear,
} IppiWarpTransformType;


typedef struct ResizeSpec_32f   IppiResizeSpec_32f;
typedef struct ResizeYUV422Spec IppiResizeYUV422Spec;
typedef struct ResizeYUV420Spec IppiResizeYUV420Spec;

typedef struct ResizeSpec_64f   IppiResizeSpec_64f;

typedef struct IppiWarpSpec     IppiWarpSpec;

typedef struct FilterBorderSpec IppiFilterBorderSpec;

typedef struct ThresholdAdaptiveSpec IppiThresholdAdaptiveSpec;

typedef struct HistogramSpec IppiHistogramSpec;

typedef struct {
   int   cvCompatible;  /* openCV compatible output format */
   int   cellSize;      /* squre cell size (pixels) */
   int   blockSize;     /* square block size (pixels) */
   int   blockStride;   /* block displacement (the same for x- and y- directions) */
   int   nbins;         /* required number of bins */
   Ipp32f   sigma;      /* gaussian factor of HOG block weights */
   Ipp32f   l2thresh;   /* normalization factor */
   IppiSize winSize;    /* detection window size (pixels) */
} IppiHOGConfig;

typedef struct FFT2DSpec_C_32fc IppiFFTSpec_C_32fc;
typedef struct FFT2DSpec_R_32f IppiFFTSpec_R_32f;

typedef struct DFT2DSpec_C_32fc IppiDFTSpec_C_32fc;
typedef struct DFT2DSpec_R_32f IppiDFTSpec_R_32f;

typedef struct DCT2DFwdSpec_32f IppiDCTFwdSpec_32f;
typedef struct DCT2DInvSpec_32f IppiDCTInvSpec_32f;

typedef struct iWTFwdSpec_32f_C1R IppiWTFwdSpec_32f_C1R;
typedef struct iWTInvSpec_32f_C1R IppiWTInvSpec_32f_C1R;
typedef struct iWTFwdSpec_32f_C3R IppiWTFwdSpec_32f_C3R;
typedef struct iWTInvSpec_32f_C3R IppiWTInvSpec_32f_C3R;

typedef struct MomentState64f IppiMomentState_64f;
typedef Ipp64f IppiHuMoment_64f[7];

typedef struct LUT_Spec IppiLUT_Spec;

#define IPP_HOG_MAX_CELL   (16)  /* max size of CELL */
#define IPP_HOG_MAX_BLOCK  (64)  /* max size of BLOCK */
#define IPP_HOG_MAX_BINS   (16)  /* max number of BINS */

typedef struct _ipHOG   IppiHOGSpec;

#endif /* _OWN_BLDPCS */


         /**** Below are 3D Image (Volume) Processing specific definitions ****/

typedef struct {
    int width;
    int height;
    int depth;
} IpprVolume;

typedef struct {
    int x;
    int y;
    int z;
    int width;
    int height;
    int depth;
} IpprCuboid;

typedef struct {
    int x;
    int y;
    int z;
} IpprPoint;

/*****************************************************************************/
/*                   Below are ippCV domain specific definitions             */
/*****************************************************************************/

typedef enum _IppiDifferentialKernel
{
    ippFilterSobelVert,
    ippFilterSobelHoriz,
    ippFilterSobel,
    ippFilterScharrVert,
    ippFilterScharrHoriz,
    ippFilterScharr,
    ippFilterCentralDiffVert,
    ippFilterCentralDiffHoriz,
    ippFilterCentralDiff,
}IppiDifferentialKernel;

#if !defined( _OWN_BLDPCS )

typedef enum _IppiKernelType {
    ippKernelSobel     =  0,
    ippKernelScharr    =  1,
    ippKernelSobelNeg  =  2
} IppiKernelType;

typedef enum _IppiNorm {
    ippiNormInf = 0,
    ippiNormL1 = 1,
    ippiNormL2 = 2,
    ippiNormFM = 3
} IppiNorm;

typedef struct ipcvMorphState IppiMorphState;
typedef struct ipcvMorphAdvState IppiMorphAdvState;
typedef struct ipcvMorphGrayState_8u IppiMorphGrayState_8u;
typedef struct ipcvMorphGrayState_32f IppiMorphGrayState_32f;

typedef struct ipcvConvState IppiConvState;

typedef struct _IppiConnectedComp {
    Ipp64f   area;    /*  area of the segmented component  */
    Ipp64f   value[3];/*  gray scale value of the segmented component  */
    IppiRect rect;    /*  bounding rectangle of the segmented component  */
} IppiConnectedComp;

typedef struct PyramidState IppiPyramidState;

typedef IppiPyramidState IppiPyramidDownState_8u_C1R;
typedef IppiPyramidState IppiPyramidDownState_16u_C1R;
typedef IppiPyramidState IppiPyramidDownState_32f_C1R;
typedef IppiPyramidState IppiPyramidDownState_8u_C3R;
typedef IppiPyramidState IppiPyramidDownState_16u_C3R;
typedef IppiPyramidState IppiPyramidDownState_32f_C3R;
typedef IppiPyramidState IppiPyramidUpState_8u_C1R;
typedef IppiPyramidState IppiPyramidUpState_16u_C1R;
typedef IppiPyramidState IppiPyramidUpState_32f_C1R;
typedef IppiPyramidState IppiPyramidUpState_8u_C3R;
typedef IppiPyramidState IppiPyramidUpState_16u_C3R;
typedef IppiPyramidState IppiPyramidUpState_32f_C3R;


typedef struct _IppiPyramid {
    Ipp8u         **pImage;
    IppiSize      *pRoi;
    Ipp64f        *pRate;
    int           *pStep;
    Ipp8u         *pState;
    int            level;
} IppiPyramid;

typedef struct OptFlowPyrLK IppiOptFlowPyrLK;

typedef IppiOptFlowPyrLK IppiOptFlowPyrLK_8u_C1R;
typedef IppiOptFlowPyrLK IppiOptFlowPyrLK_16u_C1R;
typedef IppiOptFlowPyrLK IppiOptFlowPyrLK_32f_C1R;

typedef struct ipcvHaarClassifier_32f IppiHaarClassifier_32f;
typedef struct ipcvHaarClassifier_32s IppiHaarClassifier_32s;

typedef struct ipcvFGHistogramState_8u_C1R IppFGHistogramState_8u_C1R;
typedef struct ipcvFGHistogramState_8u_C3R IppFGHistogramState_8u_C3R;

typedef struct ipcvFGGaussianState_8u_C1R IppFGGaussianState_8u_C1R;
typedef struct ipcvFGGaussianState_8u_C3R IppFGGaussianState_8u_C3R;

typedef enum _IppiInpaintFlag {
    IPP_INPAINT_TELEA    =  0,
    IPP_INPAINT_NS       =  1
} IppiInpaintFlag;

typedef struct ippcvFilterGaussianSpec IppFilterGaussianSpec;

typedef struct ipcvInpaintState_8u IppiInpaintState_8u_C1R;
typedef struct ipcvInpaintState_8u IppiInpaintState_8u_C3R;

typedef struct HoughProbSpec IppiHoughProbSpec;

typedef struct FastNSpec IppiFastNSpec;

typedef struct _IppiCornerFastN {
    int     x;
    int     y;
    int     cornerType;
    int     orientation;
    float   angle;
    float   score;
} IppiCornerFastN;

typedef struct FGMMState_8u_C3R IppFGMMState_8u_C3R;

typedef struct
{
  unsigned int numFrames;
  unsigned int maxNGauss;
  Ipp32f       varInit;
  Ipp32f       varMin;
  Ipp32f       varMax;
  Ipp32f       varWBRatio;
  Ipp32f       bckgThr;
  Ipp32f       varNGRatio;
  Ipp32f       reduction;
  Ipp8u        shadowValue;
  char         shadowFlag;
  Ipp32f       shadowRatio;
} IppFGMModel;

#endif /* _OWN_BLDPCS */

#define IPP_SEGMENT_QUEUE     0x01
#define IPP_SEGMENT_DISTANCE  0x02
#define IPP_SEGMENT_BORDER_4  0x40
#define IPP_SEGMENT_BORDER_8  0x80

#define IPP_TRUNC(a,b) ((a)&~((b)-1))
#define IPP_APPEND(a,b) (((a)+(b)-1)&~((b)-1))

/*****************************************************************************/
/*                   Below are ippCC domain specific definitions             */
/*****************************************************************************/
enum {
     IPP_UPPER        = 1,
     IPP_LEFT         = 2,
     IPP_CENTER       = 4,
     IPP_RIGHT        = 8,
     IPP_LOWER        = 16,
     IPP_UPPER_LEFT   = 32,
     IPP_UPPER_RIGHT  = 64,
     IPP_LOWER_LEFT   = 128,
     IPP_LOWER_RIGHT  = 256
};

#if !defined( _OWN_BLDPCS )

typedef enum {
    ippDitherNone,
    ippDitherFS,
    ippDitherJJN,
    ippDitherStucki,
    ippDitherBayer
} IppiDitherType;

#endif /* _OWN_BLDPCS */

/*****************************************************************************/
/*                   Below are ippCH domain specific definitions             */
/*****************************************************************************/

#if !defined( _OWN_BLDPCS )

typedef struct {
 void *pFind;
 int lenFind;
} IppRegExpFind;

typedef struct RegExpState IppRegExpState;

typedef enum {
    ippFmtASCII = 0,
    ippFmtUTF8
} IppRegExpFormat;

typedef struct RegExpReplaceState IppRegExpReplaceState;

#endif /* _OWN_BLDPCS */

/*****************************************************************************/
/*                   Below are ippDC domain specific definitions             */
/*****************************************************************************/

#if !defined ( _OWN_BLDPCS )

typedef struct MTFState_8u IppMTFState_8u;

typedef enum {
    ippBWTItohTanakaLimSort,
    ippBWTItohTanakaUnlimSort,
    ippBWTSuffixSort,
    ippBWTAutoSort
} IppBWTSortAlgorithmHint;

typedef struct LZSSState_8u IppLZSSState_8u;

typedef struct LZ77State_8u IppLZ77State_8u;
typedef enum{
   IppLZ77FastCompr,
   IppLZ77AverageCompr,
   IppLZ77BestCompr
} IppLZ77ComprLevel;
typedef enum{
   IppLZ77NoChcksm,
   IppLZ77Adler32,
   IppLZ77CRC32
} IppLZ77Chcksm;
typedef enum {
   IppLZ77NoFlush,
   IppLZ77SyncFlush,
   IppLZ77FullFlush,
   IppLZ77FinishFlush
} IppLZ77Flush;
typedef struct IppLZ77Pairs_16u {
   Ipp16u length;
   Ipp16u offset;
} IppLZ77Pair;
typedef enum {
   IppLZ77StatusInit,
   IppLZ77StatusLZ77Process,
   IppLZ77StatusHuffProcess,
   IppLZ77StatusFinal
} IppLZ77DeflateStatus;
typedef enum {
  IppLZ77UseFixed,
  IppLZ77UseDynamic,
  IppLZ77UseStored
} IppLZ77HuffMode;
typedef enum {
  IppLZ77InflateStatusInit,
  IppLZ77InflateStatusHuffProcess,
  IppLZ77InflateStatusLZ77Process,
  IppLZ77InflateStatusFinal
} IppLZ77InflateStatus;

typedef struct IppInflateState {
  const Ipp8u* pWindow;          /* pointer to the sliding window
                                    (the dictionary for the LZ77 algorithm) */
  unsigned int winSize;          /* size of the sliding window */
  unsigned int tableType;        /* type of Huffman code tables
                                    (for example, 0 - tables for Fixed Huffman codes) */
  unsigned int tableBufferSize;  /* (ENOUGH = 2048) * (sizeof(code) = 4) -
                                    sizeof(IppInflateState) */
} IppInflateState;

typedef enum { /* this type is used as a translator of the inflate_mode type from zlib */
  ippTYPE,
  ippLEN,
  ippLENEXT
} IppInflateMode;

typedef struct {
  Ipp16u freq;
  Ipp16u code;
} IppDeflateFreqTable;

typedef struct {
  Ipp16u code;
  Ipp16u len;
} IppDeflateHuffCode;

typedef struct RLEState_BZ2 IppRLEState_BZ2;

typedef struct EncodeHuffState_BZ2 IppEncodeHuffState_BZ2;

typedef struct DecodeHuffState_BZ2 IppDecodeHuffState_BZ2;

typedef enum {
    IppLZO1XST,      /* Single-threaded, generic LZO-compatible*/
    IppLZO1XMT      /* Multi-threaded */
} IppLZOMethod ;

typedef struct LZOState_8u IppLZOState_8u;

#endif /* _OWN_BLDPCS */

/* /////////////////////////////////////////////////////////////////////////////
//        The following enumerator defines a status of IPP operations
//                     negative value means error
*/
typedef enum {
    /* errors */
    ippStsNotSupportedModeErr    = -9999,/* The requested mode is currently not supported.  */
    ippStsCpuNotSupportedErr     = -9998,/* The target CPU is not supported. */
    ippStsInplaceModeNotSupportedErr = -9997,/* The inplace operation is currently not supported. */

    ippStsWarpTransformTypeErr   = -233, /* The warp transform type is illegal */
    ippStsExceededSizeErr        = -232, /* Requested size exceeded the maximum supported ROI size */
    ippStsWarpDirectionErr       = -231, /* The warp transform direction is illegal */

    ippStsFilterTypeErr          = -230, /* The filter type is incorrect or not supported */

    ippStsNormErr                = -229, /* The norm is incorrect or not supported */

    ippStsAlgTypeErr             = -228, /* Algorithm type is not supported.        */
    ippStsMisalignedOffsetErr    = -227, /* The offset is not aligned with an element. */

    ippStsQuadraticNonResidueErr = -226, /* SQRT operation on quadratic non-residue value. */

    ippStsBorderErr              = -225, /* Illegal value for border type.*/

    ippStsDitherTypeErr          = -224, /* Dithering type is not supported. */
    ippStsH264BufferFullErr      = -223, /* Buffer for the output bitstream is full. */
    ippStsWrongAffinitySettingErr= -222, /* An affinity setting does not correspond to the affinity setting that was set by f.ippSetAffinity(). */
    ippStsLoadDynErr             = -221, /* Error when loading the dynamic library. */

    ippStsPointAtInfinity        = -220, /* Point at infinity is detected.  */

    ippStsUnknownStatusCodeErr   = -216, /* Unknown status code. */

    ippStsOFBSizeErr             = -215, /* Incorrect value for crypto OFB block size. */
    ippStsLzoBrokenStreamErr     = -214, /* LZO safe decompression function cannot decode LZO stream. */

    ippStsRoundModeNotSupportedErr  = -213, /* Rounding mode is not supported. */
    ippStsDecimateFractionErr    = -212, /* Fraction in Decimate is not supported. */
    ippStsWeightErr              = -211, /* Incorrect value for weight. */

    ippStsQualityIndexErr        = -210, /* Cannot calculate the quality index for an image filled with a constant. */
    ippStsIIRPassbandRippleErr   = -209, /* Ripple in passband for Chebyshev1 design is less than zero, equal to zero, or greater than 29. */
    ippStsFilterFrequencyErr     = -208, /* Cutoff frequency of filter is less than zero, equal to zero, or greater than 0.5. */
    ippStsFIRGenOrderErr         = -207, /* Order of the FIR filter for design is less than 1.                    */
    ippStsIIRGenOrderErr         = -206, /* Order of the IIR filter for design is less than 1, or greater than 12. */

    ippStsConvergeErr            = -205, /* The algorithm does not converge. */
    ippStsSizeMatchMatrixErr     = -204, /* The sizes of the source matrices are unsuitable. */
    ippStsCountMatrixErr         = -203, /* Count value is less than, or equal to zero. */
    ippStsRoiShiftMatrixErr      = -202, /* RoiShift value is negative or not divisible by the size of the data type. */

    ippStsResizeNoOperationErr   = -201, /* One of the output image dimensions is less than 1 pixel. */
    ippStsSrcDataErr             = -200, /* The source buffer contains unsupported data. */
    ippStsMaxLenHuffCodeErr      = -199, /* Huff: Max length of Huffman code is more than the expected one. */
    ippStsCodeLenTableErr        = -198, /* Huff: Invalid codeLenTable. */
    ippStsFreqTableErr           = -197, /* Huff: Invalid freqTable. */

    ippStsIncompleteContextErr   = -196, /* Crypto: set up of context is not complete. */

    ippStsSingularErr            = -195, /* Matrix is singular. */
    ippStsSparseErr              = -194, /* Positions of taps are not in ascending order, or are negative, or repetitive. */
    ippStsBitOffsetErr           = -193, /* Incorrect bit offset value. */
    ippStsQPErr                  = -192, /* Incorrect quantization parameter value. */
    ippStsVLCErr                 = -191, /* Illegal VLC or FLC is detected during stream decoding. */
    ippStsRegExpOptionsErr       = -190, /* RegExp: Options for the pattern are incorrect. */
    ippStsRegExpErr              = -189, /* RegExp: The structure pRegExpState contains incorrect data. */
    ippStsRegExpMatchLimitErr    = -188, /* RegExp: The match limit is exhausted. */
    ippStsRegExpQuantifierErr    = -187, /* RegExp: Incorrect quantifier. */
    ippStsRegExpGroupingErr      = -186, /* RegExp: Incorrect grouping. */
    ippStsRegExpBackRefErr       = -185, /* RegExp: Incorrect back reference. */
    ippStsRegExpChClassErr       = -184, /* RegExp: Incorrect character class. */
    ippStsRegExpMetaChErr        = -183, /* RegExp: Incorrect metacharacter. */
    ippStsStrideMatrixErr        = -182,  /* Stride value is not positive or not divisible by the size of the data type. */
    ippStsCTRSizeErr             = -181,  /* Incorrect value for crypto CTR block size. */
    ippStsJPEG2KCodeBlockIsNotAttached =-180, /* Codeblock parameters are not attached to the state structure. */
    ippStsNotPosDefErr           = -179,      /* Matrix is not positive definite. */

    ippStsEphemeralKeyErr        = -178, /* ECC: Invalid ephemeral key.   */
    ippStsMessageErr             = -177, /* ECC: Invalid message digest.  */
    ippStsShareKeyErr            = -176, /* ECC: Invalid share key.   */
    ippStsIvalidPublicKey        = -175, /* ECC: Invalid public key.  */
    ippStsIvalidPrivateKey       = -174, /* ECC: Invalid private key. */
    ippStsOutOfECErr             = -173, /* ECC: Point out of EC.     */
    ippStsECCInvalidFlagErr      = -172, /* ECC: Invalid Flag.        */

    ippStsMP3FrameHeaderErr      = -171,  /* Error in fields of the IppMP3FrameHeader structure. */
    ippStsMP3SideInfoErr         = -170,  /* Error in fields of the IppMP3SideInfo structure. */

    ippStsBlockStepErr           = -169,  /* Step for Block is less than 8. */
    ippStsMBStepErr              = -168,  /* Step for MB is less than 16. */

    ippStsAacPrgNumErr           = -167,  /* AAC: Invalid number of elements for one program.   */
    ippStsAacSectCbErr           = -166,  /* AAC: Invalid section codebook.                     */
    ippStsAacSfValErr            = -164,  /* AAC: Invalid scalefactor value.                    */
    ippStsAacCoefValErr          = -163,  /* AAC: Invalid quantized coefficient value.          */
    ippStsAacMaxSfbErr           = -162,  /* AAC: Invalid coefficient index.  */
    ippStsAacPredSfbErr          = -161,  /* AAC: Invalid predicted coefficient index.  */
    ippStsAacPlsDataErr          = -160,  /* AAC: Invalid pulse data attributes.  */
    ippStsAacGainCtrErr          = -159,  /* AAC: Gain control is not supported.  */
    ippStsAacSectErr             = -158,  /* AAC: Invalid number of sections.  */
    ippStsAacTnsNumFiltErr       = -157,  /* AAC: Invalid number of TNS filters.  */
    ippStsAacTnsLenErr           = -156,  /* AAC: Invalid length of TNS region.  */
    ippStsAacTnsOrderErr         = -155,  /* AAC: Invalid order of TNS filter.  */
    ippStsAacTnsCoefResErr       = -154,  /* AAC: Invalid bit-resolution for TNS filter coefficients.  */
    ippStsAacTnsCoefErr          = -153,  /* AAC: Invalid coefficients of TNS filter. */
    ippStsAacTnsDirectErr        = -152,  /* AAC: Invalid direction TNS filter.  */
    ippStsAacTnsProfileErr       = -151,  /* AAC: Invalid TNS profile.  */
    ippStsAacErr                 = -150,  /* AAC: Internal error.  */
    ippStsAacBitOffsetErr        = -149,  /* AAC: Invalid current bit offset in bitstream.  */
    ippStsAacAdtsSyncWordErr     = -148,  /* AAC: Invalid ADTS syncword.  */
    ippStsAacSmplRateIdxErr      = -147,  /* AAC: Invalid sample rate index.  */
    ippStsAacWinLenErr           = -146,  /* AAC: Invalid window length (not short or long).  */
    ippStsAacWinGrpErr           = -145,  /* AAC: Invalid number of groups for current window length.  */
    ippStsAacWinSeqErr           = -144,  /* AAC: Invalid window sequence range.  */
    ippStsAacComWinErr           = -143,  /* AAC: Invalid common window flag.  */
    ippStsAacStereoMaskErr       = -142,  /* AAC: Invalid stereo mask.  */
    ippStsAacChanErr             = -141,  /* AAC: Invalid channel number.  */
    ippStsAacMonoStereoErr       = -140,  /* AAC: Invalid mono-stereo flag.  */
    ippStsAacStereoLayerErr      = -139,  /* AAC: Invalid this Stereo Layer flag.  */
    ippStsAacMonoLayerErr        = -138,  /* AAC: Invalid this Mono Layer flag.  */
    ippStsAacScalableErr         = -137,  /* AAC: Invalid scalable object flag.  */
    ippStsAacObjTypeErr          = -136,  /* AAC: Invalid audio object type.  */
    ippStsAacWinShapeErr         = -135,  /* AAC: Invalid window shape.  */
    ippStsAacPcmModeErr          = -134,  /* AAC: Invalid PCM output interleaving indicator.  */
    ippStsVLCUsrTblHeaderErr          = -133,  /* VLC: Invalid header inside table. */
    ippStsVLCUsrTblUnsupportedFmtErr  = -132,  /* VLC: Table format is not supported.  */
    ippStsVLCUsrTblEscAlgTypeErr      = -131,  /* VLC: Ecs-algorithm is not supported. */
    ippStsVLCUsrTblEscCodeLengthErr   = -130,  /* VLC: Esc-code length inside table header is incorrect. */
    ippStsVLCUsrTblCodeLengthErr      = -129,  /* VLC: Code length inside table is incorrect.  */
    ippStsVLCInternalTblErr           = -128,  /* VLC: Invalid internal table. */
    ippStsVLCInputDataErr             = -127,  /* VLC: Invalid input data. */
    ippStsVLCAACEscCodeLengthErr      = -126,  /* VLC: Invalid AAC-Esc code length. */
    ippStsNoiseRangeErr         = -125,  /* Noise value for Wiener Filter is out of range. */
    ippStsUnderRunErr           = -124,  /* Error in data under run. */
    ippStsPaddingErr            = -123,  /* Detected padding error indicates the possible data corruption. */
    ippStsCFBSizeErr            = -122,  /* Incorrect value for crypto CFB block size. */
    ippStsPaddingSchemeErr      = -121,  /* Invalid padding scheme.  */
    ippStsInvalidCryptoKeyErr   = -120,  /* A compromised key causes suspansion of the requested cryptographic operation.  */
    ippStsLengthErr             = -119,  /* Incorrect value for string length. */
    ippStsBadModulusErr         = -118,  /* Bad modulus caused a failure in module inversion. */
    ippStsLPCCalcErr            = -117,  /* Cannot evaluate linear prediction. */
    ippStsRCCalcErr             = -116,  /* Cannot compute reflection coefficients. */
    ippStsIncorrectLSPErr       = -115,  /* Incorrect values for Linear Spectral Pair. */
    ippStsNoRootFoundErr        = -114,  /* No roots are found for equation. */
    ippStsJPEG2KBadPassNumber   = -113,  /* Pass number exceeds allowed boundaries [0,nOfPasses-1]. */
    ippStsJPEG2KDamagedCodeBlock= -112,  /* Codeblock for decoding contains damaged data. */
    ippStsH263CBPYCodeErr       = -111,  /* Illegal Huffman code is detected through CBPY stream processing. */
    ippStsH263MCBPCInterCodeErr = -110,  /* Illegal Huffman code is detected through MCBPC Inter stream processing. */
    ippStsH263MCBPCIntraCodeErr = -109,  /* Illegal Huffman code is detected through MCBPC Intra stream processing. */
    ippStsNotEvenStepErr        = -108,  /* Step value is not pixel multiple. */
    ippStsHistoNofLevelsErr     = -107,  /* Number of levels for histogram is less than 2. */
    ippStsLUTNofLevelsErr       = -106,  /* Number of levels for LUT is less than 2. */
    ippStsMP4BitOffsetErr       = -105,  /* Incorrect bit offset value. */
    ippStsMP4QPErr              = -104,  /* Incorrect quantization parameter. */
    ippStsMP4BlockIdxErr        = -103,  /* Incorrect block index. */
    ippStsMP4BlockTypeErr       = -102,  /* Incorrect block type. */
    ippStsMP4MVCodeErr          = -101,  /* Illegal Huffman code is detected during MV stream processing. */
    ippStsMP4VLCCodeErr         = -100,  /* Illegal Huffman code is detected during VLC stream processing. */
    ippStsMP4DCCodeErr          = -99,   /* Illegal code is detected during DC stream processing. */
    ippStsMP4FcodeErr           = -98,   /* Incorrect fcode value. */
    ippStsMP4AlignErr           = -97,   /* Incorrect buffer alignment .           */
    ippStsMP4TempDiffErr        = -96,   /* Incorrect temporal difference.         */
    ippStsMP4BlockSizeErr       = -95,   /* Incorrect size of a block or macroblock. */
    ippStsMP4ZeroBABErr         = -94,   /* All BAB values are equal to zero.             */
    ippStsMP4PredDirErr         = -93,   /* Incorrect prediction direction.        */
    ippStsMP4BitsPerPixelErr    = -92,   /* Incorrect number of bits per pixel.    */
    ippStsMP4VideoCompModeErr   = -91,   /* Incorrect video component mode.       */
    ippStsMP4LinearModeErr      = -90,   /* Incorrect DC linear mode. */
    ippStsH263PredModeErr       = -83,   /* Incorrect Prediction Mode value.                                       */
    ippStsH263BlockStepErr      = -82,   /* The step value is less than 8.                                         */
    ippStsH263MBStepErr         = -81,   /* The step value is less than 16.                                        */
    ippStsH263FrameWidthErr     = -80,   /* The frame width is less than 8.                                        */
    ippStsH263FrameHeightErr    = -79,   /* The frame height is less than, or equal to zero.                        */
    ippStsH263ExpandPelsErr     = -78,   /* Expand pixels number is less than 8.                               */
    ippStsH263PlaneStepErr      = -77,   /* Step value is less than the plane width.                           */
    ippStsH263QuantErr          = -76,   /* Quantizer value is less than, or equal to zero, or greater than 31. */
    ippStsH263MVCodeErr         = -75,   /* Illegal Huffman code is detected during MV stream processing.                  */
    ippStsH263VLCCodeErr        = -74,   /* Illegal Huffman code is detected during VLC stream processing.                 */
    ippStsH263DCCodeErr         = -73,   /* Illegal code is detected during DC stream processing.                          */
    ippStsH263ZigzagLenErr      = -72,   /* Zigzag compact length is more than 64.                             */
    ippStsFBankFreqErr          = -71,   /* Incorrect value for the filter bank frequency parameter. */
    ippStsFBankFlagErr          = -70,   /* Incorrect value for the filter bank parameter.           */
    ippStsFBankErr              = -69,   /* Filter bank is not correctly initialized.              */
    ippStsNegOccErr             = -67,   /* Occupation count is negative.                     */
    ippStsCdbkFlagErr           = -66,   /* Incorrect value for the codebook flag parameter. */
    ippStsSVDCnvgErr            = -65,   /* SVD algorithm does not converge.               */
    ippStsJPEGHuffTableErr      = -64,   /* JPEG Huffman table is destroyed.        */
    ippStsJPEGDCTRangeErr       = -63,   /* JPEG DCT coefficient is out of range. */
    ippStsJPEGOutOfBufErr       = -62,   /* Attempt to access out of the buffer limits.   */
    ippStsDrawTextErr           = -61,   /* System error in the draw text operation. */
    ippStsChannelOrderErr       = -60,   /* Incorrect order of the destination channels. */
    ippStsZeroMaskValuesErr     = -59,   /* All values of the mask are equal to zero. */
    ippStsQuadErr               = -58,   /* The quadrangle is nonconvex or degenerates into triangle, line, or point */
    ippStsRectErr               = -57,   /* Size of the rectangle region is less than, or equal to 1. */
    ippStsCoeffErr              = -56,   /* Incorrect values for transformation coefficients.   */
    ippStsNoiseValErr           = -55,   /* Incorrect value for noise amplitude for dithering.             */
    ippStsDitherLevelsErr       = -54,   /* Number of dithering levels is out of range.             */
    ippStsNumChannelsErr        = -53,   /* Number of channels is incorrect, or not supported.                  */
    ippStsCOIErr                = -52,   /* COI is out of range. */
    ippStsDivisorErr            = -51,   /* Divisor is equal to zero, function is aborted. */
    ippStsAlphaTypeErr          = -50,   /* Illegal type of image compositing operation.                           */
    ippStsGammaRangeErr         = -49,   /* Gamma range bounds is less than, or equal to zero.                      */
    ippStsGrayCoefSumErr        = -48,   /* Sum of the conversion coefficients must be less than, or equal to 1.    */
    ippStsChannelErr            = -47,   /* Illegal channel number.                                                */
    ippStsToneMagnErr           = -46,   /* Tone magnitude is less than, or equal to zero.                          */
    ippStsToneFreqErr           = -45,   /* Tone frequency is negative, or greater than, or equal to 0.5.           */
    ippStsTonePhaseErr          = -44,   /* Tone phase is negative, or greater than, or equal to 2*PI.              */
    ippStsTrnglMagnErr          = -43,   /* Triangle magnitude is less than, or equal to zero.                      */
    ippStsTrnglFreqErr          = -42,   /* Triangle frequency is negative, or greater than, or equal to 0.5.       */
    ippStsTrnglPhaseErr         = -41,   /* Triangle phase is negative, or greater than, or equal to 2*PI.          */
    ippStsTrnglAsymErr          = -40,   /* Triangle asymmetry is less than -PI, or greater than, or equal to PI.   */
    ippStsHugeWinErr            = -39,   /* Kaiser window is too big.                                             */
    ippStsJaehneErr             = -38,   /* Magnitude value is negative.                                           */
    ippStsStrideErr             = -37,   /* Stride value is less than the length of the row. */
    ippStsEpsValErr             = -36,   /* Negative epsilon value.             */
    ippStsWtOffsetErr           = -35,   /* Invalid offset value for wavelet filter.                                       */
    ippStsAnchorErr             = -34,   /* Anchor point is outside the mask.                                             */
    ippStsMaskSizeErr           = -33,   /* Invalid mask size.                                                           */
    ippStsShiftErr              = -32,   /* Shift value is less than zero.                                                */
    ippStsSampleFactorErr       = -31,   /* Sampling factor is less than, or equal to zero.                                */
    ippStsSamplePhaseErr        = -30,   /* Phase value is out of range: 0 <= phase < factor.                             */
    ippStsFIRMRFactorErr        = -29,   /* MR FIR sampling factor is less than, or equal to zero.                         */
    ippStsFIRMRPhaseErr         = -28,   /* MR FIR sampling phase is negative, or greater than, or equal to the sampling factor. */
    ippStsRelFreqErr            = -27,   /* Relative frequency value is out of range.                                     */
    ippStsFIRLenErr             = -26,   /* Length of a FIR filter is less than, or equal to zero.                         */
    ippStsIIROrderErr           = -25,   /* Order of an IIR filter is not valid. */
    ippStsDlyLineIndexErr       = -24,   /* Invalid value for the delay line sample index. */
    ippStsResizeFactorErr       = -23,   /* Resize factor(s) is less than, or equal to zero. */
    ippStsInterpolationErr      = -22,   /* Invalid interpolation mode. */
    ippStsMirrorFlipErr         = -21,   /* Invalid flip mode.                                         */
    ippStsMoment00ZeroErr       = -20,   /* Moment value M(0,0) is too small to continue calculations. */
    ippStsThreshNegLevelErr     = -19,   /* Negative value of the level in the threshold operation.    */
    ippStsThresholdErr          = -18,   /* Invalid threshold bounds. */
    ippStsContextMatchErr       = -17,   /* Context parameter does not match the operation. */
    ippStsFftFlagErr            = -16,   /* Invalid value for the FFT flag parameter. */
    ippStsFftOrderErr           = -15,   /* Invalid value for the FFT order parameter. */
    ippStsStepErr               = -14,   /* Step value is not valid. */
    ippStsScaleRangeErr         = -13,   /* Scale bounds are out of range. */
    ippStsDataTypeErr           = -12,   /* Data type is incorrect or not supported. */
    ippStsOutOfRangeErr         = -11,   /* Argument is out of range, or point is outside the image. */
    ippStsDivByZeroErr          = -10,   /* An attempt to divide by zero. */
    ippStsMemAllocErr           = -9,    /* Memory allocated for the operation is not enough.*/
    ippStsNullPtrErr            = -8,    /* Null pointer error. */
    ippStsRangeErr              = -7,    /* Incorrect values for bounds: the lower bound is greater than the upper bound. */
    ippStsSizeErr               = -6,    /* Incorrect value for data size. */
    ippStsBadArgErr             = -5,    /* Incorrect arg/param of the function.  */
    ippStsNoMemErr              = -4,    /* Not enough memory for the operation. */
    ippStsSAReservedErr3        = -3,    /* Unknown/unspecified error, -3. */
    ippStsErr                   = -2,    /* Unknown/unspecified error, -2. */
    ippStsSAReservedErr1        = -1,    /* Unknown/unspecified error, -1. */

     /* no errors */
    ippStsNoErr                 =   0,   /* No errors. */

     /* warnings  */
    ippStsNoOperation       =   1,       /* No operation has been executed. */
    ippStsMisalignedBuf     =   2,       /* Misaligned pointer in operation in which it must be aligned. */
    ippStsSqrtNegArg        =   3,       /* Negative value(s) for the argument in the Sqrt function. */
    ippStsInvZero           =   4,       /* INF result. Zero value was met by InvThresh with zero level. */
    ippStsEvenMedianMaskSize=   5,       /* Even size of the Median Filter mask was replaced with the odd one. */
    ippStsDivByZero         =   6,       /* Zero value(s) for the divisor in the Div function. */
    ippStsLnZeroArg         =   7,       /* Zero value(s) for the argument in the Ln function.     */
    ippStsLnNegArg          =   8,       /* Negative value(s) for the argument in the Ln function. */
    ippStsNanArg            =   9,       /* Argument value is not a number.                  */
    ippStsJPEGMarker        =   10,      /* JPEG marker in the bitstream.                 */
    ippStsResFloor          =   11,      /* All result values are floored.                        */
    ippStsOverflow          =   12,      /* Overflow in the operation.                   */
    ippStsLSFLow            =   13,      /* Quantized LP synthesis filter stability check is applied at the low boundary of [0,pi]. */
    ippStsLSFHigh           =   14,      /* Quantized LP synthesis filter stability check is applied at the high boundary of [0,pi]. */
    ippStsLSFLowAndHigh     =   15,      /* Quantized LP synthesis filter stability check is applied at both boundaries of [0,pi]. */
    ippStsZeroOcc           =   16,      /* Zero occupation count. */
    ippStsUnderflow         =   17,      /* Underflow in the operation. */
    ippStsSingularity       =   18,      /* Singularity in the operation.                                       */
    ippStsDomain            =   19,      /* Argument is out of the function domain.                                      */
    ippStsNonIntelCpu       =   20,      /* The target CPU is not Genuine Intel.                                         */
    ippStsCpuMismatch       =   21,      /* Cannot set the library for the given CPU.                                     */
    ippStsNoIppFunctionFound =  22,      /* Application does not contain Intel IPP function calls.                            */
    ippStsDllNotFoundBestUsed = 23,      /* Dispatcher cannot find the newest version of the Intel IPP dll.                  */
    ippStsNoOperationInDll  =   24,      /* The function does nothing in the dynamic version of the library.             */
    ippStsInsufficientEntropy=  25,      /* Generation of the prime/key failed due to insufficient entropy in the random seed and stimulus bit string. */
    ippStsOvermuchStrings   =   26,      /* Number of destination strings is more than expected.                         */
    ippStsOverlongString    =   27,      /* Length of one of the destination strings is more than expected.              */
    ippStsAffineQuadChanged =   28,      /* 4th vertex of destination quad is not equal to customer's one.               */
    ippStsWrongIntersectROI =   29,      /* ROI has no intersection with the source or destination ROI. No operation. */
    ippStsWrongIntersectQuad =  30,      /* Quadrangle has no intersection with the source or destination ROI. No operation. */
    ippStsSmallerCodebook   =   31,      /* Size of created codebook is less than the cdbkSize argument. */
    ippStsSrcSizeLessExpected = 32,      /* DC: Size of the source buffer is less than the expected one. */
    ippStsDstSizeLessExpected = 33,      /* DC: Size of the destination buffer is less than the expected one. */
    ippStsStreamEnd           = 34,      /* DC: The end of stream processed. */
    ippStsDoubleSize        =   35,      /* Width or height of image is odd. */
    ippStsNotSupportedCpu   =   36,      /* The CPU is not supported. */
    ippStsUnknownCacheSize  =   37,      /* The CPU is supported, but the size of the cache is unknown. */
    ippStsSymKernelExpected =   38,      /* The Kernel is not symmetric. */
    ippStsEvenMedianWeight  =   39,      /* Even weight of the Weighted Median Filter is replaced with the odd one. */
    ippStsWrongIntersectVOI =   40,      /* VOI has no intersection with the source or destination volume. No operation.                            */
    ippStsI18nMsgCatalogInvalid=41,      /* Message Catalog is invalid, English message returned.                                                    */
    ippStsI18nGetMessageFail  = 42,      /* Failed to fetch a localized message, English message returned. For more information use errno on Linux* OS and GetLastError on Windows* OS. */
    ippStsWaterfall           = 43,      /* Cannot load required library, waterfall is used. */
    ippStsPrevLibraryUsed     = 44,      /* Cannot load required library, previous dynamic library is used. */
    ippStsLLADisabled         = 45,      /* OpenMP* Low Level Affinity is disabled. */
    ippStsNoAntialiasing      = 46,      /* The mode does not support antialiasing. */
    ippStsRepetitiveSrcData   = 47,      /* DC: The source data is too repetitive. */
    ippStsSizeWrn             = 48,      /* The size does not allow to perform full operation. */
    ippStsFeatureNotSupported = 49,      /* Current CPU doesn't support at least 1 of the desired features. */
    ippStsUnknownFeature      = 50,      /* At least one of the desired features is unknown. */
    ippStsFeaturesCombination = 51,      /* Wrong combination of features. */
    ippStsAccurateModeNotSupported = 52  /* Accurate mode is not supported. */
} IppStatus;

#define ippStsOk ippStsNoErr


#ifdef __cplusplus
}
#endif

#endif /* __IPPICV_TYPES_H__ */
