#!/usr/bin/env python

import sys, re

spaces = '[\s]*'
symbols = '[\s\w\d,.:|]*'

def pattern1(prefix, test):
    return re.compile(spaces + prefix + '_' + test + '::' + symbols + '::' + '\(' + symbols + '\)' + spaces)

def pattern2(prefix, test, param1):
    return re.compile(spaces + prefix + '_' + test + '::' + symbols + '::' + '\(' + symbols + param1 + symbols + '\)' + spaces)

def pattern3(prefix, test, param1, param2):
    return re.compile(spaces + prefix + '_' + test + '::' + symbols + '::' + '\(' + symbols + param1 + symbols + param2 + symbols + '\)' + spaces)

def pattern4(prefix, test, param1, param2, param3):
    return re.compile(spaces + prefix + '_' + test + '::' + symbols + '::' + '\(' + symbols + param1 + symbols + param2 + symbols + param3 + symbols + '\)' + spaces)

def pattern5(prefix, test, param1, param2, param3, param5):
    return re.compile(spaces + prefix + '_' + test + '::' + symbols + '::' + '\(' + symbols + param1 + symbols + param2 + symbols + param3 + symbols + param4 + symbols + '\)' + spaces)

npp_patterns = [
    ###############################################################
    # Core

    # Core_AddMat (CV_8U | CV_16U | CV_32F)
    pattern2('Core', 'AddMat', '8U'),
    pattern2('Core', 'AddMat', '16U'),
    pattern2('Core', 'AddMat', '32F'),

    # Core_AddScalar (CV_8U | CV_16U | CV_32F)
    pattern2('Core', 'AddScalar', '8U'),
    pattern2('Core', 'AddScalar', '16U'),
    pattern2('Core', 'AddScalar', '32F'),

    # Core_SubtractMat (CV_8U | CV_16U | CV_32F)
    pattern2('Core', 'SubtractMat', '8U'),
    pattern2('Core', 'SubtractMat', '16U'),
    pattern2('Core', 'SubtractMat', '32F'),

    # Core_SubtractScalar (CV_8U | CV_16U | CV_32F)
    pattern2('Core', 'SubtractScalar', '8U'),
    pattern2('Core', 'SubtractScalar', '16U'),
    pattern2('Core', 'SubtractScalar', '32F'),

    # Core_MultiplyMat (CV_8U | CV_16U | CV_32F)
    pattern2('Core', 'MultiplyMat', '8U'),
    pattern2('Core', 'MultiplyMat', '16U'),
    pattern2('Core', 'MultiplyMat', '32F'),

    # Core_MultiplyScalar (CV_8U | CV_16U | CV_32F)
    pattern2('Core', 'MultiplyScalar', '8U'),
    pattern2('Core', 'MultiplyScalar', '16U'),
    pattern2('Core', 'MultiplyScalar', '32F'),

    # Core_DivideMat (CV_8U | CV_16U | CV_32F)
    pattern2('Core', 'DivideMat', '8U'),
    pattern2('Core', 'DivideMat', '16U'),
    pattern2('Core', 'DivideMat', '32F'),

    # Core_Divide_Scalar (CV_8U | CV_16U | CV_32F)
    pattern2('Core', 'DivideScalar', '8U'),
    pattern2('Core', 'DivideScalar', '16U'),
    pattern2('Core', 'DivideScalar', '32F'),

    # Core_AbsDiff_Mat (CV_8U | CV_16U | CV_32F)
    pattern2('Core', 'AbsDiffMat', '8U'),
    pattern2('Core', 'AbsDiffMat', '16U'),
    pattern2('Core', 'AbsDiffMat', '32F'),

    # Core_AbsDiffScalar (CV_8U | CV_16U | CV_32F)
    pattern2('Core', 'AbsDiffScalar', '8U'),
    pattern2('Core', 'AbsDiffScalar', '16U'),
    pattern2('Core', 'AbsDiffScalar', '32F'),

    # Core_Abs
    pattern1('Core', 'Abs'),

    # Core_Sqr
    pattern1('Core', 'Sqr'),

    # Core_Sqrt
    pattern1('Core', 'Sqrt'),

    # Core_Log
    pattern1('Core', 'Log'),

    # Core_Exp
    pattern1('Core', 'Exp'),

    # Core_BitwiseAndScalar
    pattern1('Core', 'BitwiseAndScalar'),

    # Core_BitwiseOrScalar
    pattern1('Core', 'BitwiseOrScalar'),

    # Core_BitwiseXorScalar
    pattern1('Core', 'BitwiseXorScalar'),

    # Core_RShift
    pattern1('Core', 'RShift'),

    # Core_LShift
    pattern1('Core', 'LShift'),

    # Core_Transpose
    pattern1('Core', 'Transpose'),

    # Core_Flip
    pattern1('Core', 'Flip'),

    # Core_LutOneChannel
    pattern1('Core', 'LutOneChannel'),

    # Core_LutMultiChannel
    pattern1('Core', 'LutMultiChannel'),

    # Core_MagnitudeComplex
    pattern1('Core', 'MagnitudeComplex'),

    # Core_MagnitudeSqrComplex
    pattern1('Core', 'MagnitudeSqrComplex'),

    # Core_MeanStdDev
    pattern1('Core', 'MeanStdDev'),

    # Core_NormDiff
    pattern1('Core', 'NormDiff'),

    ##############################################################
    # Filters

    # Filters_Blur
    pattern1('Filters', 'Blur'),

    # Filters_Erode
    pattern1('Filters', 'Erode'),

    # Filters_Dilate
    pattern1('Filters', 'Dilate'),

    # Filters_MorphologyEx
    pattern1('Filters', 'MorphologyEx'),

    ##############################################################
    # ImgProc

    # ImgProc_Resize (8U, 1 | 4, INTER_NEAREST | INTER_LINEAR)
    pattern4('ImgProc', 'Resize', '8U', '1', 'INTER_NEAREST'),
    pattern4('ImgProc', 'Resize', '8U', '4', 'INTER_NEAREST'),
    pattern4('ImgProc', 'Resize', '8U', '1', 'INTER_LINEAR'),
    pattern4('ImgProc', 'Resize', '8U', '4', 'INTER_LINEAR'),

    # ImgProc_Resize (8U, 4, INTER_CUBIC)
    pattern4('ImgProc', 'Resize', '8U', '4', 'INTER_CUBIC'),

    # ImgProc_WarpAffine (8U | 32F, INTER_NEAREST | INTER_LINEAR | INTER_CUBIC, BORDER_CONSTANT)
    pattern4('ImgProc', 'WarpAffine', '8U' , 'INTER_NEAREST', 'BORDER_CONSTANT'),
    pattern4('ImgProc', 'WarpAffine', '8U' , 'INTER_LINEAR', 'BORDER_CONSTANT'),
    pattern4('ImgProc', 'WarpAffine', '8U' , 'INTER_CUBIC', 'BORDER_CONSTANT'),
    pattern4('ImgProc', 'WarpAffine', '32F', 'INTER_NEAREST', 'BORDER_CONSTANT'),
    pattern4('ImgProc', 'WarpAffine', '32F', 'INTER_LINEAR', 'BORDER_CONSTANT'),
    pattern4('ImgProc', 'WarpAffine', '32F', 'INTER_CUBIC', 'BORDER_CONSTANT'),

    # ImgProc_WarpPerspective (8U | 32F, INTER_NEAREST | INTER_LINEAR | INTER_CUBIC, BORDER_CONSTANT)
    pattern4('ImgProc', 'WarpPerspective', '8U' , 'INTER_NEAREST', 'BORDER_CONSTANT'),
    pattern4('ImgProc', 'WarpPerspective', '8U' , 'INTER_LINEAR', 'BORDER_CONSTANT'),
    pattern4('ImgProc', 'WarpPerspective', '8U' , 'INTER_CUBIC', 'BORDER_CONSTANT'),
    pattern4('ImgProc', 'WarpPerspective', '32F', 'INTER_NEAREST', 'BORDER_CONSTANT'),
    pattern4('ImgProc', 'WarpPerspective', '32F', 'INTER_LINEAR', 'BORDER_CONSTANT'),
    pattern4('ImgProc', 'WarpPerspective', '32F', 'INTER_CUBIC', 'BORDER_CONSTANT'),

    # ImgProc_CopyMakeBorder (8UC1 | 8UC4 | 32SC1 | 32FC1, BORDER_CONSTANT)
    pattern4('ImgProc', 'CopyMakeBorder', '8U' , '1', 'BORDER_CONSTANT'),
    pattern4('ImgProc', 'CopyMakeBorder', '8U' , '4', 'BORDER_CONSTANT'),
    pattern4('ImgProc', 'CopyMakeBorder', '32S', '1', 'BORDER_CONSTANT'),
    pattern4('ImgProc', 'CopyMakeBorder', '32F', '1', 'BORDER_CONSTANT'),

    # ImgProc_Threshold (32F, THRESH_TRUNC)
    pattern3('ImgProc', 'Threshold', '32F', 'THRESH_TRUNC'),

    # ImgProc_IntegralSqr
    pattern1('ImgProc', 'IntegralSqr'),

    # ImgProc_HistEven_OneChannel
    pattern1('ImgProc', 'HistEvenOneChannel'),

    # ImgProc_HistEven_FourChannel
    pattern1('ImgProc', 'HistEvenFourChannel'),

    # ImgProc_Rotate
    pattern1('ImgProc', 'Rotate'),

    # ImgProc_SwapChannels
    pattern1('ImgProc', 'SwapChannels'),

    # ImgProc_AlphaComp
    pattern1('ImgProc', 'AlphaComp'),

    # ImgProc_ImagePyramidBuild
    pattern1('ImgProc', 'ImagePyramidBuild'),

    # ImgProc_ImagePyramid_getLayer
    pattern1('ImgProc', 'ImagePyramidGetLayer'),

    ##############################################################
    # MatOp

    # MatOp_SetTo (8UC4 | 16UC1 | 16UC4 | 32FC1 | 32FC4)
    pattern3('MatOp', 'SetTo', '8U' , '4'),
    pattern3('MatOp', 'SetTo', '16U', '1'),
    pattern3('MatOp', 'SetTo', '16U', '4'),
    pattern3('MatOp', 'SetTo', '32F', '1'),
    pattern3('MatOp', 'SetTo', '32F', '4'),

    # MatOp_SetToMasked (8UC4 | 16UC1 | 16UC4 | 32FC1 | 32FC4)
    pattern3('MatOp', 'SetToMasked', '8U' , '4'),
    pattern3('MatOp', 'SetToMasked', '16U', '1'),
    pattern3('MatOp', 'SetToMasked', '16U', '4'),
    pattern3('MatOp', 'SetToMasked', '32F', '1'),
    pattern3('MatOp', 'SetToMasked', '32F', '4'),

    # MatOp_CopyToMasked (8UC1 | 8UC3 |8UC4 | 16UC1 | 16UC3 | 16UC4 | 32FC1 | 32FC3 | 32FC4)
    pattern3('MatOp', 'CopyToMasked', '8U' , '1'),
    pattern3('MatOp', 'CopyToMasked', '8U' , '3'),
    pattern3('MatOp', 'CopyToMasked', '8U' , '4'),
    pattern3('MatOp', 'CopyToMasked', '16U', '1'),
    pattern3('MatOp', 'CopyToMasked', '16U', '3'),
    pattern3('MatOp', 'CopyToMasked', '16U', '4'),
    pattern3('MatOp', 'CopyToMasked', '32F', '1'),
    pattern3('MatOp', 'CopyToMasked', '32F', '3'),
    pattern3('MatOp', 'CopyToMasked', '32F', '4'),
]

cublasPattern = pattern1('Core', 'GEMM')

cufftPattern = pattern1('ImgProc', 'Dft')

if __name__ == "__main__":
    inputFile = open(sys.argv[1], 'r')
    lines = inputFile.readlines()
    inputFile.close()


    for i in range(len(lines)):
        if cublasPattern.match(lines[i]):
            lines[i] = lines[i][:-1] + ' <font color=\"blue\">[CUBLAS]</font>\n'
        else:
            if cufftPattern.match(lines[i]):
                lines[i] = lines[i][:-1] + ' <font color=\"blue\">[CUFFT]</font>\n'
            else:
                for p in npp_patterns:
                    if p.match(lines[i]):
                        lines[i] = lines[i][:-1] + ' <font color=\"blue\">[NPP]</font>\n'

    outputFile = open(sys.argv[2], 'w')
    outputFile.writelines(lines)
    outputFile.close()
