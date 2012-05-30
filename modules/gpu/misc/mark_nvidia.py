import sys, re

spaces = '[\s]*'
symbols = '[\s\w\d,=:|]*'

def pattern1(prefix, test):
    return re.compile(spaces + 'perf::' + prefix + '/' + test + '::' + '\(' + symbols + '\)' + spaces)

def pattern2(prefix, test, cvtype):
    return re.compile(spaces + 'perf::' + prefix + '/' + test + '::' + '\(' + symbols + cvtype + symbols + '\)' + spaces)

def pattern3(prefix, test, cvtype, param1):
    return re.compile(spaces + 'perf::' + prefix + '/' + test + '::' + '\(' + symbols + cvtype + symbols + param1 + symbols + '\)' + spaces)

def pattern4(prefix, test, cvtype, param1, param2):
    return re.compile(spaces + 'perf::' + prefix + '/' + test + '::' + '\(' + symbols + cvtype + symbols + param1 + symbols + param2 + symbols + '\)' + spaces)

npp_patterns = [
    ##############################################################
    # Core
    
    # Core/Add_Mat (CV_8U | CV_16U | CV_32F)
    pattern2('Core', 'Add_Mat', '8U'),
    pattern2('Core', 'Add_Mat', '16U'),
    pattern2('Core', 'Add_Mat', '32F'),
    
    # Core/Add_Scalar (CV_8U | CV_16U | CV_32F)
    pattern2('Core', 'Add_Scalar', '8U'),
    pattern2('Core', 'Add_Scalar', '16U'),
    pattern2('Core', 'Add_Scalar', '32F'),
    
    # Core/Subtract_Mat (CV_8U | CV_16U | CV_32F)
    pattern2('Core', 'Subtract_Mat', '8U'),
    pattern2('Core', 'Subtract_Mat', '16U'),
    pattern2('Core', 'Subtract_Mat', '32F'),
    
    # Core/Subtract_Scalar (CV_8U | CV_16U | CV_32F)
    pattern2('Core', 'Subtract_Scalar', '8U'),
    pattern2('Core', 'Subtract_Scalar', '16U'),
    pattern2('Core', 'Subtract_Scalar', '32F'),
    
    # Core/Multiply_Mat (CV_8U | CV_16U | CV_32F)
    pattern2('Core', 'Multiply_Mat', '8U'),
    pattern2('Core', 'Multiply_Mat', '16U'),
    pattern2('Core', 'Multiply_Mat', '32F'),
    
    # Core/Multiply_Scalar (CV_8U | CV_16U | CV_32F)
    pattern2('Core', 'Multiply_Scalar', '8U'),
    pattern2('Core', 'Multiply_Scalar', '16U'),
    pattern2('Core', 'Multiply_Scalar', '32F'),
    
    # Core/Divide_Mat (CV_8U | CV_16U | CV_32F)
    pattern2('Core', 'Divide_Mat', '8U'),
    pattern2('Core', 'Divide_Mat', '16U'),
    pattern2('Core', 'Divide_Mat', '32F'),
    
    # Core/Divide_Scalar (CV_8U | CV_16U | CV_32F)
    pattern2('Core', 'Divide_Scalar', '8U'),
    pattern2('Core', 'Divide_Scalar', '16U'),
    pattern2('Core', 'Divide_Scalar', '32F'),
    
    # Core/AbsDiff_Mat (CV_8U | CV_16U | CV_32F)
    pattern2('Core', 'AbsDiff_Mat', '8U'),
    pattern2('Core', 'AbsDiff_Mat', '16U'),
    pattern2('Core', 'AbsDiff_Mat', '32F'),
    
    # Core/AbsDiff_Scalar (CV_8U | CV_16U | CV_32F)
    pattern2('Core', 'AbsDiff_Scalar', '8U'),
    pattern2('Core', 'AbsDiff_Scalar', '16U'),
    pattern2('Core', 'AbsDiff_Scalar', '32F'),

    # Core/Abs
    pattern1('Core', 'Abs'),

    # Core/Sqr
    pattern1('Core', 'Sqr'),

    # Core/Sqrt
    pattern1('Core', 'Sqrt'),

    # Core/Log
    pattern1('Core', 'Log'),

    # Core/Exp
    pattern1('Core', 'Exp'),

    # Core/Bitwise_And_Scalar
    pattern1('Core', 'Bitwise_And_Scalar'),

    # Core/Bitwise_Or_Scalar
    pattern1('Core', 'Bitwise_Or_Scalar'),

    # Core/Bitwise_Xor_Scalar
    pattern1('Core', 'Bitwise_Xor_Scalar'),

    # Core/RShift
    pattern1('Core', 'RShift'),

    # Core/LShift
    pattern1('Core', 'LShift'),

    # Core/Transpose
    pattern1('Core', 'Transpose'),

    # Core/Flip
    pattern1('Core', 'Flip'),

    # Core/LUT_OneChannel
    pattern1('Core', 'LUT_OneChannel'),

    # Core/LUT_MultiChannel
    pattern1('Core', 'LUT_MultiChannel'),

    # Core/Magnitude_Complex
    pattern1('Core', 'Magnitude_Complex'),

    # Core/Magnitude_Sqr_Complex
    pattern1('Core', 'Magnitude_Sqr_Complex'),

    # Core/MeanStdDev
    pattern1('Core', 'MeanStdDev'),

    # Core/NormDiff
    pattern1('Core', 'NormDiff'),
    
    ##############################################################
    # Filters

    # Filters/Blur
    pattern1('Filters', 'Blur'),
    
    # Filters/Erode
    pattern1('Filters', 'Erode'),
    
    # Filters/Dilate
    pattern1('Filters', 'Dilate'),
    
    # Filters/MorphologyEx
    pattern1('Filters', 'MorphologyEx'),
    
    ##############################################################
    # ImgProc
    
    # ImgProc/Resize (8UC1 | 8UC4, INTER_NEAREST | INTER_LINEAR)
    pattern3('ImgProc', 'Resize', '8UC1', 'INTER_NEAREST'),
    pattern3('ImgProc', 'Resize', '8UC4', 'INTER_NEAREST'),
    pattern3('ImgProc', 'Resize', '8UC1', 'INTER_LINEAR'),
    pattern3('ImgProc', 'Resize', '8UC4', 'INTER_LINEAR'),
    
    # ImgProc/Resize (8UC4, INTER_CUBIC)
    pattern3('ImgProc', 'Resize', '8UC4', 'INTER_CUBIC'),
    
    # ImgProc/WarpAffine (8UC1 | 8UC3 | 8UC4 | 32FC1 | 32FC3 | 32FC4, INTER_NEAREST | INTER_LINEAR | INTER_CUBIC, BORDER_CONSTANT)
    pattern4('ImgProc', 'WarpAffine', '8UC1', 'INTER_NEAREST', 'BORDER_CONSTANT'),
    pattern4('ImgProc', 'WarpAffine', '8UC1', 'INTER_LINEAR', 'BORDER_CONSTANT'),
    pattern4('ImgProc', 'WarpAffine', '8UC1', 'INTER_CUBIC', 'BORDER_CONSTANT'),
    pattern4('ImgProc', 'WarpAffine', '8UC3', 'INTER_NEAREST', 'BORDER_CONSTANT'),
    pattern4('ImgProc', 'WarpAffine', '8UC3', 'INTER_LINEAR', 'BORDER_CONSTANT'),
    pattern4('ImgProc', 'WarpAffine', '8UC3', 'INTER_CUBIC', 'BORDER_CONSTANT'),
    pattern4('ImgProc', 'WarpAffine', '8UC4', 'INTER_NEAREST', 'BORDER_CONSTANT'),
    pattern4('ImgProc', 'WarpAffine', '8UC4', 'INTER_LINEAR', 'BORDER_CONSTANT'),
    pattern4('ImgProc', 'WarpAffine', '8UC4', 'INTER_CUBIC', 'BORDER_CONSTANT'),
    pattern4('ImgProc', 'WarpAffine', '32FC1', 'INTER_NEAREST', 'BORDER_CONSTANT'),
    pattern4('ImgProc', 'WarpAffine', '32FC1', 'INTER_LINEAR', 'BORDER_CONSTANT'),
    pattern4('ImgProc', 'WarpAffine', '32FC1', 'INTER_CUBIC', 'BORDER_CONSTANT'),
    pattern4('ImgProc', 'WarpAffine', '32FC3', 'INTER_NEAREST', 'BORDER_CONSTANT'),
    pattern4('ImgProc', 'WarpAffine', '32FC3', 'INTER_LINEAR', 'BORDER_CONSTANT'),
    pattern4('ImgProc', 'WarpAffine', '32FC3', 'INTER_CUBIC', 'BORDER_CONSTANT'),
    pattern4('ImgProc', 'WarpAffine', '32FC4', 'INTER_NEAREST', 'BORDER_CONSTANT'),
    pattern4('ImgProc', 'WarpAffine', '32FC4', 'INTER_LINEAR', 'BORDER_CONSTANT'),
    pattern4('ImgProc', 'WarpAffine', '32FC4', 'INTER_CUBIC', 'BORDER_CONSTANT'),
    
    # ImgProc/WarpPerspective (8UC1 | 8UC3 | 8UC4 | 32FC1 | 32FC3 | 32FC4, INTER_NEAREST | INTER_LINEAR | INTER_CUBIC, BORDER_CONSTANT)
    pattern4('ImgProc', 'WarpPerspective', '8UC1', 'INTER_NEAREST', 'BORDER_CONSTANT'),
    pattern4('ImgProc', 'WarpPerspective', '8UC1', 'INTER_LINEAR', 'BORDER_CONSTANT'),
    pattern4('ImgProc', 'WarpPerspective', '8UC1', 'INTER_CUBIC', 'BORDER_CONSTANT'),
    pattern4('ImgProc', 'WarpPerspective', '8UC3', 'INTER_NEAREST', 'BORDER_CONSTANT'),
    pattern4('ImgProc', 'WarpPerspective', '8UC3', 'INTER_LINEAR', 'BORDER_CONSTANT'),
    pattern4('ImgProc', 'WarpPerspective', '8UC3', 'INTER_CUBIC', 'BORDER_CONSTANT'),
    pattern4('ImgProc', 'WarpPerspective', '8UC4', 'INTER_NEAREST', 'BORDER_CONSTANT'),
    pattern4('ImgProc', 'WarpPerspective', '8UC4', 'INTER_LINEAR', 'BORDER_CONSTANT'),
    pattern4('ImgProc', 'WarpPerspective', '8UC4', 'INTER_CUBIC', 'BORDER_CONSTANT'),
    pattern4('ImgProc', 'WarpPerspective', '32FC1', 'INTER_NEAREST', 'BORDER_CONSTANT'),
    pattern4('ImgProc', 'WarpPerspective', '32FC1', 'INTER_LINEAR', 'BORDER_CONSTANT'),
    pattern4('ImgProc', 'WarpPerspective', '32FC1', 'INTER_CUBIC', 'BORDER_CONSTANT'),
    pattern4('ImgProc', 'WarpPerspective', '32FC3', 'INTER_NEAREST', 'BORDER_CONSTANT'),
    pattern4('ImgProc', 'WarpPerspective', '32FC3', 'INTER_LINEAR', 'BORDER_CONSTANT'),
    pattern4('ImgProc', 'WarpPerspective', '32FC3', 'INTER_CUBIC', 'BORDER_CONSTANT'),
    pattern4('ImgProc', 'WarpPerspective', '32FC4', 'INTER_NEAREST', 'BORDER_CONSTANT'),
    pattern4('ImgProc', 'WarpPerspective', '32FC4', 'INTER_LINEAR', 'BORDER_CONSTANT'),
    pattern4('ImgProc', 'WarpPerspective', '32FC4', 'INTER_CUBIC', 'BORDER_CONSTANT'),
    
    # ImgProc/CopyMakeBorder (8UC1 | 8UC4 | 32SC1 | 32FC1, BORDER_CONSTANT)
    pattern3('ImgProc', 'CopyMakeBorder', '8UC1', 'BORDER_CONSTANT'),
    pattern3('ImgProc', 'CopyMakeBorder', '8UC4', 'BORDER_CONSTANT'),
    pattern3('ImgProc', 'CopyMakeBorder', '32SC1', 'BORDER_CONSTANT'),
    pattern3('ImgProc', 'CopyMakeBorder', '32FC1', 'BORDER_CONSTANT'),
    
    # ImgProc/Threshold (32F, THRESH_TRUNC)
    pattern3('ImgProc', 'Threshold', '32F', 'THRESH_TRUNC'),

    # ImgProc/Integral_Sqr
    pattern1('ImgProc', 'Integral_Sqr'),

    # ImgProc/HistEven_OneChannel
    pattern1('ImgProc', 'HistEven_OneChannel'),

    # ImgProc/HistEven_FourChannel
    pattern1('ImgProc', 'HistEven_FourChannel'),

    # ImgProc/Rotate
    pattern1('ImgProc', 'Rotate'),

    # ImgProc/SwapChannels
    pattern1('ImgProc', 'SwapChannels'),

    # ImgProc/AlphaComp
    pattern1('ImgProc', 'AlphaComp'),

    # ImgProc/ImagePyramid_build
    pattern1('ImgProc', 'ImagePyramid_build'),

    # ImgProc/ImagePyramid_getLayer
    pattern1('ImgProc', 'ImagePyramid_getLayer'),
    
    ##############################################################
    # MatOp
    
    # MatOp/SetTo (8UC4 | 16UC1 | 16UC4 | 32FC1 | 32FC4)
    pattern2('MatOp', 'SetTo', '8UC4'),
    pattern2('MatOp', 'SetTo', '16UC1'),
    pattern2('MatOp', 'SetTo', '16UC4'),
    pattern2('MatOp', 'SetTo', '32FC1'),
    pattern2('MatOp', 'SetTo', '32FC4'),
    
    # MatOp/SetToMasked (8UC4 | 16UC1 | 16UC4 | 32FC1 | 32FC4)
    pattern2('MatOp', 'SetToMasked', '8UC4'),
    pattern2('MatOp', 'SetToMasked', '16UC1'),
    pattern2('MatOp', 'SetToMasked', '16UC4'),
    pattern2('MatOp', 'SetToMasked', '32FC1'),
    pattern2('MatOp', 'SetToMasked', '32FC4'),
    
    # MatOp/CopyToMasked (8UC1 | 8UC3 |8UC4 | 16UC1 | 16UC3 | 16UC4 | 32FC1 | 32FC3 | 32FC4)
    pattern2('MatOp', 'CopyToMasked', '8UC1'),
    pattern2('MatOp', 'CopyToMasked', '8UC3'),
    pattern2('MatOp', 'CopyToMasked', '8UC4'),
    pattern2('MatOp', 'CopyToMasked', '16UC1'),
    pattern2('MatOp', 'CopyToMasked', '16UC3'),
    pattern2('MatOp', 'CopyToMasked', '16UC4'),
    pattern2('MatOp', 'CopyToMasked', '32FC1'),
    pattern2('MatOp', 'CopyToMasked', '32FC3'),
    pattern2('MatOp', 'CopyToMasked', '32FC4'),    
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

