
#
input_9Placeholder*
dtype0
Z
conv2d_9/kernelConst*3
value*B("9X8Ä0,∏0∂ê;»∂‡∂Ë7*
dtype0
@
conv2d_9/biasConst*
dtype0*
valueB"Ω≥Ω¯π
ö
conv2d_10/convolutionConv2Dinput_9conv2d_9/kernel*
use_cudnn_on_gpu(*
paddingVALID*
T0*
data_formatNHWC*
strides

b
conv2d_10/BiasAddBiasAddconv2d_10/convolutionconv2d_9/bias*
T0*
data_formatNHWC
2
conv2d_10/ReluReluconv2d_10/BiasAdd*
T0