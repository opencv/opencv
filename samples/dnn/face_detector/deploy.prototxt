input: "data"
input_shape {
  dim: 1
  dim: 3
  dim: 300
  dim: 300
}

layer {
  name: "data_bn"
  type: "BatchNorm"
  bottom: "data"
  top: "data_bn"
  param {
    lr_mult: 0.0
  }
  param {
    lr_mult: 0.0
  }
  param {
    lr_mult: 0.0
  }
}
layer {
  name: "data_scale"
  type: "Scale"
  bottom: "data_bn"
  top: "data_bn"
  param {
    lr_mult: 1.0
    decay_mult: 1.0
  }
  param {
    lr_mult: 2.0
    decay_mult: 1.0
  }
  scale_param {
    bias_term: true
  }
}
layer {
  name: "conv1_h"
  type: "Convolution"
  bottom: "data_bn"
  top: "conv1_h"
  param {
    lr_mult: 1.0
    decay_mult: 1.0
  }
  param {
    lr_mult: 2.0
    decay_mult: 1.0
  }
  convolution_param {
    num_output: 32
    pad: 3
    kernel_size: 7
    stride: 2
    weight_filler {
      type: "msra"
      variance_norm: FAN_OUT
    }
    bias_filler {
      type: "constant"
      value: 0.0
    }
  }
}
layer {
  name: "conv1_bn_h"
  type: "BatchNorm"
  bottom: "conv1_h"
  top: "conv1_h"
  param {
    lr_mult: 0.0
  }
  param {
    lr_mult: 0.0
  }
  param {
    lr_mult: 0.0
  }
}
layer {
  name: "conv1_scale_h"
  type: "Scale"
  bottom: "conv1_h"
  top: "conv1_h"
  param {
    lr_mult: 1.0
    decay_mult: 1.0
  }
  param {
    lr_mult: 2.0
    decay_mult: 1.0
  }
  scale_param {
    bias_term: true
  }
}
layer {
  name: "conv1_relu"
  type: "ReLU"
  bottom: "conv1_h"
  top: "conv1_h"
}
layer {
  name: "conv1_pool"
  type: "Pooling"
  bottom: "conv1_h"
  top: "conv1_pool"
  pooling_param {
    kernel_size: 3
    stride: 2
  }
}
layer {
  name: "layer_64_1_conv1_h"
  type: "Convolution"
  bottom: "conv1_pool"
  top: "layer_64_1_conv1_h"
  param {
    lr_mult: 1.0
    decay_mult: 1.0
  }
  convolution_param {
    num_output: 32
    bias_term: false
    pad: 1
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
      value: 0.0
    }
  }
}
layer {
  name: "layer_64_1_bn2_h"
  type: "BatchNorm"
  bottom: "layer_64_1_conv1_h"
  top: "layer_64_1_conv1_h"
  param {
    lr_mult: 0.0
  }
  param {
    lr_mult: 0.0
  }
  param {
    lr_mult: 0.0
  }
}
layer {
  name: "layer_64_1_scale2_h"
  type: "Scale"
  bottom: "layer_64_1_conv1_h"
  top: "layer_64_1_conv1_h"
  param {
    lr_mult: 1.0
    decay_mult: 1.0
  }
  param {
    lr_mult: 2.0
    decay_mult: 1.0
  }
  scale_param {
    bias_term: true
  }
}
layer {
  name: "layer_64_1_relu2"
  type: "ReLU"
  bottom: "layer_64_1_conv1_h"
  top: "layer_64_1_conv1_h"
}
layer {
  name: "layer_64_1_conv2_h"
  type: "Convolution"
  bottom: "layer_64_1_conv1_h"
  top: "layer_64_1_conv2_h"
  param {
    lr_mult: 1.0
    decay_mult: 1.0
  }
  convolution_param {
    num_output: 32
    bias_term: false
    pad: 1
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
      value: 0.0
    }
  }
}
layer {
  name: "layer_64_1_sum"
  type: "Eltwise"
  bottom: "layer_64_1_conv2_h"
  bottom: "conv1_pool"
  top: "layer_64_1_sum"
}
layer {
  name: "layer_128_1_bn1_h"
  type: "BatchNorm"
  bottom: "layer_64_1_sum"
  top: "layer_128_1_bn1_h"
  param {
    lr_mult: 0.0
  }
  param {
    lr_mult: 0.0
  }
  param {
    lr_mult: 0.0
  }
}
layer {
  name: "layer_128_1_scale1_h"
  type: "Scale"
  bottom: "layer_128_1_bn1_h"
  top: "layer_128_1_bn1_h"
  param {
    lr_mult: 1.0
    decay_mult: 1.0
  }
  param {
    lr_mult: 2.0
    decay_mult: 1.0
  }
  scale_param {
    bias_term: true
  }
}
layer {
  name: "layer_128_1_relu1"
  type: "ReLU"
  bottom: "layer_128_1_bn1_h"
  top: "layer_128_1_bn1_h"
}
layer {
  name: "layer_128_1_conv1_h"
  type: "Convolution"
  bottom: "layer_128_1_bn1_h"
  top: "layer_128_1_conv1_h"
  param {
    lr_mult: 1.0
    decay_mult: 1.0
  }
  convolution_param {
    num_output: 128
    bias_term: false
    pad: 1
    kernel_size: 3
    stride: 2
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
      value: 0.0
    }
  }
}
layer {
  name: "layer_128_1_bn2"
  type: "BatchNorm"
  bottom: "layer_128_1_conv1_h"
  top: "layer_128_1_conv1_h"
  param {
    lr_mult: 0.0
  }
  param {
    lr_mult: 0.0
  }
  param {
    lr_mult: 0.0
  }
}
layer {
  name: "layer_128_1_scale2"
  type: "Scale"
  bottom: "layer_128_1_conv1_h"
  top: "layer_128_1_conv1_h"
  param {
    lr_mult: 1.0
    decay_mult: 1.0
  }
  param {
    lr_mult: 2.0
    decay_mult: 1.0
  }
  scale_param {
    bias_term: true
  }
}
layer {
  name: "layer_128_1_relu2"
  type: "ReLU"
  bottom: "layer_128_1_conv1_h"
  top: "layer_128_1_conv1_h"
}
layer {
  name: "layer_128_1_conv2"
  type: "Convolution"
  bottom: "layer_128_1_conv1_h"
  top: "layer_128_1_conv2"
  param {
    lr_mult: 1.0
    decay_mult: 1.0
  }
  convolution_param {
    num_output: 128
    bias_term: false
    pad: 1
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
      value: 0.0
    }
  }
}
layer {
  name: "layer_128_1_conv_expand_h"
  type: "Convolution"
  bottom: "layer_128_1_bn1_h"
  top: "layer_128_1_conv_expand_h"
  param {
    lr_mult: 1.0
    decay_mult: 1.0
  }
  convolution_param {
    num_output: 128
    bias_term: false
    pad: 0
    kernel_size: 1
    stride: 2
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
      value: 0.0
    }
  }
}
layer {
  name: "layer_128_1_sum"
  type: "Eltwise"
  bottom: "layer_128_1_conv2"
  bottom: "layer_128_1_conv_expand_h"
  top: "layer_128_1_sum"
}
layer {
  name: "layer_256_1_bn1"
  type: "BatchNorm"
  bottom: "layer_128_1_sum"
  top: "layer_256_1_bn1"
  param {
    lr_mult: 0.0
  }
  param {
    lr_mult: 0.0
  }
  param {
    lr_mult: 0.0
  }
}
layer {
  name: "layer_256_1_scale1"
  type: "Scale"
  bottom: "layer_256_1_bn1"
  top: "layer_256_1_bn1"
  param {
    lr_mult: 1.0
    decay_mult: 1.0
  }
  param {
    lr_mult: 2.0
    decay_mult: 1.0
  }
  scale_param {
    bias_term: true
  }
}
layer {
  name: "layer_256_1_relu1"
  type: "ReLU"
  bottom: "layer_256_1_bn1"
  top: "layer_256_1_bn1"
}
layer {
  name: "layer_256_1_conv1"
  type: "Convolution"
  bottom: "layer_256_1_bn1"
  top: "layer_256_1_conv1"
  param {
    lr_mult: 1.0
    decay_mult: 1.0
  }
  convolution_param {
    num_output: 256
    bias_term: false
    pad: 1
    kernel_size: 3
    stride: 2
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
      value: 0.0
    }
  }
}
layer {
  name: "layer_256_1_bn2"
  type: "BatchNorm"
  bottom: "layer_256_1_conv1"
  top: "layer_256_1_conv1"
  param {
    lr_mult: 0.0
  }
  param {
    lr_mult: 0.0
  }
  param {
    lr_mult: 0.0
  }
}
layer {
  name: "layer_256_1_scale2"
  type: "Scale"
  bottom: "layer_256_1_conv1"
  top: "layer_256_1_conv1"
  param {
    lr_mult: 1.0
    decay_mult: 1.0
  }
  param {
    lr_mult: 2.0
    decay_mult: 1.0
  }
  scale_param {
    bias_term: true
  }
}
layer {
  name: "layer_256_1_relu2"
  type: "ReLU"
  bottom: "layer_256_1_conv1"
  top: "layer_256_1_conv1"
}
layer {
  name: "layer_256_1_conv2"
  type: "Convolution"
  bottom: "layer_256_1_conv1"
  top: "layer_256_1_conv2"
  param {
    lr_mult: 1.0
    decay_mult: 1.0
  }
  convolution_param {
    num_output: 256
    bias_term: false
    pad: 1
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
      value: 0.0
    }
  }
}
layer {
  name: "layer_256_1_conv_expand"
  type: "Convolution"
  bottom: "layer_256_1_bn1"
  top: "layer_256_1_conv_expand"
  param {
    lr_mult: 1.0
    decay_mult: 1.0
  }
  convolution_param {
    num_output: 256
    bias_term: false
    pad: 0
    kernel_size: 1
    stride: 2
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
      value: 0.0
    }
  }
}
layer {
  name: "layer_256_1_sum"
  type: "Eltwise"
  bottom: "layer_256_1_conv2"
  bottom: "layer_256_1_conv_expand"
  top: "layer_256_1_sum"
}
layer {
  name: "layer_512_1_bn1"
  type: "BatchNorm"
  bottom: "layer_256_1_sum"
  top: "layer_512_1_bn1"
  param {
    lr_mult: 0.0
  }
  param {
    lr_mult: 0.0
  }
  param {
    lr_mult: 0.0
  }
}
layer {
  name: "layer_512_1_scale1"
  type: "Scale"
  bottom: "layer_512_1_bn1"
  top: "layer_512_1_bn1"
  param {
    lr_mult: 1.0
    decay_mult: 1.0
  }
  param {
    lr_mult: 2.0
    decay_mult: 1.0
  }
  scale_param {
    bias_term: true
  }
}
layer {
  name: "layer_512_1_relu1"
  type: "ReLU"
  bottom: "layer_512_1_bn1"
  top: "layer_512_1_bn1"
}
layer {
  name: "layer_512_1_conv1_h"
  type: "Convolution"
  bottom: "layer_512_1_bn1"
  top: "layer_512_1_conv1_h"
  param {
    lr_mult: 1.0
    decay_mult: 1.0
  }
  convolution_param {
    num_output: 128
    bias_term: false
    pad: 1
    kernel_size: 3
    stride: 1 # 2
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
      value: 0.0
    }
  }
}
layer {
  name: "layer_512_1_bn2_h"
  type: "BatchNorm"
  bottom: "layer_512_1_conv1_h"
  top: "layer_512_1_conv1_h"
  param {
    lr_mult: 0.0
  }
  param {
    lr_mult: 0.0
  }
  param {
    lr_mult: 0.0
  }
}
layer {
  name: "layer_512_1_scale2_h"
  type: "Scale"
  bottom: "layer_512_1_conv1_h"
  top: "layer_512_1_conv1_h"
  param {
    lr_mult: 1.0
    decay_mult: 1.0
  }
  param {
    lr_mult: 2.0
    decay_mult: 1.0
  }
  scale_param {
    bias_term: true
  }
}
layer {
  name: "layer_512_1_relu2"
  type: "ReLU"
  bottom: "layer_512_1_conv1_h"
  top: "layer_512_1_conv1_h"
}
layer {
  name: "layer_512_1_conv2_h"
  type: "Convolution"
  bottom: "layer_512_1_conv1_h"
  top: "layer_512_1_conv2_h"
  param {
    lr_mult: 1.0
    decay_mult: 1.0
  }
  convolution_param {
    num_output: 256
    bias_term: false
    pad: 2 # 1
    kernel_size: 3
    stride: 1
    dilation: 2
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
      value: 0.0
    }
  }
}
layer {
  name: "layer_512_1_conv_expand_h"
  type: "Convolution"
  bottom: "layer_512_1_bn1"
  top: "layer_512_1_conv_expand_h"
  param {
    lr_mult: 1.0
    decay_mult: 1.0
  }
  convolution_param {
    num_output: 256
    bias_term: false
    pad: 0
    kernel_size: 1
    stride: 1 # 2
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
      value: 0.0
    }
  }
}
layer {
  name: "layer_512_1_sum"
  type: "Eltwise"
  bottom: "layer_512_1_conv2_h"
  bottom: "layer_512_1_conv_expand_h"
  top: "layer_512_1_sum"
}
layer {
  name: "last_bn_h"
  type: "BatchNorm"
  bottom: "layer_512_1_sum"
  top: "layer_512_1_sum"
  param {
    lr_mult: 0.0
  }
  param {
    lr_mult: 0.0
  }
  param {
    lr_mult: 0.0
  }
}
layer {
  name: "last_scale_h"
  type: "Scale"
  bottom: "layer_512_1_sum"
  top: "layer_512_1_sum"
  param {
    lr_mult: 1.0
    decay_mult: 1.0
  }
  param {
    lr_mult: 2.0
    decay_mult: 1.0
  }
  scale_param {
    bias_term: true
  }
}
layer {
  name: "last_relu"
  type: "ReLU"
  bottom: "layer_512_1_sum"
  top: "fc7"
}

layer {
  name: "conv6_1_h"
  type: "Convolution"
  bottom: "fc7"
  top: "conv6_1_h"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 128
    pad: 0
    kernel_size: 1
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "conv6_1_relu"
  type: "ReLU"
  bottom: "conv6_1_h"
  top: "conv6_1_h"
}
layer {
  name: "conv6_2_h"
  type: "Convolution"
  bottom: "conv6_1_h"
  top: "conv6_2_h"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 256
    pad: 1
    kernel_size: 3
    stride: 2
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "conv6_2_relu"
  type: "ReLU"
  bottom: "conv6_2_h"
  top: "conv6_2_h"
}
layer {
  name: "conv7_1_h"
  type: "Convolution"
  bottom: "conv6_2_h"
  top: "conv7_1_h"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 64
    pad: 0
    kernel_size: 1
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "conv7_1_relu"
  type: "ReLU"
  bottom: "conv7_1_h"
  top: "conv7_1_h"
}
layer {
  name: "conv7_2_h"
  type: "Convolution"
  bottom: "conv7_1_h"
  top: "conv7_2_h"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 128
    pad: 1
    kernel_size: 3
    stride: 2
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "conv7_2_relu"
  type: "ReLU"
  bottom: "conv7_2_h"
  top: "conv7_2_h"
}
layer {
  name: "conv8_1_h"
  type: "Convolution"
  bottom: "conv7_2_h"
  top: "conv8_1_h"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 64
    pad: 0
    kernel_size: 1
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "conv8_1_relu"
  type: "ReLU"
  bottom: "conv8_1_h"
  top: "conv8_1_h"
}
layer {
  name: "conv8_2_h"
  type: "Convolution"
  bottom: "conv8_1_h"
  top: "conv8_2_h"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 128
    pad: 1
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "conv8_2_relu"
  type: "ReLU"
  bottom: "conv8_2_h"
  top: "conv8_2_h"
}
layer {
  name: "conv9_1_h"
  type: "Convolution"
  bottom: "conv8_2_h"
  top: "conv9_1_h"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 64
    pad: 0
    kernel_size: 1
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "conv9_1_relu"
  type: "ReLU"
  bottom: "conv9_1_h"
  top: "conv9_1_h"
}
layer {
  name: "conv9_2_h"
  type: "Convolution"
  bottom: "conv9_1_h"
  top: "conv9_2_h"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 128
    pad: 1
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "conv9_2_relu"
  type: "ReLU"
  bottom: "conv9_2_h"
  top: "conv9_2_h"
}
layer {
  name: "conv4_3_norm"
  type: "Normalize"
  bottom: "layer_256_1_bn1"
  top: "conv4_3_norm"
  norm_param {
    across_spatial: false
    scale_filler {
      type: "constant"
      value: 20
    }
    channel_shared: false
  }
}
layer {
  name: "conv4_3_norm_mbox_loc"
  type: "Convolution"
  bottom: "conv4_3_norm"
  top: "conv4_3_norm_mbox_loc"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 16
    pad: 1
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "conv4_3_norm_mbox_loc_perm"
  type: "Permute"
  bottom: "conv4_3_norm_mbox_loc"
  top: "conv4_3_norm_mbox_loc_perm"
  permute_param {
    order: 0
    order: 2
    order: 3
    order: 1
  }
}
layer {
  name: "conv4_3_norm_mbox_loc_flat"
  type: "Flatten"
  bottom: "conv4_3_norm_mbox_loc_perm"
  top: "conv4_3_norm_mbox_loc_flat"
  flatten_param {
    axis: 1
  }
}
layer {
  name: "conv4_3_norm_mbox_conf"
  type: "Convolution"
  bottom: "conv4_3_norm"
  top: "conv4_3_norm_mbox_conf"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 8 # 84
    pad: 1
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "conv4_3_norm_mbox_conf_perm"
  type: "Permute"
  bottom: "conv4_3_norm_mbox_conf"
  top: "conv4_3_norm_mbox_conf_perm"
  permute_param {
    order: 0
    order: 2
    order: 3
    order: 1
  }
}
layer {
  name: "conv4_3_norm_mbox_conf_flat"
  type: "Flatten"
  bottom: "conv4_3_norm_mbox_conf_perm"
  top: "conv4_3_norm_mbox_conf_flat"
  flatten_param {
    axis: 1
  }
}
layer {
  name: "conv4_3_norm_mbox_priorbox"
  type: "PriorBox"
  bottom: "conv4_3_norm"
  bottom: "data"
  top: "conv4_3_norm_mbox_priorbox"
  prior_box_param {
    min_size: 30.0
    max_size: 60.0
    aspect_ratio: 2
    flip: true
    clip: false
    variance: 0.1
    variance: 0.1
    variance: 0.2
    variance: 0.2
    step: 8
    offset: 0.5
  }
}
layer {
  name: "fc7_mbox_loc"
  type: "Convolution"
  bottom: "fc7"
  top: "fc7_mbox_loc"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 24
    pad: 1
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "fc7_mbox_loc_perm"
  type: "Permute"
  bottom: "fc7_mbox_loc"
  top: "fc7_mbox_loc_perm"
  permute_param {
    order: 0
    order: 2
    order: 3
    order: 1
  }
}
layer {
  name: "fc7_mbox_loc_flat"
  type: "Flatten"
  bottom: "fc7_mbox_loc_perm"
  top: "fc7_mbox_loc_flat"
  flatten_param {
    axis: 1
  }
}
layer {
  name: "fc7_mbox_conf"
  type: "Convolution"
  bottom: "fc7"
  top: "fc7_mbox_conf"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 12 # 126
    pad: 1
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "fc7_mbox_conf_perm"
  type: "Permute"
  bottom: "fc7_mbox_conf"
  top: "fc7_mbox_conf_perm"
  permute_param {
    order: 0
    order: 2
    order: 3
    order: 1
  }
}
layer {
  name: "fc7_mbox_conf_flat"
  type: "Flatten"
  bottom: "fc7_mbox_conf_perm"
  top: "fc7_mbox_conf_flat"
  flatten_param {
    axis: 1
  }
}
layer {
  name: "fc7_mbox_priorbox"
  type: "PriorBox"
  bottom: "fc7"
  bottom: "data"
  top: "fc7_mbox_priorbox"
  prior_box_param {
    min_size: 60.0
    max_size: 111.0
    aspect_ratio: 2
    aspect_ratio: 3
    flip: true
    clip: false
    variance: 0.1
    variance: 0.1
    variance: 0.2
    variance: 0.2
    step: 16
    offset: 0.5
  }
}
layer {
  name: "conv6_2_mbox_loc"
  type: "Convolution"
  bottom: "conv6_2_h"
  top: "conv6_2_mbox_loc"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 24
    pad: 1
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "conv6_2_mbox_loc_perm"
  type: "Permute"
  bottom: "conv6_2_mbox_loc"
  top: "conv6_2_mbox_loc_perm"
  permute_param {
    order: 0
    order: 2
    order: 3
    order: 1
  }
}
layer {
  name: "conv6_2_mbox_loc_flat"
  type: "Flatten"
  bottom: "conv6_2_mbox_loc_perm"
  top: "conv6_2_mbox_loc_flat"
  flatten_param {
    axis: 1
  }
}
layer {
  name: "conv6_2_mbox_conf"
  type: "Convolution"
  bottom: "conv6_2_h"
  top: "conv6_2_mbox_conf"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 12 # 126
    pad: 1
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "conv6_2_mbox_conf_perm"
  type: "Permute"
  bottom: "conv6_2_mbox_conf"
  top: "conv6_2_mbox_conf_perm"
  permute_param {
    order: 0
    order: 2
    order: 3
    order: 1
  }
}
layer {
  name: "conv6_2_mbox_conf_flat"
  type: "Flatten"
  bottom: "conv6_2_mbox_conf_perm"
  top: "conv6_2_mbox_conf_flat"
  flatten_param {
    axis: 1
  }
}
layer {
  name: "conv6_2_mbox_priorbox"
  type: "PriorBox"
  bottom: "conv6_2_h"
  bottom: "data"
  top: "conv6_2_mbox_priorbox"
  prior_box_param {
    min_size: 111.0
    max_size: 162.0
    aspect_ratio: 2
    aspect_ratio: 3
    flip: true
    clip: false
    variance: 0.1
    variance: 0.1
    variance: 0.2
    variance: 0.2
    step: 32
    offset: 0.5
  }
}
layer {
  name: "conv7_2_mbox_loc"
  type: "Convolution"
  bottom: "conv7_2_h"
  top: "conv7_2_mbox_loc"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 24
    pad: 1
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "conv7_2_mbox_loc_perm"
  type: "Permute"
  bottom: "conv7_2_mbox_loc"
  top: "conv7_2_mbox_loc_perm"
  permute_param {
    order: 0
    order: 2
    order: 3
    order: 1
  }
}
layer {
  name: "conv7_2_mbox_loc_flat"
  type: "Flatten"
  bottom: "conv7_2_mbox_loc_perm"
  top: "conv7_2_mbox_loc_flat"
  flatten_param {
    axis: 1
  }
}
layer {
  name: "conv7_2_mbox_conf"
  type: "Convolution"
  bottom: "conv7_2_h"
  top: "conv7_2_mbox_conf"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 12 # 126
    pad: 1
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "conv7_2_mbox_conf_perm"
  type: "Permute"
  bottom: "conv7_2_mbox_conf"
  top: "conv7_2_mbox_conf_perm"
  permute_param {
    order: 0
    order: 2
    order: 3
    order: 1
  }
}
layer {
  name: "conv7_2_mbox_conf_flat"
  type: "Flatten"
  bottom: "conv7_2_mbox_conf_perm"
  top: "conv7_2_mbox_conf_flat"
  flatten_param {
    axis: 1
  }
}
layer {
  name: "conv7_2_mbox_priorbox"
  type: "PriorBox"
  bottom: "conv7_2_h"
  bottom: "data"
  top: "conv7_2_mbox_priorbox"
  prior_box_param {
    min_size: 162.0
    max_size: 213.0
    aspect_ratio: 2
    aspect_ratio: 3
    flip: true
    clip: false
    variance: 0.1
    variance: 0.1
    variance: 0.2
    variance: 0.2
    step: 64
    offset: 0.5
  }
}
layer {
  name: "conv8_2_mbox_loc"
  type: "Convolution"
  bottom: "conv8_2_h"
  top: "conv8_2_mbox_loc"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 16
    pad: 1
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "conv8_2_mbox_loc_perm"
  type: "Permute"
  bottom: "conv8_2_mbox_loc"
  top: "conv8_2_mbox_loc_perm"
  permute_param {
    order: 0
    order: 2
    order: 3
    order: 1
  }
}
layer {
  name: "conv8_2_mbox_loc_flat"
  type: "Flatten"
  bottom: "conv8_2_mbox_loc_perm"
  top: "conv8_2_mbox_loc_flat"
  flatten_param {
    axis: 1
  }
}
layer {
  name: "conv8_2_mbox_conf"
  type: "Convolution"
  bottom: "conv8_2_h"
  top: "conv8_2_mbox_conf"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 8 # 84
    pad: 1
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "conv8_2_mbox_conf_perm"
  type: "Permute"
  bottom: "conv8_2_mbox_conf"
  top: "conv8_2_mbox_conf_perm"
  permute_param {
    order: 0
    order: 2
    order: 3
    order: 1
  }
}
layer {
  name: "conv8_2_mbox_conf_flat"
  type: "Flatten"
  bottom: "conv8_2_mbox_conf_perm"
  top: "conv8_2_mbox_conf_flat"
  flatten_param {
    axis: 1
  }
}
layer {
  name: "conv8_2_mbox_priorbox"
  type: "PriorBox"
  bottom: "conv8_2_h"
  bottom: "data"
  top: "conv8_2_mbox_priorbox"
  prior_box_param {
    min_size: 213.0
    max_size: 264.0
    aspect_ratio: 2
    flip: true
    clip: false
    variance: 0.1
    variance: 0.1
    variance: 0.2
    variance: 0.2
    step: 100
    offset: 0.5
  }
}
layer {
  name: "conv9_2_mbox_loc"
  type: "Convolution"
  bottom: "conv9_2_h"
  top: "conv9_2_mbox_loc"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 16
    pad: 1
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "conv9_2_mbox_loc_perm"
  type: "Permute"
  bottom: "conv9_2_mbox_loc"
  top: "conv9_2_mbox_loc_perm"
  permute_param {
    order: 0
    order: 2
    order: 3
    order: 1
  }
}
layer {
  name: "conv9_2_mbox_loc_flat"
  type: "Flatten"
  bottom: "conv9_2_mbox_loc_perm"
  top: "conv9_2_mbox_loc_flat"
  flatten_param {
    axis: 1
  }
}
layer {
  name: "conv9_2_mbox_conf"
  type: "Convolution"
  bottom: "conv9_2_h"
  top: "conv9_2_mbox_conf"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 8 # 84
    pad: 1
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "conv9_2_mbox_conf_perm"
  type: "Permute"
  bottom: "conv9_2_mbox_conf"
  top: "conv9_2_mbox_conf_perm"
  permute_param {
    order: 0
    order: 2
    order: 3
    order: 1
  }
}
layer {
  name: "conv9_2_mbox_conf_flat"
  type: "Flatten"
  bottom: "conv9_2_mbox_conf_perm"
  top: "conv9_2_mbox_conf_flat"
  flatten_param {
    axis: 1
  }
}
layer {
  name: "conv9_2_mbox_priorbox"
  type: "PriorBox"
  bottom: "conv9_2_h"
  bottom: "data"
  top: "conv9_2_mbox_priorbox"
  prior_box_param {
    min_size: 264.0
    max_size: 315.0
    aspect_ratio: 2
    flip: true
    clip: false
    variance: 0.1
    variance: 0.1
    variance: 0.2
    variance: 0.2
    step: 300
    offset: 0.5
  }
}
layer {
  name: "mbox_loc"
  type: "Concat"
  bottom: "conv4_3_norm_mbox_loc_flat"
  bottom: "fc7_mbox_loc_flat"
  bottom: "conv6_2_mbox_loc_flat"
  bottom: "conv7_2_mbox_loc_flat"
  bottom: "conv8_2_mbox_loc_flat"
  bottom: "conv9_2_mbox_loc_flat"
  top: "mbox_loc"
  concat_param {
    axis: 1
  }
}
layer {
  name: "mbox_conf"
  type: "Concat"
  bottom: "conv4_3_norm_mbox_conf_flat"
  bottom: "fc7_mbox_conf_flat"
  bottom: "conv6_2_mbox_conf_flat"
  bottom: "conv7_2_mbox_conf_flat"
  bottom: "conv8_2_mbox_conf_flat"
  bottom: "conv9_2_mbox_conf_flat"
  top: "mbox_conf"
  concat_param {
    axis: 1
  }
}
layer {
  name: "mbox_priorbox"
  type: "Concat"
  bottom: "conv4_3_norm_mbox_priorbox"
  bottom: "fc7_mbox_priorbox"
  bottom: "conv6_2_mbox_priorbox"
  bottom: "conv7_2_mbox_priorbox"
  bottom: "conv8_2_mbox_priorbox"
  bottom: "conv9_2_mbox_priorbox"
  top: "mbox_priorbox"
  concat_param {
    axis: 2
  }
}

layer {
  name: "mbox_conf_reshape"
  type: "Reshape"
  bottom: "mbox_conf"
  top: "mbox_conf_reshape"
  reshape_param {
    shape {
      dim: 0
      dim: -1
      dim: 2
    }
  }
}
layer {
  name: "mbox_conf_softmax"
  type: "Softmax"
  bottom: "mbox_conf_reshape"
  top: "mbox_conf_softmax"
  softmax_param {
    axis: 2
  }
}
layer {
  name: "mbox_conf_flatten"
  type: "Flatten"
  bottom: "mbox_conf_softmax"
  top: "mbox_conf_flatten"
  flatten_param {
    axis: 1
  }
}

layer {
  name: "detection_out"
  type: "DetectionOutput"
  bottom: "mbox_loc"
  bottom: "mbox_conf_flatten"
  bottom: "mbox_priorbox"
  top: "detection_out"
  include {
    phase: TEST
  }
  detection_output_param {
    num_classes: 2
    share_location: true
    background_label_id: 0
    nms_param {
      nms_threshold: 0.45
      top_k: 400
    }
    code_type: CENTER_SIZE
    keep_top_k: 200
    confidence_threshold: 0.01
  }
}
