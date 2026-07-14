# How to schedule your network for Halide backend {#tutorial_dnn_halide_scheduling}

@tableofcontents

@prev_tutorial{tutorial_dnn_halide}
@next_tutorial{tutorial_dnn_openvino}

|    |    |
| -: | :- |
| Original author | Dmitry Kurtaev |
| Compatibility | OpenCV >= 3.3 |

## Introduction
Halide code is the same for every device we use. But for achieving the satisfied
efficiency we should schedule computations properly. In this tutorial we describe
the ways to schedule your networks using Halide backend in OpenCV deep learning module.

For better understanding of Halide scheduling you might want to read tutorials @ http://halide-lang.org/tutorials.

If it's your first meeting with Halide in OpenCV, we recommend to start from @ref tutorial_dnn_halide.

## Configuration files
You can schedule computations of Halide pipeline by writing textual configuration files.
It means that you can easily vectorize, parallelize and manage loops order of
layers computation. Pass path to file with scheduling directives for specific
device into ```cv::dnn::Net::setHalideScheduler``` before the first ```cv::dnn::Net::forward``` call.

Scheduling configuration files represented as YAML files where each node is a
scheduled function or a scheduling directive.
@code
relu1:
  reorder: [x, c, y]
  split: { y: 2, c: 8 }
  parallel: [yo, co]
  unroll: yi
  vectorize: { x: 4 }
conv1_constant_exterior:
  compute_at: { relu1: yi }
@endcode

Considered use variables `n` for batch dimension, `c` for channels,
`y` for rows and `x` for columns. For variables after split are used names
with the same prefix but `o` and `i` suffixes for outer and inner variables
correspondingly. In example, for variable `x` in range `[0, 10)` directive
`split: { x: 2 }` gives new ones `xo` in range `[0, 5)` and `xi` in range `[0, 2)`.
Variable name `x` is no longer available in the same scheduling node.

You can find scheduling examples at [opencv_extra/testdata/dnn](https://github.com/opencv/opencv_extra/tree/4.x/testdata/dnn)
and use it for schedule your networks.

## Layers fusing
Thanks to layers fusing we can schedule only the top layers of fused sets.
Because for every output value we use the fused formula.
In example, if you have three layers Convolution + Scale + ReLU one by one,
@code
conv(x, y, c, n) = sum(...) + bias(c);
scale(x, y, c, n) = conv(x, y, c, n) * weights(c);
relu(x, y, c, n) = max(scale(x, y, c, n), 0);
@endcode

fused function is something like
@code
relu(x, y, c, n) = max((sum(...) + bias(c)) * weights(c), 0);
@endcode

So only function called `relu` require scheduling.

## Scheduling patterns
Sometimes networks built using blocked structure that means some layer are
identical or quite similar. If you want to apply the same scheduling for
different layers accurate to tiling or vectorization factors, define scheduling
patterns in section `patterns` at the beginning of scheduling file.
Also, your patterns may use some parametric variables.
@code
# At the beginning of the file
patterns:
  fully_connected:
    split: { c: c_split }
    fuse: { src: [x, y, co], dst: block }
    parallel: block
    vectorize: { ci: c_split }
# Somewhere below
fc8:
  pattern: fully_connected
  params: { c_split: 8 }
@endcode

## Automatic scheduling
You can let DNN to schedule layers automatically. Just skip call of ```cv::dnn::Net::setHalideScheduler```. Sometimes it might be even more efficient than manual scheduling.
But if specific layers require be scheduled manually, you would be able to
mix both manual and automatic scheduling ways. Write scheduling file
and skip layers that you want to be scheduled automatically.
