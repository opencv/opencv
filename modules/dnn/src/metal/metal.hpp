#ifndef OPENCV_DNN_METAL_HPP
#define OPENCV_DNN_METAL_HPP

#include <opencv2/core/cvdef.h>

#include <string>

#include "ops/batch_norm.hpp"
#include "ops/blank.hpp"
#include "ops/concat.hpp"
#include "ops/const.hpp"
#include "ops/convolution.hpp"
#include "ops/deconvolution.hpp"
#include "ops/depth_space_ops.hpp"
#include "ops/elementwise.hpp"
#include "ops/eltwise.hpp"
#include "ops/flatten.hpp"
#include "ops/fully_connected.hpp"
#include "ops/gemm.hpp"
#include "ops/instance_norm.hpp"
#include "ops/layer_norm.hpp"
#include "ops/lrn.hpp"
#include "ops/matmul.hpp"
#include "ops/nary_eltwise.hpp"
#include "ops/padding.hpp"
#include "ops/permute.hpp"
#include "ops/pooling.hpp"
#include "ops/reduce.hpp"
#include "ops/reshape.hpp"
#include "ops/resize.hpp"
#include "ops/scale.hpp"
#include "ops/slice.hpp"
#include "ops/softmax.hpp"
#include "ops/topk.hpp"

namespace cv { namespace dnn { namespace metal {

CV_EXPORTS bool isAvailable() noexcept;
CV_EXPORTS bool startCapture(const std::string& path);
CV_EXPORTS void stopCapture();

}}}  // namespace cv::dnn::metal

#endif  // OPENCV_DNN_METAL_HPP
