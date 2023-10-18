// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../../precomp.hpp"
#include "fast_norm.hpp"

namespace cv { namespace dnn {

//  Mean-Variance-Normalization speedup by multi-threading.
void fastNorm(const Mat &input, Mat &output, float epsilon, int normalized_axis, bool normalize_variance) {
    const auto input_shape = shape(input);
    CV_CheckGE(normalized_axis, 0, "fastNorm: axis out of range");
    CV_CheckLT(normalized_axis, static_cast<int>(input_shape.size()), "fastNorm: axis out of range");

    size_t loops = static_cast<size_t>(total(input_shape, 0, normalized_axis)),
           norm_size = static_cast<size_t>(total(input_shape, normalized_axis));
    float inv_norm_size = 1.0 / norm_size;

    auto fn = [&](const Range &r) {
        const auto *input_data = input.ptr<const float>();
        auto *output_data = output.ptr<float>();
        for (int i = r.start; i < r.end; i++) {
            const auto *x = input_data + norm_size * i;
            auto *y = output_data + norm_size * i;

            float mean = 0.f, mean_square = 0.f;
            for (int j = 0; j < norm_size; j++) {
                float v = x[j];
                mean += v;
                mean_square += v * v;
            }

            mean *= inv_norm_size;
            mean_square = std::sqrt(std::max(0.f, mean_square * inv_norm_size - mean * mean) + epsilon);
            float inv_stdev = normalize_variance ? 1.f / mean_square : 1.f;

            for (size_t j = 0; j < norm_size; j++) {
                y[j] = (x[j] - mean) * inv_stdev;
            }
        }
    };
    double nstripes = loops * norm_size * (1 / 1024.0);
    parallel_for_(Range(0, loops), fn, nstripes);
}

// LayerNormalization speedup by multi-threading. Bias is absent.
void fastNorm(const Mat &input, const Mat &scale, Mat &output, float epsilon, int normalized_axis) {
    const auto input_shape = shape(input);
    CV_CheckGE(normalized_axis, 0, "fastNorm: axis out of range");
    CV_CheckLT(normalized_axis, static_cast<int>(input_shape.size()), "fastNorm: axis out of range");

    size_t loops = static_cast<size_t>(total(input_shape, 0, normalized_axis)),
           norm_size = static_cast<size_t>(total(input_shape, normalized_axis));
    float inv_norm_size = 1.0 / norm_size;

    auto fn = [&](const Range &r) {
        const auto *input_data = input.ptr<const float>();
        const auto *scale_data = scale.ptr<const float>();
        auto *output_data = output.ptr<float>();
        for (int i = r.start; i < r.end; i++) {
            const auto *x = input_data + norm_size * i;
            auto *y = output_data + norm_size * i;

            float mean = 0.f, mean_square = 0.f;
            for (int j = 0; j < norm_size; j++) {
                float v = x[j];
                mean += v;
                mean_square += v * v;
            }

            mean *= inv_norm_size;
            mean_square = std::sqrt(std::max(0.f, mean_square * inv_norm_size - mean * mean) + epsilon);
            float inv_stdev = 1.f / mean_square;

            for (size_t j = 0; j < norm_size; j++) {
                y[j] = scale_data[j] * (x[j] - mean) * inv_stdev;
            }
        }
    };
    double nstripes = loops * norm_size * (1 / 1024.0);
    parallel_for_(Range(0, loops), fn, nstripes);
}

/*  Normalization speedup by mutli-threading. It supports

    1. LayerNorm that has scale/bias of shape input.shape[axis:]
    2. Channel-wise scale/bias. It needs to be broadcast to shape (C)+input.shape[axis:]

*/
void fastNorm(const Mat &input, const Mat &scale, const Mat &bias, Mat &output, float epsilon, int normalized_axis) {
    const auto input_shape = shape(input);
    CV_CheckGE(normalized_axis, 0, "fastNorm: axis out of range");
    CV_CheckLT(normalized_axis, static_cast<int>(input_shape.size()), "fastNorm: axis out of range");
    CV_CheckEQ(scale.total(), bias.total(), "fastNorm: scale and bias should have the same shape");

    size_t loops = static_cast<size_t>(total(input_shape, 0, normalized_axis)),
           norm_size = static_cast<size_t>(total(input_shape, normalized_axis)),
           scale_total = scale.total();
    float inv_norm_size = 1.0 / norm_size;

    auto fn = [&](const Range &r) {
        const auto *input_data = input.ptr<const float>();
        const auto *scale_data = scale.ptr<const float>();
        const auto *bias_data = bias.ptr<const float>();
        auto *output_data = output.ptr<float>();
        for (int i = r.start; i < r.end; i++) {
            const auto *x = input_data + norm_size * i;
            const auto *s = scale_data + (norm_size * i) % scale_total;
            const auto *b = bias_data + (norm_size * i) % scale_total;
            auto *y = output_data + norm_size * i;

            float mean = 0.f, mean_square = 0.f;
            for (int j = 0; j < norm_size; j++) {
                float v = x[j];
                mean += v;
                mean_square += v * v;
            }

            mean *= inv_norm_size;
            mean_square = std::sqrt(std::max(0.f, mean_square * inv_norm_size - mean * mean) + epsilon);
            float inv_stdev = 1.f / mean_square;

            for (size_t j = 0; j < norm_size; j++) {
                y[j] = s[j] * (x[j] - mean) * inv_stdev + b[j];
            }
        }
    };
    double nstripes = loops * norm_size * (1 / 1024.0);
    parallel_for_(Range(0, loops), fn, nstripes);
}

}} // cv::dnn
