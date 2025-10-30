// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../../precomp.hpp"
#include "fast_norm.hpp"

namespace cv { namespace dnn {

void fastNorm(const Mat &input, Mat &output, float epsilon, size_t normalized_axis, bool normalize_variance) {
    const auto input_shape = shape(input);
    CV_CheckLT(normalized_axis, input_shape.size(), "fastNorm: axis out of range");

    size_t loops = static_cast<size_t>(total(input_shape, 0, static_cast<int>(normalized_axis))),
           norm_size = static_cast<size_t>(total(input_shape, static_cast<int>(normalized_axis)));
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

void fastNorm(const Mat &input, const Mat &scale, Mat &output, float epsilon, size_t normalized_axis) {
    const auto input_shape = shape(input);
    CV_CheckLT(normalized_axis, input_shape.size(), "fastNorm: axis out of range");

    size_t loops = static_cast<size_t>(total(input_shape, 0, static_cast<int>(normalized_axis))),
           norm_size = static_cast<size_t>(total(input_shape, static_cast<int>(normalized_axis)));
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

void fastNorm(const Mat &input, const Mat &scale, const Mat &bias, Mat &output, float epsilon, size_t normalized_axis) {
    const auto input_shape = shape(input);
    CV_CheckLT(normalized_axis, input_shape.size(), "fastNorm: axis out of range");
    CV_CheckEQ(scale.total(), bias.total(), "fastNorm: scale and bias should have the same shape");

    size_t loops = static_cast<size_t>(total(input_shape, 0, static_cast<int>(normalized_axis))),
           norm_size = static_cast<size_t>(total(input_shape, static_cast<int>(normalized_axis)));
    float inv_norm_size = 1.0 / norm_size;

    auto fn = [&](const Range &r) {
        const auto *input_data = input.ptr<const float>();
        const auto *scale_data = scale.ptr<const float>();
        const auto *bias_data = bias.ptr<const float>();
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
                y[j] = scale_data[j] * (x[j] - mean) * inv_stdev + bias_data[j];
            }
        }
    };
    double nstripes = loops * norm_size * (1 / 1024.0);
    parallel_for_(Range(0, loops), fn, nstripes);
}

void fastNormChannel(const Mat &input, const Mat &scale, const Mat &bias, Mat &output, float epsilon) {
    const auto input_shape = shape(input);
    size_t N = input_shape[0], C = input_shape[1];
    CV_CheckEQ(scale.total(), bias.total(), "fastNormChannel: scale and bias should have the same shape");
    CV_CheckEQ(scale.total(), C, "fastNormChannel: scale should be a 1d tensor and match the channel of input");
    CV_CheckGE(input.dims, 3, "fastNormChannel: input dimension >= 3");

    size_t loops = N * C,
           norm_size = static_cast<size_t>(total(input_shape, 2));
    float inv_norm_size = 1.0 / norm_size;

    auto fn = [&](const Range &r) {
        const auto *input_data = input.ptr<const float>();
        const auto *scale_data = scale.ptr<const float>();
        const auto *bias_data = bias.ptr<const float>();
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

            size_t c = i % C;
            float s = scale_data[c] * inv_stdev, b = bias_data[c];
            for (size_t j = 0; j < norm_size; j++) {
                y[j] = s * (x[j] - mean) + b;
            }
        }
    };
    double nstripes = loops * norm_size * (1 / 1024.0);
    parallel_for_(Range(0, loops), fn, nstripes);
}

void fastNormGroup(const Mat &input, const Mat &scale, const Mat &bias, Mat &output, float epsilon, size_t num_groups) {
    const auto input_shape = shape(input);
    size_t N = input_shape[0], C = input_shape[1];
    CV_CheckEQ(scale.total(), bias.total(), "fastNormGroup: scale and bias should have the same shape");
    CV_CheckEQ(scale.total(), C, "fastNormGroup: scale should be a 1d tensor and match the channel of input");
    CV_CheckGE(input.dims, 3, "fastNormGroup: input dimension >= 3");

    size_t channels_per_group = C / num_groups;
    size_t loops = N * num_groups;
    size_t norm_size = static_cast<size_t>(total(input_shape, 2) * channels_per_group);
    size_t step = norm_size / channels_per_group;
    float inv_norm_size = 1.0 / norm_size;

    auto fn = [&](const Range &r) {
        const auto *input_data = input.ptr<const float>();
        const auto *scale_data = scale.ptr<const float>();
        const auto *bias_data = bias.ptr<const float>();
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

            size_t group_idx = i % num_groups * channels_per_group;
            for (size_t j = 0; j < norm_size; j++) {
                size_t c = group_idx + (j / step);
                float s = scale_data[c] * inv_stdev, b = bias_data[c];
                y[j] = s * (x[j] - mean) + b;
            }
        }
    };

    double nstripes = loops * norm_size * (1 / 1024.0);
    parallel_for_(Range(0, loops), fn, nstripes);
}

}} // cv::dnn
