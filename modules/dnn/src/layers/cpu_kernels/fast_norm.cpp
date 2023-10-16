// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../../precomp.hpp"
#include "fast_norm.hpp"

namespace cv { namespace dnn {

// // Assume axes has been normalized
// static std::vector<int> getTransposePermutation(const MatShape &input_shape, const std::vector<int> normed_axes) {
//     std::vector<int> perm{};
//     bool need_trans = false;
//     for (size_t i = 0, num_axes = normed_axes.size(), num_dims = input_shape.size(); i < num_axes; i++) {
//         if (normed_axes[i] != num_dims - num_axes + i) {
//             need_trans = true;
//             break;
//         }
//     }

//     if (need_trans) {
//         perm.clear();
//         auto normed_axis = normed_axes.begin();
//         for (size_t axis = 0, num_dims = input_shape.size(); axis < num_dims; axis++) {
//             if (normed_axis != normed_axes.end() && axis == *normed_axis) {
//                 ++normed_axis;
//             } else {
//                 perm.push_back(static_cast<int>(axis));
//             }
//         }
//         perm.insert(perm.end(), normed_axes.begin(), normed_axes.end());
//     }

//     return perm;
// }

// static std::vector<int> getInvertedTransposePermutation(const std::vector<int> perm) {
//     std::vector<int> inverted_perm(perm.size());
//     for (size_t i = 0; i < perm.size(); i++) {
//         inverted_perm[perm[i]] = i;
//     }
//     return inverted_perm;
// }

/*

| Norm          | input | scale    | bias               | axis         |
| ------------- | ----- | -------- | ------------------ | ------------ |
| layer norm    | d>=n  | d[axis:] | optional, d[axis:] | -1, [-d, d]  |
| instance norm | d>=3  | d[axis]  | d[axis]            | 1            |
| mvn           | d>=4  | N/A      | N/A                | [0, 2, 3]    |

// onnx layer norm: https://github.com/onnx/onnx/blob/main/docs/Operators.md#LayerNormalization
// onnx instance norm: https://github.com/onnx/onnx/blob/main/docs/Operators.md#InstanceNormalization
// onnx mvn: https://github.com/onnx/onnx/blob/main/docs/Operators.md#MeanVarianceNormalization

*/
void fastNorm(const Mat &input, const Mat &scale, const Mat &bias, Mat &output, float epsilon, int axis) {
    // const auto input_shape = shape(input);

    // // Normalize axes
    // std::vector<int> normed_axes;
    // std::transform(axes.begin(), axes.end(), std::back_inserter(normed_axes),
    //                [input_shape] (int axis) { return normalize_axis(axis, static_cast<int>(input_shape.size())); });

    // if (axes.size() > static_cast<size_t>(1)) { // mvn
    //     std::vector<int> transpose_permutation = getTransposePermutation(input_shape, normed_axes);
    //     int axis = static_cast<int>(input_shape.size() - normed_axes.size());

    //     if (!transpose_permutation.empty()) {
    //         Mat transposed_input;
    //         cv::transposeND(input, transpose_permutation, transposed_input);

    //         Mat transposed_output(input_shape, CV_32F);
    //         fastMVNKernel(input, scale, bias, transposed_output, epsilon, axis);

    //         auto inverted_perm = getInvertedTransposePermutation(transpose_permutation);
    //         cv::transposeND(transposed_output, inverted_perm, output);

    //         return;
    //     } else {
    //         fastMVNKernel(input, scale, bias, output, epsilon, axis);
    //     }
    // }


    // TODO: check shape?

    const auto input_shape = shape(input);

    size_t loops = std::accumulate(input_shape.begin() + axis + 1, input_shape.end(), static_cast<size_t>(1), std::multiplies<size_t>()),
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

            for (size_t j = 0; j < norm_size; j++) {
                y[j] = scale_data[j] * (x[j] - mean) * inv_stdev + bias_data[j];
            }
        }
    };
    double nstripes = loops * norm_size * (1 / 1024.0);
    parallel_for_(Range(0, loops), fn, nstripes);
}

}} // cv::dnn
