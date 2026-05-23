// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../../precomp.hpp"
#include "fast_norm.hpp"
#include <opencv2/core/hal/intrin.hpp>

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

void fastNormMeanInvStdDev(const Mat& input, Mat& mean, Mat& invStdDev, float epsilon, size_t normalized_axis)
{
    CV_Assert(input.type() == CV_32F);
    CV_Assert(mean.type() == CV_32F);
    CV_Assert(invStdDev.type() == CV_32F);
    CV_Assert(input.isContinuous() && mean.isContinuous() && invStdDev.isContinuous());

    const auto input_shape = shape(input);
    CV_CheckLT(normalized_axis, input_shape.size(), "fastNormMeanInvStdDev: axis out of range");

    const size_t loops = static_cast<size_t>(total(input_shape, 0, static_cast<int>(normalized_axis)));
    const size_t norm_size = static_cast<size_t>(total(input_shape, static_cast<int>(normalized_axis)));
    const float inv_norm_size = 1.0f / (float)norm_size;

    CV_CheckEQ((size_t)mean.total(), loops, "fastNormMeanInvStdDev: mean output size mismatch");
    CV_CheckEQ((size_t)invStdDev.total(), loops, "fastNormMeanInvStdDev: invStdDev output size mismatch");

    auto fn = [&](const Range& r) {
        const float* input_data = input.ptr<float>();
        float* mean_data = mean.ptr<float>();
        float* invstd_data = invStdDev.ptr<float>();
        for (int i = r.start; i < r.end; ++i)
        {
            const float* x = input_data + norm_size * (size_t)i;
            float m = 0.f, mean_square = 0.f;
            for (size_t j = 0; j < norm_size; ++j)
            {
                float v = x[j];
                m += v;
                mean_square += v * v;
            }
            m *= inv_norm_size;
            const float var = std::max(0.f, mean_square * inv_norm_size - m * m);
            const float stdev = std::sqrt(var + epsilon);
            mean_data[i] = m;
            invstd_data[i] = 1.f / stdev;
        }
    };

    const double nstripes = loops * norm_size * (1 / 1024.0);
    parallel_for_(Range(0, (int)loops), fn, nstripes);
}

void fastNorm(const Mat &input, const Mat &scale, Mat &output, float epsilon, size_t normalized_axis, bool recenter) {
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
                if (recenter)
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
    size_t C = input_shape.layout == DATA_LAYOUT_BLOCK ? (size_t)input_shape.C : input_shape[1];
    CV_CheckEQ(scale.total(), C, "fastNormChannel: scale should be a 1d tensor and match the channel of input");
    CV_CheckEQ(bias.total(), C, "fastNormChannel: bias should be a 1d tensor and match the channel of input");

    if (input_shape.layout == DATA_LAYOUT_BLOCK) {
        CV_Assert(input.dims == 5 && output.dims == 5);
        CV_Assert(input.type() == CV_32F && output.type() == CV_32F);
        CV_Assert(input.isContinuous() && output.isContinuous());

        const int N  = input.size[0];
        const int C1 = input.size[1];
        const int H  = input.size[2];
        const int W  = input.size[3];
        const int C0 = input.size[4];
        const int Ci = (int)C;

        const float* scale_data = scale.ptr<float>();
        const float* bias_data  = bias.ptr<float>();

        const size_t inStep0 = input.step.p[0] / sizeof(float);
        const size_t inStep1 = input.step.p[1] / sizeof(float);
        const size_t inStep2 = input.step.p[2] / sizeof(float);
        const size_t inStep3 = input.step.p[3] / sizeof(float);

        const size_t outStep0 = output.step.p[0] / sizeof(float);
        const size_t outStep1 = output.step.p[1] / sizeof(float);
        const size_t outStep2 = output.step.p[2] / sizeof(float);
        const size_t outStep3 = output.step.p[3] / sizeof(float);

        const size_t norm_size = (size_t)H * (size_t)W;
        const float inv_norm_size = 1.f / (float)norm_size;

#if CV_SIMD
        const int VEC_SZ = VTraits<v_float32>::vlanes();
#endif

        parallel_for_(Range(0, N * C1), [&](const Range& r) {
            const float* inptr0 = (const float*)input.data;
            float* outptr0 = (float*)output.data;

            AutoBuffer<float> buf(C0 * 4);
            float* sum   = buf.data();
            float* sqsum = sum + C0;
            float* alpha = sqsum + C0;
            float* beta  = alpha + C0;

            for (int i = r.start; i < r.end; ++i) {
                int n  = i / C1;
                int c1 = i - n * C1;
                int cbase = c1 * C0;
                int validC0 = std::min(C0, std::max(0, Ci - cbase));

                const float* inbase  = inptr0  + n * inStep0 + c1 * inStep1;
                float*       outbase = outptr0 + n * outStep0 + c1 * outStep1;

                int c0 = 0;
#if CV_SIMD
                for (; c0 <= validC0 - VEC_SZ; c0 += VEC_SZ) {
                    v_float32 vsum = vx_setzero_f32();
                    v_float32 vsqsum = vx_setzero_f32();
                    for (int h = 0; h < H; ++h) {
                        const float* inrow = inbase + h * inStep2;
                        for (int w = 0; w < W; ++w) {
                            v_float32 v = vx_load(inrow + w * inStep3 + c0);
                            vsum = v_add(vsum, v);
                            vsqsum = v_fma(v, v, vsqsum);
                        }
                    }
                    vx_store(sum + c0, vsum);
                    vx_store(sqsum + c0, vsqsum);
                }
#endif
                for (; c0 < validC0; ++c0) {
                    float s = 0.f, sq = 0.f;
                    for (int h = 0; h < H; ++h) {
                        const float* inrow = inbase + h * inStep2;
                        for (int w = 0; w < W; ++w) {
                            float v = inrow[w * inStep3 + c0];
                            s += v;
                            sq += v * v;
                        }
                    }
                    sum[c0] = s;
                    sqsum[c0] = sq;
                }

                for (int c = 0; c < validC0; ++c) {
                    float mean = sum[c] * inv_norm_size;
                    float var = std::max(0.f, sqsum[c] * inv_norm_size - mean * mean);
                    float inv_stdev = 1.f / std::sqrt(var + epsilon);
                    alpha[c] = scale_data[cbase + c] * inv_stdev;
                    beta[c]  = bias_data[cbase + c] - alpha[c] * mean;
                }

                c0 = 0;
#if CV_SIMD
                for (; c0 <= validC0 - VEC_SZ; c0 += VEC_SZ) {
                    v_float32 va = vx_load(alpha + c0);
                    v_float32 vb = vx_load(beta + c0);
                    for (int h = 0; h < H; ++h) {
                        const float* inrow  = inbase + h * inStep2;
                        float*       outrow = outbase + h * outStep2;
                        for (int w = 0; w < W; ++w) {
                            v_float32 vin = vx_load(inrow + w * inStep3 + c0);
                            vx_store(outrow + w * outStep3 + c0, v_fma(vin, va, vb));
                        }
                    }
                }
#endif
                for (; c0 < validC0; ++c0) {
                    float a = alpha[c0], b = beta[c0];
                    for (int h = 0; h < H; ++h) {
                        const float* inrow  = inbase + h * inStep2;
                        float*       outrow = outbase + h * outStep2;
                        for (int w = 0; w < W; ++w)
                            outrow[w * outStep3 + c0] = inrow[w * inStep3 + c0] * a + b;
                    }
                }

                int c0_pad = validC0;
#if CV_SIMD
                for (; c0_pad <= C0 - VEC_SZ; c0_pad += VEC_SZ) {
                    v_float32 vzero = vx_setzero_f32();
                    for (int h = 0; h < H; ++h) {
                        float* outrow = outbase + h * outStep2;
                        for (int w = 0; w < W; ++w)
                            vx_store(outrow + w * outStep3 + c0_pad, vzero);
                    }
                }
#endif
                for (; c0_pad < C0; ++c0_pad)
                    for (int h = 0; h < H; ++h) {
                        float* outrow = outbase + h * outStep2;
                        for (int w = 0; w < W; ++w)
                            outrow[w * outStep3 + c0_pad] = 0.f;
                    }
            }
        });
        return;
    }

    size_t N = input_shape[0];
    CV_CheckGE(input.dims, 3, "fastNormChannel: input dimension >= 3");

    size_t loops = N * C,
           norm_size = static_cast<size_t>(total(input_shape, 2));

    auto fn = [&](const Range &r) {
        const auto *input_data = input.ptr<const float>();
        const auto *scale_data = scale.ptr<const float>();
        const auto *bias_data = bias.ptr<const float>();
        auto *output_data = output.ptr<float>();
        for (int i = r.start; i < r.end; i++) {
            const auto *x = input_data + norm_size * i;
            auto *y = output_data + norm_size * i;

            double dmean = 0., dmean_sq = 0.;
            for (size_t j = 0; j < norm_size; j++) {
                double v = (double)x[j];
                dmean += v;
                dmean_sq += v * v;
            }

            float mean = (float)(dmean / norm_size);
            float var = (float)std::max(0., dmean_sq / norm_size - (double)mean * (double)mean);
            float inv_stdev = 1.f / std::sqrt(var + epsilon);

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
    size_t C = input_shape.layout == DATA_LAYOUT_BLOCK ? (size_t)input_shape.C : input_shape[1];
    CV_CheckEQ(scale.total(), bias.total(), "fastNormGroup: scale and bias should have the same shape");
    CV_CheckEQ(scale.total(), C, "fastNormGroup: scale should be a 1d tensor and match the channel of input");

    // Block layout path: [N, C1, H, W, C0]
    if (input_shape.layout == DATA_LAYOUT_BLOCK) {
        CV_Assert(input.dims == 5 && output.dims == 5);
        CV_Assert(input.type() == CV_32F && output.type() == CV_32F);
        CV_Assert(input.isContinuous() && output.isContinuous());

        const int N  = input.size[0];
        const int H  = input.size[2];
        const int W  = input.size[3];
        const int C0 = input.size[4];
        const int Ci = (int)C;

        const float* scale_data = scale.ptr<float>();
        const float* bias_data  = bias.ptr<float>();

        const size_t inStep0 = input.step.p[0] / sizeof(float);
        const size_t inStep1 = input.step.p[1] / sizeof(float);
        const size_t inStep2 = input.step.p[2] / sizeof(float);
        const size_t inStep3 = input.step.p[3] / sizeof(float);

        const size_t outStep0 = output.step.p[0] / sizeof(float);
        const size_t outStep1 = output.step.p[1] / sizeof(float);
        const size_t outStep2 = output.step.p[2] / sizeof(float);
        const size_t outStep3 = output.step.p[3] / sizeof(float);

        const int channels_per_group = Ci / (int)num_groups;
        const size_t norm_size = (size_t)channels_per_group * (size_t)H * (size_t)W;
        const double inv_norm_size = 1.0 / (double)norm_size;

#if CV_SIMD
        const int VEC_SZ = VTraits<v_float32>::vlanes();
#endif

        parallel_for_(Range(0, N * (int)num_groups), [&](const Range& r) {
            const float* inptr = (const float*)input.data;
            float* outptr = (float*)output.data;

            AutoBuffer<float> buf(C0 * 2);
            float* alpha = buf.data();
            float* beta  = alpha + C0;

            for (int i = r.start; i < r.end; ++i) {
                int n = i / (int)num_groups;
                int g = i - n * (int)num_groups;
                int c_start = g * channels_per_group;
                int c_end   = c_start + channels_per_group;

                double group_sum = 0., group_sqsum = 0.;
                for (int c = c_start; c < c_end; c++) {
                    int c1 = c / C0;
                    int c0 = c % C0;
                    const float* inbase = inptr + n * inStep0 + c1 * inStep1;
                    for (int h = 0; h < H; ++h) {
                        const float* inrow = inbase + h * inStep2;
                        for (int w = 0; w < W; ++w) {
                            double v = (double)inrow[w * inStep3 + c0];
                            group_sum += v;
                            group_sqsum += v * v;
                        }
                    }
                }

                float mean = (float)(group_sum * inv_norm_size);
                float var  = std::max(0.f, (float)(group_sqsum * inv_norm_size - (double)mean * (double)mean));
                float inv_stdev = 1.f / std::sqrt(var + epsilon);

                for (int c1_start = c_start / C0, c1_end_idx = (c_end - 1) / C0 + 1,
                         c1 = c1_start; c1 < c1_end_idx; ++c1) {
                    int cbase = c1 * C0;
                    int c0_lo = std::max(0, c_start - cbase);
                    int c0_hi = std::min(C0, c_end - cbase);
                    int validC0 = std::min(C0, std::max(0, Ci - cbase));

                    for (int c0 = c0_lo; c0 < c0_hi; ++c0) {
                        alpha[c0] = scale_data[cbase + c0] * inv_stdev;
                        beta[c0]  = bias_data[cbase + c0] - alpha[c0] * mean;
                    }

                    const float* inbase  = inptr  + n * inStep0 + c1 * inStep1;
                    float*       outbase = outptr + n * outStep0 + c1 * outStep1;

                    int c0 = c0_lo;
#if CV_SIMD
                    for (; c0 <= c0_hi - VEC_SZ; c0 += VEC_SZ) {
                        v_float32 va = vx_load(alpha + c0);
                        v_float32 vb = vx_load(beta + c0);
                        for (int h = 0; h < H; ++h) {
                            const float* inrow  = inbase + h * inStep2;
                            float*       outrow = outbase + h * outStep2;
                            for (int w = 0; w < W; ++w) {
                                v_float32 vin = vx_load(inrow + w * inStep3 + c0);
                                vx_store(outrow + w * outStep3 + c0, v_fma(vin, va, vb));
                            }
                        }
                    }
#endif
                    for (; c0 < c0_hi; ++c0) {
                        float a = alpha[c0], b = beta[c0];
                        for (int h = 0; h < H; ++h) {
                            const float* inrow  = inbase + h * inStep2;
                            float*       outrow = outbase + h * outStep2;
                            for (int w = 0; w < W; ++w)
                                outrow[w * outStep3 + c0] = inrow[w * inStep3 + c0] * a + b;
                        }
                    }

                    for (int c0_pad = std::max(c0_hi, validC0); c0_pad < C0; ++c0_pad)
                        for (int h = 0; h < H; ++h) {
                            float* outrow = outbase + h * outStep2;
                            for (int w = 0; w < W; ++w)
                                outrow[w * outStep3 + c0_pad] = 0.f;
                        }
                }
            }
        });
        return;
    }

    // NCHW path
    size_t N = input_shape[0];
    CV_CheckGE(input.dims, 3, "fastNormGroup: input dimension >= 3");

    size_t channels_per_group = C / num_groups;
    size_t loops = N * num_groups;
    size_t norm_size = static_cast<size_t>(total(input_shape, 2) * channels_per_group);
    size_t step = norm_size / channels_per_group;

    auto fn = [&](const Range &r) {
        const auto *input_data = input.ptr<const float>();
        const auto *scale_data = scale.ptr<const float>();
        const auto *bias_data = bias.ptr<const float>();
        auto *output_data = output.ptr<float>();

        for (int i = r.start; i < r.end; i++) {
            const auto *x = input_data + norm_size * i;
            auto *y = output_data + norm_size * i;

            double dmean = 0., dmean_sq = 0.;
            for (size_t j = 0; j < norm_size; j++) {
                double v = (double)x[j];
                dmean += v;
                dmean_sq += v * v;
            }

            float mean = (float)(dmean / norm_size);
            float var = (float)std::max(0., dmean_sq / norm_size - (double)mean * (double)mean);
            float inv_stdev = 1.f / std::sqrt(var + epsilon);

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
