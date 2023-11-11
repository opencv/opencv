// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// This file is modified from the ficus (https://github.com/vpisarev/ficus/blob/master/lib/NN/OpNN.fx).
// Here is the original license:
/*
    This file is a part of ficus language project.
    See ficus/LICENSE for the licensing terms
*/

#include "../../precomp.hpp"
#include "softmax.hpp"

namespace cv { namespace dnn {

void softmax(Mat &dst, const Mat &src, int axis, int axisBias, int axisStep){
    CV_Assert(src.type() == CV_32F);
    CV_Assert(src.isContinuous() && dst.isContinuous());
    CV_Assert(src.size == dst.size);
    axis = normalize_axis(axis, src.dims);

    size_t outerSize = src.total(0, axis),
           innerSize = src.total(axis + 1);

    const float *srcPtr = src.ptr<float>();
    float *dstPtr = dst.ptr<float>();

    size_t outerStep = src.total(axis);
    size_t cnStep = src.total(axis + 1);

    // multi-threads
    size_t totalTasks = outerSize * innerSize;
    double nstripes = (double) totalTasks / 1024.0;
    // make the channel axis to be multiple of 8
    size_t channelAxis = (axisStep + 7) & -8;

#if (CV_SIMD || CV_SIMD_SCALABLE)
    const int nlanes = VTraits<v_float32>::vlanes();
    // the number of redundant dimension
    size_t redundantDim = nlanes - axisStep % nlanes;
#endif

    parallel_for_(Range(0, (int) totalTasks), [&](const Range &range) {
        AutoBuffer<float> axisBuf_(channelAxis);
        float *axisBuf = axisBuf_.data();

        for (size_t i = range.start; i < range.end; i++) {
            size_t outerDim = i / innerSize;
            size_t innerDim = i % innerSize;
            size_t srcOffset = outerDim * outerStep + innerDim;
            // copy data from src to buf along axis, since the data may not be continuous
            for (size_t cnDim = 0; cnDim < axisStep; cnDim++)
                axisBuf[cnDim] = srcPtr[srcOffset + (cnDim + axisBias) * cnStep];

            float s = 0.f;
#if (CV_SIMD || CV_SIMD_SCALABLE)
            // make the value of the redundant dimension to be -FLT_MAX
            if (redundantDim != nlanes) {
                for (size_t j = axisStep; j < axisStep + redundantDim; j++)
                    axisBuf[j] = -FLT_MAX;
            }
            // calculate the max value along the axis
            v_float32 vmax = vx_load(axisBuf);
            for (size_t cnDim = nlanes; cnDim < axisStep; cnDim += nlanes) {
                v_float32 val = vx_load(axisBuf + cnDim);
                vmax = v_max(vmax, val);
            }
            float maxVal = v_reduce_max(vmax);

            // calculate the exp value along the axis
            v_float32 vs = vx_setzero_f32();
            vmax = vx_setall_f32(maxVal);
            // initialize vexp constant
            v_float32 _vexp_lo = vx_setall_f32(-88.3762626647949f);
            v_float32 _vexp_hi = vx_setall_f32(88.3762626647949f);
            v_float32 _vexp_half = vx_setall_f32(0.5f);
            v_float32 _vexp_one = vx_setall_f32(1.f);
            v_float32 _vexp_LOG2EF = vx_setall_f32(1.44269504088896341f);
            v_float32 _vexp_C1 = vx_setall_f32(-0.693359375f);
            v_float32 _vexp_C2 = vx_setall_f32(2.12194440e-4f);
            v_float32 _vexp_p0 = vx_setall_f32(1.9875691500E-4f);
            v_float32 _vexp_p1 = vx_setall_f32(1.3981999507E-3f);
            v_float32 _vexp_p2 = vx_setall_f32(8.3334519073E-3f);
            v_float32 _vexp_p3 = vx_setall_f32(4.1665795894E-2f);
            v_float32 _vexp_p4 = vx_setall_f32(1.6666665459E-1f);
            v_float32 _vexp_p5 = vx_setall_f32(5.0000001201E-1f);
            // initialize temp vectors for vexp
            v_float32 val, _vexp_, _vexp_x, _vexp_y, _vexp_z;
            v_int32 _vexp_mm;

            // calculate and sum all data along axis
            for (size_t cnDim = 0; cnDim < axisStep; cnDim += nlanes) {
                val = vx_load(axisBuf + cnDim);
                val = v_sub(val, vmax);

                // compute vexp of val
                _vexp_x = v_min(val, _vexp_hi);
                _vexp_x = v_max(_vexp_x, _vexp_lo);
                _vexp_ = v_fma(_vexp_x, _vexp_LOG2EF, _vexp_half);
                _vexp_mm = v_floor(_vexp_);
                _vexp_ = v_cvt_f32(_vexp_mm);
                _vexp_mm = v_add(_vexp_mm, vx_setall_s32(0x7f));
                _vexp_mm = v_shl(_vexp_mm, 23);
                _vexp_x = v_fma(_vexp_, _vexp_C1, _vexp_x);
                _vexp_x = v_fma(_vexp_, _vexp_C2, _vexp_x);
                _vexp_z = v_mul(_vexp_x, _vexp_x);
                _vexp_y = v_fma(_vexp_x, _vexp_p0, _vexp_p1);
                _vexp_y = v_fma(_vexp_y, _vexp_x, _vexp_p2);
                _vexp_y = v_fma(_vexp_y, _vexp_x, _vexp_p3);
                _vexp_y = v_fma(_vexp_y, _vexp_x, _vexp_p4);
                _vexp_y = v_fma(_vexp_y, _vexp_x, _vexp_p5);
                _vexp_y = v_fma(_vexp_y, _vexp_z, _vexp_x);
                _vexp_y = v_add(_vexp_y, _vexp_one);
                val = v_mul(_vexp_y, v_reinterpret_as_f32(_vexp_mm));

                vs = v_add(vs, val);
                v_store(axisBuf + cnDim, val);
            }

            s = v_reduce_sum(vs);
            // subtract the value of the redundant dimension
            if (redundantDim != nlanes) {
                float _val[VTraits<v_float32>::max_nlanes];
                v_store(_val, val);
                for (size_t j = nlanes - redundantDim; j < nlanes; j++)
                    s -= _val[j];
            }
#else
            float maxVal = axisBuf[0];
            for (size_t cnDim = 1; cnDim < axisStep; cnDim++) {
                maxVal = std::max(maxVal, axisBuf[cnDim]);
            }
            for (size_t j = 0; j < axisStep; j++) {
                axisBuf[j] = expf(axisBuf[j] - maxVal);
                s += axisBuf[j];
            }
#endif
            s = 1.f / s;

            // copy back the result to src
            for (size_t cnDim = 0; cnDim < axisStep; cnDim++)
                dstPtr[srcOffset + (cnDim + axisBias) * cnStep] = axisBuf[cnDim] * s;
        }
    }, nstripes);
}

void softmax(Mat &dst, const Mat &src, int axis) {
    softmax(dst, src, axis, 0, src.size[axis]);
}

void logSoftmax(Mat &dst, const Mat &src, int axis) {
    softmax(dst, src, axis);
    log(dst, dst);
}

}} // cv::dnn
