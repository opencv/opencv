// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "layers_common.hpp"

#include <opencv2/dnn/shape_utils.hpp>
using namespace std;

namespace cv { namespace dnn {



class Clip13LayerImpl CV_FINAL : public Clip13Layer {
    public:
        Clip13LayerImpl(const LayerParams &params) {
            setParamsFrom(params);
        }

        virtual bool getMemoryShapes(const std::vector<MatShape> &inputs,
                                    const int requiredOutputs,
                                    std::vector<MatShape> &outputs,
                                    std::vector<MatShape> &internals) const CV_OVERRIDE {
            outputs.assign(1, inputs[0]);
            return false;
        }

        void forward(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr, OutputArrayOfArrays internals_arr) CV_OVERRIDE {
            std::vector<Mat> inputs, outputs;
            inputs_arr.getMatVector(inputs);
            outputs_arr.getMatVector(outputs);

            float minValue = -FLT_MAX,  maxValue = FLT_MAX;

            int inputs_size = inputs.size();

            if (inputs_size > 1){
                CV_Assert(inputs[1].total() == 1u);
                if (inputs[1].empty())
                    minValue = -FLT_MAX;
                else
                    minValue = inputs[1].at<float>(0);
            }
            if (inputs_size > 2){
                CV_Assert(inputs[2].total() == 1u);
                maxValue = inputs[2].empty() ? FLT_MAX : inputs[2].at<float>(0);
            }

            Mat input = inputs[0], output = outputs[0];

            float*src = input.ptr<float>();
            float*dst = output.ptr<float>();

            int len = input.total();
            int i = 0;

#if CV_SIMD128
            v_float32x4 minV = v_setall_f32(minValue), maxV = v_setall_f32(maxValue);

            for(; i <= len - 16; i += 16 )
            {
                v_float32x4 x0 = v_load(src + i);
                v_float32x4 x1 = v_load(src + i + 4);
                v_float32x4 x2 = v_load(src + i + 8);
                v_float32x4 x3 = v_load(src + i + 12);
                x0 = v_min(v_max(minV, x0), maxV);
                x1 = v_min(v_max(minV, x1), maxV);
                x2 = v_min(v_max(minV, x2), maxV);
                x3 = v_min(v_max(minV, x3), maxV);
                v_store(dst + i, x0);
                v_store(dst + i + 4, x1);
                v_store(dst + i + 8, x2);
                v_store(dst + i + 12, x3);
            }
#endif
            for( ; i < len; i++ )
            {
                float x = src[i];
                if (x >= minValue)
                    dst[i] = x <= maxValue ? x : maxValue;
                else
                    dst[i] = minValue;
            }
        }

};


Ptr<Clip13Layer> Clip13Layer::create(const LayerParams& params)
{
    return makePtr<Clip13LayerImpl>(params);
}

}}