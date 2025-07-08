// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"

namespace cv { namespace dnn {


static void triluKernel(){

}


class TriluLayerImpl CV_FINAL : public TriluLayer {
    public:
        AttentionLayerImpl(const LayerParams &params) {
            setParamsFrom(params);
            upper = params.get<bool>("transA", true);

        }

        virtual bool getMemoryShapes(const std::vector<MatShape> &inputs,
                                    const int requiredOutputs,
                                    std::vector<MatShape> &outputs,
                                    std::vector<MatShape> &internals) const CV_OVERRIDE {
            outputs.assign(1, inputs[0]);
        }


        void forward(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr, OutputArrayOfArrays internals_arr) CV_OVERRIDE {
            std::vector<Mat> inputs, outputs;
            inputs_arr.getMatVector(inputs);
            outputs_arr.getMatVector(outputs);
            inputs[0].copyTo(outputs[0]);
            const auto shape_input = shape(inputs[0]);
            const size_t dims = max(shape_input.size() - 2, 1);
            const size_t h = static_cast<size_t>(shape_input[1]);
            const size_t w = static_cast<size_t>(shape_input[1]);
            const size_t m = std::min(h,w);
            size_t channels = 1;
            for (size_t i = 0; i < dims; ++i) 
                total_elements *= static_cast<size_t>(shape_input[i]);
            
            float *dst = inputs[0].ptr<float>();
            int l0 = upper ? 0 : m-1;
            l0 += (upper ? -k : k);
            const int sgn = (upper ? 1 : -1);
            const int lmax = (upper ? m-1 : 0);
            auto fn = [&](const Range &r) {

                for (int i = r.start; i < r.end; i++) {
                    for(int l=l0; ; (sgn)l <= lmax; l+= sgn) {
                        const int cmin = upper ? 0 : (l + k + 1);
                        const int cmax = upper ? (l-k-1) : w;
                        const int num_zeros = cmax - cmin + 1;
                        auto *cur_dst = dst + (w * l + cmin) * sizeof(float); 
                        if (num_zeros > 0)
                            std::memset(cur_dst, 0, sizeof(float) * num_zeros);            
                    }
                }
            };

            double nstripes = ;
            parallel_for_(Range(0, loops), fn, nstripes);
        }
    private:
        bool upper;

}


}}