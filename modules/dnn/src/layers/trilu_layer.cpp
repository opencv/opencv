// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include <opencv2/dnn/shape_utils.hpp>
using namespace std;
namespace cv { namespace dnn {

class TriluLayerImpl CV_FINAL : public TriluLayer {
    public:
        TriluLayerImpl(const LayerParams &params) {
            setParamsFrom(params);
            uppertri = params.get<bool>("upper", true);

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
            inputs[0].copyTo(outputs[0]);
            int k = inputs.size() > 1 ? inputs[1].at<int64>(0,0) : 0;
            const auto shape_input = shape(inputs[0]);
            const int cdims = std::max(int(shape_input.size()) - 2, 0);
            const int h = inputs[0].size[shape_input.size() - 2];
            const int w = inputs[0].size[shape_input.size() - 1];
            const int m = std::min(h,w);
            int loops = 1;
            for (int i = 0; i < cdims; ++i)
                loops *= shape_input[i];

            float *dst = outputs[0].ptr<float>();
            auto r = Range(0, static_cast<int>(loops));
            auto fn = [&](const Range &r) {

                for (int i = r.start; i < r.end; i++) {
                    for(int l=0; l < m; l+=1) {
                        int cmin = uppertri ? 0 : (l + k + 1);
                        cmin = std::max(cmin, 0);
                        const int cmax = uppertri ? min(l+ k -1, w-1) : w-1;
                        const int num_zeros = cmax - cmin + 1;
                        auto *cur_dst = dst + ((w * h) * i + (w * l + cmin));
                        if (cmin < w && num_zeros > 0)
                            std::memset(cur_dst, 0, sizeof(float) * num_zeros);
                    }
                }
            };

            double nstripes = loops * h * w / 1024.0;
            parallel_for_(Range(0, loops), fn, nstripes);
        }


        void getTypes(const std::vector<MatType>& inputs,
            const int requiredOutputs,
            const int requiredInternals,
            std::vector<MatType>& outputs,
            std::vector<MatType>& internals) const CV_OVERRIDE
        {
            outputs.assign(1, CV_32F);
            // outputs.assign(2, CV_64SC1);
        }
    private:
        bool uppertri;

};


Ptr<TriluLayer> TriluLayer::create(const LayerParams& params)
{
    return makePtr<TriluLayerImpl>(params);
}

}}