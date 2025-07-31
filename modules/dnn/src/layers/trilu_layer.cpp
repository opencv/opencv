// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include <opencv2/dnn/shape_utils.hpp>
using namespace std;
namespace cv { namespace dnn {



template <typename T>
void trilu(
    Mat &input, Mat &output,
    int m, int w, int h,
    int k, int loops, bool uppertri
)
{
    T *dst = output.ptr<T>();
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
                    std::memset(cur_dst, 0, sizeof(T) * num_zeros);
            }
        }
    };
    double nstripes = loops * h * w / 1024.0;
    parallel_for_(r, fn, nstripes);
}


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
            const int h = shape_input.size() >= 2 ? inputs[0].size[shape_input.size() - 2] : 1;
            const int w = inputs[0].size[shape_input.size() - 1];
            const int m = std::min(h,w);
            int loops = 1;
            for (int i = 0; i < cdims; ++i)
                loops *= shape_input[i];
            CV_Assert(loops > 0);
            CV_Assert(inputs[0].type() == outputs[0].type());

            Mat inpMat = inputs[0];
            Mat outMat = outputs[0];
            if (inpMat.type() == CV_32U)
                trilu<uint32_t>(inpMat, outMat, m, w, h, k, loops, uppertri);
            else if (inpMat.type() == CV_64S)
                trilu<int64_t>(inpMat, outMat, m, w, h, k, loops, uppertri);
            else if (inpMat.type() == CV_64U)
                trilu<uint64_t>(inpMat, outMat, m, w, h, k, loops, uppertri);
            else if (inpMat.type() == CV_Bool)
                trilu<bool>(inpMat, outMat, m, w, h, k, loops, uppertri);
            else if (inpMat.type() == CV_64F)
                trilu<double>(inpMat, outMat, m, w, h, k, loops, uppertri);
            else if (inpMat.type() == CV_32F)
                trilu<float>(inpMat, outMat, m, w, h, k, loops, uppertri);
            else if (inpMat.type() == CV_32S)
                trilu<int32_t>(inpMat, outMat, m, w, h, k, loops, uppertri);
            else if (inpMat.type() == CV_16S)
                trilu<int16_t>(inpMat, outMat, m, w, h, k, loops, uppertri);
            else if (inpMat.type() == CV_16U)
                trilu<uint16_t>(inpMat, outMat, m, w, h, k, loops, uppertri);
            else if (inpMat.type() == CV_8S)
                trilu<int8_t>(inpMat, outMat, m, w, h, k, loops, uppertri);
            else if (inpMat.type() == CV_8U)
                trilu<uint8_t>(inpMat, outMat, m, w, h, k, loops, uppertri);
            else
                CV_Error(Error::StsUnsupportedFormat, "Unsupported input type: " + cv::typeToString(inpMat.type()));

            // double nstripes = loops * h * w / 1024.0;
            // parallel_for_(Range(0, loops), fn, nstripes);
        }


        void getTypes(const std::vector<MatType>& inputs,
            const int requiredOutputs,
            const int requiredInternals,
            std::vector<MatType>& outputs,
            std::vector<MatType>& internals) const CV_OVERRIDE
        {
            outputs.assign(1, inputs[0]);
        }
    private:
        bool uppertri;

};


Ptr<TriluLayer> TriluLayer::create(const LayerParams& params)
{
    return makePtr<TriluLayerImpl>(params);
}

}}