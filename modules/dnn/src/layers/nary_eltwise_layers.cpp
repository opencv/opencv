// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "layers_common.hpp"
#include <opencv2/dnn/shape_utils.hpp>
#include "small_vector.hpp"

#include <algorithm>
#include <iterator>
#include <numeric>

namespace cv
{
namespace dnn
{

template <typename T>
using VectorType = itlib::small_vector<T, 5, 5>;

class NaryEltwiseLayerImpl CV_FINAL : public NaryEltwiseLayer
{
public:
    enum class OPERATION
    {
        AND = 0,
        EQUAL,
        GREATER,
        GREATER_EQUAL,
        LESS,
        LESS_EQUAL,
        OR,
        POW,
        XOR,
        BITSHIFT,
        MAX,
        MEAN,
        MIN,
        MOD,
        PROD,
        SUB,
        SUM,
        DIV,
    } op;

    // TODO: coeffs + ActivationFunction
//    std::vector<float> coeffs;

    NaryEltwiseLayerImpl(const LayerParams& params)
    {
        setParamsFrom(params);

        String operation = toLowerCase(params.get<String>("operation", "sum"));
        if (operation == "prod")
            op = OPERATION::PROD;
        else if (operation == "sum")
            op = OPERATION::SUM;
        else if (operation == "max")
            op = OPERATION::MAX;
        else if (operation == "min")
            op = OPERATION::MIN;
        else if (operation == "div")
            op = OPERATION::DIV;
        else
            CV_Error(cv::Error::StsBadArg, "Unknown operation type \"" + operation + "\"");
    }

    virtual bool supportBackend(int backendId) CV_OVERRIDE
    {
        return backendId == DNN_BACKEND_OPENCV;
    }

    static MatShape findCommonShape(std::vector<MatShape> shapes)
    {
        CV_Assert(!shapes.empty());
        const size_t dim = std::max_element(shapes.begin(), shapes.end(),
                                            [](const MatShape& a, const MatShape& b)
                                            { return a.size() < b.size(); })->size();

        for (auto& shape : shapes)
        {
            shape.insert(shape.begin(), dim - shape.size(), 1);
        }

        MatShape outShape(dim, 1);
        for (size_t i = 0; i < dim; ++i)
        {
            for (const auto& shape : shapes)
            {
                if (shape[i] != outShape[i])
                {
                    CV_Assert(shape[i] == 1 || outShape[i] == 1);
                    outShape[i] = std::max(outShape[i], shape[i]);
                }
            }
        }

        return outShape;
    }

    bool getMemoryShapes(const std::vector<MatShape> &inputs,
                         const int requiredOutputs,
                         std::vector<MatShape> &outputs,
                         std::vector<MatShape> &internals) const CV_OVERRIDE
    {
        MatShape outShape = findCommonShape(inputs);
        outputs.assign(1, outShape);
        return false;
    }

    // doesn't take into account non-continuous matrices, because we're in DNN
    void foldShapes()
    {
        // pad shapes
        for (auto& shape : input_shapes)
        {
            shape.insert(shape.begin(), output_shape.size() - shape.size(), 1);
        }

        size_t i = 0;
        for (size_t j = 1; j < output_shape.size(); ++j)
        {
            bool foldFull = true;
            bool foldOnes = true;
            for (size_t k = 1; k < input_shapes.size(); ++k)
            {
                const auto& shape_a = input_shapes[k - 1];
                const auto& dim_a = shape_a[j];
                const auto& prev_dim_a = shape_a[j - 1];

                const auto& shape_b = input_shapes[k];
                const auto& dim_b = shape_b[j];
                const auto& prev_dim_b = shape_b[j - 1];

                foldFull &= ((prev_dim_a == prev_dim_b && dim_a == dim_b) ||
                           (std::min(prev_dim_a, dim_a) != 1 && std::max(prev_dim_b, dim_b) == 1) ||
                           (std::min(prev_dim_b, dim_b) != 1 && std::max(prev_dim_a, dim_a) == 1));

                foldOnes &= ((std::max(dim_a, dim_b) == 1) && (prev_dim_a == 1 || prev_dim_b == 1));
            }

            if (!foldFull && !foldOnes) ++i;
            for (auto& shp : input_shapes)
            {
                if (foldFull)
                {
                    shp[i] *= shp[j];
                }
                else if (foldOnes)
                {
                    shp[i] = shp[j - 1];
                }
                else
                {
                    shp[i] = shp[j];
                }
            }
            // TODO: save output_shape in the same array maybe?
            if (foldFull)
            {
                output_shape[i] *= output_shape[j];
            }
            else if (foldOnes)
            {
                output_shape[i] = output_shape[j - 1];
            }
            else
            {
                output_shape[i] = output_shape[j];
            }
        }
        for (auto& shp : input_shapes)
        {
            shp.resize(i + 1);
        }
        output_shape.resize(i + 1);
    }

    void setStrides()
    {
        input_steps.resize(input_shapes.size(), VectorType<size_t>(output_shape.size()));

        for (size_t i = 0; i < input_steps.size(); ++i)
        {
            input_steps[i].back() = 1;
            for (ptrdiff_t j = static_cast<ptrdiff_t>(output_shape.size()) - 2; j >= 0; --j)
            {
                input_steps[i][j] = input_steps[i][j + 1] * input_shapes[i][j + 1];
            }
            for (size_t j = 0; j < output_shape.size(); ++j)
            {
                if (input_shapes[i][j] != output_shape[j])
                    input_steps[i][j] = 0;
            }
        }

        // TODO: transpose
        VectorType<VectorType<size_t>> steps_transposed(output_shape.size(), VectorType<size_t>(input_steps.size()));
        for (size_t i = 0; i < input_steps.size(); ++i)
        {
            for (size_t j = 0; j < output_shape.size(); ++j)
            {
                steps_transposed[j][i] = input_steps[i][j];
            }
        }
        input_steps = std::move(steps_transposed);
    }

    void finalize(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr) CV_OVERRIDE
    {
        std::vector<Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);

        auto shapeGetter = [] (const auto& m) {
            auto v = shape(m);
            return VectorType<int>(v.begin(), v.end());
        };
        std::transform(inputs.begin(), inputs.end(), std::back_inserter(input_shapes), shapeGetter);
        output_shape = shapeGetter(outputs[0]);
        foldShapes();
        setStrides();
    }

    void forward(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr, OutputArrayOfArrays internals_arr) CV_OVERRIDE
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        if (inputs_arr.depth() == CV_16S)
        {
            forward_fallback(inputs_arr, outputs_arr, internals_arr);
            return;
        }

        std::vector<Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);

        VectorType<size_t> offsets(inputs.size(), 0);

        CV_Assert(inputs.size() >= 2 && outputs.size() == 1);

        auto dstptr = outputs[0].ptr<float>();

        const size_t n = output_shape[output_shape.size() - 1];
        const size_t total = outputs[0].total() / n;

        const ptrdiff_t dims = output_shape.size();
        const size_t ninputs = inputs.size();

        VectorType<int> indices(output_shape.size(), 0);
        auto& last_steps = input_steps[input_steps.size() - 1];

        for (size_t i = 0; i < total; ++i) {
            for (size_t j = 0; j < n; ++j)
            {
                float tmp = 0.f;
                for (size_t k = 0; k < ninputs; ++k)
                {
                    tmp += inputs[k].ptr<float>()[offsets[k] + last_steps[k] * j];
                }
                // TODO: template functor
                *dstptr++ = tmp;
            }

            for (ptrdiff_t j = dims - 2; j >= 0; --j) {
                auto& steps = input_steps[j];
                auto dim = output_shape[j];

                std::transform(offsets.begin(), offsets.end(), steps.begin(), offsets.begin(), std::plus<size_t>{});

                if (++indices[j] != dim) break;

                std::transform (offsets.begin(), offsets.end(), steps.begin(), offsets.begin(),
                    [dim] (auto& a, auto& b) { return a - dim * b; });

                indices[j] = 0;
            }

//            std::fill(offsets.begin(), offsets.end(), 0);
//            for (size_t k = 0; k < input_shapes.size(); ++k)
//            {
//                for (ptrdiff_t j = 0; j < dims - 1; ++j)
//                {
//                    offsets[k] += input_steps[k][j] * indices[j];
//                }
//            }
        }
    }

    virtual bool tryQuantize(const std::vector<std::vector<float> > &scales,
                             const std::vector<std::vector<int> > &zeropoints, LayerParams& params) CV_OVERRIDE
    {
        return false;
    }

    virtual int64 getFLOPS(const std::vector<MatShape> &inputs,
                           const std::vector<MatShape> &outputs) const CV_OVERRIDE
    {
        CV_Assert(inputs.size());
        return inputs.size() * total(outputs[0]);
    }

private:
    // TODO: flat index, transpose (INPUTS, DIMS) to (DIMS, INPUTS)
    VectorType<VectorType<size_t>> input_steps;

    VectorType<VectorType<int>> input_shapes;
    VectorType<int> output_shape;
};

Ptr<NaryEltwiseLayer> NaryEltwiseLayer::create(const LayerParams& params)
{
    return Ptr<NaryEltwiseLayer>(new NaryEltwiseLayerImpl(params));
}

}
}
