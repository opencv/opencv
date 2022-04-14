// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "layers_common.hpp"
#include <opencv2/dnn/shape_utils.hpp>

#include <algorithm>
#include <iterator>
#include <numeric>

namespace cv
{
namespace dnn
{

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
        else if (operation == "add")
            op = OPERATION::SUM; // add
        else if (operation == "max")
            op = OPERATION::MAX;
        else if (operation == "min")
            op = OPERATION::MIN;
        else if (operation == "mul")
            op = OPERATION::PROD;
        else if (operation == "div")
            op = OPERATION::DIV;
        else if (operation == "sub")
            op = OPERATION::SUB;
        else if (operation == "pow")
            op = OPERATION::POW;
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

    static bool prepare_for_broadcast_op(
        int narrays, int max_ndims, const size_t* elemsize,
        const int* ndims, const int** shape_, const size_t** step_,
        int** shape, size_t** step)
    {
        int i, j, k;

        // step 1.
        // * make all inputs and the output max_ndims-dimensional.
        // * compute proper step's
        for (i = max_ndims-1; i >= 0; i-- ) {
            for (k = 0; k < narrays; k++) {
                j = ndims[k] - (max_ndims - i);
                int sz_i = j >= 0 ? shape_[k][j] : 1;
                size_t st_i = j >= 0 && step_ && step_[k] && step_[k][j] > 0 ? step_[k][j] :
                    i == max_ndims-1 ? elemsize[k] : step[k][i+1]*shape[k][i+1];
                assert(st_i % elemsize[k] == 0);
                shape[k][i] = sz_i;
                step[k][i] = st_i;
                if (shape[k][i] == 0)
                    return false;
            }
        }

        // step 3. Let's do the flattening first,
        // since we'd need proper values of steps to check continuity.
        // this loop is probably the most tricky part
        // in the whole implementation of broadcasting.
        j = max_ndims-1;
        for (i = j - 1; i >= 0; i--) {
            bool all_contiguous = true, all_scalars = true, all_consistent = true;
            for(k = 0; k < narrays; k++) {
                size_t st = step[k][j]*shape[k][j];
                bool prev_scalar = shape[k][j] == 1;
                bool scalar = shape[k][i] == 1;
                all_contiguous = all_contiguous && (st == step[k][i]);
                all_scalars = all_scalars && scalar;
                all_consistent = all_consistent && (scalar == prev_scalar);
            }
            if (all_contiguous && (all_consistent || all_scalars)) {
                for(k = 0; k < narrays; k++)
                    shape[k][j] *= shape[k][i];
            } else {
                j--;
                if (i < j) {
                    for(k = 0; k < narrays; k++) {
                        shape[k][j] = shape[k][i];
                        step[k][j] = step[k][i];
                    }
                }
            }
        }

        // step 2. Set some step's to 0's.
        for (i = max_ndims-1; i >= j; i--) {
            for (k = 0; k < narrays; k++)
                step[k][i] = shape[k][i] == 1 ? 0 : step[k][i];
        }
        for (; i >= 0; i--) {
            for (k = 0; k < narrays; k++) {
                step[k][i] = 0;
                shape[k][i] = 1;
            }
        }
        return true;
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

    template <typename T, typename Functor>
    void do_broadcast_op(
            int ndims, const int* shape,
            const char* data1, const size_t* step1,
            const char* data2, const size_t* step2,
            char* data, const size_t* step,
            const Functor& op)
    {
        assert(ndims >= 2);
        size_t dp1 = step1[ndims-1]/sizeof(T);
        size_t dp2 = step2[ndims-1]/sizeof(T);
        size_t dp = step[ndims-1]/sizeof(T);
        int k, n1 = shape[ndims-1], n2 = shape[ndims-2];
        size_t plane_idx, nplanes = 1;
        for (k = 0; k < ndims-2; k++) nplanes *= shape[k];

        for (plane_idx = 0; plane_idx < nplanes; plane_idx++) {
            const char* ptr1_ = data1;
            const char* ptr2_ = data2;
            char* ptr_ = data;
            size_t idx = plane_idx;
            for (k = ndims-3; k >= 0; k--) {
                size_t next_idx = idx/shape[k];
                int i_k = (int)(idx - next_idx*shape[k]);
                ptr1_ += i_k*step1[k];
                ptr2_ += i_k*step2[k];
                ptr_ += i_k*step[k];
                idx = next_idx;
            }
            for (int i2 = 0; i2 < n2; i2++, ptr1_ += step1[ndims-2],
                                            ptr2_ += step2[ndims-2],
                                            ptr_ += step[ndims-2])
            {
                const T* ptr1 = (const T*)ptr1_;
                const T* ptr2 = (const T*)ptr2_;
                T* ptr = (T*)ptr_;
                if (dp1 == 1 && dp2 == 1 && dp == 1) {
                    for(int i1 = 0; i1 < n1; i1++)
                        ptr[i1] = op(ptr1[i1], ptr2[i1]);
                } else if (dp1 == 1 && dp2 == 0 && dp == 1){
                    T x2 = *ptr2;
                    for(int i1 = 0; i1 < n1; i1++)
                        ptr[i1] = op(ptr1[i1], x2);
                } else if (dp1 == 0 && dp2 == 1 && dp == 1){
                    T x1 = *ptr1;
                    for(int i1 = 0; i1 < n1; i1++)
                        ptr[i1] = op(x1, ptr2[i1]);
                } else {
                    for(int i1 = 0; i1 < n1; i1++, ptr1 += dp1, ptr2 += dp2, ptr += dp)
                        *ptr = op(*ptr1, *ptr2);
                }
            }
        }
    }

    template <typename T, typename Functor>
    void forward_impl(const Functor& f, const Mat& a, const Mat& b, Mat& out)
    {
        int *shape_buf;
        size_t *step_buf;
        int max_ndims = std::max(a.dims, std::max(b.dims, out.dims));

        // TODO: SmallVec instead of alloca as we might run out of stack
        step_buf = (size_t*)alloca(3*max_ndims*(sizeof(size_t) + sizeof(int)));
        shape_buf = (int*)(step_buf + max_ndims*3);
        size_t all_type_sizes[] = {sizeof(T), sizeof(T), sizeof(T)};
        int all_ndims[] = {out.dims, a.dims, b.dims};
        const int* orig_shapes[] = {out.size.p, a.size.p, b.size.p};
        const size_t* orig_steps[] = {out.step.p, a.step.p, b.step.p};
        int* shapes[] = {shape_buf, shape_buf + max_ndims, shape_buf + max_ndims*2};
        size_t* steps[] = {step_buf, step_buf + max_ndims, step_buf + max_ndims*2};

        if (!prepare_for_broadcast_op(3, max_ndims, all_type_sizes,
                                      all_ndims, orig_shapes, orig_steps,
                                      shapes, steps))
            return;

        do_broadcast_op<T, Functor>(
                    max_ndims, shapes[0], a.ptr<char>(), steps[1],
                    b.ptr<char>(), steps[2], out.ptr<char>(), steps[0],
                    f);
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

        Mat& a = inputs[0];
        Mat& b = inputs[1];
        Mat& out = outputs[0];
        CV_Assert(a.type() == b.type() && b.type() == out.type());

        typeDispatch(a.type(), a, b, out);
    }

    template<typename T, typename... Args>
    inline void opDispatch(Args&&... args)
    {
        switch (op)
        {
            case OPERATION::EQUAL:
            {
                auto equal = [](const T &a, const T &b) { return a == b; };
                forward_impl<T>(equal, std::forward<Args>(args)...);
                break;
            }
            case OPERATION::GREATER:
            {
                auto greater = [](const T &a, const T &b) { return a > b; };
                forward_impl<T>(greater, std::forward<Args>(args)...);
                break;
            }
            case OPERATION::GREATER_EQUAL:
            {
                auto greater_equal = [](const T &a, const T &b) { return a >= b; };
                forward_impl<T>(greater_equal, std::forward<Args>(args)...);
                break;
            }
            case OPERATION::LESS:
            {
                auto less = [](const T &a, const T &b) { return a < b; };
                forward_impl<T>(less, std::forward<Args>(args)...);
                break;
            }
            case OPERATION::LESS_EQUAL:
            {
                auto less_equal = [](const T &a, const T &b) { return a <= b; };
                forward_impl<T>(less_equal, std::forward<Args>(args)...);
                break;
            }
            case OPERATION::POW:
            {
                auto pow = [] (const T& a, const T& b) { return std::pow(a, b); };
                forward_impl<T>(pow, std::forward<Args>(args)...);
                break;
            }
            case OPERATION::BITSHIFT:
            {
                auto bitshift = [] (const uint8_t &a, const uint8_t &b) { return a << b; };
                forward_impl<T>(bitshift, std::forward<Args>(args)...);
                break;
            }
            case OPERATION::MAX:
            {
                auto max = [](const T &a, const T &b) { return std::max(a, b); };
                forward_impl<T>(max, std::forward<Args>(args)...);
                break;
            }
            case OPERATION::MEAN:
            {
                auto mean = [](const T &a, const T &b) { return (a + b) / T{2}; };
                forward_impl<T>(mean, std::forward<Args>(args)...);
                break;
            }
            case OPERATION::MIN:
            {
                auto min = [](const T &a, const T &b) { return std::min(a, b); };
                forward_impl<T>(min, std::forward<Args>(args)...);
                break;
            }
            case OPERATION::MOD:
            {
                auto mod = [](const uint8_t &a, const uint8_t &b) { return a % b; };
                forward_impl<T>(mod, std::forward<Args>(args)...);
                break;
            }
            case OPERATION::PROD:
            {
                auto prod = [](const T &a, const T &b) { return a * b; };
                forward_impl<T>(prod, std::forward<Args>(args)...);
                break;
            }
            case OPERATION::SUB:
            {
                auto sub = [](const T &a, const T &b) { return a - b; };
                forward_impl<T>(sub, std::forward<Args>(args)...);
                break;
            }
            case OPERATION::SUM:
            {
                auto sum = [](const T &a, const T &b) { return a + b; };
                forward_impl<T>(sum, std::forward<Args>(args)...);
                break;
            }
            case OPERATION::DIV:
            {
                auto div = [](const T &a, const T &b) { return a / b; };
                forward_impl<T>(div, std::forward<Args>(args)...);
                break;
            }
            default:
                CV_Error(Error::StsBadArg, "Unsupported operation.");
        };
    }

    template<typename... Args>
    inline void typeDispatch(const int type, Args&&... args)
    {
        switch (type)
        {
            case CV_8U:
                opDispatch<uint8_t>(std::forward<Args>(args)...);
                break;
            case CV_32F:
                CV_Assert(op != OPERATION::BITSHIFT && op != OPERATION::MOD);
                opDispatch<float>(std::forward<Args>(args)...);
                break;
            default:
                CV_Error(cv::Error::BadDepth, "Unsupported type.");
        };
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
};

Ptr<NaryEltwiseLayer> NaryEltwiseLayer::create(const LayerParams& params)
{
    return Ptr<NaryEltwiseLayer>(new NaryEltwiseLayerImpl(params));
}

}
}
