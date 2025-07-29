// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "layers_common.hpp"
#include "../op_cuda.hpp"
#include "../op_cann.hpp"
#include "../ie_ngraph.hpp"
#include "../op_vkcom.hpp"

#include <opencv2/dnn/shape_utils.hpp>

#include <algorithm>
#include <iterator>
#include <numeric>

#ifdef HAVE_CUDA
#include "../cuda4dnn/primitives/eltwise.hpp"
using namespace cv::dnn::cuda4dnn;
#endif

namespace cv
{
namespace dnn
{

namespace {
static int _mod(int x, int y) {
    int res = x % y;
    if ((res < 0 && y > 0) || (res > 0 && y < 0)) {
        res += y;
    }
    return res;
}
}

class NaryEltwiseHelper CV_FINAL
{
public:
    int ninputs;
    int narrays;
    int max_ndims;
    std::vector<int> all_ndims;
    std::vector<std::vector<int>> orig_shapes;
    std::vector<std::vector<size_t>> orig_steps;
    std::vector<std::vector<int>> shapes;
    std::vector<std::vector<size_t>> steps;
    std::vector<size_t> elemsize;

    NaryEltwiseHelper() {}

    void init(const std::vector<Mat>& inputs, const std::vector<Mat>& outputs)
    {
        narrays = 0;
        max_ndims = 0;
        all_ndims.clear();
        orig_shapes.clear();
        orig_steps.clear();
        shapes.clear();
        steps.clear();
        elemsize.clear();

        ninputs = inputs.size();
        narrays = ninputs + 1;

        // collect ndims
        std::vector<int> v_inp_dims;
        std::transform(inputs.begin(), inputs.end(), std::back_inserter(v_inp_dims), [] (const Mat& m) { return m.dims; });
        const int* inp_ndims = v_inp_dims.data();
        int out_ndims = outputs[0].dims;

        // find max ndims for broadcasting
        int i;
        max_ndims = out_ndims > 2 ? out_ndims : 2;
        for(i = 0; i < ninputs; i++)
            max_ndims = max_ndims > inp_ndims[i] ? max_ndims : inp_ndims[i];

        shapes = std::vector<std::vector<int>>(narrays, std::vector<int>(max_ndims, 0));
        steps = std::vector<std::vector<size_t>>(narrays, std::vector<size_t>(max_ndims, 0));

        for(i = 0; i <= ninputs; i++) {
            all_ndims.push_back(i == 0 ? out_ndims : inp_ndims[i-1]);
            std::vector<int> _size;
            std::vector<size_t> _step;
            if (!i) {
                std::transform(outputs[0].size.p, outputs[0].size.p + outputs[0].dims, std::back_inserter(_size), [](int s) { return s; });
                std::transform(outputs[0].step.p, outputs[0].step.p + outputs[0].dims, std::back_inserter(_step), [](size_t s) { return s; });
            }
            else {
                std::transform(inputs[i-1].size.p, inputs[i-1].size.p + inputs[i-1].dims, std::back_inserter(_size), [](int s) { return s; });
                std::transform(inputs[i-1].step.p, inputs[i-1].step.p + inputs[i-1].dims, std::back_inserter(_step), [](size_t s) { return s; });
            }
            orig_shapes.push_back(_size);
            orig_steps.push_back(_step);

            int esz = i == 0 ? outputs[0].elemSize() : inputs[i - 1].elemSize();
            elemsize.push_back(esz);
        }
    }

    bool prepare_for_broadcast_op()
    {
        int i, j, k;

        // step 1.
        // * make all inputs and the output max_ndims-dimensional.
        // ** prepend dimension 1 to the mat of less dims
        // * compute proper step's
        for (i = this->max_ndims-1; i >= 0; i--) {
            for (k = 0; k < this->narrays; k++) {
                j = this->all_ndims[k] - (this->max_ndims - i);
                int sz_i = j >= 0 ? this->orig_shapes[k][j] : 1;
                size_t st_i = j >= 0 && this->orig_steps[k][j] > 0 ? this->orig_steps[k][j] :
                    i == this->max_ndims-1 ? elemsize[k] : this->steps[k][i+1]*this->shapes[k][i+1];
                assert(st_i % elemsize[k] == 0);
                this->shapes[k][i] = sz_i;
                this->steps[k][i] = st_i;
                if (this->shapes[k][i] == 0)
                    return false;
            }
        }

        // step 3. Let's do the flattening first,
        // since we'd need proper values of steps to check continuity.
        // this loop is probably the most tricky part
        // in the whole implementation of broadcasting.
        j = this->max_ndims > 0 ? this->max_ndims-1 : 0;
        for (i = j - 1; i >= 0; i--) {
            bool all_contiguous = true, all_scalars = true, all_consistent = true;
            for(k = 0; k < this->narrays; k++) {
                size_t st = this->steps[k][j]*this->shapes[k][j];
                bool prev_scalar = this->shapes[k][j] == 1;
                bool scalar = this->shapes[k][i] == 1;
                all_contiguous = all_contiguous && (st == this->steps[k][i]);
                all_scalars = all_scalars && scalar;
                all_consistent = all_consistent && (scalar == prev_scalar);
            }
            if (all_contiguous && (all_consistent || all_scalars)) {
                for(k = 0; k < this->narrays; k++)
                    this->shapes[k][j] *= this->shapes[k][i];
            } else {
                j--;
                if (i < j) {
                    for(k = 0; k < this->narrays; k++) {
                        this->shapes[k][j] = this->shapes[k][i];
                        this->steps[k][j] = this->steps[k][i];
                    }
                }
            }
        }

        // step 2. Set some step's to 0's.
        for (i = this->max_ndims-1; i >= j; i--) {
            for (k = 0; k < this->narrays; k++)
                this->steps[k][i] = this->shapes[k][i] == 1 ? 0 : this->steps[k][i];
        }
        if (this->max_ndims == 0)
            i = 0;
        for (; i >= 0; i--) {
            for (k = 0; k < this->narrays; k++) {
                this->steps[k][i] = 0;
                this->shapes[k][i] = 1;
            }
        }
        return true;
    }
};

class NaryEltwiseLayerImpl CV_FINAL : public NaryEltwiseLayer
{
    NaryEltwiseHelper helper;
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
        MOD,  // Integer Mod. Reminder's sign = Divisor's sign.
        FMOD, // Floating-point Mod. Reminder's sign = Dividend's sign.
        PROD,
        SUB,
        SUM,
        ADD,
        DIV,
        WHERE,
    } op;

    NaryEltwiseLayerImpl(const LayerParams& params)
    {
        setParamsFrom(params);

        String operation = toLowerCase(params.get<String>("operation", "sum"));

        if (operation == "equal")
            op = OPERATION::EQUAL;
        else if (operation == "greater")
            op = OPERATION::GREATER;
        else if (operation == "greaterorequal")
            op = OPERATION::GREATER_EQUAL;
        else if (operation == "less")
            op = OPERATION::LESS;
        else if (operation == "lessorequal")
            op = OPERATION::LESS_EQUAL;
        else if (operation == "pow")
            op = OPERATION::POW;
        else if (operation == "bitshift")
            op = OPERATION::BITSHIFT;
        else if (operation == "max")
            op = OPERATION::MAX;
        else if (operation == "mean")
            op = OPERATION::MEAN;
        else if (operation == "min")
            op = OPERATION::MIN;
        else if (operation == "mod")
            op = OPERATION::MOD;
        else if (operation == "fmod")
            op = OPERATION::FMOD;
        else if (operation == "mul")
            op = OPERATION::PROD;
        else if (operation == "sub")
            op = OPERATION::SUB;
        else if (operation == "sum")
            op = OPERATION::SUM;
        else if (operation == "add")
            op = OPERATION::ADD;
        else if (operation == "div")
            op = OPERATION::DIV;
        else if (operation == "and")
            op = OPERATION::AND;
        else if (operation == "or")
            op = OPERATION::OR;
        else if (operation == "xor")
            op = OPERATION::XOR;
        else if (operation == "where")
            op = OPERATION::WHERE;
        else
            CV_Error(cv::Error::StsBadArg, "Unknown operation type \"" + operation + "\"");
    }

    virtual bool supportBackend(int backendId) CV_OVERRIDE
    {
#ifdef HAVE_CANN
        if (backendId == DNN_BACKEND_CANN)
            return op == OPERATION::ADD || op == OPERATION::PROD || op == OPERATION::SUB ||
                   op == OPERATION::DIV || op == OPERATION::MAX  || op == OPERATION::MIN ||
                   op == OPERATION::MOD || op == OPERATION::FMOD || op == OPERATION::POW;
#endif
        if (backendId == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
            return (op == OPERATION::ADD ||
                    op == OPERATION::PROD ||
                    op == OPERATION::EQUAL ||
                    op == OPERATION::GREATER ||
                    op == OPERATION::GREATER_EQUAL ||
                    op == OPERATION::LESS ||
                    op == OPERATION::LESS_EQUAL ||
                    op == OPERATION::AND ||
                    op == OPERATION::OR ||
                    op == OPERATION::XOR ||
                    op == OPERATION::WHERE ||
                    op == OPERATION::MOD ||
                    op == OPERATION::FMOD
            );

#ifdef HAVE_VULKAN
        if (backendId == DNN_BACKEND_VKCOM)
            return op == OPERATION::ADD || op == OPERATION::PROD || op == OPERATION::SUB ||
                   op == OPERATION::DIV;
#endif

        if (backendId == DNN_BACKEND_CUDA) {
            return op == OPERATION::MAX  || op == OPERATION::MIN  || op == OPERATION::SUM ||
                   op == OPERATION::PROD || op == OPERATION::DIV  || op == OPERATION::ADD ||
                   op == OPERATION::SUB  || op == OPERATION::MOD || op == OPERATION::FMOD;
        }
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


    virtual void finalize(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr) CV_OVERRIDE {
        std::vector<Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);

        helper.init(inputs, outputs);
        CV_CheckTrue(helper.prepare_for_broadcast_op(), "NaryEltwiseLayer: Preparation for broadcasting failed");
    }

    bool getMemoryShapes(const std::vector<MatShape> &inputs,
                         const int requiredOutputs,
                         std::vector<MatShape> &outputs,
                         std::vector<MatShape> &internals) const CV_OVERRIDE
    {
        if (inputs.size() == 1) {
            outputs.assign(1, inputs.front());
        } else {
            MatShape outShape = findCommonShape(inputs);
            outputs.assign(1, outShape);
        }
        return false;
    }

    void getTypes(const std::vector<MatType>& inputs,
        const int requiredOutputs,
        const int requiredInternals,
        std::vector<MatType>& outputs,
        std::vector<MatType>& internals) const CV_OVERRIDE
    {
        if (op == OPERATION::WHERE)
        {
            CV_CheckTypeEQ(inputs[0], CV_Bool, "");
            CV_CheckTypeEQ(inputs[1], inputs[2], "");
            outputs.assign(1, inputs[1]);
            return;
        }

        if (op == OPERATION::AND || op == OPERATION::OR || op == OPERATION::XOR)
        {
            CV_CheckTypeEQ(inputs[0], CV_Bool, "");
            CV_CheckTypeEQ(inputs[1], CV_Bool, "");
            outputs.assign(1, CV_Bool);
            return;
        }

        if (op == OPERATION::POW) {
            /*
                First input: exponent of Type T;
                Second input: power of the exponent of Type T1;
                Output: same type T as first input's.
            */
            outputs.assign(1, inputs.front());
            return;
        }

        CV_Assert(inputs.size());
        for (auto input : inputs)
        {
            CV_CheckTypeEQ(inputs[0], input, "All inputs should have equal types");
            if (preferableTarget == DNN_TARGET_OPENCL_FP16)
                CV_CheckType(input, input == CV_16F || input == CV_8S || input == CV_8U || input == CV_32S || input == CV_64S, "");
            else
                CV_CheckType(input, input == CV_32F || input == CV_8S || input == CV_8U || input == CV_32S || input == CV_64S, "");
        }

        if (op == OPERATION::EQUAL || op == OPERATION::GREATER || op == OPERATION::GREATER_EQUAL || op == OPERATION::LESS || op == OPERATION::LESS_EQUAL)
            outputs.assign(1, CV_Bool);
        else
            outputs.assign(requiredOutputs, inputs[0]);
    }


    template <typename T, typename RESULT_T, typename Functor>
    void binary_forward_impl(const Functor& op, int ndims, const std::vector<int>& shape,
                             const char* data1, const std::vector<size_t>& step1,
                             const char* data2, const std::vector<size_t>& step2,
                             char* data, const std::vector<size_t>& step, size_t block_size) {
        size_t dp1 = 0, dp2 = 0, dp = 0;
        int plane_size = 1;
        size_t nplanes = 1;

        if (ndims >= 1) {
            dp1 = step1.back() / sizeof(T);
            dp2 = step2.back() / sizeof(T);
            dp = step.back() / sizeof(RESULT_T);
            plane_size = shape.back();

            if (ndims >= 2) {
                nplanes = std::accumulate(shape.begin(), shape.end() - 1, 1, std::multiplies<size_t>());
            }
        }

        if (nplanes == 1) { // parallelize within the plane
            const T* ptr1 = (const T*)data1;
            const T* ptr2 = (const T*)data2;
            RESULT_T* ptr = (RESULT_T*)data;
            auto worker = [&](const Range &r) {
                if (dp1 == 1 && dp2 == 1 && dp == 1) {
                    for(int i = r.start; i < r.end; i++) {
                        ptr[i] = op(ptr1[i], ptr2[i]);
                    }
                } else if (dp1 == 1 && dp2 == 0 && dp == 1){
                    T x2 = *ptr2;
                    for(int i = r.start; i < r.end; i++) {
                        ptr[i] = op(ptr1[i], x2);
                    }
                } else if (dp1 == 0 && dp2 == 1 && dp == 1){
                    T x1 = *ptr1;
                    for(int i = r.start; i < r.end; i++) {
                        ptr[i] = op(x1, ptr2[i]);
                    }
                } else {
                    for(int i = r.start; i < r.end; i++, ptr1 += dp1, ptr2 += dp2, ptr += dp) {
                        *ptr = op(*ptr1, *ptr2);
                    }
                }
            };

            double nstripes = plane_size * (1.0 / double(block_size));
            parallel_for_(Range(0, plane_size), worker, nstripes);
        } else { // parallelize across planes
            auto worker = [&](const Range &r) {
                for (int plane_idx = r.start; plane_idx < r.end; plane_idx++) {
                    const char* ptr1_ = data1;
                    const char* ptr2_ = data2;
                    char* ptr_ = data;
                    size_t idx = plane_idx;
                    for (int k = ndims - 2; k >= 0; k--) {
                        size_t next_idx = idx / shape[k];
                        size_t i_k = (int)(idx - next_idx * shape[k]);
                        ptr1_ += i_k * step1[k];
                        ptr2_ += i_k * step2[k];
                        ptr_ += i_k * step[k];
                        idx = next_idx;
                    }

                    const T* ptr1 = (const T*)ptr1_;
                    const T* ptr2 = (const T*)ptr2_;
                    RESULT_T* ptr = (RESULT_T*)ptr_;
                    if (dp1 == 1 && dp2 == 1 && dp == 1) {
                        for(int i = 0; i < plane_size; i++) {
                            ptr[i] = op(ptr1[i], ptr2[i]);
                        }
                    } else if (dp1 == 1 && dp2 == 0 && dp == 1){
                        T x2 = *ptr2;
                        for(int i = 0; i < plane_size; i++) {
                            ptr[i] = op(ptr1[i], x2);
                        }
                    } else if (dp1 == 0 && dp2 == 1 && dp == 1){
                        T x1 = *ptr1;
                        for(int i = 0; i < plane_size; i++) {
                            ptr[i] = op(x1, ptr2[i]);
                        }
                    } else {
                        for(int i = 0; i < plane_size; i++, ptr1 += dp1, ptr2 += dp2, ptr += dp) {
                            *ptr = op(*ptr1, *ptr2);
                        }
                    }
                }
            };
            double nstripes = nplanes * (1.0 / double(block_size));
            parallel_for_(Range(0, nplanes), worker, nstripes);
        }
    }

    /*
        Elementwise binary operator (like +, -, x, /, etc.) which takes two operands
    */
    template <typename T, typename RESULT_T,  typename Functor>
    void binary_forward(const Functor& f, const std::vector<Mat>& inputs, std::vector<Mat>& outputs, size_t block_size = 6e6) {
        const Mat& a = inputs[0];
        const Mat& b = inputs[1];
        Mat& out = outputs[0];
        CV_Assert(helper.shapes.size() == 3 && helper.steps.size() == 3);
        binary_forward_impl<T, RESULT_T, Functor>(f, helper.max_ndims, helper.shapes[0], a.ptr<char>(), helper.steps[1],
                                        b.ptr<char>(), helper.steps[2], out.ptr<char>(), helper.steps[0], block_size);
    }

    template<typename T, typename Functor>
    void nary_forward_impl(const Functor& op, const T scale, int ninputs, int ndims, const std::vector<int>& shape,
                           const char** inp, char* out, const std::vector<std::vector<size_t>>& steps, size_t block_size) {
        CV_Assert(ndims >= 2);
        size_t dp  = steps[0].back() / sizeof(T);
        size_t dp1 = steps[1].back() / sizeof(T);
        size_t dp2 = steps[2].back() / sizeof(T);

        int plane_size = shape.back();
        int nplanes = std::accumulate(shape.begin(), shape.end() - 1, 1, std::multiplies<int>());

        if (nplanes == 1) { // parallelize within the plane
            AutoBuffer<char> buf_ptrs(steps.size());
            auto ptrs = (char**)buf_ptrs.data();
            ptrs[0] = out;
            for (int i = 0; i < ninputs; i++) {
                ptrs[i+1] = (char*)inp[i];
            }
            auto worker = [&](const Range &r) {
                const T* ptr1 = (const T*)(ptrs[1]);
                const T* ptr2 = (const T*)(ptrs[2]);
                T* ptr = (T*)(ptrs[0]);
                if (dp == 1 && dp1 == 1 && dp2 == 1) {
                    for (int i = r.start; i < r.end; i++) {
                        ptr[i] = op(ptr1[i], ptr2[i]);
                    }
                    for (int j = 2; j < ninputs; j++) {
                        size_t dpj = steps[j + 1].back() / sizeof(T);
                        const T* ptrj = (const T*)(ptrs[j + 1]);
                        if (dpj == 1) {
                            for (int i = r.start; i < r.end; i++) {
                                ptr[i] = saturate_cast<T>(op(ptr[i], ptrj[i]) * scale);
                            }
                        } else {
                            for (int i = r.start; i < r.end; i++, ptrj += dpj) {
                                ptr[i] = saturate_cast<T>(op(ptr[i], *ptrj) * scale);
                            }
                        }
                    }
                } else {
                    auto *tmp = ptr;
                    for (int i = r.start; i < r.end; i++, ptr += dp, ptr1 += dp1, ptr2 += dp2) {
                        *ptr = op(*ptr1, *ptr2);
                    }
                    ptr = tmp;
                    for (int j = 2; j < ninputs; j++) {
                        size_t dpj = steps[j + 1].back() / sizeof(T);
                        const T* ptr_j = (const T*)(ptrs[j + 1]);
                        for (int i = r.start; i < r.end; i++, ptr += dp, ptr_j += dpj) {
                            *ptr = saturate_cast<T>(op(*ptr, *ptr_j) * scale);
                        }
                    }
                }
            };
            double nstripes = plane_size * (1.0 / double(block_size));
            parallel_for_(Range(0, plane_size), worker, nstripes);
        } else { // parallelize across the plane
            auto worker = [&](const Range &r) {
                AutoBuffer<char> buf_ptrs(steps.size());
                auto ptrs = (char**)buf_ptrs.data();
                for (int plane_idx = r.start; plane_idx < r.end; plane_idx++) {
                    ptrs[0] = out;
                    for (int i = 0; i < ninputs; i++) ptrs[i+1] = (char*)inp[i];
                    size_t idx = plane_idx;
                    for (int k = ndims - 2; k >= 0; k--) {
                        size_t next_idx = idx / shape[k];
                        int i_k = (int)(idx - next_idx * shape[k]);
                        for (int i = 0; i <= ninputs; i++) {
                            ptrs[i] += i_k * steps[i][k];
                        }
                        idx = next_idx;
                    }

                    const T* ptr1 = (const T*)(ptrs[1]);
                    const T* ptr2 = (const T*)(ptrs[2]);
                    T* ptr = (T*)(ptrs[0]);
                    if (dp == 1 && dp1 == 1 && dp2 == 1) {
                        for (int i = 0; i < plane_size; i++) {
                            ptr[i] = saturate_cast<T>(op(ptr1[i], ptr2[i]) * scale);
                        }
                        for (int j = 2; j < ninputs; j++) {
                            size_t dpj = steps[j + 1].back() / sizeof(T);
                            const T* ptrj = (const T*)(ptrs[j + 1]);
                            if (dpj == 1) {
                                for (int i = 0; i < plane_size; i++) {
                                    ptr[i] = op(ptr[i], saturate_cast<T>(ptrj[i] * scale));
                                }
                            } else {
                                for (int i = 0; i < plane_size; i++, ptrj += dpj) {
                                    ptr[i] = op(ptr[i], saturate_cast<T>(*ptrj * scale));
                                }
                            }
                        }
                    } else {
                        auto *tmp = ptr;
                        for (int i = 0; i < plane_size; i++, ptr += dp, ptr1 += dp1, ptr2 += dp2) {
                            *ptr = saturate_cast<T>(op(*ptr1, *ptr2) * scale);
                        }
                        ptr = tmp;
                        for (int j = 2; j < ninputs; j++) {
                            size_t dpj = steps[j + 1].back() / sizeof(T);
                            const T* ptrj = (const T*)(ptrs[j + 1]);
                            for (int i = 0; i < plane_size; i++, ptr += dp, ptrj += dpj) {
                                *ptr = op(*ptr, saturate_cast<T>(*ptrj * scale));
                            }
                        }
                    }
                }
            };
            double nstripes = nplanes * (1.0 / double(block_size));
            parallel_for_(Range(0, nplanes), worker, nstripes);
        }
    }

    /*
        Elementwise nary operator (like sum, mean, etc.) which takes at least one operand
    */
    template <typename T, typename Functor>
    void nary_forward(const Functor& f, T scale,
                      const std::vector<Mat>& inputs, std::vector<Mat>& outputs,
                      size_t block_size = 6e6) {
        // collect all input info
        std::vector<const char*> v_inp;
        std::transform(inputs.begin(), inputs.end(), std::back_inserter(v_inp), [] (const Mat& m) { return m.template ptr<const char>(); });
        const char** inp = v_inp.data();

        // collect output info
        char* out = outputs[0].ptr<char>();

        nary_forward_impl<T, Functor>(f, scale, helper.ninputs, helper.max_ndims, helper.shapes[0], inp, out, helper.steps, block_size);
    }

    /*
        Elementwise ternary operator (like where) which takes three operands
    */
    template <typename T_INP1, typename T_INP2, typename T_INP3, typename T_OUT, typename Functor>
    void ternary_forward(const Functor& f, const std::vector<Mat>& inputs, std::vector<Mat>& outputs, size_t block_size = 6e6) {
        const Mat& a = inputs[0];
        const Mat& b = inputs[1];
        const Mat& c = inputs[2];
        Mat& out = outputs[0];

        CV_Assert(helper.shapes.size() == 4 && helper.steps.size() == 4);

        ternary_forward_impl<T_INP1, T_INP2, T_INP3, T_OUT, Functor>(f, helper.max_ndims, helper.shapes[0],
                                                                     a.ptr<char>(), helper.steps[1],
                                                                     b.ptr<char>(), helper.steps[2],
                                                                     c.ptr<char>(), helper.steps[3],
                                                                     out.ptr<char>(), helper.steps[0], block_size);
    }

    template <typename T_INP1, typename T_INP2, typename T_INP3, typename T_OUT, typename Functor>
    void ternary_forward_impl(
            const Functor& op, int ndims, const std::vector<int>& shape,
            const char* data1, const std::vector<size_t>& step1,
            const char* data2, const std::vector<size_t>& step2,
            const char* data3, const std::vector<size_t>& step3,
            char* data, const std::vector<size_t>& step, size_t block_size) {
        CV_Assert(ndims >= 2);
        size_t dp1 = step1.back() / sizeof(T_INP1);
        size_t dp2 = step2.back() / sizeof(T_INP2);
        size_t dp3 = step3.back() / sizeof(T_INP3);
        size_t dp = step.back() / sizeof(T_OUT);
        int plane_size = shape.back();
        int nplanes = std::accumulate(shape.begin(), shape.end() - 1, 1, std::multiplies<int>());

        if (nplanes == 1) { // parallelize within the plane
            const auto *ptr1 = (const T_INP1*)data1;
            const auto *ptr2 = (const T_INP2*)data2;
            const auto *ptr3 = (const T_INP3*)data3;
            auto* ptr = (T_OUT*)data;
            auto worker = [&](const Range &r) {
                if (dp1 == 1 && dp2 == 1 && dp3 == 1 && dp == 1) {
                    for (int i = r.start; i < r.end; i++) {
                        ptr[i] = op(ptr1[i], ptr2[i], ptr3[i]);
                    }
                } else if (dp1 == 0 && dp2 == 1 && dp3 == 1 && dp == 1){
                    auto x1 = *ptr1;
                    for (int i = r.start; i < r.end; i++) {
                        ptr[i] = op(x1, ptr2[i], ptr3[i]);
                    }
                } else if (dp1 == 1 && dp2 == 0 && dp3 == 1 && dp == 1){
                    auto x2 = *ptr2;
                    for (int i = r.start; i < r.end; i++) {
                        ptr[i] = op(ptr1[i], x2, ptr3[i]);
                    }
                } else if (dp1 == 1 && dp2 == 1 && dp3 == 1 && dp == 1) {
                    auto x3 = *ptr3;
                    for (int i = r.start; i < r.end; i++) {
                        ptr[i] = op(ptr1[i], ptr2[i], x3);
                    }
                } else {
                    for(int i = r.start; i < r.end; i++, ptr1 += dp1, ptr2 += dp2, ptr3 += dp3, ptr += dp) {
                        *ptr = op(*ptr1, *ptr2, *ptr3);
                    }
                }
            };
            double nstripes = plane_size * (1.0 / double(block_size));
            parallel_for_(Range(0, plane_size), worker, nstripes);
        } else { // parallelize across planes
            auto worker = [&](const Range &r) {
                for (int plane_idx = r.start; plane_idx < r.end; plane_idx++) {
                    const char* ptr1_ = data1;
                    const char* ptr2_ = data2;
                    const char* ptr3_ = data3;
                    char* ptr_ = data;
                    size_t idx = plane_idx;
                    for (int k = ndims - 2; k >= 0; k--)
                    {
                        size_t next_idx = idx / shape[k];
                        int i_k = (int)(idx - next_idx * shape[k]);
                        ptr1_ += i_k * step1[k];
                        ptr2_ += i_k * step2[k];
                        ptr3_ += i_k * step3[k];
                        ptr_ += i_k * step[k];
                        idx = next_idx;
                    }

                    const auto *ptr1 = (const T_INP1*)ptr1_;
                    const auto *ptr2 = (const T_INP2*)ptr2_;
                    const auto *ptr3 = (const T_INP3*)ptr3_;
                    auto* ptr = (T_OUT*)ptr_;
                    if (dp1 == 1 && dp2 == 1 && dp3 == 1 && dp == 1) {
                        for (int i = 0; i < plane_size; i++) {
                            ptr[i] = op(ptr1[i], ptr2[i], ptr3[i]);
                        }
                    } else if (dp1 == 0 && dp2 == 1 && dp3 == 1 && dp == 1){
                        auto x1 = *ptr1;
                        for (int i = 0; i < plane_size; i++) {
                            ptr[i] = op(x1, ptr2[i], ptr3[i]);
                        }
                    } else if (dp1 == 1 && dp2 == 0 && dp3 == 1 && dp == 1){
                        auto x2 = *ptr2;
                        for (int i = 0; i < plane_size; i++) {
                            ptr[i] = op(ptr1[i], x2, ptr3[i]);
                        }
                    } else if (dp1 == 1 && dp2 == 1 && dp3 == 1 && dp == 1) {
                        auto x3 = *ptr3;
                        for (int i = 0; i < plane_size; i++) {
                            ptr[i] = op(ptr1[i], ptr2[i], x3);
                        }
                    } else {
                        for(int i = 0; i < plane_size; i++, ptr1 += dp1, ptr2 += dp2, ptr3 += dp3, ptr += dp) {
                            *ptr = op(*ptr1, *ptr2, *ptr3);
                        }
                    }
                }
            };
            double nstripes = nplanes * (1.0 / double(block_size));
            parallel_for_(Range(0, nplanes), worker, nstripes);
        }
    }

    void forward(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr, OutputArrayOfArrays internals_arr) CV_OVERRIDE
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        if (inputs_arr.depth() == CV_16F)
        {
            forward_fallback(inputs_arr, outputs_arr, internals_arr);
            return;
        }

        std::vector<Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);

        if (inputs.size() == 1) {
            inputs[0].copyTo(outputs[0]);
            return;
        }

        int type_for_dispatch = op == OPERATION::WHERE ? outputs.front().type() : inputs.front().type();
        typeDispatch(type_for_dispatch, inputs.size(), inputs, outputs);
    }

    template<typename T, typename... Args>
    inline void opDispatch(size_t ninputs, Args&&... args)
    {
        if (ninputs == 2) { // Operators that take two operands
            switch (op) {
                case OPERATION::EQUAL: {
                    auto equal = [](const T &a, const T &b) { return a == b; };
                    binary_forward<T, bool>(equal, std::forward<Args>(args)...);
                    break;
                }
                case OPERATION::GREATER: {
                    auto greater = [](const T &a, const T &b) { return a > b; };
                    binary_forward<T, bool>(greater, std::forward<Args>(args)...);
                    break;
                }
                case OPERATION::GREATER_EQUAL: {
                    auto greater_equal = [](const T &a, const T &b) { return a >= b; };
                    binary_forward<T, bool>(greater_equal, std::forward<Args>(args)...);
                    break;
                }
                case OPERATION::LESS: {
                    auto less = [](const T &a, const T &b) { return a < b; };
                    binary_forward<T, bool>(less, std::forward<Args>(args)...);
                    break;
                }
                case OPERATION::LESS_EQUAL: {
                    auto less_equal = [](const T &a, const T &b) { return a <= b; };
                    binary_forward<T, bool>(less_equal, std::forward<Args>(args)...);
                    break;
                }
                case OPERATION::POW: {
                    auto pow = [] (const T& a, const T& b) { return std::pow(a, b); };
                    binary_forward<T, T>(pow, std::forward<Args>(args)..., 1e5);
                    break;
                }
                case OPERATION::BITSHIFT: {
                    auto bitshift = [] (const uint8_t &a, const uint8_t &b) { return a << b; };
                    binary_forward<T, T>(bitshift, std::forward<Args>(args)...);
                    break;
                }
                case OPERATION::MAX: {
                    auto max = [](const T &a, const T &b) { return std::max(a, b); };
                    binary_forward<T, T>(max, std::forward<Args>(args)...);
                    break;
                }
                case OPERATION::MEAN: {
                    auto mean = [](const T &a, const T &b) { return (a + b) / T{2}; };
                    binary_forward<T, T>(mean, std::forward<Args>(args)...);
                    break;
                }
                case OPERATION::MIN: {
                    auto min = [](const T &a, const T &b) { return std::min(a, b); };
                    binary_forward<T, T>(min, std::forward<Args>(args)...);
                    break;
                }
                case OPERATION::MOD: {
                    auto mod = [] (const T &a, const T &b) { return static_cast<T>(_mod(int(a), int(b))); };
                    binary_forward<T, T>(mod, std::forward<Args>(args)...);
                    break;
                }
                case OPERATION::FMOD: {
                    auto fmod = [](const T &a, const T &b) { return std::fmod(a, b); };
                    binary_forward<T, T>(fmod, std::forward<Args>(args)...);
                    break;
                }
                case OPERATION::PROD: {
                    auto prod = [](const T &a, const T &b) { return a * b; };
                    binary_forward<T, T>(prod, std::forward<Args>(args)...);
                    break;
                }
                case OPERATION::SUB: {
                    auto sub = [](const T &a, const T &b) { return a - b; };
                    binary_forward<T, T>(sub, std::forward<Args>(args)...);
                    break;
                }
                case OPERATION::ADD:
                case OPERATION::SUM: {
                    auto sum = [](const T &a, const T &b) { return a + b; };
                    binary_forward<T, T>(sum, std::forward<Args>(args)...);
                    break;
                }
                case OPERATION::DIV: {
                    auto div = [](const T &a, const T &b) { return a / b; };
                    binary_forward<T, T>(div, std::forward<Args>(args)...);
                    break;
                }
                default: CV_Error(Error::StsBadArg, "Unsupported operation");
            }
        } else if (ninputs == 3 && op == OPERATION::WHERE) { // Operators that take three operands
            auto where = [](const T &a, const T &b, const T &c) { return a ? b : c; };
            ternary_forward<bool, T, T, T>(where, std::forward<Args>(args)...);
        } else { // Operators that can take multiple (>= 3) operands
            switch (op)
            {
                case OPERATION::MAX: {
                    auto max = [](const T &a, const T &b) { return std::max(a, b); };
                    nary_forward<T>(max, T{1}, std::forward<Args>(args)...);
                    break;
                }
                case OPERATION::MEAN: {
                    // Sum up inputs and then calculate mean by scale = 1 / ninputs
                    auto sum = [](const T &a, const T &b) { return a + b; };
                    nary_forward<T>(sum, T{1} / ninputs, std::forward<Args>(args)...);
                    break;
                }
                case OPERATION::MIN: {
                    auto min = [](const T &a, const T &b) { return std::min(a, b); };
                    nary_forward<T>(min, T{1}, std::forward<Args>(args)...);
                    break;
                }
                case OPERATION::SUM: {
                    auto sum = [](const T &a, const T &b) { return a + b; };
                    nary_forward<T>(sum, T{1}, std::forward<Args>(args)...);
                    break;
                }
                default:
                    CV_Error(Error::StsBadArg, "Unsupported operation.");
            }
        };
    }

    template<typename... Args>
    inline void boolOpDispatch(size_t ninputs, Args&&... args)
    {
        switch (op)
        {
            case OPERATION::AND:
            {
                auto op_and = [](const bool &a, const bool &b) { return a && b; };
                binary_forward<bool, bool>(op_and, std::forward<Args>(args)...);
                break;
            }
            case OPERATION::OR:
            {
                auto op_or = [](const bool &a, const bool &b) { return a || b; };
                binary_forward<bool, bool>(op_or, std::forward<Args>(args)...);
                break;
            }
            case OPERATION::XOR:
            {
                auto op_xor = [](const bool &a, const bool &b) { return a != b; };
                binary_forward<bool, bool>(op_xor, std::forward<Args>(args)...);
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
            case CV_Bool:
                boolOpDispatch(std::forward<Args>(args)...);
                break;
            case CV_8U:
                opDispatch<uint8_t>(std::forward<Args>(args)...);
                break;
            case CV_8S:
                opDispatch<int8_t>(std::forward<Args>(args)...);
                break;
            case CV_32S:
                opDispatch<int32_t>(std::forward<Args>(args)...);
                break;
            case CV_64S:
                opDispatch<int64_t>(std::forward<Args>(args)...);
                break;
            case CV_32F:
                CV_Assert(op != OPERATION::BITSHIFT && op != OPERATION::AND &&
                          op != OPERATION::OR && op != OPERATION::XOR);
                opDispatch<float>(std::forward<Args>(args)...);
                break;
            default:
                CV_Error(cv::Error::BadDepth, "Unsupported type.");
        };
    }

#ifdef HAVE_CUDA
    Ptr<BackendNode> initCUDA(
        void *context_,
        const std::vector<Ptr<BackendWrapper>>& inputs,
        const std::vector<Ptr<BackendWrapper>>& outputs
    ) override
    {
        auto context = reinterpret_cast<csl::CSLContext*>(context_);

        cuda4dnn::EltwiseOpType op_ = cuda4dnn::EltwiseOpType::SUM;
        switch (op) {
            case OPERATION::MAX:
                op_ = cuda4dnn::EltwiseOpType::MAX;
                break;
            case OPERATION::MIN:
                op_ = cuda4dnn::EltwiseOpType::MIN;
                break;
            case OPERATION::SUM:
                op_ = cuda4dnn::EltwiseOpType::SUM;
                break;
            case OPERATION::PROD:
                op_ = cuda4dnn::EltwiseOpType::PRODUCT;
                break;
            case OPERATION::DIV:
                op_ = cuda4dnn::EltwiseOpType::DIV;
                break;
            case OPERATION::ADD:
                op_ = cuda4dnn::EltwiseOpType::SUM;
                break;
            case OPERATION::SUB:
                op_ = cuda4dnn::EltwiseOpType::SUB;
                break;
            case OPERATION::MOD:
                op_ = cuda4dnn::EltwiseOpType::MOD;
                break;
            case OPERATION::FMOD:
                op_ = cuda4dnn::EltwiseOpType::FMOD;
                break;
            case OPERATION::POW:
                op_ = cuda4dnn::EltwiseOpType::POW;
                break;
            default: return Ptr<BackendNode>(); // return empty cuda_node if the EltwiseOpType is unsupported type.
        };

        return make_cuda_node_with_type<cuda4dnn::EltwiseOp>(preferableTarget, inputs[0]->getHostMatDepth(), std::move(context->stream), op_, std::vector<float>());
    }
#endif

#ifdef HAVE_CANN
    virtual Ptr<BackendNode> initCann(const std::vector<Ptr<BackendWrapper> > &inputs,
                                      const std::vector<Ptr<BackendWrapper> > &outputs,
                                      const std::vector<Ptr<BackendNode> >& nodes) CV_OVERRIDE
    {
        CV_Assert(inputs.size() == 2);
        CV_Assert(nodes.size() == 2);

        auto op_x1 = nodes[0].dynamicCast<CannBackendNode>()->getOp();
        auto x1 = inputs[0].dynamicCast<CannBackendWrapper>();
        auto x1_desc = x1->getTensorDesc();
        auto op_x2 = nodes[1].dynamicCast<CannBackendNode>()->getOp();
        auto x2 = inputs[1].dynamicCast<CannBackendWrapper>();
        auto x2_desc = x2->getTensorDesc();
        auto output_desc = std::make_shared<ge::TensorDesc>(ge::Shape(), ge::FORMAT_NCHW, ge::DT_FLOAT);

        std::shared_ptr<ge::Operator> eltwise_operator = nullptr;
        // add, mul, sub, div, max, min
        switch (op)
        {
#define BUILD_CANN_ELTWISE_OP(op_type, class_name, op_name)                 \
            case op_type: {                                                 \
                auto eltwise_op =                                           \
                  std::make_shared<ge::op::class_name>(op_name);            \
                eltwise_op->set_input_x1_by_name(*op_x1, x1->name.c_str()); \
                eltwise_op->set_input_x2_by_name(*op_x2, x2->name.c_str()); \
                eltwise_op->update_input_desc_x1(*x1_desc);                 \
                eltwise_op->update_input_desc_x2(*x2_desc);                 \
                eltwise_op->update_output_desc_y(*output_desc);             \
                eltwise_operator = eltwise_op;                              \
            } break;
            BUILD_CANN_ELTWISE_OP(OPERATION::ADD,  Add,     name);
            BUILD_CANN_ELTWISE_OP(OPERATION::PROD, Mul,     name);
            BUILD_CANN_ELTWISE_OP(OPERATION::SUB,  Sub,     name);
            BUILD_CANN_ELTWISE_OP(OPERATION::DIV,  Xdivy,   name);
            BUILD_CANN_ELTWISE_OP(OPERATION::MAX,  Maximum, name);
            BUILD_CANN_ELTWISE_OP(OPERATION::MIN,  Minimum, name);
            BUILD_CANN_ELTWISE_OP(OPERATION::MOD,  Mod,     name);
            BUILD_CANN_ELTWISE_OP(OPERATION::FMOD, Mod,     name);
            BUILD_CANN_ELTWISE_OP(OPERATION::POW,  Pow,     name);
#undef BUILD_CANN_ELTWISE_OP
            default: CV_Error(Error::StsNotImplemented, "Unsupported eltwise operation");
        }

        return Ptr<BackendNode>(new CannBackendNode(eltwise_operator));
    }
#endif // HAVE_CANN

    virtual int64 getFLOPS(const std::vector<MatShape> &inputs,
                           const std::vector<MatShape> &outputs) const CV_OVERRIDE
    {
        CV_Assert(inputs.size());
        return inputs.size() * total(outputs[0]);
    }

#ifdef HAVE_DNN_NGRAPH
    virtual Ptr<BackendNode> initNgraph(const std::vector<Ptr<BackendWrapper> >& inputs, const std::vector<Ptr<BackendNode> >& nodes) CV_OVERRIDE
    {
        // In case only one input
        if (inputs.size() == 1) {
            auto &ieInpNode = nodes[0].dynamicCast<InfEngineNgraphNode>()->node;
            ov::OutputVector inp{ieInpNode};
            auto blank = std::make_shared<ov::op::v0::Concat>(inp, 0);
            return Ptr<BackendNode>(new InfEngineNgraphNode(blank));
        }

        // TODO: Support multiple (>=3) inputs

        if (op == OPERATION::WHERE)
            CV_CheckEQ(inputs.size(), 3u, "");
        else
            CV_CheckEQ(inputs.size(), 2u, "");
        auto& inp0 = nodes[0].dynamicCast<InfEngineNgraphNode>()->node;
        auto& inp1 = nodes[1].dynamicCast<InfEngineNgraphNode>()->node;

        if (op != OPERATION::WHERE && inp0.get_element_type() != inp1.get_element_type()) {
            CV_Assert(inp0.get_element_type() == ov::element::f16 || inp0.get_element_type() == ov::element::f32);
            CV_Assert(inp1.get_element_type() == ov::element::f16 || inp1.get_element_type() == ov::element::f32);
            auto dtype = preferableTarget == DNN_TARGET_OPENCL_FP16 || preferableTarget == DNN_TARGET_MYRIAD ?
                        ov::element::f16 : ov::element::f32;
            if (inp0.get_element_type() != dtype)
                inp0 = std::make_shared<ov::op::v0::Convert>(inp0, dtype);
            if (inp1.get_element_type() != dtype)
                inp1 = std::make_shared<ov::op::v0::Convert>(inp1, dtype);
        }

        std::shared_ptr<ov::Node> node;
        if (op == OPERATION::ADD)
            node = std::make_shared<ov::op::v1::Add>(inp0, inp1);
        else if (op == OPERATION::PROD)
            node = std::make_shared<ov::op::v1::Multiply>(inp0, inp1);
        else if (op == OPERATION::EQUAL)
            node = std::make_shared<ov::op::v1::Equal>(inp0, inp1);
        else if (op == OPERATION::GREATER)
            node = std::make_shared<ov::op::v1::Greater>(inp0, inp1);
        else if (op == OPERATION::GREATER_EQUAL)
            node = std::make_shared<ov::op::v1::GreaterEqual>(inp0, inp1);
        else if (op == OPERATION::LESS)
            node = std::make_shared<ov::op::v1::Less>(inp0, inp1);
        else if (op == OPERATION::LESS_EQUAL)
            node = std::make_shared<ov::op::v1::LessEqual>(inp0, inp1);
        else if (op == OPERATION::AND)
            node = std::make_shared<ov::op::v1::LogicalAnd>(inp0, inp1);
        else if (op == OPERATION::OR)
            node = std::make_shared<ov::op::v1::LogicalOr>(inp0, inp1);
        else if (op == OPERATION::XOR)
            node = std::make_shared<ov::op::v1::LogicalXor>(inp0, inp1);
        else if (op == OPERATION::WHERE)
        {
            auto& inp2 = nodes[2].dynamicCast<InfEngineNgraphNode>()->node;
            node = std::make_shared<ov::op::v1::Select>(inp0, inp1, inp2);
        }
        // Ideally we should do this but int32 internal blobs are converted to float32 data type in inference.
        // TODO: Remove data type convertion when we have type inference.
        else if (op == OPERATION::MOD) {
            auto inp0_i64 = std::make_shared<ov::op::v0::Convert>(inp0, ov::element::i64);
            auto inp1_i64 = std::make_shared<ov::op::v0::Convert>(inp1, ov::element::i64);
            auto mod = std::make_shared<ov::op::v1::FloorMod>(inp0_i64, inp1_i64);
            node = std::make_shared<ov::op::v0::Convert>(mod, ov::element::f32);
        }
        else if (op == OPERATION::FMOD)
            node = std::make_shared<ov::op::v1::Mod>(inp0, inp1);
        else
            CV_Error(Error::StsNotImplemented, "Operation is not implemented for nGraph backend");
        return Ptr<BackendNode>(new InfEngineNgraphNode(node));
    }
#endif

#ifdef HAVE_VULKAN
    virtual Ptr<BackendNode> initVkCom(const std::vector<Ptr<BackendWrapper> > &inputs,
                                       std::vector<Ptr<BackendWrapper> > &outputs) CV_OVERRIDE
    {
        Ptr<vkcom::OpBase> op = makePtr<vkcom::OpNary>((vkcom::OpNary::OPERATION) this->op, helper.ninputs, helper.max_ndims, helper.shapes, helper.steps);
        return Ptr<BackendNode>(makePtr<VkComBackendNode>(inputs, op, outputs));
    }
#endif

};

Ptr<NaryEltwiseLayer> NaryEltwiseLayer::create(const LayerParams& params)
{
    return Ptr<NaryEltwiseLayer>(new NaryEltwiseLayerImpl(params));
}

}
}
