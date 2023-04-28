// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "layers_common.hpp"
#include "../op_cuda.hpp"
#include "../op_cann.hpp"
#include "../ie_ngraph.hpp"

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
                   op == OPERATION::DIV || op == OPERATION::MAX  || op == OPERATION::MIN;
#endif
        if (backendId == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
            return (op == OPERATION::ADD ||
                    op == OPERATION::PROD ||
                    op == OPERATION::GREATER_EQUAL ||
                    op == OPERATION::LESS_EQUAL
            );
        if (op == OPERATION::MAX || op == OPERATION::MIN || op == OPERATION::SUM ||
            op == OPERATION::PROD || op == OPERATION::DIV || op == OPERATION::ADD)
            return backendId == DNN_BACKEND_OPENCV || backendId == DNN_BACKEND_CUDA;
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
        // ** prepend dimension 1 to the mat of less dims
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
    void binary_forward_impl(
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
    void binary_forward(const Functor& f, const std::vector<Mat>& inputs, std::vector<Mat>& outputs)
    {
        const Mat& a = inputs[0];
        const Mat& b = inputs[1];
        Mat& out = outputs[0];

        // collect info of inputs and output
        const int* in_shape[] = {a.size.p, b.size.p};
        const size_t* in_step[] = {a.step.p, b.step.p};
        const int* out_shape = out.size.p;
        const size_t* out_step = out.step.p;
        const int in_ndims[] = {a.dims, b.dims};
        int out_ndims = out.dims;

        int max_ndims = std::max(a.dims, std::max(b.dims, out.dims));

        // buf holds the folllowing for a, b & output:
        //  * orig_shapes, shapes (result_shape), orig_steps, steps (result_step), 3*4 elements in total
        //  * shape_buf & step_buf, 3*2*max_ndims elements in total
        //  * all_ndims, 3*1 elements in total
        //  * all_type_sizes, 3*1 elements in total
        AutoBuffer<size_t> buf(3 * (2 * max_ndims + 6));

        int** orig_shapes = (int**)(buf.data());
        int** shapes = orig_shapes + 3;
        size_t** orig_steps = (size_t**)(shapes + 3);
        size_t** steps = orig_steps + 3;

        int* shape_buf = (int*)(steps + 3);
        size_t* step_buf = (size_t*)(shape_buf + 3 * max_ndims);

        int* all_ndims = (int*)(step_buf + 3 * max_ndims);
        size_t* all_type_sizes = (size_t*)(all_ndims + 3);

        // assign orig_shapes, shapes, orig_steps, steps, all_ndims, all_type_sizes
        for (int i = 0; i < 3; i++)
        {
            orig_shapes[i] = (int*)(i == 0 ? out_shape : in_shape[i-1]);
            orig_steps[i] = (size_t*)(i == 0 ? out_step : in_step[i-1]);
            shapes[i] = shape_buf + i * max_ndims;
            steps[i] = step_buf + i * max_ndims;
            all_ndims[i] = i == 0 ? out_ndims : in_ndims[i-1];
            all_type_sizes[i] = sizeof(T);
        }

        if (!prepare_for_broadcast_op(3, max_ndims, all_type_sizes,
                                      all_ndims, (const int**)orig_shapes,
                                      (const size_t**)orig_steps,
                                      shapes, steps))
            return;

        binary_forward_impl<T, Functor>(
                max_ndims, shapes[0], a.ptr<char>(), steps[1],
                b.ptr<char>(), steps[2], out.ptr<char>(), steps[0],
                f);
    }

    template<typename T, typename Functor>
    void nary_forward_impl(
        const Functor& f, const T scale, int ninputs, int ndims, const int* shape,
        const char** inp, char* out,
        const size_t** steps, char** ptrs)
    {
        CV_Assert(ndims >= 2);
        size_t dp = steps[0][ndims-1]/sizeof(T);
        size_t dp1 = steps[1][ndims-1]/sizeof(T);
        size_t dp2 = steps[2][ndims-1]/sizeof(T);

        enum { BLOCK_SIZE = 1024 };
        T blck[BLOCK_SIZE];

        int k, i, di1=0, n1 = shape[ndims-1], n2 = shape[ndims-2];
        int second = ninputs == 1 ? 1 : 2;
        size_t plane_idx, nplanes = 1;
        for (k = 0; k < ndims-2; k++) nplanes *= shape[k];

        for (plane_idx = 0; plane_idx < nplanes; plane_idx++) {
            ptrs[0] = out;
            for (i = 0; i < ninputs; i++) ptrs[i+1] = (char*)inp[i];
            size_t idx = plane_idx;
            for (k = ndims-3; k >= 0; k--) {
                size_t next_idx = idx/shape[k];
                int i_k = (int)(idx - next_idx*shape[k]);
                for (i = 0; i < ninputs; i++)
                    ptrs[i] += i_k*steps[i][k];
                idx = next_idx;
            }
            for (int i2 = 0; i2 < n2; i2++)
            {
                const T* ptr1 = (const T*)(ptrs[1] + steps[1][ndims-2]*i2);
                const T* ptr2 = (const T*)(ptrs[second] + steps[second][ndims-2]*i2);
                T* ptr = (T*)(ptrs[0] + steps[0][ndims-2]*i2);
                if (ninputs <= 2) {
                    if (dp1 == 1 && dp2 == 1) {
                        for (int i1 = 0; i1 < n1; i1++)
                            ptr[i1] = saturate_cast<T>(f(ptr1[i1], ptr2[i1])*scale);
                    } else {
                        for(int i1 = 0; i1 < n1; i1++, ptr1 += dp1, ptr2 += dp2, ptr += dp)
                            *ptr = saturate_cast<T>(f(*ptr1, *ptr2)*scale);
                    }
                } else {
                    for (int i1 = 0; i1 < n1; i1 += di1, ptr += di1) {
                        di1 = BLOCK_SIZE < n1-i1 ? BLOCK_SIZE : n1-i1;
                        if (dp1 == 1 && dp2 == 1) {
                            for (int j = 0; j < di1; j++)
                                blck[j] = f(ptr1[j], ptr2[j]);
                            ptr1 += di1;
                            ptr2 += di1;
                        } else {
                            for(int j = 0; j < di1; j++, ptr1 += dp1, ptr2 += dp2)
                                blck[j] = f(*ptr1, *ptr2);
                        }
                        for(i = 2; i < ninputs; i++) {
                            int dp_i = steps[i+1][ndims-1]/sizeof(T);
                            const T* ptr_i = (const T*)(ptrs[i+1] +
                                    steps[i+1][ndims-2]*i2) + i1*dp_i;
                            if (dp_i == 1) {
                                if (i < ninputs-1) {
                                    for (int j = 0; j < di1; j++)
                                        blck[j] = f(blck[j], ptr_i[j]);
                                } else {
                                    for (int j = 0; j < di1; j++)
                                        ptr[j] = saturate_cast<T>(f(blck[j], ptr_i[j]) * scale);
                                }
                            } else {
                                if (i < ninputs-1) {
                                    for (int j = 0; j < di1; j++, ptr_i += dp_i)
                                        blck[j] = f(blck[j], *ptr_i);
                                } else {
                                    for (int j = 0; j < di1; j++, ptr_i += dp_i)
                                        ptr[j] = saturate_cast<T>(f(blck[j], *ptr_i) * scale);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    template <typename T, typename Functor>
    void nary_forward(
        const Functor& f, T scale,
        const std::vector<Mat>& inputs, std::vector<Mat>& outputs
        )
    {
        int ninputs = inputs.size();

        // collect all input
        std::vector<const char*> v_inp;
        std::transform(inputs.begin(), inputs.end(), std::back_inserter(v_inp), [] (const Mat& m) { return m.template ptr<const char>(); });
        const char** inp = v_inp.data();

        // collect ndims of all input
        std::vector<int> v_inp_dims;
        std::transform(inputs.begin(), inputs.end(), std::back_inserter(v_inp_dims), [] (const Mat& m) { return m.dims; });
        const int* inp_ndims = v_inp_dims.data();

        // collect shapes of all input
        std::vector<const int*> v_inp_shape;
        std::transform(inputs.begin(), inputs.end(), std::back_inserter(v_inp_shape), [] (const Mat& m) { return m.size.p; });
        const int** inp_shape = v_inp_shape.data();

        // collect steps of all input
        std::vector<const size_t*> v_inp_step;
        std::transform(inputs.begin(), inputs.end(), std::back_inserter(v_inp_step), [] (const Mat& m) { return m.step.p; });
        const size_t** inp_step = v_inp_step.data();

        // collect info of output (ndims, shape, step)
        char* out = outputs[0].ptr<char>();
        int out_ndims = outputs[0].dims;
        const int* out_shape = outputs[0].size.p;
        const size_t* out_step = outputs[0].step.p;

        // find max ndims for broadcasting
        int i, max_ndims = out_ndims > 2 ? out_ndims : 2;
        for(i = 0; i < ninputs; i++)
            max_ndims = max_ndims > inp_ndims[i] ? max_ndims : inp_ndims[i];

        // buf holds the following buffers for inputs & output:
        //  * orig_shapes, shapes (result_shape), orig_steps, steps (result_step), (ninputs+1)*4 elements in total
        //  * ptrs, (ninputs+1)*1 elements in total
        //  * shape_buf & step_buf, (ninputs+1)*2*max_ndims elements in total
        //  * all_ndims, (ninputs+1)*1 elements in total
        //  * all_type_sizes, (ninputs+1)*1 elements in total
        AutoBuffer<size_t> buf((ninputs + 1) * (2 * max_ndims + 7));

        int** orig_shapes = (int**)buf.data();
        int** shapes = orig_shapes + ninputs + 1;
        size_t** orig_steps = (size_t**)(shapes + ninputs + 1);
        size_t** steps = orig_steps + ninputs + 1;

        char** ptrs = (char**)(steps + ninputs + 1);

        size_t* step_buf = (size_t*)(ptrs + ninputs + 1);
        int* shape_buf = (int*)(step_buf + (ninputs + 1)*max_ndims);

        int* all_ndims = shape_buf + (ninputs + 1)*max_ndims;
        size_t* all_type_sizes = (size_t*)(all_ndims + ninputs + 1);

        for(i = 0; i <= ninputs; i++) {
            all_ndims[i] = i == 0 ? out_ndims : inp_ndims[i-1];
            all_type_sizes[i] = sizeof(T);
            orig_shapes[i] = (int*)(i == 0 ? out_shape : inp_shape ? inp_shape[i-1] : 0);
            orig_steps[i] = (size_t*)(i == 0 ? out_step : inp_step ? inp_step[i-1] : 0);
            shapes[i] = shape_buf + max_ndims*i;
            steps[i] = step_buf + max_ndims*i;
        }

        if (!prepare_for_broadcast_op(ninputs + 1, max_ndims, all_type_sizes,
                                      all_ndims, (const int**)orig_shapes,
                                      (const size_t**)orig_steps,
                                      shapes, steps))
            return;

        nary_forward_impl<T>(
                f, scale, ninputs, max_ndims, shapes[0], inp, out, (const size_t **) steps, ptrs);
    }

    template <typename T, typename Functor>
    void trinary_forward(const Functor& f, const std::vector<Mat>& inputs, std::vector<Mat>& outputs)
    {
        const Mat& a = inputs[0];
        const Mat& b = inputs[1];
        const Mat& c = inputs[2];
        Mat& out = outputs[0];

        // collect info of inputs and output
        const int* in_shape[] = {a.size.p, b.size.p, c.size.p};
        const size_t* in_step[] = {a.step.p, b.step.p, c.step.p};
        const int* out_shape = out.size.p;
        const size_t* out_step = out.step.p;
        const int in_ndims[] = {a.dims, b.dims, c.dims};
        int out_ndims = out.dims;

        int max_ndims = std::max(a.dims, std::max(b.dims, std::max(c.dims, out.dims)));

        AutoBuffer<size_t> buf(4 * (2 * max_ndims + 6));

        int** orig_shapes = (int**)(buf.data());
        int** shapes = orig_shapes + 4;
        size_t** orig_steps = (size_t**)(shapes + 4);
        size_t** steps = orig_steps + 4;

        int* shape_buf = (int*)(steps + 4);
        size_t* step_buf = (size_t*)(shape_buf + 4 * max_ndims);

        int* all_ndims = (int*)(step_buf + 4 * max_ndims);
        size_t* all_type_sizes = (size_t*)(all_ndims + 4);

        // assign orig_shapes, shapes, orig_steps, steps, all_ndims, all_type_sizes
        for (int i = 0; i < 4; i++)
        {
            orig_shapes[i] = (int*)(i == 0 ? out_shape : in_shape[i-1]);
            orig_steps[i] = (size_t*)(i == 0 ? out_step : in_step[i-1]);
            shapes[i] = shape_buf + i * max_ndims;
            steps[i] = step_buf + i * max_ndims;
            all_ndims[i] = i == 0 ? out_ndims : in_ndims[i-1];
            all_type_sizes[i] = sizeof(T);
        }

        if (!prepare_for_broadcast_op(4, max_ndims, all_type_sizes,
                                      all_ndims, (const int**)orig_shapes,
                                      (const size_t**)orig_steps,
                                      shapes, steps))
            return;

        trinary_forward_impl<T, Functor>(
                max_ndims, shapes[0], a.ptr<char>(), steps[1], b.ptr<char>(), steps[2],
                c.ptr<char>(), steps[3], out.ptr<char>(), steps[0],
                f);
    }

    template <typename T, typename Functor>
    void trinary_forward_impl(
            int ndims, const int* shape,
            const char* data1, const size_t* step1,
            const char* data2, const size_t* step2,
            const char* data3, const size_t* step3,
            char* data, const size_t* step,
            const Functor& op)
    {
        assert(ndims >= 2);
        size_t dp1 = step1[ndims-1]/sizeof(T);
        size_t dp2 = step2[ndims-1]/sizeof(T);
        size_t dp3 = step3[ndims-1]/sizeof(T);
        size_t dp = step[ndims-1]/sizeof(T);
        int k, n1 = shape[ndims-1], n2 = shape[ndims-2];
        size_t plane_idx, nplanes = 1;
        for (k = 0; k < ndims-2; k++) nplanes *= shape[k];

        for (plane_idx = 0; plane_idx < nplanes; plane_idx++)
        {
            const char* ptr1_ = data1;
            const char* ptr2_ = data2;
            const char* ptr3_ = data3;
            char* ptr_ = data;
            size_t idx = plane_idx;
            for (k = ndims-3; k >= 0; k--)
            {
                size_t next_idx = idx/shape[k];
                int i_k = (int)(idx - next_idx*shape[k]);
                ptr1_ += i_k*step1[k];
                ptr2_ += i_k*step2[k];
                ptr3_ += i_k*step3[k];
                ptr_ += i_k*step[k];
                idx = next_idx;
            }

            for (int i2 = 0; i2 < n2; i2++, ptr1_ += step1[ndims-2],
                                            ptr2_ += step2[ndims-2],
                                            ptr3_ += step3[ndims-2],
                                            ptr_ += step[ndims-2])
            {
                const T* ptr1 = (const T*)ptr1_;
                const T* ptr2 = (const T*)ptr2_;
                const T* ptr3 = (const T*)ptr3_;
                T* ptr = (T*)ptr_;

                if (dp1 == 1 && dp2 == 1 && dp3 == 1 && dp == 1)
                {
                    for(int i1 = 0; i1 < n1; i1++)
                        ptr[i1] = op(ptr1[i1], ptr2[i1], ptr3[i1]);
                }
                else
                {
                    for(int i1 = 0; i1 < n1; i1++, ptr1 += dp1, ptr2 += dp2, ptr3 += dp3, ptr += dp)
                        *ptr = op(*ptr1, *ptr2, *ptr3);
                }
            }
        }
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

        // TODO: assert types
        typeDispatch(outputs[0].type(), inputs.size(), inputs, outputs);
    }

    template<typename T, typename... Args>
    inline void opDispatch(size_t ninputs, Args&&... args)
    {
        switch (op)
        {
            case OPERATION::EQUAL:
            {
                auto equal = [](const T &a, const T &b) { return a == b; };
                binary_forward<T>(equal, std::forward<Args>(args)...);
                break;
            }
            case OPERATION::GREATER:
            {
                auto greater = [](const T &a, const T &b) { return a > b; };
                binary_forward<T>(greater, std::forward<Args>(args)...);
                break;
            }
            case OPERATION::GREATER_EQUAL:
            {
                auto greater_equal = [](const T &a, const T &b) { return a >= b; };
                binary_forward<T>(greater_equal, std::forward<Args>(args)...);
                break;
            }
            case OPERATION::LESS:
            {
                auto less = [](const T &a, const T &b) { return a < b; };
                binary_forward<T>(less, std::forward<Args>(args)...);
                break;
            }
            case OPERATION::LESS_EQUAL:
            {
                auto less_equal = [](const T &a, const T &b) { return a <= b; };
                binary_forward<T>(less_equal, std::forward<Args>(args)...);
                break;
            }
            case OPERATION::POW:
            {
                auto pow = [] (const T& a, const T& b) { return std::pow(a, b); };
                binary_forward<T>(pow, std::forward<Args>(args)...);
                break;
            }
            case OPERATION::BITSHIFT:
            {
                auto bitshift = [] (const uint8_t &a, const uint8_t &b) { return a << b; };
                binary_forward<T>(bitshift, std::forward<Args>(args)...);
                break;
            }
            case OPERATION::MAX:
            {
                auto max = [](const T &a, const T &b) { return std::max(a, b); };
                nary_forward<T>(max, T{1}, std::forward<Args>(args)...);
                break;
            }
            case OPERATION::MEAN:
            {
                auto mean = [](const T &a, const T &b) { return (a + b) / T{2}; };
                nary_forward<T>(mean, T{1} / ninputs, std::forward<Args>(args)...);
                break;
            }
            case OPERATION::MIN:
            {
                auto min = [](const T &a, const T &b) { return std::min(a, b); };
                nary_forward<T>(min, T{1}, std::forward<Args>(args)...);
                break;
            }
            case OPERATION::MOD:
            {
                auto mod = [](const uint8_t &a, const uint8_t &b) { return a % b; };
                binary_forward<T>(mod, std::forward<Args>(args)...);
                break;
            }
            case OPERATION::PROD:
            {
                auto prod = [](const T &a, const T &b) { return a * b; };
                binary_forward<T>(prod, std::forward<Args>(args)...);
                break;
            }
            case OPERATION::SUB:
            {
                auto sub = [](const T &a, const T &b) { return a - b; };
                binary_forward<T>(sub, std::forward<Args>(args)...);
                break;
            }
            case OPERATION::SUM:
            {
                auto sum = [](const T &a, const T &b) { return a + b; };
                nary_forward<T>(sum, T{1}, std::forward<Args>(args)...);
                break;
            }
            case OPERATION::ADD:
            {
                auto add = [](const T &a, const T &b) { return a + b; };
                binary_forward<T>(add, std::forward<Args>(args)...);
                break;
            }
            case OPERATION::DIV:
            {
                auto div = [](const T &a, const T &b) { return a / b; };
                binary_forward<T>(div, std::forward<Args>(args)...);
                break;
            }
            case OPERATION::AND:
            {
                auto op_and = [](const uint8_t &a, const uint8_t &b) { return a & b; };
                binary_forward<T>(op_and, std::forward<Args>(args)...);
                break;
            }
            case OPERATION::OR:
            {
                auto op_or = [](const uint8_t &a, const uint8_t &b) { return a | b; };
                binary_forward<T>(op_or, std::forward<Args>(args)...);
                break;
            }
            case OPERATION::XOR:
            {
                auto op_xor = [](const uint8_t &a, const uint8_t &b) { return a ^ b; };
                binary_forward<T>(op_xor, std::forward<Args>(args)...);
                break;
            }
            case OPERATION::WHERE:
            {
                auto op_where = [](const T &a, const T &b, const T &c) { return a ? b : c; };
                trinary_forward<T>(op_where, std::forward<Args>(args)...);
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
            case CV_32S:
                opDispatch<int32_t>(std::forward<Args>(args)...);
                break;
            case CV_32F:
                CV_Assert(op != OPERATION::BITSHIFT && op != OPERATION::MOD &&
                          op != OPERATION::AND && op != OPERATION::OR &&
                          op != OPERATION::XOR);
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

        auto input_0_shape = inputs[0].dynamicCast<CUDABackendWrapper>()->getShape();
        for (int i = 1; i < inputs.size(); i++)
        {
            auto input_i_shape = inputs[i].dynamicCast<CUDABackendWrapper>()->getShape();
            if (input_0_shape.size() != input_i_shape.size())
                return Ptr<BackendNode>();
            // check if the shape can be supported by `eltwise_ops.cu`, or return the default BackendNode
            for (int j = 0; j < input_0_shape.size(); j++)
                if (input_0_shape[j] != input_i_shape[j] &&
                    input_0_shape[j] != 1 && input_i_shape[j] != 1)
                    return Ptr<BackendNode>();
        }

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
            default: return Ptr<BackendNode>(); // return empty cuda_node if the EltwiseOpType is unsupported type.
        };

        return make_cuda_node<cuda4dnn::EltwiseOp>(preferableTarget, std::move(context->stream), op_, std::vector<float>());
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
#undef BUILD_CANN_ELTWISE_OP
            default: CV_Error(Error::StsNotImplemented, "Unsupported eltwise operation");
        }

        return Ptr<BackendNode>(new CannBackendNode(eltwise_operator));
    }
#endif // HAVE_CANN

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

#ifdef HAVE_DNN_NGRAPH
    virtual Ptr<BackendNode> initNgraph(const std::vector<Ptr<BackendWrapper> >& inputs, const std::vector<Ptr<BackendNode> >& nodes) CV_OVERRIDE
    {
        CV_Assert(inputs.size() == 2);
        auto& inp0 = nodes[0].dynamicCast<InfEngineNgraphNode>()->node;
        auto& inp1 = nodes[1].dynamicCast<InfEngineNgraphNode>()->node;

        if (inp0->get_element_type() != inp1->get_element_type()) {
            auto dtype = preferableTarget == DNN_TARGET_OPENCL_FP16 || preferableTarget == DNN_TARGET_MYRIAD ?
                        ngraph::element::f16 : ngraph::element::f32;
            if (inp0->get_element_type() != dtype)
                inp0 = std::make_shared<ngraph::op::v0::Convert>(inp0, dtype);
            if (inp1->get_element_type() != dtype)
                inp1 = std::make_shared<ngraph::op::v0::Convert>(inp1, dtype);
        }

        std::shared_ptr<ngraph::Node> node;
        if (op == OPERATION::ADD)
            node = std::make_shared<ngraph::op::v1::Add>(inp0, inp1);
        else if (op == OPERATION::PROD)
            node = std::make_shared<ngraph::op::v1::Multiply>(inp0, inp1);
        else if (op == OPERATION::GREATER_EQUAL)
            node = std::make_shared<ngraph::op::v1::GreaterEqual>(inp0, inp1);
        else if (op == OPERATION::LESS_EQUAL)
            node = std::make_shared<ngraph::op::v1::LessEqual>(inp0, inp1);
        else
            CV_Error(Error::StsNotImplemented, "Operation is not implemented for nGraph backend");
        return Ptr<BackendNode>(new InfEngineNgraphNode(node));
    }
#endif
};

Ptr<NaryEltwiseLayer> NaryEltwiseLayer::create(const LayerParams& params)
{
    return Ptr<NaryEltwiseLayer>(new NaryEltwiseLayerImpl(params));
}

}
}
