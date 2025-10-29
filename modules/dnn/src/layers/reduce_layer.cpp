// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include <opencv2/dnn/shape_utils.hpp>

#include "../op_cann.hpp"


namespace cv { namespace dnn {

class ReduceLayerImpl CV_FINAL : public ReduceLayer
{
public:
    ReduceLayerImpl(const LayerParams& params) {
        setParamsFrom(params);

        // set reduce type
        CV_Assert(params.has("reduce"));
        String op_type = toLowerCase(params.get<String>("reduce"));
        if (op_type == "max")
            reduce_type = ReduceType::MAX;
        else if (op_type == "min")
            reduce_type = ReduceType::MIN;
        else if (op_type == "mean")
            reduce_type = ReduceType::MEAN;
        else if (op_type == "sum")
            reduce_type = ReduceType::SUM;
        else if (op_type == "sum_square")
            reduce_type = ReduceType::SUM_SQUARE;
        else if (op_type == "l1")
            reduce_type = ReduceType::L1;
        else if (op_type == "l2")
            reduce_type = ReduceType::L2;
        else if (op_type == "log_sum")
            reduce_type = ReduceType::LOG_SUM;
        else if (op_type == "log_sum_exp")
            reduce_type = ReduceType::LOG_SUM_EXP;
        else if (op_type == "prod")
            reduce_type = ReduceType::PROD;
        else
            CV_Error(Error::StsBadArg, "Unknown reduce type\"" + op_type + "\"");

        keepdims = params.get<bool>("keepdims", true);
        noop_with_empty_axes = params.get<bool>("noop_with_empty_axes", false);

        // get axes if it is existed, otherwise reduce all
        if (params.has("axes")) {
            auto param_axes = params.get("axes");
            int num_axes = param_axes.size();
            axes.resize(num_axes);
            for (int i = 0; i < num_axes; ++i)
                axes[i] = param_axes.get<int>(i);
        }
    }

    virtual bool supportBackend(int backendId) CV_OVERRIDE {
#ifdef HAVE_CANN
        if (backendId == DNN_BACKEND_CANN)
            return reduce_type == ReduceType::MAX  || reduce_type == ReduceType::MIN     ||
                   reduce_type == ReduceType::MEAN || reduce_type == ReduceType::SUM     ||
                   reduce_type == ReduceType::PROD || reduce_type == ReduceType::LOG_SUM ||
                   reduce_type == ReduceType::LOG_SUM_EXP;
#endif
        return backendId == DNN_BACKEND_OPENCV;
    }

    virtual void finalize(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr) CV_OVERRIDE {
        if (axes.empty()) {
            return;
        }

        std::vector<Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);

        auto shape_input = shape(inputs[0]);
        for (auto i = 0; i < axes.size(); ++i) {
            auto norm_axis = normalize_axis(axes[i], shape_input);
            axes[i] = norm_axis;
        }
        if (shape_input.empty())
            return;

        bool do_nothing = true;
        for (auto axis : axes) {
            if (shape_input[axis] != 1 || keepdims) {
                do_nothing = false;
            }
        }
        if (do_nothing) {
            axes.clear();
            noop_with_empty_axes = true;
        }
    }

    bool getMemoryShapes(const std::vector<MatShape> &inputs,
                         const int requiredOutputs,
                         std::vector<MatShape> &outputs,
                         std::vector<MatShape> &internals) const CV_OVERRIDE
    {
        if (inputs[0].empty()){
            CV_CheckEQ(axes[0], 0, "Axis must be 0 when input is empty.");
            outputs.assign(1, MatShape());
            return false;
        }
        // empty axes
        if (axes.empty()) {
            if (noop_with_empty_axes) {
                // do nothing
                outputs.assign(1, inputs[0]);
            } else {
                // reduce all axes
                MatShape shape_output;
                if (keepdims) {
                    shape_output = inputs[0];
                    for (auto i = 0; i < shape_output.size(); ++i)
                        shape_output[i] = 1;
                } else {
                    shape_output.push_back(1);
                }
                outputs.assign(1, shape_output);
            }
        } else {
            auto shape_output_ = inputs[0];
            for (size_t i = 0; i < axes.size(); ++i) {
                auto norm_axis = normalize_axis(axes[i], inputs[0]);
                shape_output_[norm_axis] = -1;
            }
            MatShape shape_output;
            for (size_t i = 0; i < shape_output_.size(); ++i) {
                if (shape_output_[i] == -1) {
                    if (keepdims)
                        shape_output.push_back(1);
                    else
                        continue;
                } else
                    shape_output.push_back(shape_output_[i]);
            }
            if (shape_output.empty())
                shape_output.push_back(1);

            outputs.assign(1, shape_output);
        }

        return false;
    }

    virtual void getTypes(const std::vector<MatType>& inputs,
        const int requiredOutputs,
        const int requiredInternals,
        std::vector<MatType>& outputs,
        std::vector<MatType>& internals) const CV_OVERRIDE
    {
        CV_CheckType(inputs[0], inputs[0] == CV_32F || inputs[0] == CV_32S || inputs[0] == CV_64S || inputs[0] == CV_16F || inputs[0] == CV_8U || inputs[0] == CV_8S, "");
        outputs.assign(1, inputs[0]);
    }

    template <typename T>
    class ReduceBase {
    public:
        using dtype_input = T;

        ReduceBase(size_t n, const T& init) : n_(n), accumulator_(init) {}
        virtual void update(const T& a) = 0;
        virtual T get_value() { return accumulator_; }
        virtual ~ReduceBase() = default;
    protected:
        size_t n_;
        T accumulator_;
    };

    template <typename T>
    class ReduceMin : public ReduceBase<T> {
    public:
        ReduceMin(size_t n, const T& init) : ReduceBase<T>(n, init) {}
        void update(const T& a) override {
            this->accumulator_ = a > this->accumulator_ ? this->accumulator_ : a;
        }
    };

    template <typename T>
    class ReduceMax : public ReduceBase<T> {
    public:
        ReduceMax(size_t n, const T& init) : ReduceBase<T>(n, init) {}
        void update(const T& a) override {
            this->accumulator_ = a > this->accumulator_ ? a : this->accumulator_;
        }
    };

    template <typename T>
    class ReduceSum : public ReduceBase<T> {
    public:
        ReduceSum(size_t n, const T& init) : ReduceBase<T>(n, 0) {}
        void update(const T& a) override {
            this->accumulator_ += a;
        }
    };

    template <typename T>
    class ReduceMean : public ReduceSum<T> {
    public:
        ReduceMean(size_t n, const T& init) : ReduceSum<T>(n, init) {}
        T get_value() override {
            return this->accumulator_ / static_cast<T>(this->n_);
        }
    };

    template <typename T>
    class ReduceSumSquare : public ReduceBase<T> {
    public:
        ReduceSumSquare(size_t n, const T& init) : ReduceBase<T>(n, 0) {}
        void update(const T& a) override {
            this->accumulator_ += a * a;
        }
    };

    template <typename T>
    class ReduceL1 : public ReduceBase<T> {
    public:
        ReduceL1(size_t n, const T& init) : ReduceBase<T>(n, 0) {}
        void update(const T& a) override {
            this->accumulator_ += a > 0 ? a : -a;
        }
    };

    template <typename T>
    class ReduceL2 : public ReduceBase<T> {
    public:
        ReduceL2(size_t n, const T& init) : ReduceBase<T>(n, 0) {}
        void update(const T& a) override {
            this->accumulator_ += a * a;
        }
        T get_value() override {
            return std::sqrt(this->accumulator_);
        }
    };

    template <typename T>
    class ReduceProd : public ReduceBase<T> {
    public:
        ReduceProd(size_t n, const T& init) : ReduceBase<T>(n, 1) {}
        void update(const T& a) override {
            this->accumulator_ *= a;
        }
    };

    template <typename T>
    class ReduceLogSum : public ReduceBase<T> {
    public:
        ReduceLogSum(size_t n, const T& init) : ReduceBase<T>(n, 0) {}
        void update(const T& a) override {
            this->accumulator_ += a;
        }
        T get_value() override {
            return static_cast<T>(std::log(this->accumulator_));
        }
    };

    // FIXME: overflow caution
    template <typename T>
    class ReduceLogSumExp : public ReduceBase<T> {
    public:
        ReduceLogSumExp(size_t n, const T& init) : ReduceBase<T>(n, 0) {}
        void update(const T& a) override {
            this->accumulator_ += static_cast<T>(std::exp(a));
        }
        T get_value() override {
            return static_cast<T>(std::log(this->accumulator_));
        }
    };


    template <typename Op>
    class ReduceAllInvoker : public ParallelLoopBody {
    public:
        using dtype = typename Op::dtype_input;

        const Mat& src;
        Mat& dst;

        int n_reduce;
        int loop_size;

        int total;
        int cost_per_thread;

        ReduceAllInvoker(const Mat& src_, Mat& dst_) : src(src_), dst(dst_) {
            auto shape_src = shape(src);

            n_reduce = std::accumulate(shape_src.begin(), shape_src.end(), 1, std::multiplies<int>());
            loop_size = n_reduce;

            total = 1;
            cost_per_thread = 1;
        }

        void operator()(const Range& r) const CV_OVERRIDE {
            int start = r.start;
            int end = r.end;

            const dtype* p_src = src.ptr<const dtype>();
            dtype* p_dst = dst.ptr<dtype>();

            for (int i = start; i < end; ++i) {
                Op accumulator(n_reduce, *p_src);
                for (int l = 0; l < loop_size; ++l) {
                    accumulator.update(p_src[l]);
                }
                p_dst[i] = accumulator.get_value();
            }
        }
    };

    template <typename Op>
    class ReduceInvoker : public ParallelLoopBody {
    public:
        using dtype = typename Op::dtype_input;

        const Mat& src;
        Mat& dst;

        std::vector<int> reduced_axes; // assume in ascending order

        int n_reduce;
        int loop_size;

        int last_reduced_dim;
        int last_reduced_step;
        std::vector<int> projected_steps;

        int last_unreduced_dim;
        int last_unreduced_step;
        std::vector<int> unprojected_steps;

        int total;
        int cost_per_thread;

        ReduceInvoker(const Mat& src_, Mat& dst_, std::vector<int> axes_) : src(src_), dst(dst_), reduced_axes(axes_) {
            auto shape_src = shape(src);

            auto steps_src = shape_src;
            steps_src[steps_src.size() - 1] = 1;
            for (int i = static_cast<int>(steps_src.size()) - 2; i >= 0; --i)
                steps_src[i] = steps_src[i + 1] * shape_src[i + 1];

            size_t projection_size = 1;
            for (auto axis : reduced_axes) {
                projection_size *= shape_src[axis];
            }
            n_reduce = projection_size;

            last_reduced_dim = shape_src[reduced_axes.back()];
            last_reduced_step = steps_src[reduced_axes.back()];
            loop_size = last_reduced_dim * last_reduced_step;
            projection_size /= last_reduced_dim;

            // calculate projected_steps
            int last_reduced_axis = static_cast<int>(reduced_axes.size()) - 1;
            if (last_reduced_axis == 0) {
                projected_steps.resize(1, 0);
            } else {
                projected_steps.resize(projection_size);
                std::vector<int> projected_indices(last_reduced_axis, 0);
                for (size_t i = 0, current_step = 0; i < projection_size; ++i) {
                    projected_steps[i] = current_step;
                    ++projected_indices[last_reduced_axis - 1];
                    current_step += steps_src[reduced_axes[last_reduced_axis - 1]];
                    for (int j = last_reduced_axis - 1; j > 0; --j) {
                        if (projected_indices[j] < shape_src[reduced_axes[j]]) {
                            break;
                        }
                        projected_indices[j] = 0;
                        ++projected_indices[j - 1];
                        current_step = steps_src[reduced_axes[j - 1]];
                    }
                }
            }

            // calculate unprojected_steps
            std::vector<int> unreduced_axes;
            for (int i = 0; i < static_cast<int>(shape_src.size()); ++i) {
                if (std::find(reduced_axes.begin(), reduced_axes.end(), i) == reduced_axes.end()) {
                    unreduced_axes.push_back(i);
                }
            }
            size_t unprojection_size = 1;
            for (auto axis : unreduced_axes) {
                unprojection_size *= shape_src[axis];
            }
            last_unreduced_dim = shape_src[unreduced_axes.back()];
            last_unreduced_step = steps_src[unreduced_axes.back()];
            unprojection_size /= last_unreduced_dim;

            std::vector<int> unprojected_indices(unreduced_axes.size(), 0);
            unprojected_steps.reserve(unprojection_size);
            if (unprojected_indices.size() <= 1) {
                unprojected_steps.push_back(0);
            } else {
                for (size_t i = 0, current_step = 0; i < unprojection_size; ++i) {
                    unprojected_steps.push_back(current_step);
                    ++unprojected_indices[unprojected_indices.size() - 2];
                    current_step += steps_src[unreduced_axes[unreduced_axes.size() - 2]];
                    for (int j = static_cast<int>(unreduced_axes.size()) - 2; j > 0; --j) {
                        if (unprojected_indices[j] < shape_src[unreduced_axes[j]]) {
                            break;
                        }
                        unprojected_indices[j] -= shape_src[unreduced_axes[j]];
                        current_step -= shape_src[unreduced_axes[j]] * steps_src[unreduced_axes[j]];
                        ++unprojected_indices[j - 1];
                        current_step += steps_src[unreduced_axes[j - 1]];
                    }
                }
            }

            auto shape_dst = shape(dst);
            total = std::accumulate(shape_dst.begin(), shape_dst.end(), 1, std::multiplies<int>());
            cost_per_thread = static_cast<int>(projected_steps.size() * last_reduced_step);
        }

        static void run(const Mat& src, Mat& dst, std::vector<int> axes, bool noop_with_empty_axes) {
            CV_Assert(src.isContinuous());
            CV_Assert(dst.isContinuous());
            if (shape(src).empty() || (shape(src).size() == 1)){
                // since there is only one element no need for parallel compute
                // axis does not matter either (one element)
                ReduceAllInvoker<Op> p(src, dst);
                p(Range(0, p.total));
                return;
            }

            if (axes.empty()) {
                if (noop_with_empty_axes) {
                    // copyTo is not used here for the reason that we want a
                    // copy for the case when dims at all axes are 1
                    const auto p_src = src.ptr<const dtype>();
                    auto p_dst = dst.ptr<dtype>();
                    std::memcpy(p_dst, p_src, sizeof(dtype) * dst.total());
                    return;
                }

                ReduceAllInvoker<Op> p(src, dst);
                double nstripes = (size_t)p.total * (size_t)p.cost_per_thread * (1 / 1024.0);
                parallel_for_(Range(0, p.total), p, nstripes);
                return;
            }

            ReduceInvoker<Op> p(src, dst, axes);
            double nstripes = (size_t)p.total * (size_t)p.cost_per_thread * (1 / 1024.0);
            parallel_for_(Range(0, p.total), p, nstripes);
        }

        void operator()(const Range& r) const CV_OVERRIDE {
            int start = r.start;
            int end = r.end;

            const dtype* p_src = src.ptr<const dtype>();
            dtype* p_dst = dst.ptr<dtype>();

            size_t main_index = start / last_unreduced_dim;
            size_t loop = start % last_unreduced_dim;
            size_t origin = unprojected_steps[main_index] + loop * last_unreduced_step;
            for (int i = start; i < end; ++i) {
                Op accumulator(n_reduce, p_src[origin + projected_steps[0]]);
                for (auto projected_step : projected_steps) {
                    const dtype* loop_p_src = p_src + origin + projected_step;
                    for (auto l = 0; l < loop_size; l += last_reduced_step) {
                        accumulator.update(loop_p_src[l]);
                    }
                }
                p_dst[i] = accumulator.get_value();

                ++loop;
                if (loop >= last_unreduced_dim) {
                    loop = 0;
                    ++main_index;
                    if (main_index < unprojected_steps.size()) {
                        origin = unprojected_steps[main_index];
                    }
                } else {
                    origin += last_unreduced_step;
                }
            }
        }
    };

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

        typeDispatch(outputs[0].type(), inputs[0], outputs[0], axes, noop_with_empty_axes);
    }

    template <typename T, typename... Args>
    inline void opDispatch(Args&&... args) {
        switch (reduce_type) {
            case ReduceType::MAX:         ReduceInvoker<ReduceMax<T>>::run(std::forward<Args>(args)...);       break;
            case ReduceType::MIN:         ReduceInvoker<ReduceMin<T>>::run(std::forward<Args>(args)...);       break;
            case ReduceType::MEAN:        ReduceInvoker<ReduceMean<T>>::run(std::forward<Args>(args)...);      break;
            case ReduceType::SUM:         ReduceInvoker<ReduceSum<T>>::run(std::forward<Args>(args)...);       break;
            case ReduceType::L1:          ReduceInvoker<ReduceL1<T>>::run(std::forward<Args>(args)...);        break;
            case ReduceType::L2:          ReduceInvoker<ReduceL2<T>>::run(std::forward<Args>(args)...);        break;
            case ReduceType::PROD:        ReduceInvoker<ReduceProd<T>>::run(std::forward<Args>(args)...);      break;
            case ReduceType::SUM_SQUARE:  ReduceInvoker<ReduceSumSquare<T>>::run(std::forward<Args>(args)...); break;
            case ReduceType::LOG_SUM:     ReduceInvoker<ReduceLogSum<T>>::run(std::forward<Args>(args)...);    break;
            case ReduceType::LOG_SUM_EXP: ReduceInvoker<ReduceLogSumExp<T>>::run(std::forward<Args>(args)...); break;
            default: CV_Error(Error::StsBadArg, "DNN/Reduce: Unsupported operation.");
        }
    }

    template <typename... Args>
    inline void typeDispatch(const int type, Args&&... args) {
        switch (type) {
            case CV_8U: opDispatch<uint8_t>(std::forward<Args>(args)...); break;
            case CV_8S: opDispatch<int8_t>(std::forward<Args>(args)...); break;
            case CV_32S: opDispatch<int32_t>(std::forward<Args>(args)...); break;
            case CV_64S: opDispatch<int64_t>(std::forward<Args>(args)...); break;
            case CV_32F: opDispatch<float>(std::forward<Args>(args)...); break;
            default: CV_Error(cv::Error::BadDepth, "DNN/Reduce: Unsupported type.");
        }
    }

#ifdef HAVE_CANN
    virtual Ptr<BackendNode> initCann(const std::vector<Ptr<BackendWrapper> > &inputs,
                                      const std::vector<Ptr<BackendWrapper> > &outputs,
                                      const std::vector<Ptr<BackendNode> >& nodes) CV_OVERRIDE
    {
        CV_CheckFalse(axes.empty(), "DNN/CANN: Reduce layers need axes to build CANN operators");

        auto input_node = nodes[0].dynamicCast<CannBackendNode>()->getOp();
        auto input_wrapper = inputs[0].dynamicCast<CannBackendWrapper>();
        auto input_desc = input_wrapper->getTensorDesc();

        std::vector<int> axes_shape{(int)axes.size()};
        Mat axes_mat(axes_shape, CV_32S, &axes[0]);
        auto axes_node = std::make_shared<CannConstOp>(axes_mat.data, axes_mat.type(), axes_shape, cv::format("%s_axes", name.c_str()));
        auto axes_desc = axes_node->getTensorDesc();

        auto output_desc = std::make_shared<ge::TensorDesc>(ge::Shape(), ge::FORMAT_NCHW, ge::DT_FLOAT);

        std::shared_ptr<ge::Operator> reduce_op = nullptr;
        switch (reduce_type)
        {
#define BUILD_CANN_REDUCE_OP(op_type, class_name, op_name)                         \
            case op_type: {                                                        \
                auto op = std::make_shared<ge::op::class_name>(op_name);           \
                op->set_input_x_by_name(*input_node, input_wrapper->name.c_str()); \
                op->set_input_axes(*(axes_node)->getOp());                         \
                op->set_attr_keep_dims(keepdims);                                  \
                op->update_input_desc_x(*input_desc);                              \
                op->update_input_desc_axes(*axes_desc);                            \
                op->update_output_desc_y(*output_desc);                            \
                reduce_op = op;                                                    \
            } break;
            BUILD_CANN_REDUCE_OP(ReduceType::MAX,         ReduceMax,       name);
            BUILD_CANN_REDUCE_OP(ReduceType::MIN,         ReduceMin,       name);
            BUILD_CANN_REDUCE_OP(ReduceType::MEAN,        ReduceMean,      name);
            BUILD_CANN_REDUCE_OP(ReduceType::SUM,         ReduceSum,       name);
            BUILD_CANN_REDUCE_OP(ReduceType::PROD,        ReduceProd,      name);
            BUILD_CANN_REDUCE_OP(ReduceType::LOG_SUM,     ReduceLogSum,    name);
            BUILD_CANN_REDUCE_OP(ReduceType::LOG_SUM_EXP, ReduceLogSumExp, name);
#undef BUILD_CANN_REDUCE_OP
            default: CV_Error(Error::StsNotImplemented, "Unsupported reduce operation");
        }

        return Ptr<BackendNode>(new CannBackendNode(reduce_op));
    }
#endif // HAVE_CANN

private:
    enum ReduceType
    {
        MAX,
        MIN,
        MEAN,
        SUM,
        L1,
        L2,
        PROD,
        SUM_SQUARE,
        LOG_SUM,
        LOG_SUM_EXP
    } reduce_type;

    bool keepdims;
    bool noop_with_empty_axes;
    std::vector<int> axes;
};

Ptr<ReduceLayer> ReduceLayer::create(const LayerParams& params)
{
    return Ptr<ReduceLayer>(new ReduceLayerImpl(params));
}

}} // cv::dnn
