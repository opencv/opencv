// This file is part of OpenCV project.
// Internal helpers to enable type-safe std::vector handling in _InputArray/_OutputArray.

#ifndef OPENCV_CORE_DETAIL_INPUT_ARRAY_OPS_HPP
#define OPENCV_CORE_DETAIL_INPUT_ARRAY_OPS_HPP

#include <vector>
#include <cstddef>
#include "opencv2/core/cvdef.h" // for CV_EXPORTS visibility

namespace cv { namespace detail {

struct VectorOpsBase {
    virtual ~VectorOpsBase() {}
    virtual std::size_t size(const void* obj) const = 0;
    virtual const void* data(const void* obj) const = 0;
    virtual void* data(void* obj) const = 0;
    virtual void resize(void* obj, std::size_t len) const = 0;
};

template <typename T>
struct VectorOps final : VectorOpsBase {
    std::size_t size(const void* obj) const override {
        return static_cast<const std::vector<T>*>(obj)->size();
    }
    const void* data(const void* obj) const override {
        const auto& v = *static_cast<const std::vector<T>*>(obj);
        return v.empty() ? nullptr : static_cast<const void*>(v.data());
    }
    void* data(void* obj) const override {
        auto& v = *static_cast<std::vector<T>*>(obj);
        return v.empty() ? nullptr : static_cast<void*>(v.data());
    }
    void resize(void* obj, std::size_t len) const override {
        static_cast<std::vector<T>*>(obj)->resize(len);
    }
    static const VectorOps& instance() {
        static const VectorOps inst{};
        return inst;
    }
};

struct VectorVectorOpsBase {
    virtual ~VectorVectorOpsBase() {}
    virtual std::size_t outer_size(const void* obj) const = 0;
    virtual void outer_resize(void* obj, std::size_t len) const = 0;
    virtual std::size_t inner_size(const void* obj, int idx) const = 0;
    virtual const void* inner_data(const void* obj, int idx) const = 0;
    virtual void* inner_data(void* obj, int idx) const = 0;
    virtual void inner_resize(void* obj, int idx, std::size_t len) const = 0;
};

template <typename T>
struct VectorVectorOps final : VectorVectorOpsBase {
    std::size_t outer_size(const void* obj) const override {
        return static_cast<const std::vector<std::vector<T>>*>(obj)->size();
    }
    void outer_resize(void* obj, std::size_t len) const override {
        static_cast<std::vector<std::vector<T>>*>(obj)->resize(len);
    }
    std::size_t inner_size(const void* obj, int idx) const override {
        return static_cast<const std::vector<std::vector<T>>*>(obj)->at(idx).size();
    }
    const void* inner_data(const void* obj, int idx) const override {
        const auto& v = static_cast<const std::vector<std::vector<T>>*>(obj)->at(idx);
        return v.empty() ? nullptr : static_cast<const void*>(v.data());
    }
    void* inner_data(void* obj, int idx) const override {
        auto& v = static_cast<std::vector<std::vector<T>>*>(obj)->at(idx);
        return v.empty() ? nullptr : static_cast<void*>(v.data());
    }
    void inner_resize(void* obj, int idx, std::size_t len) const override {
        static_cast<std::vector<std::vector<T>>*>(obj)->at(idx).resize(len);
    }
    static const VectorVectorOps& instance() {
        static const VectorVectorOps inst{};
        return inst;
    }
};

// Registry API (thread-local) for vector ops
CV_EXPORTS void register_vector_ops(const void* key, const VectorOpsBase* ops);
CV_EXPORTS void retain_vector_ops(const void* key);
CV_EXPORTS void unregister_vector_ops(const void* key);
CV_EXPORTS const VectorOpsBase* get_vector_ops(const void* key);

// Registry API (thread-local) for vector<vector<...>> ops
CV_EXPORTS void register_vector_vector_ops(const void* key, const VectorVectorOpsBase* ops);
CV_EXPORTS void retain_vector_vector_ops(const void* key);
CV_EXPORTS void unregister_vector_vector_ops(const void* key);
CV_EXPORTS const VectorVectorOpsBase* get_vector_vector_ops(const void* key);

}} // namespace cv::detail

#endif // OPENCV_CORE_DETAIL_INPUT_ARRAY_OPS_HPP
