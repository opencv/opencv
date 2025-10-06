// This file is part of OpenCV project.
// Implements a simple thread-local registry for vector ops used by _InputArray/_OutputArray.

#include <unordered_map>

#include "opencv2/core/detail/input_array_ops.hpp"

namespace cv { namespace detail {

// Thread-local registries to avoid global contention and lifetime issues.
static thread_local std::unordered_map<const void*, const VectorOpsBase*> g_vec_ops;
static thread_local std::unordered_map<const void*, const VectorVectorOpsBase*> g_vecvec_ops;

void register_vector_ops(const void* key, const VectorOpsBase* ops) {
    if (!key || !ops) return;
    g_vec_ops[key] = ops;
}

void unregister_vector_ops(const void* key) {
    if (!key) return;
    g_vec_ops.erase(key);
}

const VectorOpsBase* get_vector_ops(const void* key) {
    if (!key) return nullptr;
    auto it = g_vec_ops.find(key);
    return it == g_vec_ops.end() ? nullptr : it->second;
}

void register_vector_vector_ops(const void* key, const VectorVectorOpsBase* ops) {
    if (!key || !ops) return;
    g_vecvec_ops[key] = ops;
}

void unregister_vector_vector_ops(const void* key) {
    if (!key) return;
    g_vecvec_ops.erase(key);
}

const VectorVectorOpsBase* get_vector_vector_ops(const void* key) {
    if (!key) return nullptr;
    auto it = g_vecvec_ops.find(key);
    return it == g_vecvec_ops.end() ? nullptr : it->second;
}

}} // namespace cv::detail
