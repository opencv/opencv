// This file is part of OpenCV project.
// Implements a simple registry for vector ops used by _InputArray/_OutputArray.

#include <unordered_map>

#include "opencv2/core/detail/input_array_ops.hpp"
#include "opencv2/core/utility.hpp"

namespace cv { namespace detail {

// Global registries guarded by a mutex to allow cross-thread access while
// keeping registration cheap. The address of the owning container is used as
// the key and remains stable for the lifetime of the std::vector object.

namespace {

using VecOpsMap = std::unordered_map<const void*, const VectorOpsBase*>;
using VecVecOpsMap = std::unordered_map<const void*, const VectorVectorOpsBase*>;

VecOpsMap& get_vec_ops_storage()
{
    static VecOpsMap storage;
    return storage;
}

VecVecOpsMap& get_vecvec_ops_storage()
{
    static VecVecOpsMap storage;
    return storage;
}

cv::Mutex& get_registry_mutex()
{
    static cv::Mutex mtx;
    return mtx;
}

} // namespace

void register_vector_ops(const void* key, const VectorOpsBase* ops) {
    if (!key || !ops) return;
    cv::AutoLock lock(get_registry_mutex());
    get_vec_ops_storage()[key] = ops;
}

void unregister_vector_ops(const void* key) {
    if (!key) return;
    cv::AutoLock lock(get_registry_mutex());
    get_vec_ops_storage().erase(key);
}

const VectorOpsBase* get_vector_ops(const void* key) {
    if (!key) return nullptr;
    cv::AutoLock lock(get_registry_mutex());
    VecOpsMap& storage = get_vec_ops_storage();
    auto it = storage.find(key);
    return it == storage.end() ? nullptr : it->second;
}

void register_vector_vector_ops(const void* key, const VectorVectorOpsBase* ops) {
    if (!key || !ops) return;
    cv::AutoLock lock(get_registry_mutex());
    get_vecvec_ops_storage()[key] = ops;
}

void unregister_vector_vector_ops(const void* key) {
    if (!key) return;
    cv::AutoLock lock(get_registry_mutex());
    get_vecvec_ops_storage().erase(key);
}

const VectorVectorOpsBase* get_vector_vector_ops(const void* key) {
    if (!key) return nullptr;
    cv::AutoLock lock(get_registry_mutex());
    VecVecOpsMap& storage = get_vecvec_ops_storage();
    auto it = storage.find(key);
    return it == storage.end() ? nullptr : it->second;
}

}} // namespace cv::detail
