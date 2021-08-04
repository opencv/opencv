// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#ifdef HAVE_ONEVPL
#include <stdlib.h>
#include <exception>
#include <mutex> //TODO spinlock

#include "streaming/vpl/vpl_cpu_accel.hpp"
#include "streaming/vpl/vpl_utils.hpp"
#include "streaming/vpl/surface/frame_adapter.hpp"
#include "streaming/vpl/surface/surface.hpp"
#include "logger.hpp"

#ifdef _WIN32
    #include <windows.h>
    #include <sysinfoapi.h>
#endif
namespace cv {
namespace gapi {
namespace wip {


#ifdef TEST_PERF    
void VPLCPUAccelerationPolicy::pool_t::reserve(size_t size) {
    surfaces.reserve(size);
}

size_t VPLCPUAccelerationPolicy::pool_t::size() const {
    return surfaces.size();
}

void VPLCPUAccelerationPolicy::pool_t::clear() {
    surfaces.clear();
    next_free_it = surfaces.begin();
    cache.clear();
}

void VPLCPUAccelerationPolicy::pool_t::push_back(surface_ptr_t &&surf) {
    cache.insert(std::make_pair(surf->get_handle(), surf));
    surfaces.push_back(std::move(surf));
    next_free_it = surfaces.begin();
}

surface_ptr_t VPLCPUAccelerationPolicy::pool_t::find_free() {

        auto it =
            std::find_if(next_free_it, surfaces.end(),
                        [](const surface_ptr_t& val) {
                GAPI_DbgAssert(val && "Pool contains empty surface");
                return !val->get_locks_count();
            });

        // Limitation realloc pool might be a future extension
        if (it == surfaces.end()) {
            it =
                std::find_if(surfaces.begin(), next_free_it,
                            [](const surface_ptr_t& val) {
                    GAPI_DbgAssert(val && "Pool contains empty surface");
                    return !val->get_locks_count();
                });
            if (it == next_free_it) {
                std::stringstream ss;
                ss << "cannot get free surface from pool, size: " << surfaces.size();
                const std::string& str = ss.str();
                GAPI_LOG_WARNING(nullptr, str);
                throw std::runtime_error(std::string(__FUNCTION__) + " - " + str);
            }
        }

        next_free_it = it;
        ++next_free_it;

    return *it;
}
surface_ptr_t VPLCPUAccelerationPolicy::pool_t::find_by_handle(mfxFrameSurface1* handle) {
    auto it = cache.find(handle);
    if (it == cache.end()) {
        std::stringstream ss;
        ss << "cannot get requested surface from pool, surf: "
           << handle << ", pool size: " << surfaces.size();
        const std::string& str = ss.str();
        GAPI_LOG_WARNING(nullptr, str);
        throw std::runtime_error(std::string(__FUNCTION__) + " - " + str);
    }
    return it->second;
}
#endif // TEST_PERF

    
VPLCPUAccelerationPolicy::VPLCPUAccelerationPolicy() {
    GAPI_LOG_INFO(nullptr, "created");
}

VPLCPUAccelerationPolicy::~VPLCPUAccelerationPolicy() {
    for (auto& pair : pool_table) {
        pair.second.clear();
        // do not free key here: last surface will release it
    }
    pool_table.clear();
    GAPI_LOG_INFO(nullptr, "destroyed");
}

void VPLCPUAccelerationPolicy::init(session_t session) {
    (void)session;
    //MFXVideoCORE_SetFrameAllocator(session, mfxFrameAllocator instance)
    GAPI_LOG_INFO(nullptr, "initialize session: " << session);
}

void VPLCPUAccelerationPolicy::deinit(session_t session) {
    (void)session;
    GAPI_LOG_INFO(nullptr, "deinitialize session: " << session);
}

VPLCPUAccelerationPolicy::pool_key_t
VPLCPUAccelerationPolicy::create_surface_pool(size_t pool_size, size_t surface_size_bytes,
                                              surface_ptr_ctr_t creator) {
    GAPI_LOG_DEBUG(nullptr, "pool size: " << pool_size << ", surface size bytes: " << surface_size_bytes);

    // create empty pool
    pool_t pool;
    pool.reserve(pool_size);

    // allocate workplace memory area
    size_t preallocated_raw_bytes = pool_size * surface_size_bytes;
    size_t page_size_bytes = 4 * 1024;
    void *preallocated_pool_memory_ptr = nullptr;

#ifdef _WIN32
    SYSTEM_INFO sysInfo;
    GetSystemInfo(&sysInfo);
    page_size_bytes = sysInfo.dwPageSize;

    GAPI_LOG_DEBUG(nullptr, "page size: " << page_size_bytes << ", preallocated_raw_bytes: " << preallocated_raw_bytes);
    preallocated_pool_memory_ptr = _aligned_malloc(preallocated_raw_bytes, page_size_bytes);
#endif

    if (!preallocated_pool_memory_ptr) {
        throw std::runtime_error("VPLCPUAccelerationPolicy::create_surface_pool - failed: not enough memory."
                                 "Requested surface count: " + std::to_string(pool_size) +
                                 ", surface bytes: " + std::to_string(surface_size_bytes));
    }

    // fill pool with surfaces
    std::shared_ptr<void> workspace_mem_owner (preallocated_pool_memory_ptr, [] (void *ptr){
        GAPI_LOG_INFO(nullptr, "Free workspace memory: " << ptr);
#ifdef _WIN32
        _aligned_free(ptr);
        GAPI_LOG_INFO(nullptr, "Released workspace memory: " << ptr);
        ptr = nullptr;
#else
        abort(); //not implemented
#endif
    
        });
    size_t i = 0;
    try {
        for (; i < pool_size; i++) {
            size_t preallocated_mem_offset = static_cast<size_t>(i) * surface_size_bytes;

            surface_ptr_t surf_ptr = creator(workspace_mem_owner,
                                             preallocated_mem_offset,
                                             preallocated_raw_bytes);
            pool.push_back(std::move(surf_ptr));
        }
    } catch (const std::exception& ex) {
        throw std::runtime_error(std::string("VPLCPUAccelerationPolicy::create_surface_pool - ") +
                                 "cannot construct surface index: " + std::to_string(i) + ", error: " +
                                 ex.what() +
                                 "Requested surface count: " + std::to_string(pool_size) +
                                 ", surface bytes: " + std::to_string(surface_size_bytes));
    }

    // remember pool by key
    GAPI_LOG_INFO(nullptr, "New pool allocated, key: " << preallocated_pool_memory_ptr <<
                           ", surface count: " << pool.size() <<
                           ", surface size bytes: " << surface_size_bytes);
    try {
        if (!pool_table.emplace(preallocated_pool_memory_ptr, std::move(pool)).second) {
            throw std::runtime_error(std::string("VPLCPUAccelerationPolicy::create_surface_pool - ") +
                                     "cannot insert pool, table size: " + std::to_string(pool_table.size()));
        }
    } catch (const std::exception&) {
        throw;
    }

    return preallocated_pool_memory_ptr;
}

VPLCPUAccelerationPolicy::surface_weak_ptr_t VPLCPUAccelerationPolicy::get_free_surface(pool_key_t key) {
    auto pool_it = pool_table.find(key);
    if (pool_it == pool_table.end()) {
        std::stringstream ss;
        ss << "key is not found: " << key << ", table size: " << pool_table.size();
        const std::string& str = ss.str();
        GAPI_LOG_WARNING(nullptr, str);
        throw std::runtime_error(std::string(__FUNCTION__) + " - " + str);
    }

    pool_t& requested_pool = pool_it->second;
#ifdef TEST_PERF
    return requested_pool.find_free();
#else // TEST_PERF
    auto it =
        std::find_if(requested_pool.begin(), requested_pool.end(),
                     [](const surface_ptr_t& val) {
            GAPI_DbgAssert(val && "Pool contains empty surface");
            return !val->get_locks_count();
        });

    // Limitation realloc pool might be a future extension
    if (it == requested_pool.end()) {
        std::stringstream ss;
        ss << "cannot get free surface from pool, key: " << key << ", size: " << requested_pool.size();
        const std::string& str = ss.str();
        GAPI_LOG_WARNING(nullptr, str);
        throw std::runtime_error(std::string(__FUNCTION__) + " - " + str);
    }

    return *it;
#endif // TEST_PERF
}

size_t VPLCPUAccelerationPolicy::get_free_surface_count(pool_key_t key) const {
    auto pool_it = pool_table.find(key);
    if (pool_it == pool_table.end()) {
        GAPI_LOG_WARNING(nullptr, "key is not found: " << key <<
                                  ", table size: " << pool_table.size());
        return 0;
    }
#ifdef TEST_PERF
    return 0;
#else // TEST_PERF
    const pool_t& requested_pool = pool_it->second;
    size_t free_surf_count =
        std::count_if(requested_pool.begin(), requested_pool.end(),
                     [](const surface_ptr_t& val) {
            GAPI_Assert(val && "Pool contains empty surface");
            return !val->get_locks_count();
        });
    return free_surf_count;
#endif // TEST_PERF
}

size_t VPLCPUAccelerationPolicy::get_surface_count(pool_key_t key) const {
    auto pool_it = pool_table.find(key);
    if (pool_it == pool_table.end()) {
        GAPI_LOG_DEBUG(nullptr, "key is not found: " << key <<
                                ", table size: " << pool_table.size());
        return 0;
    }
#ifdef TEST_PERF
    return 0;
#else // TEST_PERF
    const pool_t& requested_pool = pool_it->second;
    size_t free_surf_count =
        std::count_if(requested_pool.begin(), requested_pool.end(),
                     [](const surface_ptr_t& val) {
            GAPI_Assert(val && "Pool contains empty surface");
            return !val->get_locks_count();
        });
    return requested_pool.size() - free_surf_count;
#endif // TEST_PERF
}

cv::MediaFrame::AdapterPtr VPLCPUAccelerationPolicy::create_frame_adapter(pool_key_t key,
                                                                          mfxFrameSurface1* surface) {
    auto pool_it = pool_table.find(key);
    if (pool_it == pool_table.end()) {
        std::stringstream ss;
        ss << "key is not found: " << key << ", table size: " << pool_table.size();
        const std::string& str = ss.str();
        GAPI_LOG_WARNING(nullptr, str);
        throw std::runtime_error(std::string(__FUNCTION__) + " - " + str);
    }

    pool_t& requested_pool = pool_it->second;
#ifdef TEST_PERF
    return cv::MediaFrame::AdapterPtr{new MediaFrameAdapter(requested_pool.find_by_handle(surface))};
#else // TEST_PERF
    auto it =
        std::find_if(requested_pool.begin(), requested_pool.end(),
                     [surface](const surface_ptr_t& val) {
            GAPI_DbgAssert(val && "Pool contains empty surface");
            return val->get_handle() == surface;
        });

    // Limitation realloc pool might be a future extension
    if (it == requested_pool.end()) {
        std::stringstream ss;
        ss << "cannot get requested surface from pool, key: " << key << ", surf: "
           << surface << ", pool size: " << requested_pool.size();
        const std::string& str = ss.str();
        GAPI_LOG_WARNING(nullptr, str);
        throw std::runtime_error(std::string(__FUNCTION__) + " - " + str);
    }
    
    return cv::MediaFrame::AdapterPtr{new MediaFrameAdapter(*it)};
#endif // TEST_PERF
}
} // namespace wip
} // namespace gapi
} // namespace cv
#endif // HAVE_ONEVPL
