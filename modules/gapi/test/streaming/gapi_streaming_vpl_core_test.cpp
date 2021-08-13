// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2019-2020 Intel Corporation


#include "../test_precomp.hpp"

#include "../common/gapi_tests_common.hpp"

#include <thread> // sleep_for (Delay)

#include <opencv2/gapi/cpu/core.hpp>
#include <opencv2/gapi/cpu/imgproc.hpp>

#include <opencv2/gapi/fluid/core.hpp>
#include <opencv2/gapi/fluid/imgproc.hpp>
#include <opencv2/gapi/fluid/gfluidkernel.hpp>

#include <opencv2/gapi/ocl/core.hpp>
#include <opencv2/gapi/ocl/imgproc.hpp>

#include <opencv2/gapi/streaming/cap.hpp>
#include <opencv2/gapi/streaming/desync.hpp>
#include <opencv2/gapi/streaming/format.hpp>

#include <opencv2/gapi/streaming/onevpl/onevpl_source.hpp>

#ifdef HAVE_ONEVPL
//#include <../src/backends/common/gbackend.hpp> // asView
#include "streaming/onevpl/accelerators/surface/surface.hpp"
namespace opencv_test
{
namespace
{
TEST(OneVPL_Source_Surface, Init)
{
    // create raw MFX handle
    std::unique_ptr<mfxFrameSurface1> handle(new mfxFrameSurface1);
    memset(handle.get(), 0, sizeof(mfxFrameSurface1));
    mfxFrameSurface1 *mfx_core_handle = handle.get();

    // create preallocate surface memory: empty for test
    std::shared_ptr<void> associated_memory {};
    auto surf = Surface::create_surface(std::move(handle), out_buf_ptr);

    // check self consistency
    EXPECT_EQ(surf->get_handle(), mfx_core_handle);
    EXPECT_EQ(surf->get_locks_count(), 0);
    EXPECT_EQ(surf->obtain_lock(), 0);
    EXPECT_EQ(surf->get_locks_count(), 1);
    EXPECT_EQ(surf->release_lock(), 1);
    EXPECT_EQ(surf->get_locks_count(), 0);
}

TEST(OneVPL_Source_Surface, ConcurrentLock)
{
    // create raw MFX handle
    std::unique_ptr<mfxFrameSurface1> handle(new mfxFrameSurface1);
    memset(handle.get(), 0, sizeof(mfxFrameSurface1));
    mfxFrameSurface1 *mfx_core_handle = handle.get();

    // create preallocate surface memory: empty for test
    std::shared_ptr<void> associated_memory {};
    auto surf = Surface::create_surface(std::move(handle), out_buf_ptr);

    // check self consistency
    EXPECT_EQ(surf->get_locks_count(), 0);

    size_t lock_counter = 100000;
    std::promise<void> barrier;
    std::future<void> sync = barrier.get_future();
    std::thread worker_thread([&barrier, surf, lock_counter] () {
        barrier.set_value();

        // concurrent lock
        for (size_t i = 0; i < lock_counter; i ++) {
            surf->obtain_lock();
        }
    });

    sync.wait();
    // concurrent lock
    for (size_t i = 0; i < lock_counter; i ++) {
            surf->obtain_lock();
    }
    worker_thread.join();
    EXPECT_EQ(surf->get_locks_count(), lock_counter * 2);
}

TEST(OneVPL_Source_Surface, MemoryLifeTime)
{
    // create preallocate surface memory
    std::unique_ptr<char> preallocated_memory_ptr(new char);
    std::shared_ptr<void> associated_memory (preallocated_memory_ptr.get(),
                                             [&preallocated_memory_ptr] (void* ptr) {
                                                    EXPECT_TRUE(preallocated_memory_ptr);
                                                    EXPECT_EQ(preallocated_memory_ptr.get(), ptr);
                                                    preallocated_memory_ptr.reset();
                                            });

    // generate surfaces
    constexpr size_t surface_num = 10000;
    std::vector<std::shared_ptr<Surface>> surfaces(surface_num);
    std::generate(surfaces.begin(), surfaces.end(), [surface_num, associated_memory](){
        std::unique_ptr<mfxFrameSurface1> handle(new mfxFrameSurface1);
        memset(handle.get(), 0, sizeof(mfxFrameSurface1));
        return Surface::create_surface(std::move(handle), associated_memory));
    });

    // destroy surfaces
    {
        std::thread deleter_thread([&surfaces]() {
            surfaces.clear();
        }).join();
    }

    // workspace memory must be alive
    EXPECT_EQ(surfaces.size(), 0);
    EXPECT_TRUE(associated_memory != nullptr);
    EXPECT_TRUE(preallocated_memory_ptr.get() != nullptr);

    // generate surfaces again + 1
    constexpr size_t surface_num_plus_one = 10001;
    surfaces.resize(surface_num_plus_one);
    std::generate(surfaces.begin(), surfaces.end(), [surface_num_plus_one, associated_memory](){
        std::unique_ptr<mfxFrameSurface1> handle(new mfxFrameSurface1);
        memset(handle.get(), 0, sizeof(mfxFrameSurface1));
        return Surface::create_surface(std::move(handle), associated_memory));
    });

    // remember one surface
    std::shared_ptr<Surface> last_surface = surfaces.back();

    // destroy another surfaces
    surface.clear();

    // destroy associated_memory
    associated_memory.reset();

    // workspace memory must be still alive
    EXPECT_EQ(surfaces.size(), 0);
    EXPECT_TRUE(associated_memory == nullptr);
    EXPECT_TRUE(preallocated_memory_ptr.get() != nullptr);

    // destroy last surface
    last_surface.reset();

    // workspace memory must be freed
    EXPECT_TRUE(preallocated_memory_ptr.get() == nullptr);
}
}
} // namespace opencv_test
#endif // HAVE_ONEVPL
