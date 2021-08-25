// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2019-2020 Intel Corporation


#include "../test_precomp.hpp"

#include "../common/gapi_tests_common.hpp"

#include <future>

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

#include <opencv2/gapi/streaming/onevpl/source.hpp>

#ifdef HAVE_ONEVPL
#include "streaming/onevpl/accelerators/surface/surface.hpp"
#include "streaming/onevpl/accelerators/surface/cpu_frame_adapter.hpp"

namespace opencv_test
{
namespace
{
TEST(OneVPL_Source_Surface, InitSurface)
{
    using namespace cv::gapi::wip;

    // create raw MFX handle
    std::unique_ptr<mfxFrameSurface1> handle(new mfxFrameSurface1{});
    mfxFrameSurface1 *mfx_core_handle = handle.get();

    // create preallocate surface memory: empty for test
    std::shared_ptr<void> associated_memory {};
    auto surf = Surface::create_surface(std::move(handle), associated_memory);

    // check self consistency
    EXPECT_EQ(reinterpret_cast<void*>(surf->get_handle()),
              reinterpret_cast<void*>(mfx_core_handle));
    EXPECT_EQ(surf->get_locks_count(), 0);
    EXPECT_EQ(surf->obtain_lock(), 0);
    EXPECT_EQ(surf->get_locks_count(), 1);
    EXPECT_EQ(surf->release_lock(), 1);
    EXPECT_EQ(surf->get_locks_count(), 0);
}

TEST(OneVPL_Source_Surface, ConcurrentLock)
{
    using namespace cv::gapi::wip;

    // create raw MFX handle
    std::unique_ptr<mfxFrameSurface1> handle(new mfxFrameSurface1{});

    // create preallocate surface memory: empty for test
    std::shared_ptr<void> associated_memory {};
    auto surf = Surface::create_surface(std::move(handle), associated_memory);

    // check self consistency
    EXPECT_EQ(surf->get_locks_count(), 0);

    // MFX internal limitation: do not exceede U16 range
    // so I16 is using here
    int16_t lock_counter = std::numeric_limits<int16_t>::max() - 1;
    std::promise<void> barrier;
    std::future<void> sync = barrier.get_future();


    std::thread worker_thread([&barrier, surf, lock_counter] () {
        barrier.set_value();

        // concurrent lock
        for (int16_t i = 0; i < lock_counter; i ++) {
            surf->obtain_lock();
        }
    });
    sync.wait();

    // concurrent lock
    for (int16_t i = 0; i < lock_counter; i ++) {
            surf->obtain_lock();
    }

    worker_thread.join();
    EXPECT_EQ(surf->get_locks_count(), lock_counter * 2);
}

TEST(OneVPL_Source_Surface, MemoryLifeTime)
{
    using namespace cv::gapi::wip;

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
        std::unique_ptr<mfxFrameSurface1> handle(new mfxFrameSurface1{});
        return Surface::create_surface(std::move(handle), associated_memory);
    });

    // destroy surfaces
    {
        std::thread deleter_thread([&surfaces]() {
            surfaces.clear();
        });
        deleter_thread.join();
    }

    // workspace memory must be alive
    EXPECT_EQ(surfaces.size(), 0);
    EXPECT_TRUE(associated_memory != nullptr);
    EXPECT_TRUE(preallocated_memory_ptr.get() != nullptr);

    // generate surfaces again + 1
    constexpr size_t surface_num_plus_one = 10001;
    surfaces.resize(surface_num_plus_one);
    std::generate(surfaces.begin(), surfaces.end(), [surface_num_plus_one, associated_memory](){
        std::unique_ptr<mfxFrameSurface1> handle(new mfxFrameSurface1{});
        return Surface::create_surface(std::move(handle), associated_memory);
    });

    // remember one surface
    std::shared_ptr<Surface> last_surface = surfaces.back();

    // destroy another surfaces
    surfaces.clear();

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

TEST(OneVPL_Source_CPUFrameAdapter, InitFrameAdapter)
{
    using namespace cv::gapi::wip;

    // create raw MFX handle
    std::unique_ptr<mfxFrameSurface1> handle(new mfxFrameSurface1{});

    // create preallocate surface memory: empty for test
    std::shared_ptr<void> associated_memory {};
    auto surf = Surface::create_surface(std::move(handle), associated_memory);

    // check consistency
    EXPECT_EQ(surf->get_locks_count(), 0);

    {
        VPLMediaFrameCPUAdapter adapter(surf);
        EXPECT_EQ(surf->get_locks_count(), 1);
    }
    EXPECT_EQ(surf->get_locks_count(), 0);
}
}
} // namespace opencv_test
#endif // HAVE_ONEVPL
