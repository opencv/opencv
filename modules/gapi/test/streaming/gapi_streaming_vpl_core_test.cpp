// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation


#include "../test_precomp.hpp"

#include "../common/gapi_tests_common.hpp"

#include <chrono>
#include <future>

#include <opencv2/gapi/media.hpp>
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
#include <opencv2/gapi/streaming/onevpl/data_provider_interface.hpp>

#include "streaming/onevpl/accelerators/surface/surface.hpp"
#include "streaming/onevpl/accelerators/surface/cpu_frame_adapter.hpp"
#include "streaming/onevpl/accelerators/accel_policy_cpu.hpp"
#include "streaming/onevpl/engine/processing_engine_base.hpp"
#include "streaming/onevpl/engine/engine_session.hpp"

namespace opencv_test
{
namespace
{

struct EmptyDataProvider : public cv::gapi::wip::onevpl::IDataProvider {

    size_t fetch_data(size_t, void*) override {
        return 0;
    }
    bool empty() const override {
        return true;
    }
};

struct TestProcessingSession : public cv::gapi::wip::onevpl::EngineSession {
    TestProcessingSession(mfxSession mfx_session) :
        EngineSession(mfx_session, {}) {
    }
};

struct TestProcessingEngine: public cv::gapi::wip::onevpl::ProcessingEngineBase {

    size_t pipeline_stage_num = 0;

    TestProcessingEngine(std::unique_ptr<cv::gapi::wip::onevpl::VPLAccelerationPolicy>&& accel) :
        cv::gapi::wip::onevpl::ProcessingEngineBase(std::move(accel)) {
        using cv::gapi::wip::onevpl::EngineSession;
        create_pipeline(
            // 0)
            [this] (EngineSession&) -> ExecutionStatus
            {
                pipeline_stage_num = 0;
                return ExecutionStatus::Continue;
            },
            // 1)
            [this] (EngineSession&) -> ExecutionStatus
            {
                pipeline_stage_num = 1;
                return ExecutionStatus::Continue;
            },
            // 2)
            [this] (EngineSession&) -> ExecutionStatus
            {
                pipeline_stage_num = 2;
                return ExecutionStatus::Continue;
            },
            // 3)
            [this] (EngineSession&) -> ExecutionStatus
            {
                pipeline_stage_num = 3;
                ready_frames.emplace(cv::MediaFrame());
                return ExecutionStatus::Processed;
            }
        );
    }

    void initialize_session(mfxSession mfx_session,
                            cv::gapi::wip::onevpl::DecoderParams&&,
                            std::shared_ptr<cv::gapi::wip::onevpl::IDataProvider>) override {

        register_session<TestProcessingSession>(mfx_session);
    }
};

cv::gapi::wip::onevpl::surface_ptr_t create_test_surface(std::shared_ptr<void> out_buf_ptr,
                                                 size_t, size_t) {
    std::unique_ptr<mfxFrameSurface1> handle(new mfxFrameSurface1{});
    return cv::gapi::wip::onevpl::Surface::create_surface(std::move(handle), out_buf_ptr);
}

TEST(OneVPL_Source_Surface, InitSurface)
{
    using namespace cv::gapi::wip::onevpl;

    // create raw MFX handle
    std::unique_ptr<mfxFrameSurface1> handle(new mfxFrameSurface1{});
    mfxFrameSurface1 *mfx_core_handle = handle.get();

    // create preallocate surface memory: empty for test
    std::shared_ptr<void> associated_memory {};
    auto surf = Surface::create_surface(std::move(handle), associated_memory);

    // check self consistency
    EXPECT_EQ(reinterpret_cast<void*>(surf->get_handle()),
              reinterpret_cast<void*>(mfx_core_handle));
    EXPECT_EQ(0, surf->get_locks_count());
    EXPECT_EQ(0, surf->obtain_lock());
    EXPECT_EQ(1, surf->get_locks_count());
    EXPECT_EQ(1, surf->release_lock());
    EXPECT_EQ(0, surf->get_locks_count());
}

TEST(OneVPL_Source_Surface, ConcurrentLock)
{
    using namespace cv::gapi::wip::onevpl;

    // create raw MFX handle
    std::unique_ptr<mfxFrameSurface1> handle(new mfxFrameSurface1{});

    // create preallocate surface memory: empty for test
    std::shared_ptr<void> associated_memory {};
    auto surf = Surface::create_surface(std::move(handle), associated_memory);

    // check self consistency
    EXPECT_EQ(0, surf->get_locks_count());

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
    EXPECT_EQ(lock_counter * 2, surf->get_locks_count());
}

TEST(OneVPL_Source_Surface, MemoryLifeTime)
{
    using namespace cv::gapi::wip::onevpl;

    // create preallocate surface memory
    std::unique_ptr<char> preallocated_memory_ptr(new char);
    std::shared_ptr<void> associated_memory (preallocated_memory_ptr.get(),
                                             [&preallocated_memory_ptr] (void* ptr) {
                                                    EXPECT_TRUE(preallocated_memory_ptr);
                                                    EXPECT_EQ(ptr, preallocated_memory_ptr.get());
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
    EXPECT_EQ(0, surfaces.size());
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
    EXPECT_EQ(0, surfaces.size());
    EXPECT_TRUE(associated_memory == nullptr);
    EXPECT_TRUE(preallocated_memory_ptr.get() != nullptr);

    // destroy last surface
    last_surface.reset();

    // workspace memory must be freed
    EXPECT_TRUE(preallocated_memory_ptr.get() == nullptr);
}

TEST(OneVPL_Source_CPU_FrameAdapter, InitFrameAdapter)
{
    using namespace cv::gapi::wip::onevpl;

    // create raw MFX handle
    std::unique_ptr<mfxFrameSurface1> handle(new mfxFrameSurface1{});

    // create preallocate surface memory: empty for test
    std::shared_ptr<void> associated_memory {};
    auto surf = Surface::create_surface(std::move(handle), associated_memory);

    // check consistency
    EXPECT_EQ(0, surf->get_locks_count());

    {
        VPLMediaFrameCPUAdapter adapter(surf);
        EXPECT_EQ(1, surf->get_locks_count());
    }
    EXPECT_EQ(0, surf->get_locks_count());
}

TEST(OneVPL_Source_CPU_Accelerator, InitDestroy)
{
    using cv::gapi::wip::onevpl::VPLCPUAccelerationPolicy;
    using cv::gapi::wip::onevpl::VPLAccelerationPolicy;

    auto acceleration_policy = std::make_shared<VPLCPUAccelerationPolicy>();

    size_t surface_count = 10;
    size_t surface_size_bytes = 1024;
    size_t pool_count = 3;
    std::vector<VPLAccelerationPolicy::pool_key_t> pool_export_keys;
    pool_export_keys.reserve(pool_count);

    // create several pools
    for (size_t i = 0; i < pool_count; i++)
    {
        VPLAccelerationPolicy::pool_key_t key =
                acceleration_policy->create_surface_pool(surface_count,
                                                         surface_size_bytes,
                                                         create_test_surface);
        // check consistency
        EXPECT_EQ(surface_count, acceleration_policy->get_surface_count(key));
        EXPECT_EQ(surface_count, acceleration_policy->get_free_surface_count(key));

        pool_export_keys.push_back(key);
    }

    EXPECT_NO_THROW(acceleration_policy.reset());
}

TEST(OneVPL_Source_CPU_Accelerator, PoolProduceConsume)
{
    using cv::gapi::wip::onevpl::VPLCPUAccelerationPolicy;
    using cv::gapi::wip::onevpl::VPLAccelerationPolicy;
    using cv::gapi::wip::onevpl::Surface;

    auto acceleration_policy = std::make_shared<VPLCPUAccelerationPolicy>();

    size_t surface_count = 10;
    size_t surface_size_bytes = 1024;

    VPLAccelerationPolicy::pool_key_t key =
                acceleration_policy->create_surface_pool(surface_count,
                                                         surface_size_bytes,
                                                         create_test_surface);
    // check consistency
    EXPECT_EQ(surface_count, acceleration_policy->get_surface_count(key));
    EXPECT_EQ(surface_count, acceleration_policy->get_free_surface_count(key));

    // consume available surfaces
    std::vector<std::shared_ptr<Surface>> surfaces;
    surfaces.reserve(surface_count);
    for (size_t i = 0; i < surface_count; i++) {
        std::shared_ptr<Surface> surf = acceleration_policy->get_free_surface(key).lock();
        EXPECT_TRUE(surf.get() != nullptr);
        EXPECT_EQ(0, surf->obtain_lock());
        surfaces.push_back(std::move(surf));
    }

    // check consistency (no free surfaces)
    EXPECT_EQ(acceleration_policy->get_surface_count(key), surface_count);
    EXPECT_EQ(0, acceleration_policy->get_free_surface_count(key));

    // fail consume non-free surfaces
    for (size_t i = 0; i < surface_count; i++) {
        EXPECT_THROW(acceleration_policy->get_free_surface(key), std::runtime_error);
    }

    // release surfaces
    for (auto& surf : surfaces) {
        EXPECT_EQ(1, surf->release_lock());
    }
    surfaces.clear();

    // check consistency
    EXPECT_EQ(surface_count, acceleration_policy->get_surface_count(key));
    EXPECT_EQ(surface_count, acceleration_policy->get_free_surface_count(key));

    //check availability after release
    for (size_t i = 0; i < surface_count; i++) {
        std::shared_ptr<Surface> surf = acceleration_policy->get_free_surface(key).lock();
        EXPECT_TRUE(surf.get() != nullptr);
        EXPECT_EQ(0, surf->obtain_lock());
    }
}

TEST(OneVPL_Source_CPU_Accelerator, PoolProduceConcurrentConsume)
{
    using cv::gapi::wip::onevpl::VPLCPUAccelerationPolicy;
    using cv::gapi::wip::onevpl::VPLAccelerationPolicy;
    using cv::gapi::wip::onevpl::Surface;

    auto acceleration_policy = std::make_shared<VPLCPUAccelerationPolicy>();

    size_t surface_count = 10;
    size_t surface_size_bytes = 1024;

    VPLAccelerationPolicy::pool_key_t key =
                acceleration_policy->create_surface_pool(surface_count,
                                                         surface_size_bytes,
                                                         create_test_surface);

    // check consistency
    EXPECT_EQ(surface_count, acceleration_policy->get_surface_count(key));
    EXPECT_EQ(surface_count, acceleration_policy->get_free_surface_count(key));

    // consume available surfaces
    std::vector<std::shared_ptr<Surface>> surfaces;
    surfaces.reserve(surface_count);
    for (size_t i = 0; i < surface_count; i++) {
        std::shared_ptr<Surface> surf = acceleration_policy->get_free_surface(key).lock();
        EXPECT_TRUE(surf.get() != nullptr);
        EXPECT_EQ(0, surf->obtain_lock());
        surfaces.push_back(std::move(surf));
    }

    std::promise<void> launch_promise;
    std::future<void> sync = launch_promise.get_future();
    std::promise<size_t> surface_released_promise;
    std::future<size_t> released_result = surface_released_promise.get_future();
    std::thread worker_thread([&launch_promise, &surface_released_promise, &surfaces] () {
        launch_promise.set_value();

        // concurrent release surfaces
        size_t surfaces_count = surfaces.size();
        for (auto& surf : surfaces) {
            EXPECT_EQ(1, surf->release_lock());
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
        surfaces.clear();

        surface_released_promise.set_value(surfaces_count);
    });
    sync.wait();

    // check free surface concurrently
    std::future_status status;
    size_t free_surface_count = 0;
    size_t free_surface_count_prev = 0;
    do {
        status = released_result.wait_for(std::chrono::seconds(1));
        free_surface_count = acceleration_policy->get_free_surface_count(key);
        EXPECT_TRUE(free_surface_count >= free_surface_count_prev);
        free_surface_count_prev = free_surface_count;
    } while (status != std::future_status::ready);
    std::cerr<< "Ready" << std::endl;
    free_surface_count = acceleration_policy->get_free_surface_count(key);
    worker_thread.join();
    EXPECT_TRUE(free_surface_count >= free_surface_count_prev);
}

TEST(OneVPL_Source_ProcessingEngine, Init)
{
    using namespace cv::gapi::wip::onevpl;
    std::unique_ptr<VPLAccelerationPolicy> accel;
    TestProcessingEngine engine(std::move(accel));

    mfxSession mfx_session{};
    engine.initialize_session(mfx_session, DecoderParams{}, std::shared_ptr<IDataProvider>{});

    EXPECT_EQ(0, engine.get_ready_frames_count());
    ProcessingEngineBase::ExecutionStatus ret = engine.process(mfx_session);
    EXPECT_EQ(ret, ProcessingEngineBase::ExecutionStatus::Continue);
    EXPECT_EQ(0, engine.pipeline_stage_num);

    ret = engine.process(mfx_session);
    EXPECT_EQ(ret, ProcessingEngineBase::ExecutionStatus::Continue);
    EXPECT_EQ(1, engine.pipeline_stage_num);

    ret = engine.process(mfx_session);
    EXPECT_EQ(ret, ProcessingEngineBase::ExecutionStatus::Continue);
    EXPECT_EQ(2, engine.pipeline_stage_num);

    ret = engine.process(mfx_session);
    EXPECT_EQ(ret, ProcessingEngineBase::ExecutionStatus::Processed);
    EXPECT_EQ(3, engine.pipeline_stage_num);
    EXPECT_EQ(1, engine.get_ready_frames_count());

    ret = engine.process(mfx_session);
    EXPECT_EQ(ret, ProcessingEngineBase::ExecutionStatus::SessionNotFound);
    EXPECT_EQ(3, engine.pipeline_stage_num);
    EXPECT_EQ(1, engine.get_ready_frames_count());

    cv::gapi::wip::Data frame;
    engine.get_frame(frame);
}
}
} // namespace opencv_test
#endif // HAVE_ONEVPL
