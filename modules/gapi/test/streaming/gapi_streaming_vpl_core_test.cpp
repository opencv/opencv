// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation


#include "../test_precomp.hpp"

#include "../common/gapi_streaming_tests_common.hpp"
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

#ifdef HAVE_ONEVPL
#include <opencv2/gapi/streaming/onevpl/data_provider_interface.hpp>
#include <opencv2/gapi/streaming/onevpl/default.hpp>
#include "streaming/onevpl/file_data_provider.hpp"
#include "streaming/onevpl/cfg_param_device_selector.hpp"

#include "streaming/onevpl/accelerators/surface/surface.hpp"
#include "streaming/onevpl/accelerators/surface/cpu_frame_adapter.hpp"
#include "streaming/onevpl/accelerators/accel_policy_cpu.hpp"
#include "streaming/onevpl/accelerators/accel_policy_dx11.hpp"
#include "streaming/onevpl/accelerators/accel_policy_va_api.hpp"
#include "streaming/onevpl/accelerators/dx11_alloc_resource.hpp"
#include "streaming/onevpl/accelerators/utils/shared_lock.hpp"
#define private public
#define protected public
#include "streaming/onevpl/engine/transcode/transcode_engine_legacy.hpp"
#include "streaming/onevpl/engine/transcode/transcode_session.hpp"
#undef protected
#undef private
#include "logger.hpp"

#define ALIGN16(value)           (((value + 15) >> 4) << 4)

namespace opencv_test
{
namespace
{

struct EmptyDataProvider : public cv::gapi::wip::onevpl::IDataProvider {

    bool empty() const override {
        return true;
    }
    mfx_codec_id_type get_mfx_codec_id() const override {
        return std::numeric_limits<uint32_t>::max();
    }
    bool fetch_bitstream_data(std::shared_ptr<mfx_bitstream> &) override {
        return false;
    }
};

struct TestProcessingSession : public cv::gapi::wip::onevpl::EngineSession {
    TestProcessingSession(mfxSession mfx_session) :
        EngineSession(mfx_session) {
    }

    const mfxFrameInfo& get_video_param() const override {
        static mfxVideoParam empty;
        return empty.mfx.FrameInfo;
    }
};

struct TestProcessingEngine: public cv::gapi::wip::onevpl::ProcessingEngineBase {

    int pipeline_stage_num = 0;

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

    std::shared_ptr<cv::gapi::wip::onevpl::EngineSession>
            initialize_session(mfxSession mfx_session,
                               const std::vector<cv::gapi::wip::onevpl::CfgParam>&,
                               std::shared_ptr<cv::gapi::wip::onevpl::IDataProvider>) override {

        return register_session<TestProcessingSession>(mfx_session);
    }
};

template <class LockProcessor, class UnlockProcessor>
class TestLockableAllocator {
public :
    using self_t = TestLockableAllocator<LockProcessor, UnlockProcessor>;
    mfxFrameAllocator get() {
        return m_allocator;
    }
private:
    TestLockableAllocator(mfxFrameAllocator allocator) :
        m_allocator(allocator) {
    }

    static mfxStatus MFX_CDECL lock_cb(mfxHDL, mfxMemId mid, mfxFrameData *ptr) {
        auto it = lock_processor_table.find(mid);
        EXPECT_TRUE(it != lock_processor_table.end());
        return it->second(mid, ptr);
    }
    static mfxStatus MFX_CDECL unlock_cb(mfxHDL, mfxMemId mid, mfxFrameData *ptr) {
        auto it = unlock_processor_table.find(mid);
        EXPECT_TRUE(it != unlock_processor_table.end());
        return it->second(mid, ptr);
    }

    template <class L, class U>
    friend TestLockableAllocator<L,U> create_test_allocator(mfxMemId, L, U);

    static std::map<mfxMemId, LockProcessor> lock_processor_table;
    static std::map<mfxMemId, UnlockProcessor> unlock_processor_table;

    mfxFrameAllocator m_allocator;
};
template <class LockProcessor, class UnlockProcessor>
std::map<mfxMemId, LockProcessor> TestLockableAllocator<LockProcessor, UnlockProcessor>::lock_processor_table {};

template <class LockProcessor, class UnlockProcessor>
std::map<mfxMemId, UnlockProcessor> TestLockableAllocator<LockProcessor, UnlockProcessor>::unlock_processor_table {};

template <class LockProcessor, class UnlockProcessor>
TestLockableAllocator<LockProcessor, UnlockProcessor>
create_test_allocator(mfxMemId mid, LockProcessor lock_p, UnlockProcessor unlock_p) {
    mfxFrameAllocator allocator {};

    TestLockableAllocator<LockProcessor, UnlockProcessor>::lock_processor_table[mid] = lock_p;
    allocator.Lock = &TestLockableAllocator<LockProcessor, UnlockProcessor>::lock_cb;

    TestLockableAllocator<LockProcessor, UnlockProcessor>::unlock_processor_table[mid] = unlock_p;
    allocator.Unlock = &TestLockableAllocator<LockProcessor, UnlockProcessor>::unlock_cb;

    return TestLockableAllocator<LockProcessor, UnlockProcessor> {allocator};
}

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
    EXPECT_TRUE(0 == surf->get_locks_count());
    EXPECT_TRUE(0 == surf->obtain_lock());
    EXPECT_TRUE(1 == surf->get_locks_count());
    EXPECT_TRUE(1 == surf->release_lock());
    EXPECT_TRUE(0 == surf->get_locks_count());
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
    EXPECT_TRUE(0 == surf->get_locks_count());

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
    EXPECT_TRUE(static_cast<size_t>(lock_counter * 2) == surf->get_locks_count());
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
    EXPECT_TRUE(0 == surfaces.size());
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
    EXPECT_TRUE(0 == surfaces.size());
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
    EXPECT_TRUE(0 == surf->get_locks_count());

    {
        mfxSession stub_session = reinterpret_cast<mfxSession>(0x1);
        VPLMediaFrameCPUAdapter adapter(surf, stub_session);
        EXPECT_TRUE(1 == surf->get_locks_count());
    }
    EXPECT_TRUE(0 == surf->get_locks_count());
}

TEST(OneVPL_Source_Default_Source_With_OCL_Backend, Accuracy)
{
    using namespace cv::gapi::wip::onevpl;

    auto create_from_string = [](const std::string& line){
        std::string::size_type name_endline_pos = line.find(':');
        std::string name = line.substr(0, name_endline_pos);
        std::string value = line.substr(name_endline_pos + 1);
        return CfgParam::create(name, value);
    };

    std::vector<CfgParam> source_cfgs;
    source_cfgs.push_back(create_from_string("mfxImplDescription.AccelerationMode:MFX_ACCEL_MODE_VIA_D3D11"));

    // Create VPL-based source
    std::shared_ptr<IDeviceSelector> default_device_selector = getDefaultDeviceSelector(source_cfgs);

    cv::gapi::wip::IStreamSource::Ptr source;
    cv::gapi::wip::IStreamSource::Ptr source_cpu;

    auto input = findDataFile("cv/video/768x576.avi");
    try {
        source = cv::gapi::wip::make_onevpl_src(input, source_cfgs, default_device_selector);
        source_cpu = cv::gapi::wip::make_onevpl_src(input, source_cfgs, default_device_selector);
    } catch(...) {
        throw SkipTestException("Video file can not be opened");
    }

    // Build the graph w/ OCL backend
    cv::GFrame in; // input frame from VPL source
    auto bgr_gmat = cv::gapi::streaming::BGR(in); // conversion from VPL source frame to BGR UMat
    auto out = cv::gapi::blur(bgr_gmat, cv::Size(4,4)); // ocl kernel of blur operation

    cv::GStreamingCompiled pipeline = cv::GComputation(cv::GIn(in), cv::GOut(out))
        .compileStreaming(std::move(cv::compile_args(cv::gapi::core::ocl::kernels())));
    pipeline.setSource(std::move(source));

    cv::GStreamingCompiled pipeline_cpu = cv::GComputation(cv::GIn(in), cv::GOut(out))
        .compileStreaming(std::move(cv::compile_args(cv::gapi::core::cpu::kernels())));
    pipeline_cpu.setSource(std::move(source_cpu));

    // The execution part
    cv::Mat out_mat;
    std::vector<cv::Mat> ocl_mats, cpu_mats;

    // Run the pipelines
    pipeline.start();
    while (pipeline.pull(cv::gout(out_mat)))
    {
        ocl_mats.push_back(out_mat);
    }

    pipeline_cpu.start();
    while (pipeline_cpu.pull(cv::gout(out_mat)))
    {
        cpu_mats.push_back(out_mat);
    }

    // Compare results
    // FIXME: investigate why 2 sources produce different number of frames sometimes
    for (size_t i = 0; i < std::min(ocl_mats.size(), cpu_mats.size()); ++i)
    {
        EXPECT_TRUE(AbsTolerance(1).to_compare_obj()(ocl_mats[i], cpu_mats[i]));
    }
}

TEST(OneVPL_Source_CPU_Accelerator, InitDestroy)
{
    using cv::gapi::wip::onevpl::VPLCPUAccelerationPolicy;
    using cv::gapi::wip::onevpl::VPLAccelerationPolicy;
    using cv::gapi::wip::onevpl::CfgParamDeviceSelector;

    auto acceleration_policy =
            std::make_shared<VPLCPUAccelerationPolicy>(std::make_shared<CfgParamDeviceSelector>());

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
    using cv::gapi::wip::onevpl::CfgParamDeviceSelector;
    using cv::gapi::wip::onevpl::Surface;

    auto acceleration_policy =
            std::make_shared<VPLCPUAccelerationPolicy>(std::make_shared<CfgParamDeviceSelector>());

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
        EXPECT_TRUE(0 == surf->obtain_lock());
        surfaces.push_back(std::move(surf));
    }

    // check consistency (no free surfaces)
    EXPECT_EQ(acceleration_policy->get_surface_count(key), surface_count);
    EXPECT_TRUE(0 == acceleration_policy->get_free_surface_count(key));

    // fail consume non-free surfaces
    for (size_t i = 0; i < surface_count; i++) {
        EXPECT_THROW(acceleration_policy->get_free_surface(key), std::runtime_error);
    }

    // release surfaces
    for (auto& surf : surfaces) {
        EXPECT_TRUE(1 == surf->release_lock());
    }
    surfaces.clear();

    // check consistency
    EXPECT_EQ(surface_count, acceleration_policy->get_surface_count(key));
    EXPECT_EQ(surface_count, acceleration_policy->get_free_surface_count(key));

    //check availability after release
    for (size_t i = 0; i < surface_count; i++) {
        std::shared_ptr<Surface> surf = acceleration_policy->get_free_surface(key).lock();
        EXPECT_TRUE(surf.get() != nullptr);
        EXPECT_TRUE(0 == surf->obtain_lock());
    }
}

TEST(OneVPL_Source_CPU_Accelerator, PoolProduceConcurrentConsume)
{
    using cv::gapi::wip::onevpl::VPLCPUAccelerationPolicy;
    using cv::gapi::wip::onevpl::VPLAccelerationPolicy;
    using cv::gapi::wip::onevpl::CfgParamDeviceSelector;
    using cv::gapi::wip::onevpl::Surface;

    auto acceleration_policy =
            std::make_shared<VPLCPUAccelerationPolicy>(std::make_shared<CfgParamDeviceSelector>());

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
        EXPECT_TRUE(0 == surf->obtain_lock());
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
            EXPECT_TRUE(1 == surf->release_lock());
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
    engine.initialize_session(mfx_session, {}, std::shared_ptr<IDataProvider>{});

    EXPECT_TRUE(0 == engine.get_ready_frames_count());
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
    EXPECT_TRUE(1 == engine.get_ready_frames_count());

    ret = engine.process(mfx_session);
    EXPECT_EQ(ret, ProcessingEngineBase::ExecutionStatus::SessionNotFound);
    EXPECT_EQ(3, engine.pipeline_stage_num);
    EXPECT_TRUE(1 == engine.get_ready_frames_count());

    cv::gapi::wip::Data frame;
    engine.get_frame(frame);
}

#ifdef HAVE_DIRECTX
#ifdef HAVE_D3D11
TEST(OneVPL_Source_DX11_Accel, Init)
{
    using namespace cv::gapi::wip::onevpl;

    std::vector<CfgParam> cfg_params_w_dx11;
    cfg_params_w_dx11.push_back(CfgParam::create_acceleration_mode(MFX_ACCEL_MODE_VIA_D3D11));
    VPLDX11AccelerationPolicy accel(std::make_shared<CfgParamDeviceSelector>(cfg_params_w_dx11));

    mfxLoader test_mfx_handle = MFXLoad();

    mfxConfig cfg_inst_0 = MFXCreateConfig(test_mfx_handle);
    EXPECT_TRUE(cfg_inst_0);
    mfxVariant mfx_param_0;
    mfx_param_0.Type = MFX_VARIANT_TYPE_U32;
    mfx_param_0.Data.U32 = MFX_IMPL_TYPE_HARDWARE;
    EXPECT_EQ(MFXSetConfigFilterProperty(cfg_inst_0,(mfxU8 *)CfgParam::implementation_name(),
                                                    mfx_param_0), MFX_ERR_NONE);

    mfxConfig cfg_inst_1 = MFXCreateConfig(test_mfx_handle);
    EXPECT_TRUE(cfg_inst_1);
    mfxVariant mfx_param_1;
    mfx_param_1.Type = MFX_VARIANT_TYPE_U32;
    mfx_param_1.Data.U32 = MFX_ACCEL_MODE_VIA_D3D11;
    EXPECT_EQ(MFXSetConfigFilterProperty(cfg_inst_1,(mfxU8 *)CfgParam::acceleration_mode_name(),
                                                    mfx_param_1), MFX_ERR_NONE);

    mfxConfig cfg_inst_2 = MFXCreateConfig(test_mfx_handle);
    EXPECT_TRUE(cfg_inst_2);
    mfxVariant mfx_param_2;
    mfx_param_2.Type = MFX_VARIANT_TYPE_U32;
    mfx_param_2.Data.U32 = MFX_CODEC_HEVC;
    EXPECT_EQ(MFXSetConfigFilterProperty(cfg_inst_2,(mfxU8 *)CfgParam::decoder_id_name(),
                                                    mfx_param_2), MFX_ERR_NONE);

    // create session
    mfxSession mfx_session{};
    mfxStatus sts = MFXCreateSession(test_mfx_handle, 0, &mfx_session);
    EXPECT_EQ(MFX_ERR_NONE, sts);

    // assign acceleration
    EXPECT_NO_THROW(accel.init(mfx_session));

    // create proper bitstream
    mfxBitstream bitstream{};
    const int BITSTREAM_BUFFER_SIZE = 2000000;
    bitstream.MaxLength = BITSTREAM_BUFFER_SIZE;
    bitstream.Data = (mfxU8 *)calloc(bitstream.MaxLength, sizeof(mfxU8));
    EXPECT_TRUE(bitstream.Data);

    // simulate read stream
    bitstream.DataOffset = 0;
    bitstream.DataLength = sizeof(streaming::onevpl::hevc_header) * sizeof(streaming::onevpl::hevc_header[0]);
    memcpy(bitstream.Data, streaming::onevpl::hevc_header, bitstream.DataLength);
    bitstream.CodecId = MFX_CODEC_HEVC;

    // prepare dec params
    mfxVideoParam mfxDecParams {};
    mfxDecParams.mfx.CodecId = bitstream.CodecId;
    mfxDecParams.IOPattern = MFX_IOPATTERN_OUT_VIDEO_MEMORY;
    sts = MFXVideoDECODE_DecodeHeader(mfx_session, &bitstream, &mfxDecParams);
    EXPECT_EQ(MFX_ERR_NONE, sts);

    mfxFrameAllocRequest request{};
    memset(&request, 0, sizeof(request));
    sts = MFXVideoDECODE_QueryIOSurf(mfx_session, &mfxDecParams, &request);
    EXPECT_EQ(MFX_ERR_NONE, sts);

    // Allocate surfaces for decoder
    VPLAccelerationPolicy::pool_key_t key = accel.create_surface_pool(request,
                                                                      mfxDecParams.mfx.FrameInfo);
    auto cand_surface = accel.get_free_surface(key).lock();

    sts = MFXVideoDECODE_Init(mfx_session, &mfxDecParams);
    EXPECT_EQ(MFX_ERR_NONE, sts);

    MFXVideoDECODE_Close(mfx_session);
    EXPECT_EQ(MFX_ERR_NONE, sts);

    EXPECT_NO_THROW(accel.deinit(mfx_session));
    MFXClose(mfx_session);
    MFXUnload(test_mfx_handle);
}
#endif // HAVE_DIRECTX
#endif // HAVE_D3D11

#ifdef __linux__
#if defined(HAVE_VA) || defined(HAVE_VA_INTEL)
TEST(OneVPL_Source_VAAPI_Accel, Init)
{
    using namespace cv::gapi::wip::onevpl;

    std::vector<CfgParam> cfg_params_w_vaapi;
    cfg_params_w_vaapi.push_back(CfgParam::create_acceleration_mode(MFX_ACCEL_MODE_VIA_VAAPI));
    VPLVAAPIAccelerationPolicy accel(std::make_shared<CfgParamDeviceSelector>(cfg_params_w_vaapi));

    mfxLoader test_mfx_handle = MFXLoad();

    mfxConfig cfg_inst_0 = MFXCreateConfig(test_mfx_handle);
    EXPECT_TRUE(cfg_inst_0);
    mfxVariant mfx_param_0;
    mfx_param_0.Type = MFX_VARIANT_TYPE_U32;
    mfx_param_0.Data.U32 = MFX_IMPL_TYPE_HARDWARE;
    EXPECT_EQ(MFXSetConfigFilterProperty(cfg_inst_0,(mfxU8 *)CfgParam::implementation_name(),
                                                    mfx_param_0), MFX_ERR_NONE);

    mfxConfig cfg_inst_1 = MFXCreateConfig(test_mfx_handle);
    EXPECT_TRUE(cfg_inst_1);
    mfxVariant mfx_param_1;
    mfx_param_1.Type = MFX_VARIANT_TYPE_U32;
    mfx_param_1.Data.U32 = MFX_ACCEL_MODE_VIA_VAAPI;
    EXPECT_EQ(MFXSetConfigFilterProperty(cfg_inst_1,(mfxU8 *)CfgParam::acceleration_mode_name(),
                                                    mfx_param_1), MFX_ERR_NONE);

    mfxConfig cfg_inst_2 = MFXCreateConfig(test_mfx_handle);
    EXPECT_TRUE(cfg_inst_2);
    mfxVariant mfx_param_2;
    mfx_param_2.Type = MFX_VARIANT_TYPE_U32;
    mfx_param_2.Data.U32 = MFX_CODEC_HEVC;
    EXPECT_EQ(MFXSetConfigFilterProperty(cfg_inst_2,(mfxU8 *)CfgParam::decoder_id_name(),
                                                    mfx_param_2), MFX_ERR_NONE);

    // create session
    mfxSession mfx_session{};
    mfxStatus sts = MFXCreateSession(test_mfx_handle, 0, &mfx_session);
    EXPECT_EQ(MFX_ERR_NONE, sts);

    // assign acceleration
    EXPECT_NO_THROW(accel.init(mfx_session));

    // create proper bitstream
    mfxBitstream bitstream{};
    const int BITSTREAM_BUFFER_SIZE = 2000000;
    bitstream.MaxLength = BITSTREAM_BUFFER_SIZE;
    bitstream.Data = (mfxU8 *)calloc(bitstream.MaxLength, sizeof(mfxU8));
    EXPECT_TRUE(bitstream.Data);

    // simulate read stream
    bitstream.DataOffset = 0;
    bitstream.DataLength = sizeof(streaming::onevpl::hevc_header) * sizeof(streaming::onevpl::hevc_header[0]);
    memcpy(bitstream.Data, streaming::onevpl::hevc_header, bitstream.DataLength);
    bitstream.CodecId = MFX_CODEC_HEVC;

    // prepare dec params
    mfxVideoParam mfxDecParams {};
    mfxDecParams.mfx.CodecId = bitstream.CodecId;
    mfxDecParams.IOPattern = MFX_IOPATTERN_OUT_SYSTEM_MEMORY;
    sts = MFXVideoDECODE_DecodeHeader(mfx_session, &bitstream, &mfxDecParams);
    EXPECT_EQ(MFX_ERR_NONE, sts);

    mfxFrameAllocRequest request{};
    memset(&request, 0, sizeof(request));
    sts = MFXVideoDECODE_QueryIOSurf(mfx_session, &mfxDecParams, &request);
    EXPECT_EQ(MFX_ERR_NONE, sts);

    // Allocate surfaces for decoder
    VPLAccelerationPolicy::pool_key_t key = accel.create_surface_pool(request,
                                                                      mfxDecParams.mfx.FrameInfo);
    auto cand_surface = accel.get_free_surface(key).lock();

    sts = MFXVideoDECODE_Init(mfx_session, &mfxDecParams);
    EXPECT_EQ(MFX_ERR_NONE, sts);

    MFXVideoDECODE_Close(mfx_session);
    EXPECT_EQ(MFX_ERR_NONE, sts);

    EXPECT_NO_THROW(accel.deinit(mfx_session));
    MFXClose(mfx_session);
    MFXUnload(test_mfx_handle);
}
#endif // defined(HAVE_VA) || defined(HAVE_VA_INTEL)
#endif // __linux__

#ifdef HAVE_DIRECTX
#ifdef HAVE_D3D11
TEST(OneVPL_Source_DX11_Accel_VPL, Init)
{
    using namespace cv::gapi::wip::onevpl;

    std::vector<CfgParam> cfg_params_w_dx11;
    cfg_params_w_dx11.push_back(CfgParam::create_acceleration_mode(MFX_ACCEL_MODE_VIA_D3D11));
    std::unique_ptr<VPLAccelerationPolicy> acceleration_policy (new VPLDX11AccelerationPolicy(std::make_shared<CfgParamDeviceSelector>(cfg_params_w_dx11)));

    mfxLoader test_mfx_handle = MFXLoad();

    mfxConfig cfg_inst_0 = MFXCreateConfig(test_mfx_handle);
    EXPECT_TRUE(cfg_inst_0);
    mfxVariant mfx_param_0;
    mfx_param_0.Type = MFX_VARIANT_TYPE_U32;
    mfx_param_0.Data.U32 = MFX_IMPL_TYPE_HARDWARE;
    EXPECT_EQ(MFXSetConfigFilterProperty(cfg_inst_0,(mfxU8 *)CfgParam::implementation_name(),
                                                    mfx_param_0), MFX_ERR_NONE);

    mfxConfig cfg_inst_1 = MFXCreateConfig(test_mfx_handle);
    EXPECT_TRUE(cfg_inst_1);
    mfxVariant mfx_param_1;
    mfx_param_1.Type = MFX_VARIANT_TYPE_U32;
    mfx_param_1.Data.U32 = MFX_ACCEL_MODE_VIA_D3D11;
    EXPECT_EQ(MFXSetConfigFilterProperty(cfg_inst_1,(mfxU8 *)CfgParam::acceleration_mode_name(),
                                                    mfx_param_1), MFX_ERR_NONE);

    mfxConfig cfg_inst_2 = MFXCreateConfig(test_mfx_handle);
    EXPECT_TRUE(cfg_inst_2);
    mfxVariant mfx_param_2;
    mfx_param_2.Type = MFX_VARIANT_TYPE_U32;
    mfx_param_2.Data.U32 = MFX_CODEC_HEVC;
    EXPECT_EQ(MFXSetConfigFilterProperty(cfg_inst_2,(mfxU8 *)CfgParam::decoder_id_name(),
                                                    mfx_param_2), MFX_ERR_NONE);

    mfxConfig cfg_inst_3 = MFXCreateConfig(test_mfx_handle);
    EXPECT_TRUE(cfg_inst_3);
    mfxVariant mfx_param_3;
    mfx_param_3.Type = MFX_VARIANT_TYPE_U32;
    mfx_param_3.Data.U32 = MFX_EXTBUFF_VPP_SCALING;
    EXPECT_EQ(MFXSetConfigFilterProperty(cfg_inst_3,
                                         (mfxU8 *)"mfxImplDescription.mfxVPPDescription.filter.FilterFourCC",
                                         mfx_param_3), MFX_ERR_NONE);
    // create session
    mfxSession mfx_session{};
    mfxStatus sts = MFXCreateSession(test_mfx_handle, 0, &mfx_session);
    EXPECT_EQ(MFX_ERR_NONE, sts);

    // assign acceleration
    EXPECT_NO_THROW(acceleration_policy->init(mfx_session));

    // create proper bitstream
    std::string file_path = findDataFile("highgui/video/big_buck_bunny.h265");
    std::shared_ptr<IDataProvider> data_provider(new FileDataProvider(file_path,
                                                                      {CfgParam::create_decoder_id(MFX_CODEC_HEVC)}));
    IDataProvider::mfx_codec_id_type decoder_id_name = data_provider->get_mfx_codec_id();

    // Prepare video param
    mfxVideoParam mfxDecParams {};
    mfxDecParams.mfx.CodecId = decoder_id_name;
    mfxDecParams.IOPattern = MFX_IOPATTERN_OUT_VIDEO_MEMORY;

    // try fetch & decode input data
    sts = MFX_ERR_NONE;
    std::shared_ptr<IDataProvider::mfx_bitstream> bitstream{};
    do {
        EXPECT_TRUE(data_provider->fetch_bitstream_data(bitstream));
        sts = MFXVideoDECODE_DecodeHeader(mfx_session, bitstream.get(), &mfxDecParams);
        EXPECT_TRUE(MFX_ERR_NONE == sts || MFX_ERR_MORE_DATA == sts);
    } while (sts == MFX_ERR_MORE_DATA && !data_provider->empty());

    EXPECT_EQ(MFX_ERR_NONE, sts);

    mfxFrameAllocRequest request{};
    memset(&request, 0, sizeof(request));
    sts = MFXVideoDECODE_QueryIOSurf(mfx_session, &mfxDecParams, &request);
    EXPECT_EQ(MFX_ERR_NONE, sts);

    // Allocate surfaces for decoder
    request.Type |= MFX_MEMTYPE_EXTERNAL_FRAME | MFX_MEMTYPE_FROM_DECODE | MFX_MEMTYPE_FROM_VPPIN;
    VPLAccelerationPolicy::pool_key_t decode_pool_key = acceleration_policy->create_surface_pool(request,
                                                                      mfxDecParams.mfx.FrameInfo);
    sts = MFXVideoDECODE_Init(mfx_session, &mfxDecParams);
    EXPECT_EQ(MFX_ERR_NONE, sts);

    // initialize VPLL
    mfxU16 vppOutImgWidth  = 672;
    mfxU16 vppOutImgHeight = 382;

    mfxVideoParam mfxVPPParams{0};
    mfxVPPParams.vpp.In = mfxDecParams.mfx.FrameInfo;

    mfxVPPParams.vpp.Out.FourCC        = MFX_FOURCC_NV12;
    mfxVPPParams.vpp.Out.ChromaFormat  = MFX_CHROMAFORMAT_YUV420;
    mfxVPPParams.vpp.Out.Width         = ALIGN16(vppOutImgWidth);
    mfxVPPParams.vpp.Out.Height        = ALIGN16(vppOutImgHeight);
    mfxVPPParams.vpp.Out.CropX = 0;
    mfxVPPParams.vpp.Out.CropY = 0;
    mfxVPPParams.vpp.Out.CropW         = vppOutImgWidth;
    mfxVPPParams.vpp.Out.CropH         = vppOutImgHeight;
    mfxVPPParams.vpp.Out.PicStruct     = MFX_PICSTRUCT_PROGRESSIVE;
    mfxVPPParams.vpp.Out.FrameRateExtN = 30;
    mfxVPPParams.vpp.Out.FrameRateExtD = 1;

    mfxVPPParams.IOPattern = MFX_IOPATTERN_IN_VIDEO_MEMORY | MFX_IOPATTERN_OUT_VIDEO_MEMORY;

    mfxFrameAllocRequest vppRequests[2];
    memset(&vppRequests, 0, sizeof(mfxFrameAllocRequest) * 2);
    EXPECT_EQ(MFXVideoVPP_QueryIOSurf(mfx_session, &mfxVPPParams, vppRequests), MFX_ERR_NONE);

    vppRequests[1].AllocId = 666;
    VPLAccelerationPolicy::pool_key_t vpp_out_pool_key =
                acceleration_policy->create_surface_pool(vppRequests[1], mfxVPPParams.vpp.Out);
    EXPECT_EQ(MFXVideoVPP_Init(mfx_session, &mfxVPPParams), MFX_ERR_NONE);

    // finalize session creation
    DecoderParams d_param{bitstream, mfxDecParams};
    TranscoderParams t_param{mfxVPPParams};
    VPLLegacyTranscodeEngine engine(std::move(acceleration_policy));
    std::shared_ptr<LegacyTranscodeSession> sess_ptr =
                                engine.register_session<LegacyTranscodeSession>(
                                                        mfx_session,
                                                        std::move(d_param),
                                                        std::move(t_param),
                                                        data_provider);

    sess_ptr->init_surface_pool(decode_pool_key);
    sess_ptr->init_transcode_surface_pool(vpp_out_pool_key);

    // prepare working surfaces
    sess_ptr->swap_decode_surface(engine);
    sess_ptr->swap_transcode_surface(engine);

    // launch pipeline
    LegacyTranscodeSession & my_sess = *sess_ptr;
    {
        if (!my_sess.data_provider) {
                my_sess.last_status = MFX_ERR_MORE_DATA;
        } else {
            my_sess.last_status = MFX_ERR_NONE;
            if (!my_sess.data_provider->fetch_bitstream_data(my_sess.stream)) {
                my_sess.last_status = MFX_ERR_MORE_DATA;
                my_sess.data_provider.reset(); //close source
            }
        }

        // 2) enqueue ASYNC decode operation
        // prepare sync object for new surface
        LegacyTranscodeSession::op_handle_t sync_pair{};

        // enqueue decode operation with current session surface
        {
            my_sess.last_status =
                    MFXVideoDECODE_DecodeFrameAsync(my_sess.session,
                                                    my_sess.get_mfx_bitstream_ptr(),
                                                    my_sess.processing_surface_ptr.lock()->get_handle(),
                                                    &sync_pair.second,
                                                    &sync_pair.first);

            // process wait-like statuses in-place:
            // It had better to use up all VPL decoding resources in pipeline
            // as soon as possible. So waiting more free-surface or device free
            while (my_sess.last_status == MFX_ERR_MORE_SURFACE ||
                   my_sess.last_status == MFX_WRN_DEVICE_BUSY) {
                try {
                    if (my_sess.last_status == MFX_ERR_MORE_SURFACE) {
                        my_sess.swap_decode_surface(engine);
                    }
                    my_sess.last_status =
                    MFXVideoDECODE_DecodeFrameAsync(my_sess.session,
                                                    my_sess.get_mfx_bitstream_ptr(),
                                                    my_sess.processing_surface_ptr.lock()->get_handle(),
                                                    &sync_pair.second,
                                                    &sync_pair.first);

                } catch (const std::runtime_error&) {
                    // NB: not an error, yield CPU ticks to check
                    // surface availability at a next phase.
                    break;
                }
            }
        }
        // 4) transcode
        {
            auto *dec_surface = sync_pair.second;
            if(my_sess.vpp_surface_ptr.lock())
            {
                mfxFrameSurface1* out_surf = my_sess.vpp_surface_ptr.lock()->get_handle();
                my_sess.last_status = MFXVideoVPP_RunFrameVPPAsync(my_sess.session, dec_surface,
                                                                out_surf,
                                                                nullptr, &sync_pair.first);
                sync_pair.second = out_surf;

                my_sess.last_status = MFXVideoCORE_SyncOperation(my_sess.session, sync_pair.first, 11000);
            }
            try {
                my_sess.swap_transcode_surface(engine);
            } catch (... ) {
                my_sess.vpp_surface_ptr.reset();
            }
        }
    }
}

TEST(OneVPL_Source_DX11_Accel_VPL, preproc)
{
    using namespace cv::gapi::wip::onevpl;

    std::vector<CfgParam> cfg_params_w_dx11;
    cfg_params_w_dx11.push_back(CfgParam::create_acceleration_mode(MFX_ACCEL_MODE_VIA_D3D11));
    std::unique_ptr<VPLAccelerationPolicy> acceleration_policy (new VPLDX11AccelerationPolicy(std::make_shared<CfgParamDeviceSelector>(cfg_params_w_dx11)));

    mfxLoader test_mfx_handle = MFXLoad();

    mfxConfig cfg_inst_0 = MFXCreateConfig(test_mfx_handle);
    EXPECT_TRUE(cfg_inst_0);
    mfxVariant mfx_param_0;
    mfx_param_0.Type = MFX_VARIANT_TYPE_U32;
    mfx_param_0.Data.U32 = MFX_IMPL_TYPE_HARDWARE;
    EXPECT_EQ(MFXSetConfigFilterProperty(cfg_inst_0,(mfxU8 *)CfgParam::implementation_name(),
                                                    mfx_param_0), MFX_ERR_NONE);

    mfxConfig cfg_inst_1 = MFXCreateConfig(test_mfx_handle);
    EXPECT_TRUE(cfg_inst_1);
    mfxVariant mfx_param_1;
    mfx_param_1.Type = MFX_VARIANT_TYPE_U32;
    mfx_param_1.Data.U32 = MFX_ACCEL_MODE_VIA_D3D11;
    EXPECT_EQ(MFXSetConfigFilterProperty(cfg_inst_1,(mfxU8 *)CfgParam::acceleration_mode_name(),
                                                    mfx_param_1), MFX_ERR_NONE);

    mfxConfig cfg_inst_2 = MFXCreateConfig(test_mfx_handle);
    EXPECT_TRUE(cfg_inst_2);
    mfxVariant mfx_param_2;
    mfx_param_2.Type = MFX_VARIANT_TYPE_U32;
    mfx_param_2.Data.U32 = MFX_CODEC_HEVC;
    EXPECT_EQ(MFXSetConfigFilterProperty(cfg_inst_2,(mfxU8 *)CfgParam::decoder_id_name(),
                                                    mfx_param_2), MFX_ERR_NONE);

    mfxConfig cfg_inst_3 = MFXCreateConfig(test_mfx_handle);
    EXPECT_TRUE(cfg_inst_3);
    mfxVariant mfx_param_3;
    mfx_param_3.Type = MFX_VARIANT_TYPE_U32;
    mfx_param_3.Data.U32 = MFX_EXTBUFF_VPP_SCALING;
    EXPECT_EQ(MFXSetConfigFilterProperty(cfg_inst_3,
                                         (mfxU8 *)"mfxImplDescription.mfxVPPDescription.filter.FilterFourCC",
                                         mfx_param_3), MFX_ERR_NONE);
    // create session
    mfxSession mfx_decode_session{};
    mfxStatus sts = MFXCreateSession(test_mfx_handle, 0, &mfx_decode_session);
    EXPECT_EQ(MFX_ERR_NONE, sts);

    // assign acceleration
    EXPECT_NO_THROW(acceleration_policy->init(mfx_decode_session));

    // create proper bitstream
    std::string file_path = findDataFile("highgui/video/big_buck_bunny.h265");
    std::shared_ptr<IDataProvider> data_provider(new FileDataProvider(file_path,
                                                                      {CfgParam::create_decoder_id(MFX_CODEC_HEVC)}));
    IDataProvider::mfx_codec_id_type decoder_id_name = data_provider->get_mfx_codec_id();

    // Prepare video param
    mfxVideoParam mfxDecParams {};
    mfxDecParams.mfx.CodecId = decoder_id_name;
    mfxDecParams.IOPattern = MFX_IOPATTERN_OUT_VIDEO_MEMORY;

    // try fetch & decode input data
    sts = MFX_ERR_NONE;
    std::shared_ptr<IDataProvider::mfx_bitstream> bitstream{};
    do {
        EXPECT_TRUE(data_provider->fetch_bitstream_data(bitstream));
        sts = MFXVideoDECODE_DecodeHeader(mfx_decode_session, bitstream.get(), &mfxDecParams);
        EXPECT_TRUE(MFX_ERR_NONE == sts || MFX_ERR_MORE_DATA == sts);
    } while (sts == MFX_ERR_MORE_DATA && !data_provider->empty());

    EXPECT_EQ(MFX_ERR_NONE, sts);

    mfxFrameAllocRequest request{};
    memset(&request, 0, sizeof(request));
    sts = MFXVideoDECODE_QueryIOSurf(mfx_decode_session, &mfxDecParams, &request);
    EXPECT_EQ(MFX_ERR_NONE, sts);

    // Allocate surfaces for decoder
    request.Type |= MFX_MEMTYPE_EXTERNAL_FRAME | MFX_MEMTYPE_FROM_DECODE | MFX_MEMTYPE_FROM_VPPIN;
    VPLAccelerationPolicy::pool_key_t decode_pool_key = acceleration_policy->create_surface_pool(request,
                                                                      mfxDecParams.mfx.FrameInfo);
    sts = MFXVideoDECODE_Init(mfx_decode_session, &mfxDecParams);
    EXPECT_EQ(MFX_ERR_NONE, sts);

    // initialize VPL session
    mfxSession mfx_vpl_session{};
    sts = MFXCreateSession(test_mfx_handle, 0, &mfx_vpl_session);
    // assign acceleration
    EXPECT_NO_THROW(acceleration_policy->init(mfx_vpl_session));
    EXPECT_EQ(MFX_ERR_NONE, sts);

    // request VPL surface
    mfxU16 vppOutImgWidth  = 672;
    mfxU16 vppOutImgHeight = 382;

    mfxVideoParam mfxVPPParams{0};
    mfxVPPParams.vpp.In = mfxDecParams.mfx.FrameInfo;

    mfxVPPParams.vpp.Out.FourCC        = MFX_FOURCC_NV12;
    mfxVPPParams.vpp.Out.ChromaFormat  = MFX_CHROMAFORMAT_YUV420;
    mfxVPPParams.vpp.Out.Width         = ALIGN16(vppOutImgWidth);
    mfxVPPParams.vpp.Out.Height        = ALIGN16(vppOutImgHeight);
    mfxVPPParams.vpp.Out.CropX = 0;
    mfxVPPParams.vpp.Out.CropY = 0;
    mfxVPPParams.vpp.Out.CropW         = vppOutImgWidth;
    mfxVPPParams.vpp.Out.CropH         = vppOutImgHeight;
    mfxVPPParams.vpp.Out.PicStruct     = MFX_PICSTRUCT_PROGRESSIVE;
    mfxVPPParams.vpp.Out.FrameRateExtN = 30;
    mfxVPPParams.vpp.Out.FrameRateExtD = 1;

    mfxVPPParams.IOPattern = MFX_IOPATTERN_IN_VIDEO_MEMORY | MFX_IOPATTERN_OUT_VIDEO_MEMORY;

    mfxFrameAllocRequest vppRequests[2];
    memset(&vppRequests, 0, sizeof(mfxFrameAllocRequest) * 2);
    EXPECT_EQ(MFXVideoVPP_QueryIOSurf(mfx_vpl_session, &mfxVPPParams, vppRequests), MFX_ERR_NONE);

    vppRequests[1].AllocId = 666;
    VPLAccelerationPolicy::pool_key_t vpp_out_pool_key =
                acceleration_policy->create_surface_pool(vppRequests[1], mfxVPPParams.vpp.Out);
    EXPECT_EQ(MFXVideoVPP_Init(mfx_vpl_session, &mfxVPPParams), MFX_ERR_NONE);

    // finalize session creation
    DecoderParams d_param{bitstream, mfxDecParams};
    TranscoderParams t_param{mfxVPPParams};
    VPLLegacyDecodeEngine engine(std::move(acceleration_policy));
    std::shared_ptr<LegacyDecodeSession> sess_ptr =
                                engine.register_session<LegacyDecodeSession>(
                                                        mfx_decode_session,
                                                        std::move(d_param),
                                                        data_provider);

    sess_ptr->init_surface_pool(decode_pool_key);

    // prepare working surfaces
    sess_ptr->swap_decode_surface(engine);

    // launch pipeline
    LegacyDecodeSession &my_sess = *sess_ptr;

    size_t min_available_frames_count =
                std::min(engine.get_accel()->get_surface_count(decode_pool_key),
                         engine.get_accel()->get_surface_count(vpp_out_pool_key));
    size_t frame_num = 0;
    do {
        if (!my_sess.data_provider) {
                my_sess.last_status = MFX_ERR_MORE_DATA;
        } else {
            my_sess.last_status = MFX_ERR_NONE;
            if (!my_sess.data_provider->fetch_bitstream_data(my_sess.stream)) {
                my_sess.last_status = MFX_ERR_MORE_DATA;
                my_sess.data_provider.reset(); //close source
            }
        }

        // 2) enqueue ASYNC decode operation
        // prepare sync object for new surface
        LegacyTranscodeSession::op_handle_t sync_pair{};

        // enqueue decode operation with current session surface
        {
            my_sess.last_status =
                    MFXVideoDECODE_DecodeFrameAsync(my_sess.session,
                                                    my_sess.get_mfx_bitstream_ptr(),
                                                    my_sess.processing_surface_ptr.lock()->get_handle(),
                                                    &sync_pair.second,
                                                    &sync_pair.first);

            // process wait-like statuses in-place:
            // It had better to use up all VPL decoding resources in pipeline
            // as soon as possible. So waiting more free-surface or device free
            while (my_sess.last_status == MFX_ERR_MORE_SURFACE ||
                   my_sess.last_status == MFX_WRN_DEVICE_BUSY) {
                try {
                    if (my_sess.last_status == MFX_ERR_MORE_SURFACE) {
                        my_sess.swap_decode_surface(engine);
                    }
                    my_sess.last_status =
                    MFXVideoDECODE_DecodeFrameAsync(my_sess.session,
                                                    my_sess.get_mfx_bitstream_ptr(),
                                                    my_sess.processing_surface_ptr.lock()->get_handle(),
                                                    &sync_pair.second,
                                                    &sync_pair.first);

                } catch (const std::runtime_error&) {
                    // NB: not an error, yield CPU ticks to check
                    // surface availability at a next phase.
                    EXPECT_TRUE(false);
                }
            }
        }
        {
            do {
                my_sess.last_status = MFXVideoCORE_SyncOperation(my_sess.session, sync_pair.first, 0);
                // put frames in ready queue on success
                if (MFX_ERR_NONE == my_sess.last_status) {
                    break;
                }
            } while (MFX_WRN_IN_EXECUTION == my_sess.last_status);
            EXPECT_EQ(my_sess.last_status, MFX_ERR_NONE);
        }

        // perform VPP operation on decoder synchronized surface

        auto vpp_out = engine.get_accel()->get_free_surface(vpp_out_pool_key).lock();
        EXPECT_TRUE(vpp_out.get());
        my_sess.last_status = MFXVideoVPP_RunFrameVPPAsync(mfx_vpl_session,
                                                           sync_pair.second,
                                                           vpp_out->get_handle(),
                                                           nullptr, &sync_pair.first);
        if (my_sess.last_status == MFX_ERR_MORE_SURFACE ||
            my_sess.last_status == MFX_ERR_NONE) {
            my_sess.last_status = MFXVideoCORE_SyncOperation(mfx_vpl_session, sync_pair.first, INFINITE);
            EXPECT_EQ(my_sess.last_status, MFX_ERR_NONE);
            frame_num++;
        }
    } while(frame_num < min_available_frames_count);
}

TEST(OneVPL_Source_DX11_FrameLockable, LockUnlock_without_Adaptee)
{
    using namespace cv::gapi::wip::onevpl;
    mfxMemId mid = 0;
    int lock_counter = 0;
    int unlock_counter = 0;

    std::function<mfxStatus(mfxMemId, mfxFrameData *)> lock =
    [&lock_counter] (mfxMemId, mfxFrameData *) {
        lock_counter ++;
        return MFX_ERR_NONE;
    };
    std::function<mfxStatus(mfxMemId, mfxFrameData *)> unlock =
    [&unlock_counter] (mfxMemId, mfxFrameData *) {
        unlock_counter++;
        return MFX_ERR_NONE;
    };

    auto test_allocator = create_test_allocator(mid, lock, unlock);
    LockAdapter adapter(test_allocator.get());

    mfxFrameData data;
    const int exec_count = 123;
    for (int i = 0; i < exec_count; i ++) {
        EXPECT_EQ(adapter.read_lock(mid, data), 0);
        adapter.write_lock(mid, data);
        EXPECT_EQ(adapter.unlock_read(mid, data), 0);
        adapter.unlock_write(mid, data);
    }

    EXPECT_EQ(lock_counter, exec_count * 2);
    EXPECT_EQ(unlock_counter, exec_count * 2);
}

TEST(OneVPL_Source_DX11_FrameLockable, LockUnlock_with_Adaptee)
{
    using namespace cv::gapi::wip::onevpl;
    mfxMemId mid = 0;
    int r_lock_counter = 0;
    int r_unlock_counter = 0;
    int w_lock_counter = 0;
    int w_unlock_counter = 0;

    SharedLock adaptee;
    std::function<mfxStatus(mfxMemId, mfxFrameData *)> lock =
    [&r_lock_counter, &w_lock_counter, &adaptee] (mfxMemId, mfxFrameData *) {
        if (adaptee.owns()) {
            w_lock_counter ++;
        } else {
            r_lock_counter ++;
        }
        return MFX_ERR_NONE;
    };
    std::function<mfxStatus(mfxMemId, mfxFrameData *)> unlock =
    [&r_unlock_counter, &w_unlock_counter, &adaptee] (mfxMemId, mfxFrameData *) {
        if (adaptee.owns()) {
            w_unlock_counter ++;
        } else {
            r_unlock_counter ++;
        }
        return MFX_ERR_NONE;
    };

    auto test_allocator = create_test_allocator(mid, lock, unlock);
    LockAdapter adapter(test_allocator.get());

    adapter.set_adaptee(&adaptee);

    mfxFrameData data;
    const int exec_count = 123;
    for (int i = 0; i < exec_count; i ++) {
        EXPECT_EQ(adapter.read_lock(mid, data), 0);
        EXPECT_FALSE(adaptee.try_lock());

        EXPECT_EQ(adapter.unlock_read(mid, data), 1);
        EXPECT_TRUE(adaptee.try_lock());
        adaptee.unlock();

        adapter.write_lock(mid, data);
        adapter.unlock_write(mid, data);
    }

    EXPECT_EQ(r_lock_counter, exec_count);
    EXPECT_EQ(r_unlock_counter, exec_count);
    EXPECT_EQ(w_lock_counter, exec_count);
    EXPECT_EQ(w_unlock_counter, exec_count);
}
#endif // HAVE_DIRECTX
#endif // HAVE_D3D11
}
} // namespace opencv_test
#endif // HAVE_ONEVPL
