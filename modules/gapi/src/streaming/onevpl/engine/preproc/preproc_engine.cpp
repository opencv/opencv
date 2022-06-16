// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2022 Intel Corporation

#ifdef HAVE_ONEVPL

#include <algorithm>
#include <exception>

#include <opencv2/gapi/streaming/onevpl/data_provider_interface.hpp>

#include "streaming/onevpl/engine/preproc/preproc_engine.hpp"
#include "streaming/onevpl/engine/preproc/preproc_session.hpp"

#include "streaming/onevpl/accelerators/accel_policy_interface.hpp"
#include "streaming/onevpl/accelerators/surface/surface.hpp"
#include "streaming/onevpl/cfg_params_parser.hpp"
#include "logger.hpp"

#define ALIGN16(value)           (((value + 15) >> 4) << 4)

namespace cv {
namespace gapi {
namespace wip {
namespace onevpl {

bool FrameInfoComparator::operator()(const mfxFrameInfo& lhs, const mfxFrameInfo& rhs) const {
    return lhs < rhs;
}

bool FrameInfoComparator::equal_to(const mfxFrameInfo& lhs, const mfxFrameInfo& rhs) {
    return lhs == rhs;
}

static void apply_roi(mfxFrameSurface1* surface_handle,
                      const cv::util::optional<cv::Rect> &opt_roi) {
    if (opt_roi.has_value()) {
        const cv::Rect &roi = opt_roi.value();
        surface_handle->Info.CropX = static_cast<mfxU16>(roi.x);
        surface_handle->Info.CropY = static_cast<mfxU16>(roi.y);
        surface_handle->Info.CropW = static_cast<mfxU16>(roi.width);
        surface_handle->Info.CropH = static_cast<mfxU16>(roi.height);
        GAPI_LOG_DEBUG(nullptr, "applied ROI {" << surface_handle->Info.CropX <<
                                ", " << surface_handle->Info.CropY << "}, "
                                "{ " << surface_handle->Info.CropX + surface_handle->Info.CropW <<
                                ", " << surface_handle->Info.CropY + surface_handle->Info.CropH << "}");
    }
}

VPPPreprocEngine::VPPPreprocEngine(std::unique_ptr<VPLAccelerationPolicy>&& accel) :
    ProcessingEngineBase(std::move(accel)) {
    GAPI_LOG_DEBUG(nullptr, "Create VPP preprocessing engine");
    preprocessed_frames_count = 0;
    create_pipeline(
        // 0) preproc decoded surface with VPP params
        [this] (EngineSession& sess) -> ExecutionStatus
        {
            session_type &my_sess = static_cast<session_type&>(sess);
            while (!my_sess.sync_in_queue.empty()) {
                do {
                    if (!my_sess.processing_surface_ptr.expired()) {
                        session_type::incoming_task pending_op = my_sess.sync_in_queue.front();
                        GAPI_LOG_DEBUG(nullptr, "pending IN operations count: " <<
                                                my_sess.sync_in_queue.size() <<
                                                ", sync id:  " <<
                                                pending_op.sync_handle <<
                                                ", surface:  " <<
                                                pending_op.decoded_surface_ptr);

                        my_sess.sync_in_queue.pop();
                        auto *vpp_suface = my_sess.processing_surface_ptr.lock()->get_handle();

                        apply_roi(pending_op.decoded_surface_ptr, pending_op.roi);

                        mfxSyncPoint vpp_sync_handle{};
                        my_sess.last_status = MFXVideoVPP_RunFrameVPPAsync(my_sess.session,
                                                                           pending_op.decoded_surface_ptr,
                                                                           vpp_suface,
                                                                           nullptr, &vpp_sync_handle);
                        session_type::outgoing_task vpp_pending_op {vpp_sync_handle,
                                                                    vpp_suface,
                                                                    std::move(pending_op) };
                        GAPI_LOG_DEBUG(nullptr, "Got VPP async operation" <<
                                                ", sync id:  " <<
                                                vpp_pending_op.sync_handle <<
                                                ", dec surface:  " <<
                                                vpp_pending_op.original_surface_ptr <<
                                                ", trans surface: " <<
                                                vpp_pending_op.vpp_surface_ptr <<
                                                ", status: " <<
                                                mfxstatus_to_string(my_sess.last_status));
                        // NB: process status
                        if (my_sess.last_status == MFX_ERR_MORE_SURFACE ||
                            my_sess.last_status == MFX_ERR_NONE) {
                            vpp_pending_op.vpp_surface_ptr->Data.Locked++; // TODO -S- workaround
                            my_sess.vpp_out_queue.emplace(vpp_pending_op);
                        }
                    }

                    try {
                        my_sess.swap_surface(*this);
                    } catch (const std::runtime_error& ex) {
                        // NB: not an error, yield CPU ticks to check
                        // surface availability at a next phase.
                        // But print WARNING to notify user about pipeline stuck
                        GAPI_LOG_WARNING(nullptr, "[" << my_sess.session <<
                                                    "] has no VPP surface, reason: " <<
                                                  ex.what());
                        my_sess.processing_surface_ptr.reset();
                        break;
                    }
                } while(my_sess.last_status == MFX_ERR_MORE_SURFACE);

                if (my_sess.processing_surface_ptr.expired()) {
                    // TODO break main loop
                    break;
                }
            }
            return ExecutionStatus::Continue;
        },
        // 1) Wait for ASYNC decode result
        [this] (EngineSession& sess) -> ExecutionStatus
        {
            session_type& my_sess = static_cast<session_type&>(sess);
            do {
                if (!my_sess.vpp_out_queue.empty()) { // FIFO: check the oldest async operation complete
                    session_type::outgoing_task& pending_op = my_sess.vpp_out_queue.front();
                    sess.last_status = MFXVideoCORE_SyncOperation(sess.session, pending_op.sync_handle, 0);

                    GAPI_LOG_DEBUG(nullptr, "pending VPP operations count: " <<
                                            my_sess.vpp_out_queue.size() <<
                                            ", sync id:  " <<
                                            pending_op.sync_handle <<
                                            ", surface:  " <<
                                            pending_op.vpp_surface_ptr <<
                                            ", status: " <<
                                            mfxstatus_to_string(my_sess.last_status));

                    // put frames in ready queue on success
                    if (MFX_ERR_NONE == sess.last_status) {
                        pending_op.release_frame();
                        on_frame_ready(my_sess, pending_op.vpp_surface_ptr);
                    }
                }
            } while (MFX_ERR_NONE == sess.last_status && !my_sess.vpp_out_queue.empty());
            return ExecutionStatus::Continue;
        },
        // 2) Falls back on generic status processing
        [this] (EngineSession& sess) -> ExecutionStatus
        {
            return this->process_error(sess.last_status, static_cast<session_type&>(sess));
        }
    );
}

cv::util::optional<pp_params> VPPPreprocEngine::is_applicable(const cv::MediaFrame& in_frame) {
    // TODO consider something smarter than RTI
    cv::util::optional<pp_params> ret;
    BaseFrameAdapter *vpl_adapter = in_frame.get<BaseFrameAdapter>();
    GAPI_LOG_DEBUG(nullptr, "validate VPP preprocessing is applicable for frame");
    if (vpl_adapter) {
        ret = cv::util::make_optional<pp_params>(
                        pp_params::create<vpp_pp_params>(vpl_adapter->get_session_handle(),
                                                         vpl_adapter->get_surface()->get_info(),
                                                         vpl_adapter));
        GAPI_LOG_DEBUG(nullptr, "VPP preprocessing applicable, session [" <<
                                vpl_adapter->get_session_handle() << "]");
    }
    return ret;
}

pp_session VPPPreprocEngine::initialize_preproc(const pp_params& initial_frame_param,
                                                const GFrameDesc& required_frame_descr) {
    const vpp_pp_params &params = initial_frame_param.get<vpp_pp_params>();

    // adjust preprocessing settings
    mfxVideoParam mfxVPPParams{};
    memset(&mfxVPPParams, 0, sizeof(mfxVideoParam));
    // NB: IN params for VPP session must be equal to decoded surface params
    mfxVPPParams.vpp.In = params.info;

    // NB: OUT params must refer to IN params of a network
    GAPI_LOG_DEBUG(nullptr, "network input size: " << required_frame_descr.size.width <<
                            "x" << required_frame_descr.size.height);
    mfxVPPParams.vpp.Out = mfxVPPParams.vpp.In;
    switch (required_frame_descr.fmt) {
        case MediaFormat::NV12:
            mfxVPPParams.vpp.Out.FourCC = MFX_FOURCC_NV12;
            break;
        default:
            GAPI_LOG_WARNING(nullptr, "Unsupported MediaFormat in preprocessing: " <<
                                      static_cast<int>(required_frame_descr.fmt) <<
                                      ". Frame will be rejected");
            throw std::runtime_error("unsupported MediaFormat value in VPP preprocessing");
    }

    mfxVPPParams.vpp.Out.ChromaFormat  = MFX_CHROMAFORMAT_YUV420;
    mfxVPPParams.vpp.Out.Width         = static_cast<mfxU16>(required_frame_descr.size.width);
    mfxVPPParams.vpp.Out.Height        = static_cast<mfxU16>(required_frame_descr.size.height);
    mfxVPPParams.vpp.Out.CropW         = mfxVPPParams.vpp.Out.Width;
    mfxVPPParams.vpp.Out.CropH         = mfxVPPParams.vpp.Out.Height;

    // check In & Out equally to bypass preproc
    if (mfxVPPParams.vpp.Out == mfxVPPParams.vpp.In) {
        GAPI_LOG_DEBUG(nullptr, "no preproc required");
        return pp_session::create<vpp_pp_session>(nullptr);
    }

    // recalculate size param according to VPP alignment
    mfxVPPParams.vpp.Out.Width = ALIGN16(mfxVPPParams.vpp.Out.Width);
    mfxVPPParams.vpp.Out.Height = ALIGN16(mfxVPPParams.vpp.Out.Height);
    mfxVPPParams.vpp.Out.CropW = mfxVPPParams.vpp.Out.Width;
    mfxVPPParams.vpp.Out.CropH = mfxVPPParams.vpp.Out.Height;

    GAPI_LOG_DEBUG(nullptr, "\nFrom:\n{\n" << mfx_frame_info_to_string(mfxVPPParams.vpp.In) <<
                            "}\nTo:\n{\n" << mfx_frame_info_to_string(mfxVPPParams.vpp.Out) << "}");

    // find existing session
    GAPI_LOG_DEBUG(nullptr, "Find existing VPPPreprocSession for requested frame params"
                            ", total sessions: " << preproc_session_map.size());
    auto it = preproc_session_map.find(mfxVPPParams.vpp.In);
    if (it != preproc_session_map.end()) {
        GAPI_LOG_DEBUG(nullptr, "[" << it->second->session << "] found");
        return pp_session::create<vpp_pp_session>(std::static_pointer_cast<EngineSession>(it->second));
    }

    // NB: make some sanity checks
    IDeviceSelector::DeviceScoreTable devices = acceleration_policy->get_device_selector()->select_devices();
    GAPI_Assert(devices.size() == 1 && "Multiple(or zero) acceleration devices case is unsupported");
    AccelType accel_type = devices.begin()->second.get_type();
    // assign acceleration
    if (accel_type == AccelType::DX11) {
        mfxVPPParams.IOPattern = MFX_IOPATTERN_IN_VIDEO_MEMORY | MFX_IOPATTERN_OUT_VIDEO_MEMORY;
    } else {
        mfxVPPParams.IOPattern = MFX_IOPATTERN_IN_SYSTEM_MEMORY | MFX_IOPATTERN_OUT_SYSTEM_MEMORY;
    }

    // clone existing VPL session to inherit VPL loader configuration
    // and avoid refer to any global state
    // TODO no clone due to clone issue

    mfxSession mfx_vpp_session = params.handle;
    mfxStatus sts = MFX_ERR_NONE;

    // TODO: simply use clone after VPL bug fixing
    //sts = MFXCloneSession(params.handle, &mfx_vpp_session);
    sts = MFXCreateSession(mfx_handle, impl_number, &mfx_vpp_session);
    if (sts != MFX_ERR_NONE) {
        GAPI_LOG_WARNING(nullptr, "Cannot clone VPP session, error: " << mfxstatus_to_string(sts));
        GAPI_Assert(false && "Cannot continue VPP preprocessing");
    }

    sts = MFXJoinSession(params.handle, mfx_vpp_session);
    if (sts != MFX_ERR_NONE) {
        GAPI_LOG_WARNING(nullptr, "Cannot join VPP sessions, error: " << mfxstatus_to_string(sts));
        GAPI_Assert(false && "Cannot continue VPP preprocessing");
    }

    GAPI_LOG_INFO(nullptr, "[" << mfx_vpp_session << "] starting pool allocation");
    VPLAccelerationPolicy::pool_key_t vpp_out_pool_key {};
    try {
        // assign HW acceleration processor
        acceleration_policy->init(mfx_vpp_session);
        try {
            // ask to allocate external memory pool
            mfxFrameAllocRequest vppRequests[2];
            memset(&vppRequests, 0, sizeof(mfxFrameAllocRequest) * 2);
            sts = MFXVideoVPP_QueryIOSurf(mfx_vpp_session, &mfxVPPParams, vppRequests);
            if (MFX_ERR_NONE != sts) {
                GAPI_LOG_WARNING(nullptr, "cannot execute MFXVideoVPP_QueryIOSurf, error: " <<
                                          mfxstatus_to_string(sts));
                throw std::runtime_error("Cannot execute MFXVideoVPP_QueryIOSurf");
            }

            // NB: Assign ID as upper limit descendant to distinguish specific VPP allocation
            // from decode allocations witch started from 0: by local module convention

            static uint16_t request_id = 0;
            vppRequests[1].AllocId = std::numeric_limits<uint16_t>::max() - request_id++;
            GAPI_Assert(request_id != std::numeric_limits<uint16_t>::max() && "Something wrong");

            vppRequests[1].Type |= MFX_MEMTYPE_FROM_VPPIN;
            vpp_out_pool_key = acceleration_policy->create_surface_pool(vppRequests[1],
                                                                        mfxVPPParams.vpp.Out);

            sts = MFXVideoVPP_Init(mfx_vpp_session, &mfxVPPParams);
            if (MFX_ERR_NONE != sts) {
                GAPI_LOG_WARNING(nullptr, "cannot Init VPP, error: " <<
                                          mfxstatus_to_string(sts));
                // TODO consider deallocate pool
                // but not necessary now cause every fail processed as GAPI_Assert
                throw std::runtime_error("Cannot init VPP, error: " +
                                         mfxstatus_to_string(sts));
            }
        } catch (const std::exception&) {
            GAPI_LOG_WARNING(nullptr, "[" << mfx_vpp_session << "] allocation failed, rollback");
            acceleration_policy->deinit(mfx_vpp_session);
            throw;
        }
    } catch (const std::exception&) {
        MFXClose(mfx_vpp_session);
        GAPI_Assert(false && "Cannot init preproc resources");
    }

    // create engine session after all
    session_ptr_type sess_ptr = register_session<session_type>(mfx_vpp_session,
                                                               mfxVPPParams);
    sess_ptr->init_surface_pool(vpp_out_pool_key);
    sess_ptr->swap_surface(*this);

    bool inserted = preproc_session_map.emplace(mfxVPPParams.vpp.In, sess_ptr).second;
    GAPI_Assert(inserted && "preproc session is exist");
    GAPI_LOG_INFO(nullptr, "VPPPreprocSession created, total sessions: " << preproc_session_map.size());
    return pp_session::create<vpp_pp_session>(std::static_pointer_cast<EngineSession>(sess_ptr));
}

void VPPPreprocEngine::on_frame_ready(session_type& sess,
                                      mfxFrameSurface1* ready_surface)
{
    GAPI_LOG_DEBUG(nullptr, "[" << sess.session << "], frame ready");

    // manage memory ownership rely on acceleration policy
    ready_surface->Data.Locked--;  // TODO -S- workaround
    VPLAccelerationPolicy::FrameConstructorArgs args{ready_surface, sess.session};
    auto frame_adapter = acceleration_policy->create_frame_adapter(sess.vpp_pool_id,
                                                                   args);
    ready_frames.emplace(cv::MediaFrame(std::move(frame_adapter)), sess.generate_frame_meta());

    // pop away synced out object
    sess.vpp_out_queue.pop();
}

VPPPreprocEngine::session_ptr
VPPPreprocEngine::initialize_session(mfxSession,
                                     const std::vector<CfgParam>&,
                                     std::shared_ptr<IDataProvider>) {
    return {};
}

cv::MediaFrame VPPPreprocEngine::run_sync(const pp_session& sess, const cv::MediaFrame& in_frame,
                                          const cv::util::optional<cv::Rect> &roi) {
    vpp_pp_session pp_sess_impl = sess.get<vpp_pp_session>();
    if (!pp_sess_impl.handle) {
        // bypass case
        return in_frame;
    }
    session_ptr_type s = std::static_pointer_cast<session_type>(pp_sess_impl.handle);
    GAPI_DbgAssert(s && "Session is nullptr");
    GAPI_DbgAssert(is_applicable(in_frame) &&
                   "VPP preproc is not applicable for the given frame");
    BaseFrameAdapter *vpl_adapter = in_frame.get<BaseFrameAdapter>();
    if (!vpl_adapter) {
        GAPI_LOG_WARNING(nullptr, "VPP preproc is inapplicable for a given frame. "
                                  "Make sure the frame is collected using onevpl::GSource");
        throw std::runtime_error("VPP preproc is inapplicable for given frame");
    }

    // schedule decoded surface into preproc queue
    session_type::incoming_task in_preproc_request {nullptr,
                                                    vpl_adapter->get_surface()->get_handle(),
                                                    vpl_adapter->get_surface()->get_info(),
                                                    in_frame,
                                                    roi};
    s->sync_in_queue.emplace(in_preproc_request);

    // invoke pipeline to transform decoded surface into preprocessed surface
    try
    {
        ExecutionStatus status = ExecutionStatus::Continue;
        while (0 == get_ready_frames_count() &&
            status == ExecutionStatus::Continue) {
            status = process(s->session);
        }

        if (get_ready_frames_count() == 0) {
            GAPI_LOG_WARNING(nullptr, "failed: cannot obtain preprocessed frames, last status: " <<
                                        ProcessingEngineBase::status_to_string(status));
            throw std::runtime_error("cannot finalize VPP preprocessing operation");
        }
    } catch(const std::exception&) {
        throw;
    }
    // obtain new frame is available
    cv::gapi::wip::Data data;
    get_frame(data);
    preprocessed_frames_count++;
    GAPI_LOG_DEBUG(nullptr, "processed frames count: " << preprocessed_frames_count);
    return cv::util::get<cv::MediaFrame>(data);
}

ProcessingEngineBase::ExecutionStatus VPPPreprocEngine::process_error(mfxStatus status, session_type& sess) {
    GAPI_LOG_DEBUG(nullptr, "status: " << mfxstatus_to_string(status));

    switch (status) {
        case MFX_ERR_NONE:
        {
            // prepare sync object for new surface
            try {
                sess.swap_surface(*this);
                return ExecutionStatus::Continue;
            } catch (const std::runtime_error& ex) {
                GAPI_LOG_WARNING(nullptr, "[" << sess.session << "] error: " << ex.what());
                return ExecutionStatus::Continue; // read more data
            }
        }
        case MFX_ERR_MORE_DATA: // The function requires more bitstream at input before decoding can proceed
            return ExecutionStatus::Processed;
        case MFX_ERR_MORE_SURFACE:
        {
            // The function requires more frame surface at output before decoding can proceed.
            // This applies to external memory allocations and should not be expected for
            // a simple internal allocation case like this
            try {
                sess.swap_surface(*this);
                return ExecutionStatus::Continue;
            } catch (const std::runtime_error& ex) {
                GAPI_LOG_WARNING(nullptr, "[" << sess.session << "] error: " << ex.what());
                 return ExecutionStatus::Continue; // read more data
            }
            break;
        }
        case MFX_ERR_DEVICE_LOST:
            // For non-CPU implementations,
            // Cleanup if device is lost
            GAPI_DbgAssert(false && "VPPPreprocEngine::process_error - "
                                    "MFX_ERR_DEVICE_LOST is not processed");
            break;
        case MFX_WRN_DEVICE_BUSY:
            // For non-CPU implementations,
            // Wait a few milliseconds then try again
            GAPI_DbgAssert(false && "VPPPreprocEngine::process_error - "
                                    "MFX_WRN_DEVICE_BUSY is not processed");
            break;
        case MFX_WRN_VIDEO_PARAM_CHANGED:
            // The decoder detected a new sequence header in the bitstream.
            // Video parameters may have changed.
            // In external memory allocation case, might need to reallocate the output surface
            GAPI_LOG_WARNING(nullptr, "[" << sess.session << "] got MFX_WRN_VIDEO_PARAM_CHANGED");
            return ExecutionStatus::Continue;
            break;
        case MFX_ERR_INCOMPATIBLE_VIDEO_PARAM:
            // The function detected that video parameters provided by the application
            // are incompatible with initialization parameters.
            // The application should close the component and then reinitialize it
            GAPI_DbgAssert(false && "VPPPreprocEngine::process_error - "
                                    "MFX_ERR_INCOMPATIBLE_VIDEO_PARAM is not processed");
            break;
        case MFX_ERR_REALLOC_SURFACE:
            // Bigger surface_work required. May be returned only if
            // mfxInfoMFX::EnableReallocRequest was set to ON during initialization.
            // This applies to external memory allocations and should not be expected for
            // a simple internal allocation case like this
            GAPI_DbgAssert(false && "VPPPreprocEngine::process_error - "
                                    "MFX_ERR_REALLOC_SURFACE is not processed");
            break;
        case MFX_WRN_IN_EXECUTION:
            GAPI_LOG_DEBUG(nullptr, "[" << sess.session << "] got MFX_WRN_IN_EXECUTION");
            return ExecutionStatus::Continue;
        default:
            GAPI_LOG_WARNING(nullptr, "Unknown status code: " << mfxstatus_to_string(status) <<
                                      ", decoded frames: " << sess.preprocessed_frames_count);
            break;
    }

    return ExecutionStatus::Failed;
}
} // namespace onevpl
} // namespace wip
} // namespace gapi
} // namespace cv
#endif // HAVE_ONEVPL
