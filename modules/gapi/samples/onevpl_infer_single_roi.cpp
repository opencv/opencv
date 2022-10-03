#include <algorithm>
#include <fstream>
#include <iostream>
#include <cctype>
#include <tuple>

#include <opencv2/imgproc.hpp>
#include <opencv2/gapi.hpp>
#include <opencv2/gapi/core.hpp>
#include <opencv2/gapi/cpu/gcpukernel.hpp>
#include <opencv2/gapi/infer/ie.hpp>
#include <opencv2/gapi/render.hpp>
#include <opencv2/gapi/streaming/onevpl/source.hpp>
#include <opencv2/highgui.hpp> // CommandLineParser
#include <opencv2/gapi/infer/parsers.hpp>

#ifdef HAVE_INF_ENGINE
#include <inference_engine.hpp> // ParamMap
#endif // HAVE_INF_ENGINE

#ifdef HAVE_DIRECTX
#ifdef HAVE_D3D11

// get rid of generate macro max/min/etc from DX side
#define D3D11_NO_HELPERS
#define NOMINMAX
#include <d3d11.h>
#undef NOMINMAX
#undef D3D11_NO_HELPERS
#endif // HAVE_D3D11
#endif // HAVE_DIRECTX

#ifdef __linux__
#if defined(HAVE_VA) || defined(HAVE_VA_INTEL)
#include "va/va.h"
#include "va/va_drm.h"

#include <fcntl.h>
#include <unistd.h>
#endif // defined(HAVE_VA) || defined(HAVE_VA_INTEL)
#endif // __linux__


const std::string about =
    "This is an OpenCV-based version of oneVPLSource decoder example";
const std::string keys =
    "{ h help                       |                                           | Print this help message }"
    "{ input                        |                                           | Path to the input demultiplexed video file }"
    "{ output                       |                                           | Path to the output RAW video file. Use .avi extension }"
    "{ facem                        | face-detection-adas-0001.xml              | Path to OpenVINO IE face detection model (.xml) }"
    "{ faced                        | GPU                                       | Target device for face detection model (e.g. AUTO, GPU, VPU, ...) }"
    "{ cfg_params                   |                                           | Semicolon separated list of oneVPL mfxVariants which is used for configuring source (see `MFXSetConfigFilterProperty` by https://spec.oneapi.io/versions/latest/elements/oneVPL/source/index.html) }"
    "{ streaming_queue_capacity     | 1                                         | Streaming executor queue capacity. Calculated automatically if 0 }"
    "{ frames_pool_size             | 0                                         | OneVPL source applies this parameter as preallocated frames pool size}"
    "{ vpp_frames_pool_size         | 0                                         | OneVPL source applies this parameter as preallocated frames pool size for VPP preprocessing results}"
    "{ roi                          | -1,-1,-1,-1                               | Region of interest (ROI) to use for inference. Identified automatically when not set }"
    "{ source_device                | CPU                                       | choose device for decoding }"
    "{ preproc_device               |                                           | choose device for preprocessing }";


namespace {
bool is_gpu(const std::string &device_name) {
    return device_name.find("GPU") != std::string::npos;
}

std::string get_weights_path(const std::string &model_path) {
    const auto EXT_LEN = 4u;
    const auto sz = model_path.size();
    GAPI_Assert(sz > EXT_LEN);

    auto ext = model_path.substr(sz - EXT_LEN);
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c){
            return static_cast<unsigned char>(std::tolower(c));
        });
    GAPI_Assert(ext == ".xml");
    return model_path.substr(0u, sz - EXT_LEN) + ".bin";
}

// TODO: It duplicates infer_single_roi sample
cv::util::optional<cv::Rect> parse_roi(const std::string &rc) {
    cv::Rect rv;
    char delim[3];

    std::stringstream is(rc);
    is >> rv.x >> delim[0] >> rv.y >> delim[1] >> rv.width >> delim[2] >> rv.height;
    if (is.bad()) {
        return cv::util::optional<cv::Rect>(); // empty value
    }
    const auto is_delim = [](char c) {
        return c == ',';
    };
    if (!std::all_of(std::begin(delim), std::end(delim), is_delim)) {
        return cv::util::optional<cv::Rect>(); // empty value
    }
    if (rv.x < 0 || rv.y < 0 || rv.width <= 0 || rv.height <= 0) {
        return cv::util::optional<cv::Rect>(); // empty value
    }
    return cv::util::make_optional(std::move(rv));
}

#ifdef HAVE_DIRECTX
#ifdef HAVE_D3D11

// Since ATL headers might not be available on specific MSVS Build Tools
// we use simple `CComPtr` implementation like as `ComPtrGuard`
// which is not supposed to be the full functional replacement of `CComPtr`
// and it uses as RAII to make sure utilization is correct
template <typename COMNonManageableType>
void release(COMNonManageableType *ptr) {
    if (ptr) {
        ptr->Release();
    }
}

template <typename COMNonManageableType>
using ComPtrGuard = std::unique_ptr<COMNonManageableType, decltype(&release<COMNonManageableType>)>;

template <typename COMNonManageableType>
ComPtrGuard<COMNonManageableType> createCOMPtrGuard(COMNonManageableType *ptr = nullptr) {
    return ComPtrGuard<COMNonManageableType> {ptr, &release<COMNonManageableType>};
}


using AccelParamsType = std::tuple<ComPtrGuard<ID3D11Device>, ComPtrGuard<ID3D11DeviceContext>>;

AccelParamsType create_device_with_ctx(IDXGIAdapter* adapter) {
    UINT flags = 0;
    D3D_FEATURE_LEVEL feature_levels[] = { D3D_FEATURE_LEVEL_11_1,
                                           D3D_FEATURE_LEVEL_11_0,
                                         };
    D3D_FEATURE_LEVEL featureLevel;
    ID3D11Device* ret_device_ptr = nullptr;
    ID3D11DeviceContext* ret_ctx_ptr = nullptr;
    HRESULT err = D3D11CreateDevice(adapter, D3D_DRIVER_TYPE_UNKNOWN,
                                    nullptr, flags,
                                    feature_levels,
                                    ARRAYSIZE(feature_levels),
                                    D3D11_SDK_VERSION, &ret_device_ptr,
                                    &featureLevel, &ret_ctx_ptr);
    if (FAILED(err)) {
        throw std::runtime_error("Cannot create D3D11CreateDevice, error: " +
                                 std::to_string(HRESULT_CODE(err)));
    }

    return std::make_tuple(createCOMPtrGuard(ret_device_ptr),
                           createCOMPtrGuard(ret_ctx_ptr));
}
#endif // HAVE_D3D11
#endif // HAVE_DIRECTX
} // anonymous namespace

namespace custom {
G_API_NET(FaceDetector,   <cv::GMat(cv::GMat)>, "face-detector");

using GDetections = cv::GArray<cv::Rect>;
using GRect       = cv::GOpaque<cv::Rect>;
using GSize       = cv::GOpaque<cv::Size>;
using GPrims      = cv::GArray<cv::gapi::wip::draw::Prim>;

G_API_OP(ParseSSD, <GDetections(cv::GMat, GRect, GSize)>, "sample.custom.parse-ssd") {
    static cv::GArrayDesc outMeta(const cv::GMatDesc &, const cv::GOpaqueDesc &, const cv::GOpaqueDesc &) {
        return cv::empty_array_desc();
    }
};

// TODO: It duplicates infer_single_roi sample
G_API_OP(LocateROI, <GRect(GSize)>, "sample.custom.locate-roi") {
    static cv::GOpaqueDesc outMeta(const cv::GOpaqueDesc &) {
        return cv::empty_gopaque_desc();
    }
};

G_API_OP(BBoxes, <GPrims(GDetections, GRect)>, "sample.custom.b-boxes") {
    static cv::GArrayDesc outMeta(const cv::GArrayDesc &, const cv::GOpaqueDesc &) {
        return cv::empty_array_desc();
    }
};

GAPI_OCV_KERNEL(OCVLocateROI, LocateROI) {
    // This is the place where we can run extra analytics
    // on the input image frame and select the ROI (region
    // of interest) where we want to detect our objects (or
    // run any other inference).
    //
    // Currently it doesn't do anything intelligent,
    // but only crops the input image to square (this is
    // the most convenient aspect ratio for detectors to use)

    static void run(const cv::Size& in_size,
                    cv::Rect &out_rect) {

        // Identify the central point & square size (- some padding)
        const auto center = cv::Point{in_size.width/2, in_size.height/2};
        auto sqside = std::min(in_size.width, in_size.height);

        // Now build the central square ROI
        out_rect = cv::Rect{ center.x - sqside/2
                             , center.y - sqside/2
                             , sqside
                             , sqside
                            };
    }
};

GAPI_OCV_KERNEL(OCVBBoxes, BBoxes) {
    // This kernel converts the rectangles into G-API's
    // rendering primitives
    static void run(const std::vector<cv::Rect> &in_face_rcs,
                    const             cv::Rect  &in_roi,
                          std::vector<cv::gapi::wip::draw::Prim> &out_prims) {
        out_prims.clear();
        const auto cvt = [](const cv::Rect &rc, const cv::Scalar &clr) {
            return cv::gapi::wip::draw::Rect(rc, clr, 2);
        };
        out_prims.emplace_back(cvt(in_roi, CV_RGB(0,255,255))); // cyan
        for (auto &&rc : in_face_rcs) {
            out_prims.emplace_back(cvt(rc, CV_RGB(0,255,0)));   // green
        }
    }
};

GAPI_OCV_KERNEL(OCVParseSSD, ParseSSD) {
    static void run(const cv::Mat &in_ssd_result,
                    const cv::Rect &in_roi,
                    const cv::Size &in_parent_size,
                    std::vector<cv::Rect> &out_objects) {
        const auto &in_ssd_dims = in_ssd_result.size;
        GAPI_Assert(in_ssd_dims.dims() == 4u);

        const int MAX_PROPOSALS = in_ssd_dims[2];
        const int OBJECT_SIZE   = in_ssd_dims[3];
        GAPI_Assert(OBJECT_SIZE  == 7); // fixed SSD object size

        const cv::Size up_roi = in_roi.size();
        const cv::Rect surface({0,0}, in_parent_size);

        out_objects.clear();

        const float *data = in_ssd_result.ptr<float>();
        for (int i = 0; i < MAX_PROPOSALS; i++) {
            const float image_id   = data[i * OBJECT_SIZE + 0];
            const float label      = data[i * OBJECT_SIZE + 1];
            const float confidence = data[i * OBJECT_SIZE + 2];
            const float rc_left    = data[i * OBJECT_SIZE + 3];
            const float rc_top     = data[i * OBJECT_SIZE + 4];
            const float rc_right   = data[i * OBJECT_SIZE + 5];
            const float rc_bottom  = data[i * OBJECT_SIZE + 6];
            (void) label; // unused

            if (image_id < 0.f) {
                break;    // marks end-of-detections
            }
            if (confidence < 0.5f) {
                continue; // skip objects with low confidence
            }

            // map relative coordinates to the original image scale
            // taking the ROI into account
            cv::Rect rc;
            rc.x      = static_cast<int>(rc_left   * up_roi.width);
            rc.y      = static_cast<int>(rc_top    * up_roi.height);
            rc.width  = static_cast<int>(rc_right  * up_roi.width)  - rc.x;
            rc.height = static_cast<int>(rc_bottom * up_roi.height) - rc.y;
            rc.x += in_roi.x;
            rc.y += in_roi.y;
            out_objects.emplace_back(rc & surface);
        }
    }
};

} // namespace custom

namespace cfg {
typename cv::gapi::wip::onevpl::CfgParam create_from_string(const std::string &line);

struct flow {
    flow(bool preproc, bool rctx) :
        vpl_preproc_enable(preproc),
        ie_remote_ctx_enable(rctx) {
    }
    bool vpl_preproc_enable = false;
    bool ie_remote_ctx_enable = false;
};

using support_matrix =
        std::map <std::string/*source_dev_id*/,
                  std::map<std::string/*preproc_device_id*/,
                           std::map <std::string/*rctx device_id*/, std::shared_ptr<flow>>>>;
support_matrix resolved_conf{{
                            {"GPU", {{
                                        {"",    {{ "CPU", std::make_shared<flow>(false, false)},
                                                 { "GPU", {/* unsupported:
                                                           * ie GPU preproc isn't available */}}
                                                }},

                                        {"CPU", {{ "CPU", {/* unsupported: preproc mix */}},
                                                 { "GPU", {/* unsupported: preproc mix */}}
                                                }},
#if defined(HAVE_DIRECTX) && defined(HAVE_D3D11)
                                        {"GPU", {{ "CPU", std::make_shared<flow>(true, false)},
                                                 { "GPU", std::make_shared<flow>(true, true)}}}
#else   // TODO VAAPI under linux doesn't support GPU IE remote context
                                        {"GPU", {{ "CPU", std::make_shared<flow>(true, false)},
                                                 { "GPU", std::make_shared<flow>(true, false)}}}
#endif
                                    }}
                            },
                            {"CPU", {{
                                        {"",    {{ "CPU", std::make_shared<flow>(false, false)},
                                                 { "GPU", std::make_shared<flow>(false, false)}
                                                }},

                                        {"CPU", {{ "CPU", std::make_shared<flow>(true, false)},
                                                 { "GPU", std::make_shared<flow>(true, false)}
                                                }},

                                        {"GPU", {{ "CPU", {/* unsupported: preproc mix */}},
                                                 { "GPU", {/* unsupported: preproc mix */}}}}
                                    }}
                            }
                        }};

static void print_available_cfg(std::ostream &out,
                                const std::string &source_device,
                                const std::string &preproc_device,
                                const std::string &ie_device_id) {
    const std::string source_device_cfg_name("--source_device=");
    const std::string preproc_device_cfg_name("--preproc_device=");
    const std::string ie_cfg_name("--faced=");
    out << "unsupported acceleration param combinations:\n"
                     << source_device_cfg_name << source_device << " "
                     << preproc_device_cfg_name << preproc_device << " "
                     << ie_cfg_name << ie_device_id <<
                     "\n\nSupported matrix:\n\n" << std::endl;
    for (const auto &s_d : cfg::resolved_conf) {
        std::string prefix = source_device_cfg_name + s_d.first;
        for (const auto &p_d : s_d.second) {
            std::string mid_prefix = prefix + +"\t" + preproc_device_cfg_name +
                                    (p_d.first.empty() ? "" : p_d.first);
            for (const auto &i_d : p_d.second) {
                if (i_d.second) {
                    std::cerr << mid_prefix << "\t" << ie_cfg_name <<i_d.first << std::endl;
                }
            }
        }
    }
}
}

int main(int argc, char *argv[]) {

    cv::CommandLineParser cmd(argc, argv, keys);
    cmd.about(about);
    if (cmd.has("help")) {
        cmd.printMessage();
        return 0;
    }

    // get file name
    const auto file_path = cmd.get<std::string>("input");
    const auto output = cmd.get<std::string>("output");
    const auto opt_roi = parse_roi(cmd.get<std::string>("roi"));
    const auto face_model_path = cmd.get<std::string>("facem");
    const auto streaming_queue_capacity = cmd.get<uint32_t>("streaming_queue_capacity");
    const auto source_decode_queue_capacity = cmd.get<uint32_t>("frames_pool_size");
    const auto source_vpp_queue_capacity = cmd.get<uint32_t>("vpp_frames_pool_size");
    const auto device_id = cmd.get<std::string>("faced");
    const auto source_device = cmd.get<std::string>("source_device");
    const auto preproc_device = cmd.get<std::string>("preproc_device");

    // validate support matrix
    std::shared_ptr<cfg::flow> flow_settings = cfg::resolved_conf[source_device][preproc_device][device_id];
    if (!flow_settings) {
        cfg::print_available_cfg(std::cerr, source_device, preproc_device, device_id);
        return -1;
    }

    // check output file extension
    if (!output.empty()) {
        auto ext = output.find_last_of(".");
        if (ext == std::string::npos || (output.substr(ext + 1) != "avi")) {
            std::cerr << "Output file should have *.avi extension for output video" << std::endl;
            return -1;
        }
    }

    // get oneVPL cfg params from cmd
    std::stringstream params_list(cmd.get<std::string>("cfg_params"));
    std::vector<cv::gapi::wip::onevpl::CfgParam> source_cfgs;
    try {
        std::string line;
        while (std::getline(params_list, line, ';')) {
            source_cfgs.push_back(cfg::create_from_string(line));
        }
    } catch (const std::exception& ex) {
        std::cerr << "Invalid cfg parameter: " << ex.what() << std::endl;
        return -1;
    }

    // apply VPL source optimization params
    if (source_decode_queue_capacity != 0) {
        source_cfgs.push_back(cv::gapi::wip::onevpl::CfgParam::create_frames_pool_size(source_decode_queue_capacity));
    }
    if (source_vpp_queue_capacity != 0) {
        source_cfgs.push_back(cv::gapi::wip::onevpl::CfgParam::create_vpp_frames_pool_size(source_vpp_queue_capacity));
    }

    auto face_net = cv::gapi::ie::Params<custom::FaceDetector> {
        face_model_path,                 // path to topology IR
        get_weights_path(face_model_path),   // path to weights
        device_id
    };

    // It is allowed (and highly recommended) to reuse predefined device_ptr & context_ptr objects
    // received from user application. Current sample demonstrate how to deal with this situation.
    //
    // But if you do not need this fine-grained acceleration devices configuration then
    // just use default constructors for onevpl::GSource, IE and preprocessing module.
    // But please pay attention that default pipeline construction in this case will be
    // very inefficient and carries out multiple CPU-GPU memory copies
    //
    // If you want to reach max performance and seize copy-free approach for specific
    // device & context selection then follow the steps below.
    // The situation is complicated a little bit in comparison with default configuration, thus
    // let's focusing this:
    //
    // - all component-participants (Source, Preprocessing, Inference)
    // must share the same device & context instances
    //
    // - you must wrapping your available device & context instancs into thin
    // `cv::gapi::wip::Device` & `cv::gapi::wip::Context`.
    // !!! Please pay attention that both objects are weak wrapper so you must ensure
    // that device & context would be alived before full pipeline created !!!
    //
    // - you should pass such wrappers as constructor arguments for each component in pipeline:
    //      a) use extended constructor for `onevpl::GSource` for activating predefined device & context
    //      b) use `cfgContextParams` method of `cv::gapi::ie::Params` to enable `PreprocesingEngine`
    // for predefined device & context
    //      c) use `InferenceEngine::ParamMap` to activate remote ctx in Inference Engine for given
    // device & context
    //
    //
    //// P.S. the current sample supports heterogenous pipeline construction also.
    //// It is possible to make up mixed device approach.
    //// Please feel free to explore different configurations!

    cv::util::optional<cv::gapi::wip::onevpl::Device> gpu_accel_device;
    cv::util::optional<cv::gapi::wip::onevpl::Context> gpu_accel_ctx;
    cv::gapi::wip::onevpl::Device cpu_accel_device = cv::gapi::wip::onevpl::create_host_device();
    cv::gapi::wip::onevpl::Context cpu_accel_ctx = cv::gapi::wip::onevpl::create_host_context();
    // create GPU device if requested
    if (is_gpu(device_id)
        || is_gpu(source_device)
        || is_gpu(preproc_device)) {
#ifdef HAVE_DIRECTX
#ifdef HAVE_D3D11
        // create DX11 device & context owning handles.
        // wip::Device & wip::Context provide non-owning semantic of resources and act
        // as weak references API wrappers in order to carry type-erased resources type
        // into appropriate modules: onevpl::GSource, PreprocEngine and InferenceEngine
        // Until modules are not created owner handles must stay alive
        auto dx11_dev = createCOMPtrGuard<ID3D11Device>();
        auto dx11_ctx = createCOMPtrGuard<ID3D11DeviceContext>();

        auto adapter_factory = createCOMPtrGuard<IDXGIFactory>();
        {
            IDXGIFactory* out_factory = nullptr;
            HRESULT err = CreateDXGIFactory(__uuidof(IDXGIFactory),
                                        reinterpret_cast<void**>(&out_factory));
            if (FAILED(err)) {
                std::cerr << "Cannot create CreateDXGIFactory, error: " << HRESULT_CODE(err) << std::endl;
                return -1;
            }
            adapter_factory = createCOMPtrGuard(out_factory);
        }

        auto intel_adapter = createCOMPtrGuard<IDXGIAdapter>();
        UINT adapter_index = 0;
        const unsigned int refIntelVendorID = 0x8086;
        IDXGIAdapter* out_adapter = nullptr;

        while (adapter_factory->EnumAdapters(adapter_index, &out_adapter) != DXGI_ERROR_NOT_FOUND) {
            DXGI_ADAPTER_DESC desc{};
            out_adapter->GetDesc(&desc);
            if (desc.VendorId == refIntelVendorID) {
                intel_adapter = createCOMPtrGuard(out_adapter);
                break;
            }
            ++adapter_index;
        }

        if (!intel_adapter) {
            std::cerr << "No Intel GPU adapter on aboard. Exit" << std::endl;
            return -1;
        }

        std::tie(dx11_dev, dx11_ctx) = create_device_with_ctx(intel_adapter.get());
        gpu_accel_device = cv::util::make_optional(
                            cv::gapi::wip::onevpl::create_dx11_device(
                                                        reinterpret_cast<void*>(dx11_dev.release()),
                                                        "GPU"));
        gpu_accel_ctx = cv::util::make_optional(
                            cv::gapi::wip::onevpl::create_dx11_context(
                                                        reinterpret_cast<void*>(dx11_ctx.release())));
#endif // HAVE_D3D11
#endif // HAVE_DIRECTX
#ifdef __linux__
#if defined(HAVE_VA) || defined(HAVE_VA_INTEL)
        static const char *predefined_vaapi_devices_list[] {"/dev/dri/renderD128",
                                                            "/dev/dri/renderD129",
                                                            "/dev/dri/card0",
                                                            "/dev/dri/card1",
                                                            nullptr};
        std::stringstream ss;
        int device_fd = -1;
        VADisplay va_handle = nullptr;
        for (const char **device_path = predefined_vaapi_devices_list;
            *device_path != nullptr; device_path++) {
            device_fd = open(*device_path, O_RDWR);
            if (device_fd < 0) {
                std::string info("Cannot open GPU file: \"");
                info = info + *device_path + "\", error: " + strerror(errno);
                ss << info << std::endl;
                continue;
            }
            va_handle = vaGetDisplayDRM(device_fd);
            if (!va_handle) {
                close(device_fd);
                std::string info("VAAPI device vaGetDisplayDRM failed, error: ");
                info += strerror(errno);
                ss << info << std::endl;
                continue;
            }
            int major_version = 0, minor_version = 0;
            VAStatus status {};
            status = vaInitialize(va_handle, &major_version, &minor_version);
            if (VA_STATUS_SUCCESS != status) {
                close(device_fd);
                va_handle = nullptr;

                std::string info("Cannot initialize VAAPI device, error: ");
                info += vaErrorStr(status);
                ss << info << std::endl;
                continue;
            }
            std::cout << "VAAPI created for device: " << *device_path << ", version: "
                      << major_version << "." << minor_version << std::endl;
            break;
        }

        // check device creation
        if (!va_handle) {
            std::cerr << "Cannot create VAAPI device. Log:\n" << ss.str() << std::endl;
            return -1;
        }
        gpu_accel_device = cv::util::make_optional(
                            cv::gapi::wip::onevpl::create_vaapi_device(reinterpret_cast<void*>(va_handle),
                                                                       "GPU"));
        gpu_accel_ctx = cv::util::make_optional(
                            cv::gapi::wip::onevpl::create_vaapi_context(nullptr));
#endif // defined(HAVE_VA) || defined(HAVE_VA_INTEL)
#endif // #ifdef __linux__
    }

#ifdef HAVE_INF_ENGINE
    // activate remote ctx in Inference Engine for GPU device
    // when other pipeline component use the GPU device too
    if (flow_settings->ie_remote_ctx_enable) {
        InferenceEngine::ParamMap ctx_config({{"CONTEXT_TYPE", "VA_SHARED"},
                                              {"VA_DEVICE", gpu_accel_device.value().get_ptr()} });
        face_net.cfgContextParams(ctx_config);
        std::cout << "enforce InferenceEngine remote context on device: " << device_id << std::endl;

        // NB: consider NV12 surface because it's one of native GPU image format
        face_net.pluginConfig({{"GPU_NV12_TWO_INPUTS", "YES" }});
        std::cout << "enforce InferenceEngine NV12 blob" << std::endl;
    }
#endif // HAVE_INF_ENGINE

    // turn on VPP PreprocesingEngine if available & requested
    if (flow_settings->vpl_preproc_enable) {
        if (is_gpu(preproc_device)) {
            // activate VPP PreprocesingEngine on GPU
            face_net.cfgPreprocessingParams(gpu_accel_device.value(),
                                            gpu_accel_ctx.value());
        } else {
            // activate VPP PreprocesingEngine on CPU
            face_net.cfgPreprocessingParams(cpu_accel_device,
                                            cpu_accel_ctx);
        }
        std::cout << "enforce VPP preprocessing on device: " << preproc_device << std::endl;
    } else {
        std::cout << "use InferenceEngine default preprocessing" << std::endl;
    }

    auto kernels = cv::gapi::kernels
        < custom::OCVLocateROI
        , custom::OCVParseSSD
        , custom::OCVBBoxes>();
    auto networks = cv::gapi::networks(face_net);
    auto face_detection_args = cv::compile_args(networks, kernels);
    if (streaming_queue_capacity != 0) {
        face_detection_args += cv::compile_args(cv::gapi::streaming::queue_capacity{ streaming_queue_capacity });
    }

    // Create source
    cv::gapi::wip::IStreamSource::Ptr cap;
    try {
        if (is_gpu(source_device)) {
            std::cout << "enforce VPL Source deconding on device: " << source_device << std::endl;
            // use special 'Device' constructor for `onevpl::GSource`
            cap = cv::gapi::wip::make_onevpl_src(file_path, source_cfgs,
                                                 gpu_accel_device.value(),
                                                 gpu_accel_ctx.value());
        } else {
            cap = cv::gapi::wip::make_onevpl_src(file_path, source_cfgs);
        }
        std::cout << "oneVPL source description: " << cap->descr_of() << std::endl;
    } catch (const std::exception& ex) {
        std::cerr << "Cannot create source: " << ex.what() << std::endl;
        return -1;
    }

    cv::GMetaArg descr = cap->descr_of();
    auto frame_descr = cv::util::get<cv::GFrameDesc>(descr);
    cv::GOpaque<cv::Rect> in_roi;
    auto inputs = cv::gin(cap);

    // Now build the graph
    cv::GFrame in;
    auto size = cv::gapi::streaming::size(in);
    auto graph_inputs = cv::GIn(in);
    if (!opt_roi.has_value()) {
        // Automatically detect ROI to infer. Make it output parameter
        std::cout << "ROI is not set or invalid. Locating it automatically"
                  << std::endl;
        in_roi = custom::LocateROI::on(size);
    } else {
        // Use the value provided by user
        std::cout << "Will run inference for static region "
                  << opt_roi.value()
                  << " only"
                  << std::endl;
        graph_inputs += cv::GIn(in_roi);
        inputs += cv::gin(opt_roi.value());
    }
    auto blob = cv::gapi::infer<custom::FaceDetector>(in_roi, in);
    cv::GArray<cv::Rect> rcs = custom::ParseSSD::on(blob, in_roi, size);
    auto out_frame = cv::gapi::wip::draw::renderFrame(in, custom::BBoxes::on(rcs, in_roi));
    auto out = cv::gapi::streaming::BGR(out_frame);
    cv::GStreamingCompiled pipeline = cv::GComputation(std::move(graph_inputs), cv::GOut(out))   // and move here
                                        .compileStreaming(std::move(face_detection_args));
    // The execution part
    pipeline.setSource(std::move(inputs));
    pipeline.start();

    size_t frames = 0u;
    cv::TickMeter tm;
    cv::VideoWriter writer;
    if (!output.empty() && !writer.isOpened()) {
        const auto sz = cv::Size{frame_descr.size.width, frame_descr.size.height};
        writer.open(output, cv::VideoWriter::fourcc('M','J','P','G'), 25.0, sz);
        GAPI_Assert(writer.isOpened());
    }

    cv::Mat outMat;
    tm.start();
    while (pipeline.pull(cv::gout(outMat))) {
        cv::imshow("Out", outMat);
        cv::waitKey(1);
        if (!output.empty()) {
            writer << outMat;
        }
        ++frames;
    }
    tm.stop();
    std::cout << "Processed " << frames << " frames" << " (" << frames / tm.getTimeSec() << " FPS)" << std::endl;

    return 0;
}


namespace cfg {
typename cv::gapi::wip::onevpl::CfgParam create_from_string(const std::string &line) {
    using namespace cv::gapi::wip;

    if (line.empty()) {
        throw std::runtime_error("Cannot parse CfgParam from emply line");
    }

    std::string::size_type name_endline_pos = line.find(':');
    if (name_endline_pos == std::string::npos) {
        throw std::runtime_error("Cannot parse CfgParam from: " + line +
                                 "\nExpected separator \":\"");
    }

    std::string name = line.substr(0, name_endline_pos);
    std::string value = line.substr(name_endline_pos + 1);

    return cv::gapi::wip::onevpl::CfgParam::create(name, value,
                                                   /* vpp params strongly optional */
                                                   name.find("vpp.") == std::string::npos);
}
}
