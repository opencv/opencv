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
#include <opencv2/gapi/streaming/onevpl/data_provider_interface.hpp>
#include <opencv2/highgui.hpp> // CommandLineParser
#include <opencv2/gapi/infer/parsers.hpp>

#ifdef HAVE_INF_ENGINE
#include <inference_engine.hpp> // ParamMap

#ifdef HAVE_DIRECTX
#ifdef HAVE_D3D11
#pragma comment(lib,"d3d11.lib")

// get rid of generate macro max/min/etc from DX side
#define D3D11_NO_HELPERS
#define NOMINMAX
#include <cldnn/cldnn_config.hpp>
#include <d3d11.h>
#pragma comment(lib, "dxgi")
#undef NOMINMAX
#undef D3D11_NO_HELPERS

#endif // HAVE_D3D11
#endif // HAVE_DIRECTX
#endif // HAVE_INF_ENGINE

const std::string about =
    "This is an OpenCV-based version of oneVPLSource decoder example";
const std::string keys =
    "{ h help                       |                                           | Print this help message }"
    "{ input                        |                                           | Path to the input demultiplexed video file }"
    "{ output                       |                                           | Path to the output RAW video file. Use .avi extension }"
    "{ facem                        | face-detection-adas-0001.xml              | Path to OpenVINO IE face detection model (.xml) }"
    "{ faced                        | AUTO                                      | Target device for face detection model (e.g. AUTO, GPU, VPU, ...) }"
    "{ cfg_params                   | <prop name>:<value>;<prop name>:<value>   | Semicolon separated list of oneVPL mfxVariants which is used for configuring source (see `MFXSetConfigFilterProperty` by https://spec.oneapi.io/versions/latest/elements/oneVPL/source/index.html) }"
    "{ streaming_queue_capacity     | 1                                         | Streaming executor queue capacity. Calculated automaticaly if 0 }"
    "{ frames_pool_size             | 0                                         | OneVPL source applies this parameter as preallocated frames pool size}"
    "{ vpp_frames_pool_size         | 0                                         | OneVPL source applies this parameter as preallocated frames pool size for VPP preprocessing results}"
    "{ source_preproc_enable        | 0                                         | Turn on OneVPL source frame preprocessing using network input description instead of IE plugin preprocessing}";

namespace {
bool is_gpu(const std::string &device_name) {
    return device_name.find("GPU") != std::string::npos;
}

std::string get_weights_path(const std::string &model_path) {
    const auto EXT_LEN = 4u;
    const auto sz = model_path.size();
    CV_Assert(sz > EXT_LEN);

    auto ext = model_path.substr(sz - EXT_LEN);
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c){
            return static_cast<unsigned char>(std::tolower(c));
        });
    CV_Assert(ext == ".xml");
    return model_path.substr(0u, sz - EXT_LEN) + ".bin";
}

#ifdef HAVE_INF_ENGINE
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
#endif // HAVE_INF_ENGINE
} // anonymous namespace

namespace custom {
G_API_NET(FaceDetector,   <cv::GMat(cv::GMat)>, "face-detector");

using GDetections = cv::GArray<cv::Rect>;
using GRect       = cv::GOpaque<cv::Rect>;
using GSize       = cv::GOpaque<cv::Size>;
using GPrims      = cv::GArray<cv::gapi::wip::draw::Prim>;

G_API_OP(LocateROI, <GRect(GSize, std::reference_wrapper<const std::string>)>, "sample.custom.locate-roi") {
    static cv::GOpaqueDesc outMeta(const cv::GOpaqueDesc &,
                                   std::reference_wrapper<const std::string>) {
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
                    std::reference_wrapper<const std::string> device_id_ref,
                    cv::Rect &out_rect) {

        // Identify the central point & square size (- some padding)
        // NB: GPU plugin in InferenceEngine doesn't support ROI at now
        if (!is_gpu(device_id_ref.get())) {
            const auto center = cv::Point{in_size.width/2, in_size.height/2};
            auto sqside = std::min(in_size.width, in_size.height);

            // Now build the central square ROI
            out_rect = cv::Rect{ center.x - sqside/2
                                , center.y - sqside/2
                                , sqside
                                , sqside
                                };
        } else {
            // use whole frame for GPU device
            out_rect = cv::Rect{ 0
                                , 0
                                , in_size.width
                                , in_size.height
                                };
        }
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

} // namespace custom

namespace cfg {
typename cv::gapi::wip::onevpl::CfgParam create_from_string(const std::string &line);
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
    const auto face_model_path = cmd.get<std::string>("facem");
    const auto streaming_queue_capacity = cmd.get<uint32_t>("streaming_queue_capacity");
    const auto source_decode_queue_capacity = cmd.get<uint32_t>("frames_pool_size");
    const auto source_vpp_queue_capacity = cmd.get<uint32_t>("vpp_frames_pool_size");
    const auto vpl_source_preproc_enable = cmd.get<uint32_t>("source_preproc_enable");
    const auto device_id = cmd.get<std::string>("faced");

    // check ouput file extension
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
            if (vpl_source_preproc_enable == 0) {
                if (line.find("vpp.") != std::string::npos) {
                    // skip VPP preprocessing primitives if not requested
                    continue;
                }
            }
            source_cfgs.push_back(cfg::create_from_string(line));
        }
    } catch (const std::exception& ex) {
        std::cerr << "Invalid cfg parameter: " << ex.what() << std::endl;
        return -1;
    }

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

    // Create device_ptr & context_ptr using graphic API
    // InferenceEngine requires such device & context to create its own
    // remote shared context through InferenceEngine::ParamMap in
    // GAPI InferenceEngine backend to provide interoperability with onevpl::GSource
    // So GAPI InferenceEngine backend and onevpl::GSource MUST share the same
    // device and context
    void* accel_device_ptr = nullptr;
    void* accel_ctx_ptr = nullptr;

#ifdef HAVE_INF_ENGINE
#ifdef HAVE_DIRECTX
#ifdef HAVE_D3D11
    auto dx11_dev = createCOMPtrGuard<ID3D11Device>();
    auto dx11_ctx = createCOMPtrGuard<ID3D11DeviceContext>();

    if (is_gpu(device_id)) {
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
        accel_device_ptr = reinterpret_cast<void*>(dx11_dev.get());
        accel_ctx_ptr = reinterpret_cast<void*>(dx11_ctx.get());

        // put accel type description for VPL source
        source_cfgs.push_back(cfg::create_from_string(
                                        "mfxImplDescription.AccelerationMode"
                                        ":"
                                        "MFX_ACCEL_MODE_VIA_D3D11"));
    }

#endif // HAVE_D3D11
#endif // HAVE_DIRECTX
    // set ctx_config for GPU device only - no need in case of CPU device type
    if (is_gpu(device_id)) {
        InferenceEngine::ParamMap ctx_config({{"CONTEXT_TYPE", "VA_SHARED"},
                                            {"VA_DEVICE", accel_device_ptr} });

        face_net.cfgContextParams(ctx_config);
        face_net.pluginConfig({{"GPU_NV12_TWO_INPUTS", "YES" }});

        std::cout <<"/*******************************************************/\n"
                    "ATTENTION: GPU Inference Engine preprocessing is not vital as expected!"
                     " Please consider param \"source_preproc_enable=1\" and specify "
                     " appropriated media frame transformation using oneVPL::VPP primitives"
                     " which force onevpl::GSource to produce tranformed media frames."
                     " For exploring list of supported transformations please find out "
                     " vpp_* related stuff in"
                     " gapi/include/opencv2/gapi/streaming/onevpl/cfg_params.hpp"
                     " Pay attention that to obtain expected result In this case VPP "
                     " transformation must match network input params."
                     " Please vote/create issue about exporting network params using GAPI\n"
                     "/******************************************************/" << std::endl;
    }
#endif // HAVE_INF_ENGINE

    auto kernels = cv::gapi::kernels
        < custom::OCVLocateROI
        , custom::OCVBBoxes>();
    auto networks = cv::gapi::networks(face_net);
    auto face_detection_args = cv::compile_args(networks, kernels);
    if (streaming_queue_capacity != 0) {
        face_detection_args += cv::compile_args(cv::gapi::streaming::queue_capacity{ streaming_queue_capacity });
    }

    // Create source
    cv::Ptr<cv::gapi::wip::IStreamSource> cap;
    try {
        if (is_gpu(device_id)) {
            cap = cv::gapi::wip::make_onevpl_src(file_path, source_cfgs,
                                                 device_id,
                                                 accel_device_ptr,
                                                 accel_ctx_ptr);
        } else {
            cap = cv::gapi::wip::make_onevpl_src(file_path, source_cfgs);
        }
        std::cout << "oneVPL source desription: " << cap->descr_of() << std::endl;
    } catch (const std::exception& ex) {
        std::cerr << "Cannot create source: " << ex.what() << std::endl;
        return -1;
    }

    cv::GMetaArg descr = cap->descr_of();
    auto frame_descr = cv::util::get<cv::GFrameDesc>(descr);

    // Now build the graph
    cv::GFrame in;
    auto size = cv::gapi::streaming::size(in);
    auto roi = custom::LocateROI::on(size, std::cref(device_id));
    auto blob = cv::gapi::infer<custom::FaceDetector>(roi, in);
    cv::GArray<cv::Rect> rcs = cv::gapi::parseSSD(blob, size, 0.5f, true, true);
    auto out_frame = cv::gapi::wip::draw::renderFrame(in, custom::BBoxes::on(rcs, roi));
    auto out = cv::gapi::streaming::BGR(out_frame);

    cv::GStreamingCompiled pipeline;
    try {
        pipeline = cv::GComputation(cv::GIn(in), cv::GOut(out))
                .compileStreaming(std::move(face_detection_args));
    } catch (const std::exception& ex) {
        std::cerr << "Exception occured during pipeline construction: " << ex.what() << std::endl;
        return -1;
    }
    // The execution part

    // TODO USE may set pool size from outside and set queue_capacity size,
    // compile arg: cv::gapi::streaming::queue_capacity
    pipeline.setSource(std::move(cap));
    pipeline.start();

    size_t frames = 0u;
    cv::TickMeter tm;
    cv::VideoWriter writer;
    if (!output.empty() && !writer.isOpened()) {
        const auto sz = cv::Size{frame_descr.size.width, frame_descr.size.height};
        writer.open(output, cv::VideoWriter::fourcc('M','J','P','G'), 25.0, sz);
        CV_Assert(writer.isOpened());
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
