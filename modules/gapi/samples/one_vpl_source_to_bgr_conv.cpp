#include <algorithm>
#include <fstream>
#include <iostream>
#include <cctype>
#include <tuple>

#include <opencv2/imgproc.hpp>
#include <opencv2/gapi.hpp>
#include <opencv2/gapi/core.hpp>
#include <opencv2/gapi/cpu/gcpukernel.hpp>
#include <opencv2/gapi/gpu/ggpukernel.hpp>
#include <opencv2/gapi/render.hpp>
#include <opencv2/gapi/streaming/onevpl/source.hpp>
#include <opencv2/gapi/streaming/onevpl/data_provider_interface.hpp>
#include <opencv2/highgui.hpp> // CommandLineParser
#include <opencv2/gapi/infer/parsers.hpp>
#include <opencv2/gapi/ocl/core.hpp>

#ifdef HAVE_DIRECTX
#ifdef HAVE_D3D11
#pragma comment(lib,"d3d11.lib")

// get rid of generate macro max/min/etc from DX side
#define D3D11_NO_HELPERS
#define NOMINMAX
#include <d3d11.h>
#pragma comment(lib, "dxgi")
#undef NOMINMAX
#undef D3D11_NO_HELPERS

#endif // HAVE_D3D11
#endif // HAVE_DIRECTX

namespace cfg {
typename cv::gapi::wip::onevpl::CfgParam create_from_string(const std::string &line);
}

namespace {
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

#ifdef HAVE_DIRECTX
#ifdef HAVE_D3D11
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
}

const std::string about =
    "This is an example that covers case when a MediaFrame is received from GPU using VPL Source and converted to BGR UMat";
const std::string keys =
    "{ h help                       |                                           | Print this help message }"
    "{ input                        |                                           | Path to the input video file }"
    "{ output                       |                                           | Path to the output RAW video file. Use .avi extension }";

int main(int argc, char *argv[]) {
    cv::CommandLineParser cmd(argc, argv, keys);
    cmd.about(about);
    if (cmd.has("help")) {
        cmd.printMessage();
        return 0;
    }
#ifdef HAVE_DIRECTX
#ifdef HAVE_D3D11
    // Get file name
    const auto file_path = "C:\work\opencv_extra\testdata\cv\video\768x576.avi";// cmd.get<std::string>("input");
    const auto output = cmd.get<std::string>("output");

    // check ouput file extension
    if (!output.empty()) {
        auto ext = output.find_last_of(".");
        if (ext == std::string::npos || (output.substr(ext + 1) != "avi")) {
            std::cerr << "Output file should have *.avi extension for output video" << std::endl;
            return -1;
        }
    }

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
    auto dx11_dev = createCOMPtrGuard<ID3D11Device>();
    auto dx11_ctx = createCOMPtrGuard<ID3D11DeviceContext>();
    std::tie(dx11_dev, dx11_ctx) = create_device_with_ctx(intel_adapter.get());
    void* accel_device_ptr = nullptr;
    void* accel_ctx_ptr = nullptr;
    accel_device_ptr = reinterpret_cast<void*>(dx11_dev.get());
    accel_ctx_ptr = reinterpret_cast<void*>(dx11_ctx.get());

    cv::Ptr<cv::gapi::wip::IStreamSource> cap;
    std::vector<cv::gapi::wip::onevpl::CfgParam> source_cfgs;
    source_cfgs.push_back(cfg::create_from_string(
                                        "mfxImplDescription.AccelerationMode"
                                        ":"
                                        "MFX_ACCEL_MODE_VIA_D3D11"));
    cap = cv::gapi::wip::make_onevpl_src(file_path, source_cfgs,
                                         "GPU",
                                         accel_device_ptr,
                                         accel_ctx_ptr);

    // Now build the graph
    cv::GFrame in; // input frame from VPL source
    auto bgr_gmat = cv::gapi::streaming::BGR(in); // conversion from VPL source frame to BGR UMat
    auto out = cv::gapi::blur(bgr_gmat, cv::Size(4,4)); // ocl version of blur kernel

    cv::GStreamingCompiled pipeline = cv::GComputation(cv::GIn(in), cv::GOut(out))
        .compileStreaming(std::move(cv::compile_args(cv::gapi::core::ocl::kernels())));
    pipeline.setSource(std::move(cap));

    // The execution part
    size_t frames = 0u;
    cv::TickMeter tm;
    cv::VideoWriter writer;
    cv::GMetaArg descr = cap->descr_of();
    auto frame_descr = cv::util::get<cv::GFrameDesc>(descr);
    if (!output.empty() && !writer.isOpened()) {
        const auto sz = cv::Size{frame_descr.size.width, frame_descr.size.height};
        writer.open(output, cv::VideoWriter::fourcc('M','J','P','G'), 25.0, sz);
        GAPI_Assert(writer.isOpened());
    }

    pipeline.start();
    tm.start();
    cv::Mat outMat;
    while (pipeline.pull(cv::gout(outMat))) {
        cv::imshow("OutVideo", outMat);
        if (!output.empty()) {
            writer << outMat;
        }
        cv::waitKey(1);
        ++frames;
    }
    tm.stop();
    std::cout << "Processed " << frames << " frames" << " (" << frames / tm.getTimeSec() << " FPS)" << std::endl;
#else
    GAPI_Assert(false && "Assembled without DX11 support");
#endif // HAVE_DIRECTX
#endif // HAVE_D3D11
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
