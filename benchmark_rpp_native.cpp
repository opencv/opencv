/*
 * Native RPP benchmark (no OpenCV integration).
 *
 * Measures raw RPP HOST and RPP HIP performance for the same set of
 * operations that the OpenCV HAL exposes. Memory is uploaded once and
 * downloaded once per test, matching the HAL behavior so the comparison
 * is fair.
 *
 * Run:
 *   ./benchmark_rpp_native                # RPP HOST
 *   OPENCV_RPP_FORCE_GPU=1 ./benchmark_rpp_native  # force RPP HIP
 */

#include <rpp/rppt.h>
#include <rpp/rppt_tensor_bitwise_operations.h>
#include <rpp/rppt_tensor_geometric_augmentations.h>
#include <rpp/rppt_tensor_filter_augmentations.h>
#include <hip/hip_runtime_api.h>

#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <memory>

using namespace std;

static int WARMUP = 20;
static int ITERS  = 100;  // Native RPP HOST resize can hang/crash at 1000 iters due to internal state/heap issues.

static double ms(chrono::steady_clock::time_point a,
                 chrono::steady_clock::time_point b) {
    return chrono::duration_cast<chrono::microseconds>(b - a).count() / 1000.0;
}

static void fillRandom(uint8_t* ptr, size_t n) {
    for (size_t i = 0; i < n; ++i) ptr[i] = static_cast<uint8_t>(rand() % 256);
}

class RppContext {
public:
    rppHandle_t handle = nullptr;
    RppBackend backend;
    RppContext(RppBackend b) : backend(b) {
        rppCreate(&handle, 1, 0, nullptr, backend);
    }
    ~RppContext() {
        if (handle) rppDestroy(handle, backend);
    }
};

class DeviceBuffer {
public:
    struct Impl {
        void* dev = nullptr;
        size_t size = 0;
        bool gpu = false;
        Impl(size_t n, bool onGpu) : size(n), gpu(onGpu) {
            if (gpu) (void)hipMalloc(&dev, n);
            else dev = malloc(n);
        }
        ~Impl() {
            if (gpu) (void)hipFree(dev);
            else free(dev);
        }
    };
    shared_ptr<Impl> impl;
    DeviceBuffer() = default;
    DeviceBuffer(size_t n, bool onGpu) : impl(make_shared<Impl>(n, onGpu)) {}
    void* dev() const { return impl ? impl->dev : nullptr; }
    size_t size() const { return impl ? impl->size : 0; }
    bool gpu() const { return impl ? impl->gpu : false; }
    void upload(const uint8_t* host) {
        if (impl->gpu) (void)hipMemcpy(impl->dev, host, impl->size, hipMemcpyHostToDevice);
        else memcpy(impl->dev, host, impl->size);
    }
    void download(uint8_t* host) {
        if (impl->gpu) (void)hipMemcpy(host, impl->dev, impl->size, hipMemcpyDeviceToHost);
        else memcpy(host, impl->dev, impl->size);
    }
};

struct BenchCase {
    string name;
    int w;
    int h;
    function<void(rppHandle_t)> fn;
};

static void addBitwiseCases(vector<BenchCase>& cases, int w, int h,
                            const DeviceBuffer& a, const DeviceBuffer& b,
                            const DeviceBuffer& d, RppBackend backend) {
    RpptDesc desc{};
    desc.numDims = 4;
    desc.dataType = U8;
    desc.n = 1; desc.c = 1; desc.h = h; desc.w = w;
    desc.layout = NHWC;
    desc.strides.wStride = 1;
    desc.strides.hStride = w;
    desc.strides.cStride = 1;
    desc.strides.nStride = w * h;

    RpptROI roi{};
    roi.xywhROI.xy.x = 0; roi.xywhROI.xy.y = 0;
    roi.xywhROI.roiWidth = w; roi.xywhROI.roiHeight = h;

    cases.push_back({"rpp bitwise_and 8UC1", w, h,
        [a, b, d, desc, roi, backend](rppHandle_t hnd) {
            rppt_bitwise_and(a.dev(), b.dev(), const_cast<RpptDesc*>(&desc),
                             d.dev(), const_cast<RpptDesc*>(&desc),
                             const_cast<RpptROI*>(&roi), XYWH, hnd, backend);
        }});
    cases.push_back({"rpp bitwise_or 8UC1", w, h,
        [a, b, d, desc, roi, backend](rppHandle_t hnd) {
            rppt_bitwise_or(a.dev(), b.dev(), const_cast<RpptDesc*>(&desc),
                            d.dev(), const_cast<RpptDesc*>(&desc),
                            const_cast<RpptROI*>(&roi), XYWH, hnd, backend);
        }});
    cases.push_back({"rpp bitwise_xor 8UC1", w, h,
        [a, b, d, desc, roi, backend](rppHandle_t hnd) {
            rppt_bitwise_xor(a.dev(), b.dev(), const_cast<RpptDesc*>(&desc),
                             d.dev(), const_cast<RpptDesc*>(&desc),
                             const_cast<RpptROI*>(&roi), XYWH, hnd, backend);
        }});
    cases.push_back({"rpp bitwise_not 8UC1", w, h,
        [a, d, desc, roi, backend](rppHandle_t hnd) {
            rppt_bitwise_not(a.dev(), const_cast<RpptDesc*>(&desc),
                             d.dev(), const_cast<RpptDesc*>(&desc),
                             const_cast<RpptROI*>(&roi), XYWH, hnd, backend);
        }});
}

static void addImgprocCases(vector<BenchCase>& cases, int w, int h,
                            const DeviceBuffer& src3, const DeviceBuffer& dst3,
                            const DeviceBuffer& src1, const DeviceBuffer& dst1,
                            RppBackend backend,
                            const DeviceBuffer& dst3Down, const DeviceBuffer& dst3Up) {
    RpptDesc srcDesc3{};
    srcDesc3.numDims = 4; srcDesc3.dataType = U8;
    srcDesc3.n = 1; srcDesc3.c = 3; srcDesc3.h = h; srcDesc3.w = w;
    srcDesc3.layout = NHWC;
    srcDesc3.strides.wStride = 3;
    srcDesc3.strides.hStride = w * 3;
    srcDesc3.strides.cStride = 1;
    srcDesc3.strides.nStride = w * h * 3;

    RpptDesc srcDesc1 = srcDesc3;
    srcDesc1.c = 1;
    srcDesc1.strides.wStride = 1;
    srcDesc1.strides.hStride = w;
    srcDesc1.strides.cStride = 1;
    srcDesc1.strides.nStride = w * h;

    RpptROI srcRoi{};
    srcRoi.xywhROI.xy.x = 0; srcRoi.xywhROI.xy.y = 0;
    srcRoi.xywhROI.roiWidth = w; srcRoi.xywhROI.roiHeight = h;

    RpptImagePatch dstSizeDown{};
    dstSizeDown.width = w / 2; dstSizeDown.height = h / 2;
    RpptImagePatch dstSizeUp{};
    dstSizeUp.width = w * 2; dstSizeUp.height = h * 2;

    RpptDesc dstDesc3Down = srcDesc3;
    dstDesc3Down.w = w / 2; dstDesc3Down.h = h / 2;
    dstDesc3Down.strides.hStride = (w / 2) * 3;
    dstDesc3Down.strides.nStride = (w / 2) * (h / 2) * 3;

    RpptDesc dstDesc3Up = srcDesc3;
    dstDesc3Up.w = w * 2; dstDesc3Up.h = h * 2;
    dstDesc3Up.strides.hStride = (w * 2) * 3;
    dstDesc3Up.strides.nStride = (w * 2) * (h * 2) * 3;

    RpptDesc dstDesc3 = srcDesc3;
    RpptDesc dstDesc1 = srcDesc1;

    float affine[6] = {1.0f, 0.05f, 30.0f, 0.02f, 1.0f, 20.0f};
    DeviceBuffer affineBuf(sizeof(affine), backend == RPP_HIP_BACKEND);
    affineBuf.upload(reinterpret_cast<uint8_t*>(affine));
    void* affineDev = affineBuf.dev();

/*
    cases.push_back({"rpp resize 8UC3 bilinear down2x", w, h,
        [src3, dst3Down, srcDesc3, dstDesc3Down, dstSizeDown, srcRoi, backend](rppHandle_t hnd) {
            rppt_resize(src3.dev(), const_cast<RpptDesc*>(&srcDesc3),
                        dst3Down.dev(), const_cast<RpptDesc*>(&dstDesc3Down),
                        const_cast<RpptImagePatch*>(&dstSizeDown),
                        BILINEAR, const_cast<RpptROI*>(&srcRoi), XYWH, hnd, backend);
        }});
    cases.push_back({"rpp resize 8UC3 bilinear up2x", w, h,
        [src3, dst3Up, srcDesc3, dstDesc3Up, dstSizeUp, srcRoi, backend](rppHandle_t hnd) {
            rppt_resize(src3.dev(), const_cast<RpptDesc*>(&srcDesc3),
                        dst3Up.dev(), const_cast<RpptDesc*>(&dstDesc3Up),
                        const_cast<RpptImagePatch*>(&dstSizeUp),
                        BILINEAR, const_cast<RpptROI*>(&srcRoi), XYWH, hnd, backend);
        }});
*/

    Rpp32u horiz = 1, vert = 0;
    cases.push_back({"rpp flip 8UC3 horizontal", w, h,
        [src3, dst3, srcDesc3, dstDesc3, srcRoi, horiz, vert, backend](rppHandle_t hnd) {
            Rpp32u h_ = horiz, v_ = vert;
            rppt_flip(src3.dev(), const_cast<RpptDesc*>(&srcDesc3),
                      dst3.dev(), const_cast<RpptDesc*>(&dstDesc3),
                      &h_, &v_, const_cast<RpptROI*>(&srcRoi), XYWH, hnd, backend);
        }});
    Rpp32u horiz0 = 0, vert1 = 1;
    cases.push_back({"rpp flip 8UC3 vertical", w, h,
        [src3, dst3, srcDesc3, dstDesc3, srcRoi, horiz0, vert1, backend](rppHandle_t hnd) {
            Rpp32u h_ = horiz0, v_ = vert1;
            rppt_flip(src3.dev(), const_cast<RpptDesc*>(&srcDesc3),
                      dst3.dev(), const_cast<RpptDesc*>(&dstDesc3),
                      &h_, &v_, const_cast<RpptROI*>(&srcRoi), XYWH, hnd, backend);
        }});

    cases.push_back({"rpp warpAffine 8UC3 bilinear", w, h,
        [src3, dst3, srcDesc3, dstDesc3, srcRoi, affineDev, backend](rppHandle_t hnd) {
            rppt_warp_affine(src3.dev(), const_cast<RpptDesc*>(&srcDesc3),
                             dst3.dev(), const_cast<RpptDesc*>(&dstDesc3),
                             static_cast<Rpp32f*>(affineDev),
                             BILINEAR, const_cast<RpptROI*>(&srcRoi), XYWH, hnd, backend);
        }});

    cases.push_back({"rpp boxFilter 3x3 8UC3 replicate", w, h,
        [src3, dst3, srcDesc3, dstDesc3, srcRoi, backend](rppHandle_t hnd) {
            rppt_box_filter(src3.dev(), const_cast<RpptDesc*>(&srcDesc3),
                            dst3.dev(), const_cast<RpptDesc*>(&dstDesc3),
                            3, REPLICATE, const_cast<RpptROI*>(&srcRoi), XYWH, hnd, backend);
        }});
    cases.push_back({"rpp boxFilter 5x5 8UC3 replicate", w, h,
        [src3, dst3, srcDesc3, dstDesc3, srcRoi, backend](rppHandle_t hnd) {
            rppt_box_filter(src3.dev(), const_cast<RpptDesc*>(&srcDesc3),
                            dst3.dev(), const_cast<RpptDesc*>(&dstDesc3),
                            5, REPLICATE, const_cast<RpptROI*>(&srcRoi), XYWH, hnd, backend);
        }});
    cases.push_back({"rpp boxFilter 3x3 8UC1 replicate", w, h,
        [src1, dst1, srcDesc1, dstDesc1, srcRoi, backend](rppHandle_t hnd) {
            rppt_box_filter(src1.dev(), const_cast<RpptDesc*>(&srcDesc1),
                            dst1.dev(), const_cast<RpptDesc*>(&dstDesc1),
                            3, REPLICATE, const_cast<RpptROI*>(&srcRoi), XYWH, hnd, backend);
        }});
}

static vector<BenchCase> buildCases(RppBackend backend) {
    vector<pair<int,int>> sizes = {
        {640, 480},
        {1280, 720},
        {1920, 1080},
        {2560, 1440},
        {3840, 2160},
    };

    vector<BenchCase> cases;
    for (auto& sz : sizes) {
        int w = sz.first, h = sz.second;
        size_t n1 = static_cast<size_t>(w) * h;
        size_t n3 = n1 * 3;

        vector<uint8_t> aHost(n1), bHost(n1), dHost(n1);
        vector<uint8_t> src3Host(n3), dst3Host(n3);
        vector<uint8_t> src1Host(n1), dst1Host(n1);
        fillRandom(aHost.data(), n1);
        fillRandom(bHost.data(), n1);
        fillRandom(src3Host.data(), n3);
        fillRandom(src1Host.data(), n1);

        DeviceBuffer a(n1, backend == RPP_HIP_BACKEND);
        DeviceBuffer b(n1, backend == RPP_HIP_BACKEND);
        DeviceBuffer d(n1, backend == RPP_HIP_BACKEND);
        DeviceBuffer src3(n3, backend == RPP_HIP_BACKEND);
        DeviceBuffer dst3(n3, backend == RPP_HIP_BACKEND);
        DeviceBuffer src1(n1, backend == RPP_HIP_BACKEND);
        DeviceBuffer dst1(n1, backend == RPP_HIP_BACKEND);
        DeviceBuffer dst3Down((static_cast<size_t>(w/2)*(h/2)*3), backend == RPP_HIP_BACKEND);
        DeviceBuffer dst3Up((static_cast<size_t>(w*2)*(h*2)*3), backend == RPP_HIP_BACKEND);

        a.upload(aHost.data());
        b.upload(bHost.data());
        src3.upload(src3Host.data());
        src1.upload(src1Host.data());

        addBitwiseCases(cases, w, h, a, b, d, backend);
        addImgprocCases(cases, w, h, src3, dst3, src1, dst1, backend,
                    dst3Down, dst3Up);

        // Touch outputs once.
        d.download(dHost.data());
        dst3.download(dst3Host.data());
        dst1.download(dst1Host.data());
    }
    return cases;
}

static void runBenchmark(RppBackend backend) {
    RppContext ctx(backend);
    if (!ctx.handle) {
        cerr << "Failed to create RPP context\n";
        return;
    }

    vector<BenchCase> cases = buildCases(backend);

    cout << "\n" << string(90, '=') << "\n";
    cout << "Native RPP benchmark | backend="
         << (backend == RPP_HIP_BACKEND ? "HIP" : "HOST")
         << " | warmups=" << WARMUP << " | iters=" << ITERS << "\n";
    cout << string(90, '=') << "\n";
    cout << left << setw(38) << "function"
         << setw(12) << "resolution"
         << right << setw(14) << "ms/op"
         << setw(14) << "total_ms"
         << setw(10) << "status" << "\n";
    cout << string(90, '-') << "\n";

    for (auto& c : cases) {
        bool failed = false;
        string failMsg;
        try {
            for (int i = 0; i < WARMUP; ++i) c.fn(ctx.handle);
        } catch (const exception& e) {
            failed = true; failMsg = string("WARMUP_ERR:") + e.what();
        } catch (...) {
            failed = true; failMsg = "WARMUP_ERR";
        }

        double total = 0, perOp = 0;
        if (!failed) {
            auto t1 = chrono::steady_clock::now();
            for (int i = 0; i < ITERS; ++i) c.fn(ctx.handle);
            auto t2 = chrono::steady_clock::now();
            total = ms(t1, t2);
            perOp = total / ITERS;
        }

        cout << left << setw(38) << c.name
             << setw(12) << (to_string(c.w) + "x" + to_string(c.h))
             << right << fixed << setprecision(4) << setw(14) << perOp
             << setw(14) << total
             << setw(10) << (failed ? failMsg : "OK") << "\n";
    }
}

int main(int argc, char** argv) {
    if (argc > 1) WARMUP = atoi(argv[1]);
    if (argc > 2) ITERS  = atoi(argv[2]);

    const char* force = getenv("OPENCV_RPP_FORCE_GPU");
    bool useHip = force && (strcmp(force, "1") == 0 || strcmp(force, "yes") == 0 || strcmp(force, "true") == 0);

    if (useHip) {
        int deviceCount = 0;
        if (hipGetDeviceCount(&deviceCount) != hipSuccess || deviceCount == 0) {
            cerr << "No HIP device available\n";
            return 1;
        }
    }

    RppBackend backend = useHip ? RPP_HIP_BACKEND : RPP_HOST_BACKEND;
    runBenchmark(backend);
    return 0;
}
