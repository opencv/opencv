#include <algorithm>
#include <iostream>
#include <cctype>

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/gapi.hpp>
#include <opencv2/gapi/core.hpp>
#include <opencv2/gapi/imgproc.hpp>
#include <opencv2/gapi/infer.hpp>
#include <opencv2/gapi/render.hpp>
#include <opencv2/gapi/infer/ie.hpp>
#include <opencv2/gapi/cpu/gcpukernel.hpp>
#include <opencv2/gapi/streaming/cap.hpp>
#include <opencv2/highgui.hpp>

const std::string about =
    "This is an OpenCV-based version of Privacy Masking Camera example";
const std::string keys =
    "{ h help |                                                  | Print this help message }"
    "{ input  |                                                  | Path to the input video file }"
    "{ platm  | vehicle-license-plate-detection-barrier-0106.xml | Path to OpenVINO IE vehicle/plate detection model (.xml) }"
    "{ platd  | CPU                                              | Target device for vehicle/plate detection model (e.g. CPU, GPU, VPU, ...) }"
    "{ facem  | face-detection-retail-0005.xml                   | Path to OpenVINO IE face detection model (.xml) }"
    "{ faced  | CPU                                              | Target device for face detection model (e.g. CPU, GPU, VPU, ...) }"
    "{ trad   | false                                            | Run processing in a traditional (non-pipelined) way }"
    "{ noshow | false                                            | Don't display UI (improves performance) }";

namespace {

std::string weights_path(const std::string &model_path) {
    const auto EXT_LEN = 4u;
    const auto sz = model_path.size();
    CV_Assert(sz > EXT_LEN);

    auto ext = model_path.substr(sz - EXT_LEN);

    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c){ return static_cast<unsigned char>(std::tolower(c)); });
    CV_Assert(ext == ".xml");

    return model_path.substr(0u, sz - EXT_LEN) + ".bin";
}
} // namespace

namespace custom {

G_API_NET(VehLicDetector, <cv::GMat(cv::GMat)>, "vehicle-license-plate-detector");
G_API_NET(FaceDetector,   <cv::GMat(cv::GMat)>,                  "face-detector");

using GDetections = cv::GArray<cv::Rect>;

G_API_OP(ParseSSD, <GDetections(cv::GMat, cv::GMat, int)>, "custom.privacy_masking.postproc") {
    static cv::GArrayDesc outMeta(const cv::GMatDesc &, const cv::GMatDesc &, int) {
        return cv::empty_array_desc();
    }
};

using GPrims = cv::GArray<cv::gapi::wip::draw::Prim>;

G_API_OP(ToMosaic, <GPrims(GDetections, GDetections)>, "custom.privacy_masking.to_mosaic") {
    static cv::GArrayDesc outMeta(const cv::GArrayDesc &, const cv::GArrayDesc &) {
        return cv::empty_array_desc();
    }
};

GAPI_OCV_KERNEL(OCVParseSSD, ParseSSD) {
    static void run(const cv::Mat &in_ssd_result,
                    const cv::Mat &in_frame,
                    const int      filter_label,
                    std::vector<cv::Rect> &out_objects) {
        const auto &in_ssd_dims = in_ssd_result.size;
        CV_Assert(in_ssd_dims.dims() == 4u);

        const int MAX_PROPOSALS = in_ssd_dims[2];
        const int OBJECT_SIZE   = in_ssd_dims[3];
        CV_Assert(OBJECT_SIZE  == 7); // fixed SSD object size

        const cv::Size upscale = in_frame.size();
        const cv::Rect surface({0,0}, upscale);

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

            if (image_id < 0.f) {
                break;    // marks end-of-detections
            }
            if (confidence < 0.5f) {
                continue; // skip objects with low confidence
            }
            if (filter_label != -1 && static_cast<int>(label) != filter_label) {
                continue; // filter out object classes if filter is specified
            }

            cv::Rect rc;  // map relative coordinates to the original image scale
            rc.x      = static_cast<int>(rc_left   * upscale.width);
            rc.y      = static_cast<int>(rc_top    * upscale.height);
            rc.width  = static_cast<int>(rc_right  * upscale.width)  - rc.x;
            rc.height = static_cast<int>(rc_bottom * upscale.height) - rc.y;
            out_objects.emplace_back(rc & surface);
        }
    }
};

GAPI_OCV_KERNEL(OCVToMosaic, ToMosaic) {
    static void run(const std::vector<cv::Rect> &in_plate_rcs,
                    const std::vector<cv::Rect> &in_face_rcs,
                          std::vector<cv::gapi::wip::draw::Prim> &out_prims) {
        out_prims.clear();
        const auto cvt = [](cv::Rect rc) {
            // Align the mosaic region to mosaic block size
            const int BLOCK_SIZE = 24;
            const int dw = BLOCK_SIZE - (rc.width  % BLOCK_SIZE);
            const int dh = BLOCK_SIZE - (rc.height % BLOCK_SIZE);
            rc.width  += dw;
            rc.height += dh;
            rc.x      -= dw / 2;
            rc.y      -= dh / 2;
            return cv::gapi::wip::draw::Mosaic{rc, BLOCK_SIZE, 0};
        };
        for (auto &&rc : in_plate_rcs) { out_prims.emplace_back(cvt(rc)); }
        for (auto &&rc : in_face_rcs)  { out_prims.emplace_back(cvt(rc)); }
    }
};

} // namespace custom

int main(int argc, char *argv[])
{
    cv::CommandLineParser cmd(argc, argv, keys);
    cmd.about(about);
    if (cmd.has("help")) {
        cmd.printMessage();
        return 0;
    }
    const std::string input = cmd.get<std::string>("input");
    const bool no_show = cmd.get<bool>("noshow");
    const bool run_trad = cmd.get<bool>("trad");

    cv::GMat in;
    cv::GMat blob_plates = cv::gapi::infer<custom::VehLicDetector>(in);
    cv::GMat blob_faces  = cv::gapi::infer<custom::FaceDetector>(in);
    // VehLicDetector from Open Model Zoo marks vehicles with label "1" and
    // license plates with label "2", filter out license plates only.
    cv::GArray<cv::Rect> rc_plates = custom::ParseSSD::on(blob_plates, in, 2);
    // Face detector produces faces only so there's no need to filter by label,
    // pass "-1".
    cv::GArray<cv::Rect> rc_faces  = custom::ParseSSD::on(blob_faces, in, -1);
    cv::GMat out = cv::gapi::wip::draw::render3ch(in, custom::ToMosaic::on(rc_plates, rc_faces));
    cv::GComputation graph(in, out);

    const auto plate_model_path = cmd.get<std::string>("platm");
    auto plate_net = cv::gapi::ie::Params<custom::VehLicDetector> {
        plate_model_path,                // path to topology IR
        weights_path(plate_model_path),  // path to weights
        cmd.get<std::string>("platd"),   // device specifier
    };
    const auto face_model_path = cmd.get<std::string>("facem");
    auto face_net = cv::gapi::ie::Params<custom::FaceDetector> {
        face_model_path,                 // path to topology IR
        weights_path(face_model_path),   // path to weights
        cmd.get<std::string>("faced"),   // device specifier
    };
    auto kernels = cv::gapi::kernels<custom::OCVParseSSD, custom::OCVToMosaic>();
    auto networks = cv::gapi::networks(plate_net, face_net);

    cv::TickMeter tm;
    cv::Mat out_frame;
    std::size_t frames = 0u;
    std::cout << "Reading " << input << std::endl;

    if (run_trad) {
        cv::Mat in_frame;
        cv::VideoCapture cap(input);
        cap >> in_frame;

        auto exec = graph.compile(cv::descr_of(in_frame), cv::compile_args(kernels, networks));
        tm.start();
        do {
            exec(in_frame, out_frame);
            if (!no_show) {
                cv::imshow("Out", out_frame);
                cv::waitKey(1);
            }
            frames++;
        } while (cap.read(in_frame));
        tm.stop();
    } else {
        auto pipeline = graph.compileStreaming(cv::compile_args(kernels, networks));
        pipeline.setSource(cv::gapi::wip::make_src<cv::gapi::wip::GCaptureSource>(input));
        pipeline.start();
        tm.start();

        while (pipeline.pull(cv::gout(out_frame))) {
            frames++;
            if (!no_show) {
                cv::imshow("Out", out_frame);
                cv::waitKey(1);
            }
        }

        tm.stop();
    }

    std::cout << "Processed " << frames << " frames"
              << " (" << frames / tm.getTimeSec() << " FPS)" << std::endl;
    return 0;
}
