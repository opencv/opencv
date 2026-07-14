#include "opencv2/opencv_modules.hpp"
#include <iostream>
#if defined(HAVE_OPENCV_GAPI)

#include <chrono>
#include <iomanip>

#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/gapi.hpp"
#include "opencv2/gapi/core.hpp"
#include "opencv2/gapi/imgproc.hpp"
#include "opencv2/gapi/infer.hpp"
#include "opencv2/gapi/infer/ie.hpp"
#include "opencv2/gapi/cpu/gcpukernel.hpp"
#include "opencv2/gapi/streaming/cap.hpp"
#include "opencv2/highgui.hpp"

const std::string about =
    "This is an OpenCV-based version of Security Barrier Camera example";
const std::string keys =
    "{ h help |   | print this help message }"
    "{ input  |   | Path to an input video file }"
    "{ detm   |   | IE vehicle/license plate detection model IR }"
    "{ detw   |   | IE vehicle/license plate detection model weights }"
    "{ detd   |   | IE vehicle/license plate detection model device }"
    "{ vehm   |   | IE vehicle attributes model IR }"
    "{ vehw   |   | IE vehicle attributes model weights }"
    "{ vehd   |   | IE vehicle attributes model device }"
    "{ lprm   |   | IE license plate recognition model IR }"
    "{ lprw   |   | IE license plate recognition model weights }"
    "{ lprd   |   | IE license plate recognition model device }"
    "{ pure   |   | When set, no output is displayed. Useful for benchmarking }"
    "{ ser    |   | When set, runs a regular (serial) pipeline }";

namespace {
struct Avg {
    struct Elapsed {
        explicit Elapsed(double ms) : ss(ms/1000.), mm(static_cast<int>(ss)/60) {}
        const double ss;
        const int    mm;
    };

    using MS = std::chrono::duration<double, std::ratio<1, 1000>>;
    using TS = std::chrono::time_point<std::chrono::high_resolution_clock>;
    TS started;

    void    start() { started = now(); }
    TS      now() const { return std::chrono::high_resolution_clock::now(); }
    double  tick() const { return std::chrono::duration_cast<MS>(now() - started).count(); }
    Elapsed elapsed() const { return Elapsed{tick()}; }
    double  fps(std::size_t n) const { return static_cast<double>(n) / (tick() / 1000.); }
};
std::ostream& operator<<(std::ostream &os, const Avg::Elapsed &e) {
    os << e.mm << ':' << (e.ss - 60*e.mm);
    return os;
}
} // namespace


namespace custom {
G_API_NET(VehicleLicenseDetector, <cv::GMat(cv::GMat)>, "vehicle-license-plate-detector");

using Attrs = std::tuple<cv::GMat, cv::GMat>;
G_API_NET(VehicleAttributes,      <Attrs(cv::GMat)>,    "vehicle-attributes");
G_API_NET(LPR,                    <cv::GMat(cv::GMat)>, "license-plate-recognition");

using GVehiclesPlates = std::tuple< cv::GArray<cv::Rect>
                                  , cv::GArray<cv::Rect> >;
G_API_OP_M(ProcessDetections,
           <GVehiclesPlates(cv::GMat, cv::GMat)>,
           "custom.security_barrier.detector.postproc") {
    static std::tuple<cv::GArrayDesc,cv::GArrayDesc>
    outMeta(const cv::GMatDesc &, const cv::GMatDesc) {
        // FIXME: Need to get rid of this - literally there's nothing useful
        return std::make_tuple(cv::empty_array_desc(), cv::empty_array_desc());
    }
};

GAPI_OCV_KERNEL(OCVProcessDetections, ProcessDetections) {
    static void run(const cv::Mat &in_ssd_result,
                    const cv::Mat &in_frame,
                    std::vector<cv::Rect> &out_vehicles,
                    std::vector<cv::Rect> &out_plates) {
        const int MAX_PROPOSALS = 200;
        const int OBJECT_SIZE   =   7;
        const cv::Size upscale = in_frame.size();
        const cv::Rect surface({0,0}, upscale);

        out_vehicles.clear();
        out_plates.clear();

        const float *data = in_ssd_result.ptr<float>();
        for (int i = 0; i < MAX_PROPOSALS; i++) {
            const float image_id   = data[i * OBJECT_SIZE + 0]; // batch id
            const float label      = data[i * OBJECT_SIZE + 1];
            const float confidence = data[i * OBJECT_SIZE + 2];
            const float rc_left    = data[i * OBJECT_SIZE + 3];
            const float rc_top     = data[i * OBJECT_SIZE + 4];
            const float rc_right   = data[i * OBJECT_SIZE + 5];
            const float rc_bottom  = data[i * OBJECT_SIZE + 6];

            if (image_id < 0.f) {  // indicates end of detections
                break;
            }
            if (confidence < 0.5f) { // fixme: hard-coded snapshot
                continue;
            }

            cv::Rect rc;
            rc.x      = static_cast<int>(rc_left   * upscale.width);
            rc.y      = static_cast<int>(rc_top    * upscale.height);
            rc.width  = static_cast<int>(rc_right  * upscale.width)  - rc.x;
            rc.height = static_cast<int>(rc_bottom * upscale.height) - rc.y;

            using PT = cv::Point;
            using SZ = cv::Size;
            switch (static_cast<int>(label)) {
            case 1: out_vehicles.push_back(rc & surface); break;
            case 2: out_plates.emplace_back((rc-PT(15,15)+SZ(30,30)) & surface); break;
            default: CV_Assert(false && "Unknown object class");
            }
        }
    }
};
} // namespace custom

namespace labels {
const std::string colors[] = {
    "white", "gray", "yellow", "red", "green", "blue", "black"
};
const std::string types[] = {
    "car", "van", "truck", "bus"
};
const std::vector<std::string> license_text = {
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "<Anhui>", "<Beijing>", "<Chongqing>", "<Fujian>",
    "<Gansu>", "<Guangdong>", "<Guangxi>", "<Guizhou>",
    "<Hainan>", "<Hebei>", "<Heilongjiang>", "<Henan>",
    "<HongKong>", "<Hubei>", "<Hunan>", "<InnerMongolia>",
    "<Jiangsu>", "<Jiangxi>", "<Jilin>", "<Liaoning>",
    "<Macau>", "<Ningxia>", "<Qinghai>", "<Shaanxi>",
    "<Shandong>", "<Shanghai>", "<Shanxi>", "<Sichuan>",
    "<Tianjin>", "<Tibet>", "<Xinjiang>", "<Yunnan>",
    "<Zhejiang>", "<police>",
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
    "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
    "U", "V", "W", "X", "Y", "Z"
};
namespace {
void DrawResults(cv::Mat &frame,
                 const std::vector<cv::Rect> &vehicles,
                 const std::vector<cv::Mat>  &out_colors,
                 const std::vector<cv::Mat>  &out_types,
                 const std::vector<cv::Rect> &plates,
                 const std::vector<cv::Mat>  &out_numbers) {
    CV_Assert(vehicles.size() == out_colors.size());
    CV_Assert(vehicles.size() == out_types.size());
    CV_Assert(plates.size()   == out_numbers.size());

    for (auto it = vehicles.begin(); it != vehicles.end(); ++it) {
        const auto idx = std::distance(vehicles.begin(), it);
        const auto &rc = *it;

        const float *colors_data = out_colors[idx].ptr<float>();
        const float *types_data  = out_types [idx].ptr<float>();
        const auto color_id = std::max_element(colors_data, colors_data + 7) - colors_data;
        const auto  type_id = std::max_element(types_data,  types_data  + 4) - types_data;

        const int ATTRIB_OFFSET = 25;
        cv::rectangle(frame, rc, {0, 255, 0},  4);
        cv::putText(frame, labels::colors[color_id],
                    cv::Point(rc.x + 5, rc.y + ATTRIB_OFFSET),
                    cv::FONT_HERSHEY_COMPLEX_SMALL,
                    1,
                    cv::Scalar(255, 0, 0));
        cv::putText(frame, labels::types[type_id],
                    cv::Point(rc.x + 5, rc.y + ATTRIB_OFFSET * 2),
                    cv::FONT_HERSHEY_COMPLEX_SMALL,
                    1,
                    cv::Scalar(255, 0, 0));
    }

    for (auto it = plates.begin(); it != plates.end(); ++it) {
        const int MAX_LICENSE = 88;
        const int LPR_OFFSET  = 50;

        const auto &rc   = *it;
        const auto idx   = std::distance(plates.begin(), it);

        std::string result;
        const auto *lpr_data = out_numbers[idx].ptr<float>();
        for (int i = 0; i < MAX_LICENSE; i++) {
            if (lpr_data[i] == -1) break;
            result += labels::license_text[static_cast<size_t>(lpr_data[i])];
        }

        const int y_pos = std::max(0, rc.y + rc.height - LPR_OFFSET);
        cv::rectangle(frame, rc, {0, 0, 255},  4);
        cv::putText(frame, result,
                    cv::Point(rc.x, y_pos),
                    cv::FONT_HERSHEY_COMPLEX_SMALL,
                    1,
                    cv::Scalar(0, 0, 255));
    }
}

void DrawFPS(cv::Mat &frame, std::size_t n, double fps) {
    std::ostringstream out;
    out << "FRAME " << n << ": "
        << std::fixed << std::setprecision(2) << fps
        << " FPS (AVG)";
    cv::putText(frame, out.str(),
                cv::Point(0, frame.rows),
                cv::FONT_HERSHEY_SIMPLEX,
                1,
                cv::Scalar(0, 0, 0),
                2);
}
} // anonymous namespace
} // namespace labels

int main(int argc, char *argv[])
{
    cv::CommandLineParser cmd(argc, argv, keys);
    cmd.about(about);
    if (cmd.has("help")) {
        cmd.printMessage();
        return 0;
    }
    const std::string input = cmd.get<std::string>("input");
    const bool no_show = cmd.get<bool>("pure");

    cv::GComputation pp([]() {
            cv::GMat in;
            cv::GMat detections          = cv::gapi::infer<custom::VehicleLicenseDetector>(in);
            cv::GArray<cv::Rect> vehicles;
            cv::GArray<cv::Rect> plates;
            std::tie(vehicles, plates)   = custom::ProcessDetections::on(detections, in);
            cv::GArray<cv::GMat> colors;
            cv::GArray<cv::GMat> types;
            std::tie(colors, types)      = cv::gapi::infer<custom::VehicleAttributes>(vehicles, in);
            cv::GArray<cv::GMat> numbers = cv::gapi::infer<custom::LPR>(plates, in);
            cv::GMat frame = cv::gapi::copy(in); // pass-through the input frame
            return cv::GComputation(cv::GIn(in),
                                    cv::GOut(frame, vehicles, colors, types, plates, numbers));
        });

    // Note: it might be very useful to have dimensions loaded at this point!
    auto det_net = cv::gapi::ie::Params<custom::VehicleLicenseDetector> {
        cmd.get<std::string>("detm"),   // path to topology IR
        cmd.get<std::string>("detw"),   // path to weights
        cmd.get<std::string>("detd"),   // device specifier
    };

    auto attr_net = cv::gapi::ie::Params<custom::VehicleAttributes> {
        cmd.get<std::string>("vehm"),   // path to topology IR
        cmd.get<std::string>("vehw"),   // path to weights
        cmd.get<std::string>("vehd"),   // device specifier
    }.cfgOutputLayers({ "color", "type" });

    // Fill a special LPR input (seq_ind) with a predefined value
    // First element is 0.f, the rest 87 are 1.f
    const std::vector<int> lpr_seq_dims = {88,1};
    cv::Mat lpr_seq(lpr_seq_dims, CV_32F, cv::Scalar(1.f));
    lpr_seq.ptr<float>()[0] = 0.f;
    auto lpr_net = cv::gapi::ie::Params<custom::LPR> {
        cmd.get<std::string>("lprm"),   // path to topology IR
        cmd.get<std::string>("lprw"),   // path to weights
        cmd.get<std::string>("lprd"),   // device specifier
    }.constInput("seq_ind", lpr_seq);

    auto kernels = cv::gapi::kernels<custom::OCVProcessDetections>();
    auto networks = cv::gapi::networks(det_net, attr_net, lpr_net);

    Avg avg;
    cv::Mat frame;
    std::vector<cv::Rect> vehicles, plates;
    std::vector<cv::Mat> out_colors;
    std::vector<cv::Mat> out_types;
    std::vector<cv::Mat> out_numbers;
    std::size_t frames = 0u;

    std::cout << "Reading " << input << std::endl;

    if (cmd.get<bool>("ser")) {
        std::cout << "Going serial..." << std::endl;
        cv::VideoCapture cap(input);

        auto cc = pp.compile(cv::GMatDesc{CV_8U,3,cv::Size(1920,1080)},
                             cv::compile_args(kernels, networks));

        avg.start();
        while (cv::waitKey(1) < 0) {
            cap >> frame;
            if (frame.empty()) break;

            cc(cv::gin(frame),
               cv::gout(frame, vehicles, out_colors, out_types, plates, out_numbers));
            frames++;
            labels::DrawResults(frame, vehicles, out_colors, out_types, plates, out_numbers);
            labels::DrawFPS(frame, frames, avg.fps(frames));
            if (!no_show) cv::imshow("Out", frame);
        }
    } else {
        std::cout << "Going pipelined..." << std::endl;

        auto cc = pp.compileStreaming(cv::GMatDesc{CV_8U,3,cv::Size(1920,1080)},
                                      cv::compile_args(kernels, networks));

        cc.setSource(cv::gapi::wip::make_src<cv::gapi::wip::GCaptureSource>(input));

        avg.start();
        cc.start();

        // Implement different execution policies depending on the display option
        // for the best performance.
        while (cc.running()) {
            auto out_vector = cv::gout(frame, vehicles, out_colors, out_types, plates, out_numbers);
            if (no_show) {
                // This is purely a video processing. No need to balance with UI rendering.
                // Use a blocking pull() to obtain data. Break the loop if the stream is over.
                if (!cc.pull(std::move(out_vector)))
                    break;
            } else if (!cc.try_pull(std::move(out_vector))) {
                // Use a non-blocking try_pull() to obtain data.
                // If there's no data, let UI refresh (and handle keypress)
                if (cv::waitKey(1) >= 0) break;
                else continue;
            }
            // At this point we have data for sure (obtained in either blocking or non-blocking way).
            frames++;
            labels::DrawResults(frame, vehicles, out_colors, out_types, plates, out_numbers);
            labels::DrawFPS(frame, frames, avg.fps(frames));
            if (!no_show) cv::imshow("Out", frame);
        }
        cc.stop();
    }
    std::cout << "Processed " << frames << " frames in " << avg.elapsed() << std::endl;

    return 0;
}
#else
int main()
{
    std::cerr << "This tutorial code requires G-API module "
                 "with Inference Engine backend to run"
              << std::endl;
    return 1;
}
#endif  // HAVE_OPECV_GAPI
