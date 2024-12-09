#include "opencv2/opencv_modules.hpp"
#if defined(HAVE_OPENCV_GAPI)

#include <chrono>
#include <iomanip>

#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

#include "opencv2/gapi.hpp"
#include "opencv2/gapi/core.hpp"
#include "opencv2/gapi/imgproc.hpp"
#include "opencv2/gapi/infer.hpp"
#include "opencv2/gapi/infer/ie.hpp"
#include "opencv2/gapi/cpu/gcpukernel.hpp"
#include "opencv2/gapi/streaming/cap.hpp"

namespace {
const std::string about =
    "This is an OpenCV-based version of Security Barrier Camera example";
const std::string keys =
    "{ h help |   | print this help message }"
    "{ input  |   | Path to an input video file }"
    "{ fdm    |   | IE face detection model IR }"
    "{ fdw    |   | IE face detection model weights }"
    "{ fdd    |   | IE face detection device }"
    "{ agem   |   | IE age/gender recognition model IR }"
    "{ agew   |   | IE age/gender recognition model weights }"
    "{ aged   |   | IE age/gender recognition model device }"
    "{ emom   |   | IE emotions recognition model IR }"
    "{ emow   |   | IE emotions recognition model weights }"
    "{ emod   |   | IE emotions recognition model device }"
    "{ pure   |   | When set, no output is displayed. Useful for benchmarking }"
    "{ ser    |   | Run serially (no pipelining involved). Useful for benchmarking }";

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
// Describe networks we use in our program.
// In G-API, topologies act like "operations". Here we define our
// topologies as operations which have inputs and outputs.

// Every network requires three parameters to define:
// 1) Network's TYPE name - this TYPE is then used as a template
//    parameter to generic functions like cv::gapi::infer<>(),
//    and is used to define network's configuration (per-backend).
// 2) Network's SIGNATURE - a std::function<>-like record which defines
//    networks' input and output parameters (its API)
// 3) Network's IDENTIFIER - a string defining what the network is.
//    Must be unique within the pipeline.

// Note: these definitions are neutral to _how_ the networks are
// executed. The _how_ is defined at graph compilation stage (via parameters),
// not on the graph construction stage.

//! [G_API_NET]
// Face detector: takes one Mat, returns another Mat
G_API_NET(Faces, <cv::GMat(cv::GMat)>, "face-detector");

// Age/Gender recognition - takes one Mat, returns two:
// one for Age and one for Gender. In G-API, multiple-return-value operations
// are defined using std::tuple<>.
using AGInfo = std::tuple<cv::GMat, cv::GMat>;
G_API_NET(AgeGender, <AGInfo(cv::GMat)>,   "age-gender-recoginition");

// Emotion recognition - takes one Mat, returns another.
G_API_NET(Emotions, <cv::GMat(cv::GMat)>, "emotions-recognition");
//! [G_API_NET]

//! [Postproc]
// SSD Post-processing function - this is not a network but a kernel.
// The kernel body is declared separately, this is just an interface.
// This operation takes two Mats (detections and the source image),
// and returns a vector of ROI (filtered by a default threshold).
// Threshold (or a class to select) may become a parameter, but since
// this kernel is custom, it doesn't make a lot of sense.
G_API_OP(PostProc, <cv::GArray<cv::Rect>(cv::GMat, cv::GMat)>, "custom.fd_postproc") {
    static cv::GArrayDesc outMeta(const cv::GMatDesc &, const cv::GMatDesc &) {
        // This function is required for G-API engine to figure out
        // what the output format is, given the input parameters.
        // Since the output is an array (with a specific type),
        // there's nothing to describe.
        return cv::empty_array_desc();
    }
};

// OpenCV-based implementation of the above kernel.
GAPI_OCV_KERNEL(OCVPostProc, PostProc) {
    static void run(const cv::Mat &in_ssd_result,
                    const cv::Mat &in_frame,
                    std::vector<cv::Rect> &out_faces) {
        const int MAX_PROPOSALS = 200;
        const int OBJECT_SIZE   =   7;
        const cv::Size upscale = in_frame.size();
        const cv::Rect surface({0,0}, upscale);

        out_faces.clear();

        const float *data = in_ssd_result.ptr<float>();
        for (int i = 0; i < MAX_PROPOSALS; i++) {
            const float image_id   = data[i * OBJECT_SIZE + 0]; // batch id
            const float confidence = data[i * OBJECT_SIZE + 2];
            const float rc_left    = data[i * OBJECT_SIZE + 3];
            const float rc_top     = data[i * OBJECT_SIZE + 4];
            const float rc_right   = data[i * OBJECT_SIZE + 5];
            const float rc_bottom  = data[i * OBJECT_SIZE + 6];

            if (image_id < 0.f) {  // indicates end of detections
                break;
            }
            if (confidence < 0.5f) { // a hard-coded snapshot
                continue;
            }

            // Convert floating-point coordinates to the absolute image
            // frame coordinates; clip by the source image boundaries.
            cv::Rect rc;
            rc.x      = static_cast<int>(rc_left   * upscale.width);
            rc.y      = static_cast<int>(rc_top    * upscale.height);
            rc.width  = static_cast<int>(rc_right  * upscale.width)  - rc.x;
            rc.height = static_cast<int>(rc_bottom * upscale.height) - rc.y;
            out_faces.push_back(rc & surface);
        }
    }
};
//! [Postproc]

} // namespace custom

namespace labels {
const std::string genders[] = {
    "Female", "Male"
};
const std::string emotions[] = {
    "neutral", "happy", "sad", "surprise", "anger"
};
namespace {
void DrawResults(cv::Mat &frame,
                 const std::vector<cv::Rect> &faces,
                 const std::vector<cv::Mat>  &out_ages,
                 const std::vector<cv::Mat>  &out_genders,
                 const std::vector<cv::Mat>  &out_emotions) {
    CV_Assert(faces.size() == out_ages.size());
    CV_Assert(faces.size() == out_genders.size());
    CV_Assert(faces.size() == out_emotions.size());

    for (auto it = faces.begin(); it != faces.end(); ++it) {
        const auto idx = std::distance(faces.begin(), it);
        const auto &rc = *it;

        const float *ages_data     = out_ages[idx].ptr<float>();
        const float *genders_data  = out_genders[idx].ptr<float>();
        const float *emotions_data = out_emotions[idx].ptr<float>();
        const auto gen_id = std::max_element(genders_data,  genders_data  + 2) - genders_data;
        const auto emo_id = std::max_element(emotions_data, emotions_data + 5) - emotions_data;

        std::stringstream ss;
        ss << static_cast<int>(ages_data[0]*100)
           << ' '
           << genders[gen_id]
           << ' '
           << emotions[emo_id];

        const int ATTRIB_OFFSET = 15;
        cv::rectangle(frame, rc, {0, 255, 0},  4);
        cv::putText(frame, ss.str(),
                    cv::Point(rc.x, rc.y - ATTRIB_OFFSET),
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
                cv::Scalar(0, 255, 0),
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
    const bool be_serial = cmd.get<bool>("ser");

    // Express our processing pipeline. Lambda-based constructor
    // is used to keep all temporary objects in a dedicated scope.
    //! [GComputation]
    cv::GComputation pp([]() {
            // Declare an empty GMat - the beginning of the pipeline.
            cv::GMat in;

            // Run face detection on the input frame. Result is a single GMat,
            // internally representing an 1x1x200x7 SSD output.
            // This is a single-patch version of infer:
            // - Inference is running on the whole input image;
            // - Image is converted and resized to the network's expected format
            //   automatically.
            cv::GMat detections = cv::gapi::infer<custom::Faces>(in);

            // Parse SSD output to a list of ROI (rectangles) using
            // a custom kernel. Note: parsing SSD may become a "standard" kernel.
            cv::GArray<cv::Rect> faces = custom::PostProc::on(detections, in);

            // Now run Age/Gender model on every detected face. This model has two
            // outputs (for age and gender respectively).
            // A special ROI-list-oriented form of infer<>() is used here:
            // - First input argument is the list of rectangles to process,
            // - Second one is the image where to take ROI from;
            // - Crop/Resize/Layout conversion happens automatically for every image patch
            //   from the list
            // - Inference results are also returned in form of list (GArray<>)
            // - Since there're two outputs, infer<> return two arrays (via std::tuple).
            cv::GArray<cv::GMat> ages;
            cv::GArray<cv::GMat> genders;
            std::tie(ages, genders) = cv::gapi::infer<custom::AgeGender>(faces, in);

            // Recognize emotions on every face.
            // ROI-list-oriented infer<>() is used here as well.
            // Since custom::Emotions network produce a single output, only one
            // GArray<> is returned here.
            cv::GArray<cv::GMat> emotions = cv::gapi::infer<custom::Emotions>(faces, in);

            // Return the decoded frame as a result as well.
            // Input matrix can't be specified as output one, so use copy() here
            // (this copy will be optimized out in the future).
            cv::GMat frame = cv::gapi::copy(in);

            // Now specify the computation's boundaries - our pipeline consumes
            // one images and produces five outputs.
            return cv::GComputation(cv::GIn(in),
                                    cv::GOut(frame, faces, ages, genders, emotions));
        });
    //! [GComputation]

    // Note: it might be very useful to have dimensions loaded at this point!
    // After our computation is defined, specify how it should be executed.
    // Execution is defined by inference backends and kernel backends we use to
    // compile the pipeline (it is a different step).

    // Declare IE parameters for FaceDetection network. Note here custom::Face
    // is the type name we specified in GAPI_NETWORK() previously.
    // cv::gapi::ie::Params<> is a generic configuration description which is
    // specialized to every particular network we use.
    //
    // OpenCV DNN backend will have its own parmater structure with settings
    // relevant to OpenCV DNN module. Same applies to other possible inference
    // backends...
    //! [Param_Cfg]
    auto det_net = cv::gapi::ie::Params<custom::Faces> {
        cmd.get<std::string>("fdm"),   // read cmd args: path to topology IR
        cmd.get<std::string>("fdw"),   // read cmd args: path to weights
        cmd.get<std::string>("fdd"),   // read cmd args: device specifier
    };

    auto age_net = cv::gapi::ie::Params<custom::AgeGender> {
        cmd.get<std::string>("agem"),   // read cmd args: path to topology IR
        cmd.get<std::string>("agew"),   // read cmd args: path to weights
        cmd.get<std::string>("aged"),   // read cmd args: device specifier
    }.cfgOutputLayers({ "age_conv3", "prob" });

    auto emo_net = cv::gapi::ie::Params<custom::Emotions> {
        cmd.get<std::string>("emom"),   // read cmd args: path to topology IR
        cmd.get<std::string>("emow"),   // read cmd args: path to weights
        cmd.get<std::string>("emod"),   // read cmd args: device specifier
    };
    //! [Param_Cfg]

    //! [Compile]
    // Form a kernel package (with a single OpenCV-based implementation of our
    // post-processing) and a network package (holding our three networks).
    auto kernels = cv::gapi::kernels<custom::OCVPostProc>();
    auto networks = cv::gapi::networks(det_net, age_net, emo_net);

    // Compile our pipeline and pass our kernels & networks as
    // parameters.  This is the place where G-API learns which
    // networks & kernels we're actually operating with (the graph
    // description itself known nothing about that).
    auto cc = pp.compileStreaming(cv::compile_args(kernels, networks));
    //! [Compile]

    Avg avg;
    std::size_t frames = 0u;            // Frame counter (not produced by the graph)

    std::cout << "Reading " << input << std::endl;
    // Duplicate huge portions of the code in if/else branches in the sake of
    // better documentation snippets
    if (!be_serial) {
        //! [Source]
        auto in_src = cv::gapi::wip::make_src<cv::gapi::wip::GCaptureSource>(input);
        cc.setSource(cv::gin(in_src));
        //! [Source]

        avg.start();

        //! [Run]
        // After data source is specified, start the execution
        cc.start();

        // Declare data objects we will be receiving from the pipeline.
        cv::Mat frame;                      // The captured frame itself
        std::vector<cv::Rect> faces;        // Array of detected faces
        std::vector<cv::Mat> out_ages;      // Array of inferred ages (one blob per face)
        std::vector<cv::Mat> out_genders;   // Array of inferred genders (one blob per face)
        std::vector<cv::Mat> out_emotions;  // Array of classified emotions (one blob per face)

        // Implement different execution policies depending on the display option
        // for the best performance.
        while (cc.running()) {
            auto out_vector = cv::gout(frame, faces, out_ages, out_genders, out_emotions);
            if (no_show) {
                // This is purely a video processing. No need to balance
                // with UI rendering.  Use a blocking pull() to obtain
                // data. Break the loop if the stream is over.
                if (!cc.pull(std::move(out_vector)))
                    break;
            } else if (!cc.try_pull(std::move(out_vector))) {
                // Use a non-blocking try_pull() to obtain data.
                // If there's no data, let UI refresh (and handle keypress)
                if (cv::waitKey(1) >= 0) break;
                else continue;
            }
            // At this point we have data for sure (obtained in either
            // blocking or non-blocking way).
            frames++;
            labels::DrawResults(frame, faces, out_ages, out_genders, out_emotions);
            labels::DrawFPS(frame, frames, avg.fps(frames));
            if (!no_show) cv::imshow("Out", frame);
        }
        //! [Run]
    } else { // (serial flag)
        //! [Run_Serial]
        cv::VideoCapture cap(input);
        cv::Mat in_frame, frame;            // The captured frame itself
        std::vector<cv::Rect> faces;        // Array of detected faces
        std::vector<cv::Mat> out_ages;      // Array of inferred ages (one blob per face)
        std::vector<cv::Mat> out_genders;   // Array of inferred genders (one blob per face)
        std::vector<cv::Mat> out_emotions;  // Array of classified emotions (one blob per face)

        while (cap.read(in_frame)) {
            pp.apply(cv::gin(in_frame),
                     cv::gout(frame, faces, out_ages, out_genders, out_emotions),
                     cv::compile_args(kernels, networks));
            labels::DrawResults(frame, faces, out_ages, out_genders, out_emotions);
            frames++;
            if (frames == 1u) {
                // Start timer only after 1st frame processed -- compilation
                // happens on-the-fly here
                avg.start();
            } else {
                // Measurfe & draw FPS for all other frames
                labels::DrawFPS(frame, frames, avg.fps(frames-1));
            }
            if (!no_show) {
                cv::imshow("Out", frame);
                if (cv::waitKey(1) >= 0) break;
            }
        }
        //! [Run_Serial]
    }
    std::cout << "Processed " << frames << " frames in " << avg.elapsed()
              << " (" << avg.fps(frames) << " FPS)" << std::endl;
    return 0;
}
#else
#include <iostream>
int main()
{
    std::cerr << "This tutorial code requires G-API module "
                 "with Inference Engine backend to run"
              << std::endl;
    return 1;
}
#endif  // HAVE_OPECV_GAPI
