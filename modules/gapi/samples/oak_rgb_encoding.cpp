#include <fstream>

#include <opencv2/gapi.hpp>
#include <opencv2/gapi/oak/oak.hpp>
#include <opencv2/gapi/oak/oak_media_adapter.hpp>

#include <opencv2/highgui.hpp> // CommandLineParser

// FIXME: extend parameters?
const std::string keys =
    "{ h help  |              | Print this help message }"
    "{ output  | output.h265  | Path to the output .h265 video file }";

int main(int argc, char *argv[]) {
    cv::CommandLineParser cmd(argc, argv, keys);
    if (cmd.has("help")) {
        cmd.printMessage();
        return 0;
    }

    const std::string output = cmd.get<std::string>("output");

    cv::GFrame in, h265;
    h265 = cv::gapi::oak::encode(in, {});

    auto args = cv::compile_args(cv::gapi::oak::ColorCameraParams{}, cv::gapi::oak::kernels());

    auto pipeline = cv::GComputation(cv::GIn(in), cv::GOut(h265)).compileStreaming(std::move(args));

    // Graph execution /////////////////////////////////////////////////////////
    pipeline.setSource(cv::gapi::wip::make_src<cv::gapi::oak::ColorCamera>());
    pipeline.start();

    cv::MediaFrame out_frame;
    std::ofstream out_h265_file;

    // Open H265 file for writing
    out_h265_file.open(output, std::ofstream::out | std::ofstream::binary | std::ofstream::trunc);

    // Pull 300 frames from the camera
    uint32_t frames = 300;
    uint32_t pulled = 0;

    while (pulled++ < frames &&
           pipeline.pull(cv::gout(out_frame))) {
        cv::MediaFrame::View view = out_frame.access(cv::MediaFrame::Access::R);
        auto adapter = out_frame.get<cv::gapi::oak::OAKMediaAdapter>();
        out_h265_file.write(reinterpret_cast<const char*>(view.ptr[0]), adapter->getDataSize());
    }
}
