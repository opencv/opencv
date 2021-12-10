#include <opencv2/gapi.hpp>
#include <opencv2/gapi/core.hpp>
#include <opencv2/gapi/gframe.hpp>
#include <opencv2/gapi/media.hpp>

#include <opencv2/gapi/oak/oak.hpp>
#include <opencv2/gapi/streaming/format.hpp> // BGR accessor

#include <opencv2/highgui.hpp> // CommandLineParser

const std::string keys =
    "{ h help  |              | Print this help message }"
    "{ output  | output.png   | Path to the output file }";

int main(int argc, char *argv[]) {
    cv::CommandLineParser cmd(argc, argv, keys);
    if (cmd.has("help")) {
        cmd.printMessage();
        return 0;
    }

    const std::string output_name = cmd.get<std::string>("output");

    // Heterogeneous pipeline:
    // OAK camera -> streaming accessor -> CPU
    cv::GFrame in;
    cv::GMat acc = cv::gapi::streaming::BGR(in);
    cv::GMat out = cv::gapi::core::add(acc, acc);

    auto args = cv::compile_args(cv::gapi::oak::ColorCameraParams{},
                                 cv::gapi::combine(cv::gapi::oak::kernels(),
                                                   cv::gapi::streaming::kernels(),
                                                   cv::gapi::core::kernels()));

    auto pipeline = cv::GComputation(cv::GIn(in), cv::GOut(out)).compileStreaming(std::move(args));

    // Graph execution /////////////////////////////////////////////////////////
    cv::Mat out_mat(1920, 1080, CV_8UC3);

    pipeline.setSource(cv::gapi::wip::make_src<cv::gapi::oak::ColorCamera>());
    pipeline.start();

    // pull 1 frame
    pipeline.pull(cv::gout(out_mat));

    cv::imwrite(output_name, out_mat);
}
