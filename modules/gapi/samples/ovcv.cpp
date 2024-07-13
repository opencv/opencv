#include <algorithm>
#include <iostream>
#include <cctype>

#include <opencv2/gapi.hpp>
#include <opencv2/gapi/core.hpp>
#include <opencv2/gapi/infer/ov.hpp>
#include <opencv2/gapi/fluid/core.hpp>
#include <opencv2/gapi/streaming/cap.hpp>
#include <opencv2/highgui.hpp>

const std::string about =
    "This is an OVCV example";
const std::string keys =
    "{ h help   |  | Print this help message }"
    "{ i input  |  | Path to the input video file }";

int main(int argc, char *argv[])
{
    cv::CommandLineParser cmd(argc, argv, keys);
    cmd.about(about);
    if (cmd.has("help")) {
        cmd.printMessage();
        return 0;
    }
    const std::string input = cmd.get<std::string>("input");

    cv::GMat in;
    cv::GMat res = cv::gapi::resize(in, {320, 240});
    cv::GMat r, g, b;
    std::tie(b, g, r) = cv::gapi::split3(res);

    cv::GMat panes[4];
    panes[0] = res;
    panes[1] = cv::gapi::merge3(r, r, r);
    panes[2] = cv::gapi::merge3(g, g, g);
    panes[3] = cv::gapi::merge3(b, b, b);

    cv::GMat out = cv::gapi::concatVert(cv::gapi::concatHor(panes[0], panes[1]),
                                        cv::gapi::concatHor(panes[2], panes[3]));

    try {
        auto pipeline = cv::GComputation(in, out)
            .compileStreaming(cv::compile_args(cv::gapi::use_only{cv::gapi::ov::kernels()}));

        pipeline.setSource<cv::gapi::wip::GCaptureSource>(input);
        pipeline.start();

        cv::Mat pic;
        while (pipeline.pull(cv::gout(pic))) {
            cv::imshow("Out", pic);
            cv::waitKey(15);
        }
    } catch (std::exception &ex) {
        std::cout << ex.what() << std::endl;
        return 1;
    }

    return 0;
}
