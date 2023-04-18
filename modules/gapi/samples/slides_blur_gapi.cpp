#include <opencv2/gapi.hpp>                            // G-API framework header
#include <opencv2/gapi/imgproc.hpp>                    // cv::gapi::blur()
#include <opencv2/highgui.hpp>                         // cv::imread/imwrite

int main(int argc, char *argv[]) {
    if (argc < 3) return 1;

    cv::GMat in;                                       // Express the graph:
    cv::GMat out = cv::gapi::blur(in, cv::Size(3,3));  // `out` is a result of `blur` of `in`

    cv::Mat in_mat = cv::imread(argv[1]);              // Get the real data
    cv::Mat out_mat;                                   // Output buffer (may be empty)

    cv::GComputation(cv::GIn(in), cv::GOut(out))       // Declare a graph from `in` to `out`
        .apply(cv::gin(in_mat), cv::gout(out_mat));    // ...and run it immediately

    cv::imwrite(argv[2], out_mat);                     // Save the result
    return 0;
}
