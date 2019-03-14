#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/gapi.hpp>
#include <opencv2/gapi/core.hpp>
#include <opencv2/gapi/imgproc.hpp>

#include <opencv2/gapi/cpu/gcpukernel.hpp>

#include <opencv2/gapi/fluid/core.hpp>
#include <opencv2/gapi/fluid/imgproc.hpp>

G_TYPED_KERNEL(IAdd, <cv::GMat(cv::GMat)>, "test.custom.add") {
    static cv::GMatDesc outMeta(const cv::GMatDesc &in) { return in; }
};
G_TYPED_KERNEL(IFilter2D, <cv::GMat(cv::GMat)>, "test.custom.filter2d") {
    static cv::GMatDesc outMeta(const cv::GMatDesc &in) { return in; }
};
G_TYPED_KERNEL(IRGB2YUV, <cv::GMat(cv::GMat)>, "test.custom.add") {
    static cv::GMatDesc outMeta(const cv::GMatDesc &in) { return in; }
};
GAPI_OCV_KERNEL(CustomAdd,      IAdd)      { static void run(cv::Mat, cv::Mat &) {} };
GAPI_OCV_KERNEL(CustomFilter2D, IFilter2D) { static void run(cv::Mat, cv::Mat &) {} };
GAPI_OCV_KERNEL(CustomRGB2YUV,  IRGB2YUV)  { static void run(cv::Mat, cv::Mat &) {} };

int main(int argc, char *argv[])
{
    if (argc < 3)
        return -1;

    cv::Mat input = cv::imread(argv[1]);
    cv::Mat output;

    {
    //! [graph_def]
    cv::GMat in;
    cv::GMat gx = cv::gapi::Sobel(in, CV_32F, 1, 0);
    cv::GMat gy = cv::gapi::Sobel(in, CV_32F, 0, 1);
    cv::GMat g  = cv::gapi::sqrt(cv::gapi::mul(gx, gx) + cv::gapi::mul(gy, gy));
    cv::GMat out = cv::gapi::convertTo(g, CV_8U);
    //! [graph_def]

    //! [graph_decl_apply]
    //! [graph_cap_full]
    cv::GComputation sobelEdge(cv::GIn(in), cv::GOut(out));
    //! [graph_cap_full]
    sobelEdge.apply(input, output);
    //! [graph_decl_apply]

    //! [apply_with_param]
    cv::gapi::GKernelPackage kernels = cv::gapi::combine
        (cv::gapi::core::fluid::kernels(),
         cv::gapi::imgproc::fluid::kernels(),
         cv::unite_policy::KEEP);
    sobelEdge.apply(input, output, cv::compile_args(kernels));
    //! [apply_with_param]

    //! [graph_cap_sub]
    cv::GComputation sobelEdgeSub(cv::GIn(gx, gy), cv::GOut(out));
    //! [graph_cap_sub]
    }
    //! [graph_gen]
    cv::GComputation sobelEdgeGen([](){
            cv::GMat in;
            cv::GMat gx = cv::gapi::Sobel(in, CV_32F, 1, 0);
            cv::GMat gy = cv::gapi::Sobel(in, CV_32F, 0, 1);
            cv::GMat g  = cv::gapi::sqrt(cv::gapi::mul(gx, gx) + cv::gapi::mul(gy, gy));
            cv::GMat out = cv::gapi::convertTo(g, CV_8U);
            return cv::GComputation(in, out);
        });
    //! [graph_gen]

    cv::imwrite(argv[2], output);

    //! [kernels_snippet]
    cv::gapi::GKernelPackage pkg = cv::gapi::kernels
        < CustomAdd
        , CustomFilter2D
        , CustomRGB2YUV
        >();
    //! [kernels_snippet]
    return 0;
}
