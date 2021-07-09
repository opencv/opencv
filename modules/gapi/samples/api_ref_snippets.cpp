#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/gapi.hpp>
#include <opencv2/gapi/core.hpp>
#include <opencv2/gapi/imgproc.hpp>

#include <opencv2/gapi/cpu/gcpukernel.hpp>

#include <opencv2/gapi/fluid/core.hpp>
#include <opencv2/gapi/fluid/imgproc.hpp>

static void gscalar_example()
{
    //! [gscalar_implicit]
    cv::GMat a;
    cv::GMat b = a + 1;
    //! [gscalar_implicit]
}

static void typed_example()
{
    const cv::Size sz(32, 32);
    cv::Mat
        in_mat1        (sz, CV_8UC1),
        in_mat2        (sz, CV_8UC1),
        out_mat_untyped(sz, CV_8UC1),
        out_mat_typed1 (sz, CV_8UC1),
        out_mat_typed2 (sz, CV_8UC1);
    cv::randu(in_mat1, cv::Scalar::all(0), cv::Scalar::all(255));
    cv::randu(in_mat2, cv::Scalar::all(0), cv::Scalar::all(255));

    //! [Untyped_Example]
    // Untyped G-API ///////////////////////////////////////////////////////////
    cv::GComputation cvtU([]()
    {
        cv::GMat in1, in2;
        cv::GMat out = cv::gapi::add(in1, in2);
        return cv::GComputation({in1, in2}, {out});
    });
    std::vector<cv::Mat> u_ins  = {in_mat1, in_mat2};
    std::vector<cv::Mat> u_outs = {out_mat_untyped};
    cvtU.apply(u_ins, u_outs);
    //! [Untyped_Example]

    //! [Typed_Example]
    // Typed G-API /////////////////////////////////////////////////////////////
    cv::GComputationT<cv::GMat (cv::GMat, cv::GMat)> cvtT([](cv::GMat m1, cv::GMat m2)
    {
        return m1+m2;
    });
    cvtT.apply(in_mat1, in_mat2, out_mat_typed1);

    auto cvtTC =  cvtT.compile(cv::descr_of(in_mat1), cv::descr_of(in_mat2));
    cvtTC(in_mat1, in_mat2, out_mat_typed2);
    //! [Typed_Example]
}

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
         cv::gapi::imgproc::fluid::kernels());
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

    // Just call typed example with no input/output - avoid warnings about
    // unused functions
    typed_example();
    gscalar_example();
    return 0;
}
