#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/gapi.hpp>
#include <opencv2/gapi/core.hpp>
#include <opencv2/gapi/imgproc.hpp>
#include <opencv2/gapi/cpu/gcpukernel.hpp>
#include <opencv2/gapi/render.hpp>

#include "../src/api/render_ocv.hpp"
#include <opencv2/gapi/fluid/core.hpp>
#include <opencv2/gapi/fluid/imgproc.hpp>
#include <opencv2/gapi/cpu/core.hpp>
#include <opencv2/gapi/cpu/imgproc.hpp>


//using namespace cv;

/*int main(int argc, char *argv[])
{
    cv::VideoCapture cap;
    if (argc > 1) cap.open(argv[1]);
    else cap.open(0);
    CV_Assert(cap.isOpened());

    cv::GMat in;
    cv::GMat vga      = cv::gapi::resize(in, cv::Size(), 0.5, 0.5);
    cv::GMat gray     = cv::gapi::BGR2Gray(vga);
    cv::GMat blurred  = cv::gapi::blur(gray, cv::Size(5,5));
    cv::GMat edges    = cv::gapi::Canny(blurred, 32, 128, 3);
    cv::GMat b,g,r;
    std::tie(b,g,r)   = cv::gapi::split3(vga);
    cv::GMat out      = cv::gapi::merge3(b, g | edges, r);
    cv::GComputation ac(in, out);

    cv::Mat input_frame;
    cv::Mat output_frame;
    CV_Assert(cap.read(input_frame));
    do
    {
        ac.apply(input_frame, output_frame);
        cv::imshow("output", output_frame);
    } while (cap.read(input_frame) && cv::waitKey(30) < 0);

    return 0;
}*/

void HoughLinesToPrims(const std::vector<cv::Vec2f> &lines, std::vector<cv::gapi::wip::draw::Prim> &prims)
{
    prims.clear();
    for (size_t i = 0; i < lines.size(); i++)
    {
        float r = lines[i][0], t = lines[i][1];
        double cos_t = cos(t), sin_t = sin(t);
        double x0 = r * cos_t, y0 = r * sin_t;
        double alpha = 1000;

        prims.emplace_back(cv::gapi::wip::draw::Prim{ cv::gapi::wip::draw::Line{cv::Point{cvRound(x0 + alpha * (-sin_t)), cvRound(y0 + alpha * cos_t)},
                           cv::Point{cvRound(x0 - alpha * (-sin_t)), cvRound(y0 - alpha * cos_t)},
                           cv::Scalar{100, 0, 155}, 1, cv::LINE_AA, 0} });
    }
    return;
}

G_TYPED_KERNEL(GHoughLines, <cv::GArray<cv::Vec2f>(cv::GMat, double, double, int)>, "org.opencv.sample.hough_lines")
{
    static cv::GArrayDesc outMeta(cv::GMatDesc in, double, double, int) { return cv::empty_array_desc(); }
};

GAPI_OCV_KERNEL(GOCVHoughLines, GHoughLines)
{
    static void
    run(const cv::Mat& in,
        double rho,
        double theta,
        int threshhold,
        std::vector<cv::Vec2f> &lines)
    {
        cv::HoughLines(in, lines, rho, theta, threshhold);
    }
};

G_TYPED_KERNEL(GHoughLinesToPrims, <cv::GArray<cv::gapi::wip::draw::Prim>(cv::GArray<cv::Vec2f>)>, "org.opencv.sample.houghlines2prim")
{
    static cv::GArrayDesc outMeta(cv::GArrayDesc lines) { return cv::empty_array_desc(); }
};

GAPI_OCV_KERNEL(GOCVHoughLinesToPrims, GHoughLinesToPrims)
{
    static void
    run(const std::vector<cv::Vec2f> &lines,
        std::vector<cv::gapi::wip::draw::Prim> &prims)
    {
        HoughLinesToPrims(lines, prims);
    }
};

int main(int argc, char* argv[])
{
    cv::Mat src;
    int min_threshold = 300;

    cv::String imageName("building.jpg");
    if (argc > 1)
    {
        imageName = argv[1];
    }

    src = cv::imread(cv::samples::findFile("building.jpg"), cv::IMREAD_COLOR);

    if (src.empty())
    {
        std::cerr << "Invalid input image\n";
        std::cout << "Usage : " << argv[0] << " <path_to_input_image>\n";
        return -1;
    }

    cv::namedWindow("Original", cv::WINDOW_AUTOSIZE);
    cv::imshow("Original", src);
    cv::waitKey(0);

    cv::GMat input;
    cv::GMat grey = cv::gapi::BGR2Gray(input);
    cv::GMat blur = cv::gapi::gaussianBlur(grey, cv::Size(3, 3), 3.5, 3.5);//blur(grey, cv::Size(3, 3));
    cv::GMat edge = cv::gapi::Canny(blur, 40, 170);
    cv::GArray<cv::Vec2f> houghLines = GHoughLines::on(edge, 1, CV_PI/180, min_threshold);
    cv::GArray<cv::gapi::wip::draw::Prim> prims = GHoughLinesToPrims::on(houghLines);
    cv::GMat render = cv::gapi::wip::draw::GRenderBGR::on(input, prims);

    cv::GComputation comp(cv::GIn(input), cv::GOut(render));
    
    const cv::gapi::GKernelPackage custom = cv::gapi::kernels<GOCVHoughLines, GOCVHoughLinesToPrims>();
    const cv::gapi::GKernelPackage fluid = cv::gapi::combine(cv::gapi::core::fluid::kernels(),
                                                            cv::gapi::imgproc::fluid::kernels());
    const cv::gapi::GKernelPackage full_custom = cv::gapi::combine(cv::gapi::ocv::kernels(), custom);
    const auto full = cv::gapi::combine(full_custom, fluid);
    
    cv::Mat output;
    //std::vector<cv::Vec2f> lines;
    //std::vector<cv::gapi::wip::draw::Prim> prims;
    comp.apply(cv::gin(src), cv::gout(output), cv::compile_args(full));
    /*std::cout << "Lines number:" << lines.size() << std::endl;
    for (size_t i = 0; i < lines.size(); i++)
    {
        float r = lines[i][0], t = lines[i][1];
        double cos_t = cos(t), sin_t = sin(t);
        double x0 = r * cos_t, y0 = r * sin_t;
        double alpha = 1000;

        cv::Point pt1(cvRound(x0 + alpha * (-sin_t)), cvRound(y0 + alpha * cos_t));
        cv::Point pt2(cvRound(x0 - alpha * (-sin_t)), cvRound(y0 - alpha * cos_t));
        cv::line(src, pt1, pt2, cv::Scalar(100, 0, 155), 2, cv::LINE_AA);
    }

    cv::namedWindow("StandartHough", cv::WINDOW_AUTOSIZE);*/
    //cv::imshow("StandartHough", output);
    //cv::waitKey(0);

    return 0;
}
