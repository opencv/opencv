#include <opencv2/gapi/cpu/core.hpp>
#include <opencv2/gapi/cpu/imgproc.hpp>

int main(int argc, char *argv[])
{
    (void) argc;
    (void) argv;

    bool need_first_resize  = true;
    bool need_seocnd_resize = false;

    cv::Size szOut(4, 4);
    cv::GComputation cc([&](){
// ! [GIOProtoArgs usage]
        auto ins = GIn();
        cv::GMat in1;
        if (need_first_resize)
            ins += GIn(in1);

        cv::GMat in2;
        if (need_seocnd_resize)
            ins += GIn(in2);

        cv::GProtoOutputArgs outs = GOut();
        cv::GMat out1 = cv::gapi::resize(in1, szOut);
        if (need_first_resize)
            outs += GOut(out1);

        cv::GMat out2 = cv::gapi::resize(in2, szOut);
        if (need_seocnd_resize)
            outs += GOut(out2);
// ! [GIOProtoArgs usage]
        return cv::GComputation(std::move(ins), std::move(outs));
    });

// ! [GRunArgs usage]
    auto in_vector = cv::gin();

    cv::Mat in_mat1( 8,  8, CV_8UC3);
    cv::Mat in_mat2(16, 16, CV_8UC3);
    cv::randu(in_mat1, cv::Scalar::all(0), cv::Scalar::all(255));
    cv::randu(in_mat2, cv::Scalar::all(0), cv::Scalar::all(255));

    if (need_first_resize)
        in_vector += cv::gin(in_mat1);
    if (need_seocnd_resize)
        in_vector += cv::gin(in_mat2);
// ! [GRunArgs usage]

// ! [GRunArgsP usage]
    auto out_vector = cv::gout();
    cv::Mat out_mat1, out_mat2;
    if (need_first_resize)
        out_vector += cv::gout(out_mat1);
    if (need_seocnd_resize)
        out_vector += cv::gout(out_mat2);
// ! [GRunArgsP usage]

    auto stream = cc.compileStreaming(cv::compile_args(cv::gapi::core::cpu::kernels()));
    stream.setSource(std::move(in_vector));

    stream.start();
    stream.pull(std::move(out_vector));
    stream.stop();

    return 0;
}
