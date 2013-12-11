#include <opencv2/core/utility.hpp>
#include <opencv2/cuda.hpp>
#include <opencv2/softcascade.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

typedef cv::softcascade::Detection Detection;

int main(int argc, char** argv)
{
    const std::string keys =
        "{help h usage ?    |     | print this message }"
        "{cascade c         |     | path to configuration xml }"
        "{frames f          |     | path to configuration xml }"
        "{min_scale         |0.4f | path to configuration xml }"
        "{max_scale         |5.0f | path to configuration xml }"
        "{total_scales      |55   | path to configuration xml }"
        "{device d          |0    | path to configuration xml }"
    ;

    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("Soft cascade training application.");

    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }

    if (!parser.check())
    {
        parser.printErrors();
        return 1;
    }

    cv::cuda::setDevice(parser.get<int>("device"));

    std::string cascadePath = parser.get<std::string>("cascade");

    cv::FileStorage fs(cascadePath, cv::FileStorage::READ);
    if(!fs.isOpened())
    {
        std::cout << "Soft Cascade file " << cascadePath << " can't be opened." << std::endl << std::flush;
        return 1;
    }

    std::cout << "Read cascade from file " << cascadePath << std::endl;

    float minScale =  parser.get<float>("min_scale");
    float maxScale =  parser.get<float>("max_scale");
    int scales     =  parser.get<int>("total_scales");

    using cv::softcascade::SCascade;
    SCascade cascade(minScale, maxScale, scales);

    if (!cascade.load(fs.getFirstTopLevelNode()))
    {
        std::cout << "Soft Cascade can't be parsed." << std::endl << std::flush;
        return 1;
    }

    std::string frames = parser.get<std::string>("frames");
    cv::VideoCapture capture(frames);
    if(!capture.isOpened())
    {
        std::cout << "Frame source " << frames << " can't be opened." << std::endl << std::flush;
        return 1;
    }

    cv::cuda::GpuMat objects(1, sizeof(Detection) * 10000, CV_8UC1);
    cv::cuda::printShortCudaDeviceInfo(parser.get<int>("device"));
    for (;;)
    {
        cv::Mat frame;
        if (!capture.read(frame))
        {
            std::cout << "Nothing to read. " << std::endl << std::flush;
            return 0;
        }

        cv::cuda::GpuMat dframe(frame), roi(frame.rows, frame.cols, CV_8UC1);
        roi.setTo(cv::Scalar::all(1));
        cascade.detect(dframe, roi, objects);

        cv::Mat dt(objects);

        Detection* dts = ((Detection*)dt.data) + 1;
        int* count = dt.ptr<int>(0);

        std::cout << *count << std::endl;

        cv::Mat result;
        frame.copyTo(result);


        for (int i = 0; i < *count; ++i)
        {
            Detection d = dts[i];
            cv::rectangle(result, cv::Rect(d.x, d.y, d.w, d.h), cv::Scalar(255, 0, 0, 255), 1);
        }

        std::cout << "working..." << std::endl;
        cv::imshow("Soft Cascade demo", result);
        if (27 == cv::waitKey(10))
            break;
    }

    return 0;
}
