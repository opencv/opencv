#include <opencv2/softcascade/softcascade.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>

#include <iostream>
#include <vector>
#include <string>
#include <fstream>

#include <glob.h>

namespace {

using std::string;

void glob(const std::string& path, std::vector<string>& ret)
{
    glob_t glob_result;
    glob(path.c_str(), GLOB_TILDE, 0, &glob_result);

    ret.clear();
    ret.reserve(glob_result.gl_pathc);

    for(unsigned int i = 0; i < glob_result.gl_pathc; ++i)
    {
    ret.push_back(std::string(glob_result.gl_pathv[i]));
    std::cout << ret[i] << std::endl;
    }

    globfree(&glob_result);
}

}

int main(int argc, char** argv)
{
    const std::string keys =
    "{help h usage ?    |     | print this message }"
    "{cascade c         |     | path to cascade xml xml }"
    "{frame f           |     | path to frame source xml }"
    "{min_scale         |0.4  | minimum scale to detect }"
    "{max_scale         |5.0  | maxamum scale to detect }"
    "{total_scales      |55   | prefered number of scales between min and max }"
    "{write_file wf     |0    | write to .txt. Enabled by default.}"
    "{write_image wi    |0    | write to image. Disabled by default.}"
    "{threshold thr     |0    | detection threshold}"
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

    std::string cascadePath = parser.get<std::string>("cascade");

    cv::FileStorage fs(cascadePath, cv::FileStorage::READ);
    if( !fs.isOpened())
    {
        std::cout << "Soft Cascade file " << cascadePath << " can't be opened." << std::endl << std::flush;
        return 1;
    }

    std::cout << "Read cascade from file " << cascadePath << std::endl;

    float minScale =  parser.get<float>("min_scale");
    float maxScale =  parser.get<float>("max_scale");
    int scales     =  parser.get<int>("total_scales");
    int thr        =  parser.get<int>("threshold");

    cv::softcascade::Detector cascade(minScale, maxScale, scales, cv::softcascade::Detector::DOLLAR);

    if (!cascade.load(fs.getFirstTopLevelNode()))
    {
        std::cout << "Soft Cascade can't be parsed." << std::endl << std::flush;
        return 1;
    }

    std::string src = parser.get<std::string>("frame");
    std::vector<std::string> frames;
    glob(parser.get<std::string>("frame"), frames);
    std::cout << "collected " << src << " " << frames.size() << " frames." << std::endl;

    int wf = parser.get<int>("write_file");

    if (wf)
        std::cout << "resulte will be stored to .txt file with the same name as image." << std::endl;

    int wi = parser.get<int>("write_image");

    if (wi)
        std::cout << "resulte will be stored to image with the same name as input plus dt." << std::endl;

    for (int i = 0; i < (int)frames.size(); ++i)
    {
        std::string& frame_sourse = frames[i];
        cv::Mat frame = cv::imread(frame_sourse);

        if(frame.empty())
        {
            std::cout << "Frame source " << frame_sourse << " can't be opened." << std::endl << std::flush;
            return 1;
        }

        std::vector<cv::softcascade::Detection> objects;
        cascade.detect(frame,  cv::noArray(), objects);

        std::cout << "collected: " << (int)objects.size() << " detections." << std::endl;

        std::ofstream myfile;

        if (wf)
            myfile.open((frame_sourse + ".txt").c_str(), std::ios::out);

        for (int obj = 0; obj  < (int)objects.size(); ++obj)
        {
            cv::softcascade::Detection d = objects[obj];

            if(d.confidence > thr)
            {
                float b = d.confidence * 1.5f;

                std::stringstream conf(std::stringstream::in | std::stringstream::out);
                conf << d.confidence;

                cv::rectangle(frame, cv::Rect(d.bb.x, d.bb.y, d.bb.width, d.bb.height), cv::Scalar(b, 0, 255 - b, 255), 2);
                cv::putText(frame, conf.str() , cv::Point(d.bb.x + 10, d.bb.y - 5),1, 1.1, cv::Scalar(25, 133, 255, 0), 1, CV_AA);

                if (wf)
                    myfile << d.bb.x     << " " <<  d.bb.y      << " "
                           << d.bb.width << " " <<  d.bb.height << " " << d.confidence << "\n";
            }
        }

        if (wi)
            cv::imwrite(frame_sourse + ".dt.png", frame);

        if (wf)
        {
            myfile.close();
        }

        cv::imshow("frame", frame);
        cv::waitKey(10);
    }

    return 0;
}