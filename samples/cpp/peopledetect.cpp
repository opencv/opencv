#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/softcascade.hpp>

#include <iostream>
#include <vector>
#include <string>
#include <fstream>

void filter_rects(const std::vector<cv::Rect>& candidates, std::vector<cv::Rect>& objects);

int main(int argc, char** argv)
{
    const std::string keys =
    "{help h usage ?    |     | print this message and exit }"
    "{cascade c         |     | path to cascade xml, if empty HOG detector will be executed }"
    "{frame f           |     | wildchart pattern to frame source}"
    "{min_scale         |0.4  | minimum scale to detect }"
    "{max_scale         |5.0  | maxamum scale to detect }"
    "{total_scales      |55   | prefered number of scales between min and max }"
    "{write_file wf     |0    | write to .txt. Disabled by default.}"
    "{write_image wi    |0    | write to image. Disabled by default.}"
    "{show_image si     |1    | show image. Enabled by default.}"
    "{threshold thr     |-1   | detection threshold. Detections with score less then threshold will be ignored.}"
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

    int wf = parser.get<int>("write_file");
    if (wf) std::cout << "resulte will be stored to .txt file with the same name as image." << std::endl;

    int wi = parser.get<int>("write_image");
    if (wi) std::cout << "resulte will be stored to image with the same name as input plus dt." << std::endl;

    int si = parser.get<int>("show_image");

    float minScale =  parser.get<float>("min_scale");
    float maxScale =  parser.get<float>("max_scale");
    int scales     =  parser.get<int>("total_scales");
    int thr        =  parser.get<int>("threshold");

    cv::HOGDescriptor hog;
    cv::softcascade::Detector cascade;

    bool useHOG = false;
    std::string cascadePath = parser.get<std::string>("cascade");
    if (cascadePath.empty())
    {
        useHOG = true;
        hog.setSVMDetector(cv::HOGDescriptor::getDefaultPeopleDetector());
        std::cout << "going to use HOG detector." << std::endl;
    }
    else
    {
        cv::FileStorage fs(cascadePath, cv::FileStorage::READ);
        if( !fs.isOpened())
        {
            std::cout << "Soft Cascade file " << cascadePath << " can't be opened." << std::endl << std::flush;
            return 1;
        }

        cascade = cv::softcascade::Detector(minScale, maxScale, scales, cv::softcascade::Detector::DOLLAR);

        if (!cascade.load(fs.getFirstTopLevelNode()))
        {
            std::cout << "Soft Cascade can't be parsed." << std::endl << std::flush;
            return 1;
        }
    }

    std::string src = parser.get<std::string>("frame");
    std::vector<cv::String> frames;
    cv::glob(parser.get<std::string>("frame"), frames);
    std::cout << "collected " << src << " " << frames.size() << " frames." << std::endl;

    for (int i = 0; i < (int)frames.size(); ++i)
    {
        std::string frame_sourse = frames[i];
        cv::Mat frame = cv::imread(frame_sourse);

        if(frame.empty())
        {
            std::cout << "Frame source " << frame_sourse << " can't be opened." << std::endl << std::flush;
            continue;
        }

        std::ofstream myfile;
        if (wf)
            myfile.open((frame_sourse.replace(frame_sourse.end() - 3, frame_sourse.end(), "txt")).c_str(), std::ios::out);

        ////
        if (useHOG)
        {
            std::vector<cv::Rect> found, found_filtered;
            // run the detector with default parameters. to get a higher hit-rate
            // (and more false alarms, respectively), decrease the hitThreshold and
            // groupThreshold (set groupThreshold to 0 to turn off the grouping completely).
            hog.detectMultiScale(frame, found, 0, cv::Size(8,8), cv::Size(32,32), 1.05, 2);

            filter_rects(found, found_filtered);
            std::cout << "collected: " << (int)found_filtered.size() << " detections." << std::endl;

            for (size_t ff = 0; ff < found_filtered.size(); ++ff)
            {
                cv::Rect r = found_filtered[ff];
                cv::rectangle(frame, r.tl(), r.br(), cv::Scalar(0,255,0), 3);

                if (wf) myfile << r.x << "," << r.y << "," << r.width << "," << r.height << "," << 0.f << "\n";
            }
        }
        else
        {
            std::vector<cv::softcascade::Detection> objects;
            cascade.detect(frame,  cv::noArray(), objects);
            std::cout << "collected: " << (int)objects.size() << " detections." << std::endl;

            for (int obj = 0; obj  < (int)objects.size(); ++obj)
            {
                cv::softcascade::Detection d = objects[obj];

                if(d.confidence > thr)
                {
                    float b = d.confidence * 1.5f;

                    std::stringstream conf(std::stringstream::in | std::stringstream::out);
                    conf << d.confidence;

                    cv::rectangle(frame, cv::Rect((int)d.x, (int)d.y, (int)d.w, (int)d.h), cv::Scalar(b, 0, 255 - b, 255), 2);
                    cv::putText(frame, conf.str() , cv::Point((int)d.x + 10, (int)d.y - 5),1, 1.1, cv::Scalar(25, 133, 255, 0), 1, cv::LINE_AA);

                    if (wf)
                        myfile << d.x << "," <<  d.y << "," << d.w << "," <<  d.h << "," << d.confidence << "\n";
                }
            }
        }

        if (wi) cv::imwrite(frame_sourse + ".dt.png", frame);
        if (wf) myfile.close();

        if (si)
        {
            cv::imshow("pedestrian detector", frame);
            cv::waitKey(10);
        }
    }

    if (si) cv::waitKey(0);
    return 0;
}

void filter_rects(const std::vector<cv::Rect>& candidates, std::vector<cv::Rect>& objects)
{
    size_t i, j;
    for (i = 0; i < candidates.size(); ++i)
    {
        cv::Rect r = candidates[i];

        for (j = 0; j < candidates.size(); ++j)
            if (j != i && (r & candidates[j]) == r)
                break;

        if (j == candidates.size())
            objects.push_back(r);
    }
}
