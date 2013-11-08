#include <iostream>
#include <cstring>

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/ocl.hpp"
#include "opencv2/bioinspired.hpp"

using namespace cv;
using namespace cv::ocl;
using namespace std;

const int total_loop_count = 50;

static void help(CommandLineParser cmd, const String& errorMessage)
{
    cout << errorMessage << endl;
    cout << "Avaible options:" << endl;
    cmd.printMessage();
}

int main(int argc, char* argv[])
{
    //set this to save kernel compile time from second time you run
    ocl::setBinaryDiskCache();
    const char* keys =
        "{ h   | help     | false           | print help message }"
        "{ c   | cpu      | false           | use cpu (original version) or gpu(OpenCL) to process the image }"
        "{ i   | image    | cat.jpg         | specify the input image }";

    CommandLineParser cmd(argc, argv, keys);

    if(cmd.get<bool>("help"))
    {
        help(cmd, "Usage: ./retina_ocl [options]");
        return EXIT_FAILURE;
    }

    String fname = cmd.get<String>("i");
    bool useCPU = cmd.get<bool>("c");

    cv::Mat input = imread(fname);
    if(input.empty())
    {
        help(cmd, "Error opening: " + fname);
        return EXIT_FAILURE;
    }
    //////////////////////////////////////////////////////////////////////////////
    // Program start in a try/catch safety context (Retina may throw errors)
    try
    {
        // create a retina instance with default parameters setup, uncomment the initialisation you wanna test
        cv::Ptr<cv::bioinspired::Retina> oclRetina;
        cv::Ptr<cv::bioinspired::Retina> retina;
        // declare retina output buffers
        cv::ocl::oclMat retina_parvo_ocl;
        cv::ocl::oclMat retina_magno_ocl;
        cv::Mat retina_parvo;
        cv::Mat retina_magno;

        if(useCPU)
        {
            retina = cv::bioinspired::createRetina(input.size());
            retina->clearBuffers();
        }
        else
        {
            oclRetina = cv::bioinspired::createRetina_OCL(input.size());
            oclRetina->clearBuffers();
        }

        int64 temp_time = 0, total_time = 0;

        int loop_counter = 0;
        for(; loop_counter <= total_loop_count; ++loop_counter)
        {
            if(useCPU)
            {
                temp_time = cv::getTickCount();
                retina->run(input);
                retina->getParvo(retina_parvo);
                retina->getMagno(retina_magno);
            }
            else
            {
                cv::ocl::oclMat input_ocl(input);
                temp_time = cv::getTickCount();
                oclRetina->run(input_ocl);
                oclRetina->getParvo(retina_parvo_ocl);
                oclRetina->getMagno(retina_magno_ocl);
            }
            // will not count the first loop, which is considered as warm-up period
            if(loop_counter > 0)
            {
                temp_time = (cv::getTickCount() - temp_time);
                total_time += temp_time;
                printf("Frame id %2d: %3.4fms\n", loop_counter, (double)temp_time / cv::getTickFrequency() * 1000.0);
            }
            if(!useCPU)
            {
                retina_parvo = retina_parvo_ocl;
                retina_magno = retina_magno_ocl;
            }
            cv::imshow("retina input", input);
            cv::imshow("Retina Parvo", retina_parvo);
            cv::imshow("Retina Magno", retina_magno);
            cv::waitKey(10);
        }
        printf("Average: %.4fms\n", (double)total_time / total_loop_count / cv::getTickFrequency() * 1000.0);
    }
    catch(cv::Exception e)
    {
        std::cerr << "Error using Retina : " << e.what() << std::endl;
    }
    // Program end message
    std::cout << "Retina demo end" << std::endl;
    return EXIT_SUCCESS;
}
