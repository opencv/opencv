//============================================================================
// Name        : retinademo.cpp
// Author      : Alexandre Benoit, benoit.alexandre.vision@gmail.com
// Version     : 0.1
// Copyright   : LISTIC/GIPSA French Labs, july 2011
// Description : Gipsa/LISTIC Labs retina demo in C++, Ansi-style
//============================================================================

#include <iostream>
#include <cstring>

#include "opencv2/opencv.hpp"

static void help(std::string errorMessage)
{
    std::cout<<"Program init error : "<<errorMessage<<std::endl;
    std::cout<<"\nProgram call procedure : retinaDemo [processing mode] [Optional : media target] [Optional LAST parameter: \"log\" to activate retina log sampling]"<<std::endl;
    std::cout<<"\t[processing mode] :"<<std::endl;
    std::cout<<"\t -image : for still image processing"<<std::endl;
    std::cout<<"\t -video : for video stream processing"<<std::endl;
    std::cout<<"\t[Optional : media target] :"<<std::endl;
    std::cout<<"\t if processing an image or video file, then, specify the path and filename of the target to process"<<std::endl;
    std::cout<<"\t leave empty if processing video stream coming from a connected video device"<<std::endl;
    std::cout<<"\t[Optional : activate retina log sampling] : an optional last parameter can be specified for retina spatial log sampling"<<std::endl;
    std::cout<<"\t set \"log\" without quotes to activate this sampling, output frame size will be divided by 4"<<std::endl;
    std::cout<<"\nExamples:"<<std::endl;
    std::cout<<"\t-Image processing : ./retinaDemo -image lena.jpg"<<std::endl;
    std::cout<<"\t-Image processing with log sampling : ./retinaDemo -image lena.jpg log"<<std::endl;
    std::cout<<"\t-Video processing : ./retinaDemo -video myMovie.mp4"<<std::endl;
    std::cout<<"\t-Live video processing : ./retinaDemo -video"<<std::endl;
    std::cout<<"\nPlease start again with new parameters"<<std::endl;
}

int main(int argc, char* argv[]) {
    // welcome message
    std::cout<<"****************************************************"<<std::endl;
    std::cout<<"* Retina demonstration : demonstrates the use of is a wrapper class of the Gipsa/Listic Labs retina model."<<std::endl;
    std::cout<<"* This retina model allows spatio-temporal image processing (applied on still images, video sequences)."<<std::endl;
    std::cout<<"* As a summary, these are the retina model properties:"<<std::endl;
    std::cout<<"* => It applies a spectral whithening (mid-frequency details enhancement)"<<std::endl;
    std::cout<<"* => high frequency spatio-temporal noise reduction"<<std::endl;
    std::cout<<"* => low frequency luminance to be reduced (luminance range compression)"<<std::endl;
    std::cout<<"* => local logarithmic luminance compression allows details to be enhanced in low light conditions\n"<<std::endl;
    std::cout<<"* for more information, reer to the following papers :"<<std::endl;
    std::cout<<"* Benoit A., Caplier A., Durette B., Herault, J., \"USING HUMAN VISUAL SYSTEM MODELING FOR BIO-INSPIRED LOW LEVEL IMAGE PROCESSING\", Elsevier, Computer Vision and Image Understanding 114 (2010), pp. 758-773, DOI: http://dx.doi.org/10.1016/j.cviu.2010.01.011"<<std::endl;
    std::cout<<"* Vision: Images, Signals and Neural Networks: Models of Neural Processing in Visual Perception (Progress in Neural Processing),By: Jeanny Herault, ISBN: 9814273686. WAPI (Tower ID): 113266891."<<std::endl;
    std::cout<<"* => reports comments/remarks at benoit.alexandre.vision@gmail.com"<<std::endl;
    std::cout<<"* => more informations and papers at : http://sites.google.com/site/benoitalexandrevision/"<<std::endl;
    std::cout<<"****************************************************"<<std::endl;
    std::cout<<" NOTE : this program generates the default retina parameters file 'RetinaDefaultParameters.xml'"<<std::endl;
    std::cout<<" => you can use this to fine tune parameters and load them if you save to file 'RetinaSpecificParameters.xml'"<<std::endl;

    // basic input arguments checking
    if (argc<2)
    {
        help("bad number of parameter");
        return -1;
    }

    bool useLogSampling = !strcmp(argv[argc-1], "log"); // check if user wants retina log sampling processing

    std::string inputMediaType=argv[1];

    // declare the retina input buffer... that will be fed differently in regard of the input media
    cv::Mat inputFrame;
    cv::VideoCapture videoCapture; // in case a video media is used, its manager is declared here

    //////////////////////////////////////////////////////////////////////////////
    // checking input media type (still image, video file, live video acquisition)
    if (!strcmp(inputMediaType.c_str(), "-image") && argc >= 3)
    {
        std::cout<<"RetinaDemo: processing image "<<argv[2]<<std::endl;
        // image processing case
        inputFrame = cv::imread(std::string(argv[2]), 1); // load image in RGB mode
    }else
        if (!strcmp(inputMediaType.c_str(), "-video"))
        {
            if (argc == 2 || (argc == 3 && useLogSampling)) // attempt to grab images from a video capture device
            {
                videoCapture.open(0);
            }else// attempt to grab images from a video filestream
            {
                std::cout<<"RetinaDemo: processing video stream "<<argv[2]<<std::endl;
                videoCapture.open(argv[2]);
            }

            // grab a first frame to check if everything is ok
            videoCapture>>inputFrame;
        }else
        {
            // bad command parameter
            help("bad command parameter");
            return -1;
        }

    if (inputFrame.empty())
    {
        help("Input media could not be loaded, aborting");
        return -1;
    }


    //////////////////////////////////////////////////////////////////////////////
    // Program start in a try/catch safety context (Retina may throw errors)
    try
    {
        // create a retina instance with default parameters setup, uncomment the initialisation you wanna test
        cv::Ptr<cv::Retina> myRetina;

        // if the last parameter is 'log', then activate log sampling (favour foveal vision and subsamples peripheral vision)
        if (useLogSampling)
                {
                        myRetina = new cv::Retina(inputFrame.size(), true, cv::RETINA_COLOR_BAYER, true, 2.0, 10.0);
                }
        else// -> else allocate "classical" retina :
            myRetina = new cv::Retina(inputFrame.size());

        // save default retina parameters file in order to let you see this and maybe modify it and reload using method "setup"
        myRetina->write("RetinaDefaultParameters.xml");

        // load parameters if file exists
        myRetina->setup("RetinaSpecificParameters.xml");
        myRetina->clearBuffers();

        // declare retina output buffers
        cv::Mat retinaOutput_parvo;
        cv::Mat retinaOutput_magno;

        // processing loop with stop condition
        bool continueProcessing=true; // FIXME : not yet managed during process...
        while(continueProcessing)
        {
            // if using video stream, then, grabbing a new frame, else, input remains the same
            if (videoCapture.isOpened())
                videoCapture>>inputFrame;

            // run retina filter
            myRetina->run(inputFrame);
            // Retrieve and display retina output
            myRetina->getParvo(retinaOutput_parvo);
            myRetina->getMagno(retinaOutput_magno);
            cv::imshow("retina input", inputFrame);
            cv::imshow("Retina Parvo", retinaOutput_parvo);
            cv::imshow("Retina Magno", retinaOutput_magno);
            cv::waitKey(10);
        }
    }catch(cv::Exception e)
    {
        std::cerr<<"Error using Retina : "<<e.what()<<std::endl;
    }

    // Program end message
    std::cout<<"Retina demo end"<<std::endl;

    return 0;
}

