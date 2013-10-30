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

using namespace std;
using namespace cv;

static void help(string errorMessage)
{
    cout<<"Program init error : "<<errorMessage<<endl;
    cout<<"\nProgram call procedure : retinaDemo [processing mode] [Optional : media target] [Optional LAST parameter: \"log\" to activate retina log sampling]"<<endl;
    cout<<"\t[processing mode] :"<<endl;
    cout<<"\t -image : for still image processing"<<endl;
    cout<<"\t -video : for video stream processing"<<endl;
    cout<<"\t[Optional : media target] :"<<endl;
    cout<<"\t if processing an image or video file, then, specify the path and filename of the target to process"<<endl;
    cout<<"\t leave empty if processing video stream coming from a connected video device"<<endl;
    cout<<"\t[Optional : activate retina log sampling] : an optional last parameter can be specified for retina spatial log sampling"<<endl;
    cout<<"\t set \"log\" without quotes to activate this sampling, output frame size will be divided by 4"<<endl;
    cout<<"\nExamples:"<<endl;
    cout<<"\t-Image processing : ./retinaDemo -image lena.jpg"<<endl;
    cout<<"\t-Image processing with log sampling : ./retinaDemo -image lena.jpg log"<<endl;
    cout<<"\t-Video processing : ./retinaDemo -video myMovie.mp4"<<endl;
    cout<<"\t-Live video processing : ./retinaDemo -video"<<endl;
    cout<<"\nPlease start again with new parameters"<<endl;
}

int main(int argc, char* argv[]) {
    // welcome message
    cout<<"****************************************************"<<endl;
    cout<<"* Retina demonstration : demonstrates the use of is a wrapper class of the Gipsa/Listic Labs retina model."<<endl;
    cout<<"* This retina model allows spatio-temporal image processing (applied on still images, video sequences)."<<endl;
    cout<<"* As a summary, these are the retina model properties:"<<endl;
    cout<<"* => It applies a spectral whithening (mid-frequency details enhancement)"<<endl;
    cout<<"* => high frequency spatio-temporal noise reduction"<<endl;
    cout<<"* => low frequency luminance to be reduced (luminance range compression)"<<endl;
    cout<<"* => local logarithmic luminance compression allows details to be enhanced in low light conditions\n"<<endl;
    cout<<"* for more information, reer to the following papers :"<<endl;
    cout<<"* Benoit A., Caplier A., Durette B., Herault, J., \"USING HUMAN VISUAL SYSTEM MODELING FOR BIO-INSPIRED LOW LEVEL IMAGE PROCESSING\", Elsevier, Computer Vision and Image Understanding 114 (2010), pp. 758-773, DOI: http://dx.doi.org/10.1016/j.cviu.2010.01.011"<<endl;
    cout<<"* Vision: Images, Signals and Neural Networks: Models of Neural Processing in Visual Perception (Progress in Neural Processing),By: Jeanny Herault, ISBN: 9814273686. WAPI (Tower ID): 113266891."<<endl;
    cout<<"* => reports comments/remarks at benoit.alexandre.vision@gmail.com"<<endl;
    cout<<"* => more informations and papers at : http://sites.google.com/site/benoitalexandrevision/"<<endl;
    cout<<"****************************************************"<<endl;
    cout<<" NOTE : this program generates the default retina parameters file 'RetinaDefaultParameters.xml'"<<endl;
    cout<<" => you can use this to fine tune parameters and load them if you save to file 'RetinaSpecificParameters.xml'"<<endl;

    // basic input arguments checking
    if (argc<2)
    {
        help("bad number of parameter");
        return -1;
    }

    bool useLogSampling = !strcmp(argv[argc-1], "log"); // check if user wants retina log sampling processing

    string inputMediaType=argv[1];

    // declare the retina input buffer... that will be fed differently in regard of the input media
    Mat inputFrame;
    VideoCapture videoCapture; // in case a video media is used, its manager is declared here

    //////////////////////////////////////////////////////////////////////////////
    // checking input media type (still image, video file, live video acquisition)
    if (!strcmp(inputMediaType.c_str(), "-image") && argc >= 3)
    {
        cout<<"RetinaDemo: processing image "<<argv[2]<<endl;
        // image processing case
        inputFrame = imread(string(argv[2]), 1); // load image in RGB mode
    }else
        if (!strcmp(inputMediaType.c_str(), "-video"))
        {
            if (argc == 2 || (argc == 3 && useLogSampling)) // attempt to grab images from a video capture device
            {
                videoCapture.open(0);
            }else// attempt to grab images from a video filestream
            {
                cout<<"RetinaDemo: processing video stream "<<argv[2]<<endl;
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
        Ptr<Retina> myRetina;

        // if the last parameter is 'log', then activate log sampling (favour foveal vision and subsamples peripheral vision)
        if (useLogSampling)
                {
                        myRetina = new Retina(inputFrame.size(), true, RETINA_COLOR_BAYER, true, 2.0, 10.0);
                }
        else// -> else allocate "classical" retina :
            myRetina = new Retina(inputFrame.size());

        // save default retina parameters file in order to let you see this and maybe modify it and reload using method "setup"
        myRetina->write("RetinaDefaultParameters.xml");

        // load parameters if file exists
        myRetina->setup("RetinaSpecificParameters.xml");
        myRetina->clearBuffers();

        // declare retina output buffers
        Mat retinaOutput_parvo;
        Mat retinaOutput_magno;

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
            imshow("retina input", inputFrame);
            imshow("Retina Parvo", retinaOutput_parvo);
            imshow("Retina Magno", retinaOutput_magno);
            waitKey(10);
        }
    }catch(Exception e)
    {
        cerr<<"Error using Retina : "<<e.what()<<endl;
    }

    // Program end message
    cout<<"Retina demo end"<<endl;

    return 0;
}
